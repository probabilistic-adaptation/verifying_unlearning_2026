"""
Evaluates every checkpoint saved during pretraining (see master_pretraining.ipynb) on the
train and test splits of the dataset it was trained on, and writes the results out in the
same format run_pretraining() itself produces -- one

    results/seed_<SEED>/pretraining/<DATASET>_<MODEL_CLASS>_<num_epochs>_epochs/results_<i>.json

per checkpoint, loadable as-is by visualize_pretraining_results.ipynb's gather_results().

Meant for checkpoints that exist but have no (or stale) results jsons -- e.g. pretraining runs
whose logging didn't complete, or where accuracy needs to be recomputed after a bug fix.

Usage:
    python pretraining_results.py
"""

import glob
import json
import os
import re

import torch
import torch.nn as nn

from data.dataloaders import load_dataloaders_for_experiment, unmark_dataset
from data.utils import setup_seed
from models.archs.utils import init_model
from trainer.utils import init_folder_if_not_exists
from trainer.val import validate


# ============================================================
# CONFIG
# ============================================================

SEED = 4
DATASET = "CIFAR10"
MODEL_CLASS = "ResNet"
BATCH_SIZE = 4096
NUM_WORKERS = 4
PRINT_FREQ = 3

CHECKPOINTS_FOLDER = f"models/model_checkpoints/seed_{SEED}/pretrained"
RESULTS_FOLDER = f"results/seed_{SEED}/pretraining"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def infer_num_classes(checkpoint_path, device):
    """
    Reads the output dimension straight off the checkpoint's classification head, so
    checkpoints trained with different num_classes (e.g. 10 for CIFAR10 vs. 1000 for an
    ImageNet-style head) can sit side by side in the same pretraining folder without a fixed
    NUM_CLASSES config having to match every one of them.

    Every arch in models/archs/utils.py assigns its classification head (fc/classifier/head/
    final) last in __init__, so state_dict() -- which walks submodules in registration order --
    always ends on that head's weight; its out_features is num_classes.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    weight_keys = [k for k, v in state_dict.items() if k.endswith(".weight") and v.dim() == 2]
    if not weight_keys:
        raise ValueError(f"couldn't find a classifier weight (2D '*.weight' entry) in {checkpoint_path}")

    return state_dict[weight_keys[-1]].shape[0]


def discover_epoch_folders(checkpoints_folder, dataset, model_class):
    """
    Every '<dataset>_<model_class>_<N>_epochs' subfolder of checkpoints_folder, paired with
    the N it names. Skips subfolders for other datasets/architectures that might also live
    under the same seed's checkpoint directory.
    """
    pattern = re.compile(rf"^{re.escape(dataset)}_{re.escape(model_class)}_(\d+)_epochs$")

    folders = []
    for name in sorted(os.listdir(checkpoints_folder)):
        path = os.path.join(checkpoints_folder, name)
        if not os.path.isdir(path):
            continue
        m = pattern.match(name)
        if not m:
            continue
        folders.append((int(m.group(1)), name, path))
    return sorted(folders)


def main():
    device = get_device()
    criterion = nn.CrossEntropyLoss()

    print(f"[pretraining_results] seed={SEED}, dataset={DATASET}, model_class={MODEL_CLASS}, device={device}")

    setup_seed(SEED)
    train_loader, _, test_loader = load_dataloaders_for_experiment(
        name=DATASET,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=SEED,
        replace_type="class",
        value_to_replace=5,
        val=False,
        only_mark = True
    )

    unmark_dataset(train_loader.dataset)

    epoch_folders = discover_epoch_folders(CHECKPOINTS_FOLDER, DATASET, MODEL_CLASS)
    if not epoch_folders:
        raise SystemExit(f"no '{DATASET}_{MODEL_CLASS}_*_epochs' folders under {CHECKPOINTS_FOLDER}")
    print(f"[pretraining_results] found {len(epoch_folders)} epoch settings: {[n for n, _, _ in epoch_folders]}\n")

    for num_epochs, folder_name, folder_path in epoch_folders:
        checkpoint_paths = sorted(glob.glob(os.path.join(folder_path, "*.pth")))
        if not checkpoint_paths:
            print(f"[pretraining_results] {folder_name}: no checkpoints, skipping")
            continue

        results_folder = init_folder_if_not_exists(os.path.join(RESULTS_FOLDER, folder_name))

        for run, checkpoint_path in enumerate(checkpoint_paths, start=1):
            num_classes = infer_num_classes(checkpoint_path, device)
            print(f"[pretraining_results] {folder_name} run {run}/{len(checkpoint_paths)}: {checkpoint_path} (num_classes={num_classes})")

            model = init_model(
                model_class=MODEL_CLASS, num_classes=num_classes,
                checkpoint_path=checkpoint_path, device=device,
            ).to(device)

            train_out = validate(train_loader, model, criterion, print_freq=PRINT_FREQ, device=device, w_and_b=False)
            test_out = validate(test_loader, model, criterion, print_freq=PRINT_FREQ, device=device, w_and_b=False)

            print(f"  train_acc={train_out['avg_acc']:.4f}  test_acc={test_out['avg_acc']:.4f}")

            results = {
                "num_epochs": num_epochs,
                "train_acc": train_out["avg_acc"],
                "test_acc": test_out["avg_acc"],
            }
            with open(os.path.join(results_folder, f"results_{run}.json"), "w") as f:
                json.dump(results, f, indent=4)

            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    print("\n[pretraining_results] done.")


if __name__ == "__main__":
    main()
