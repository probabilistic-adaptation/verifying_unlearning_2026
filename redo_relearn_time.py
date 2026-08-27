"""
Re-executes the relearn-time metric (evaluation/relearn_time.py) for every unlearned model
checkpoint under one seed, and writes the refreshed "relearn_time" block back into that
checkpoint's existing results JSON in place.

Reconstructs the forget/retain dataloaders and base model output the same way
run_experiment() does in master_unlearning.ipynb, using the seed's saved
experiment_config.json for every other setting (model class, dataset, training
hyperparams, forget_set_type/item).
"""

import glob
import hashlib
import json
import os
import re

import torch

from data.dataloaders import load_dataloaders_for_experiment
from data.utils import setup_seed, split_forget_retain
from evaluation.relearn_time import relearn_time
from models.archs.utils import init_model


# ============================================================
# CONFIG
# ============================================================

SEED = 4

RESULTS_FOLDER = f"results/seed_{SEED}"
CHECKPOINT_FOLDER = f"models/model_checkpoints/seed_{SEED}"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def discover_unlearn_checkpoints(checkpoint_folder, forget_set_type, unlearning_item):
    """
    Every top-level '<method>_epoch_<k>_<forget_set_type>_<unlearning_item>[_run_<run>].pth'
    file in checkpoint_folder (i.e. unlearned-model checkpoints -- NOT the 'pretrained' or
    'retrain_from_scratch' subfolders), parsed into its (method, epoch, run) components.

    forget_set_type/unlearning_item are taken from the seed's experiment_config.json rather
    than parsed out of the filename, since unlearning_item can itself contain an underscore or
    decimal point (e.g. percent-forgetting's "random_0.1"), which would make that split
    ambiguous. The "_run_<run>" suffix is optional and defaults to run=1 -- some older
    checkpoints were saved before per-run filenames were introduced.
    """
    suffix = re.escape(f"{forget_set_type}_{unlearning_item}")
    pattern = re.compile(rf"^(?P<method>.+)_epoch_(?P<epoch>\d+)_{suffix}(?:_run_(?P<run>\d+))?\.pth$")

    entries = []
    for pth_path in sorted(glob.glob(os.path.join(checkpoint_folder, "*.pth"))):
        m = pattern.match(os.path.basename(pth_path))
        if not m:
            print(f"[redo_relearn_time] skipping unrecognized checkpoint filename: {pth_path}")
            continue
        entries.append({
            "pth_path": pth_path,
            "method": m.group("method"),
            "epoch": int(m.group("epoch")),
            "run": int(m.group("run")) if m.group("run") else 1,
        })
    return entries


def result_json_path(results_folder, entry, forget_set_type, unlearning_item):
    return os.path.join(
        results_folder, "unlearn", f"run_{entry['run']}", entry["method"],
        f"epoch_{entry['epoch']}", f"{forget_set_type}_{unlearning_item}.json"
    )


def deterministic_seed(*parts):
    """A stable, reproducible int seed derived from `parts` -- doesn't need to match the
    original run's seed, just needs to be deterministic across reruns of this script."""
    digest = hashlib.md5("_".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest, 16) % (2**31)


def main():
    device = get_device()

    with open(os.path.join(RESULTS_FOLDER, "experiment_config.json")) as fh:
        config = json.load(fh)

    model_class = config["model_class"]
    num_classes = config["data"]["num_classes"]
    forget_set_type = config["unlearning_type"]
    unlearning_item = config["data"]["item_to_unlearn"]
    training_hp = config["training"]

    print(f"[redo_relearn_time] seed={SEED}, device={device}, model_class={model_class}, "
          f"forget_set_type={forget_set_type}, unlearning_item={unlearning_item}")

    # ---- rebuild the forget/retain loaders exactly as run_experiment() does ----
    setup_seed(SEED)
    marked_train_loader, _, _ = load_dataloaders_for_experiment(
        name=config["data"]["dataset"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        seed=SEED,
        replace_type=forget_set_type,
        value_to_replace=unlearning_item,
        only_mark=True,
        val=False,
    )
    forget_loader, retain_loader = split_forget_retain(
        marked_train_loader, batch_size=config["data"]["batch_size"],
        shuffle=True, num_workers=config["data"]["num_workers"],
    )
    dataloaders = {"forget": forget_loader, "retain": retain_loader}

    # ---- base model output (relearn_time's target forget-set loss) ----
    base_out_path = os.path.join(RESULTS_FOLDER, "base", f"base_{forget_set_type}_{unlearning_item}_out.pth")
    base_out = torch.load(base_out_path, weights_only=False)

    entries = discover_unlearn_checkpoints(CHECKPOINT_FOLDER, forget_set_type, unlearning_item)
    print(f"[redo_relearn_time] found {len(entries)} unlearned-model checkpoints under {CHECKPOINT_FOLDER}\n")

    for i, entry in enumerate(entries, start=1):
        json_path = result_json_path(RESULTS_FOLDER, entry, forget_set_type, unlearning_item)
        if not os.path.exists(json_path):
            print(f"[redo_relearn_time] ({i}/{len(entries)}) skipping {entry['pth_path']}: no matching results file at {json_path}")
            continue

        print(f"[redo_relearn_time] ({i}/{len(entries)}) {entry['method']} epoch {entry['epoch']} run {entry['run']}")

        model = init_model(
            model_class=model_class, num_classes=num_classes,
            checkpoint_path=entry["pth_path"], device=device,
        ).to(device)
        model.eval()

        seed = deterministic_seed(SEED, entry["method"], entry["epoch"], entry["run"])
        new_relearn_time = relearn_time(
            model=model,
            dataloaders=dataloaders,
            base_out=base_out,
            model_class=model_class,
            num_classes=num_classes,
            device=device,
            seed=seed,
            training_hp=training_hp,
        )

        with open(json_path) as fh:
            results = json.load(fh)
        results["relearn_time"] = new_relearn_time
        with open(json_path, "w") as fh:
            json.dump(results, fh, indent=4)

        print(f"[redo_relearn_time] updated {json_path}: avg_epochs={new_relearn_time['avg_epochs']:.2f}\n")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print("[redo_relearn_time] done.")


if __name__ == "__main__":
    main()
