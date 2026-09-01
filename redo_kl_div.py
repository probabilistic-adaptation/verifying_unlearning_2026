"""
Recompute the KL-divergence metric for every unlearned checkpoint of a given seed.

The kl_div_metric in evaluation/distances.py wasn't behaving as expected, so this
script recomputes it directly, following the pattern:

    unlearned_all = cat([unlearned forget probs, unlearned retain probs])
    retrain_all   = cat([retrain   forget probs, retrain   retain probs])
    kl = kl_divergence(retrain_all, unlearned_all).mean().item()   # retrained first (Chien et al.)

Every unlearned checkpoint is compared to retrain_run_1 of that seed.

The result is written back into each unlearned checkpoint's own results json under
    ["outputs"]["retrain_vs_unlearned"]["full_train"]["kl_divergence"]

Usage:
    python redo_kl_div.py                # seed 4
    python redo_kl_div.py --seed 5       # another seed
    python redo_kl_div.py --seed 4 --dry-run
"""

import argparse
import glob
import json
import os

import torch

from evaluation.distances import kl_divergence

RESULTS_DIR = "results"


def all_probs(out):
    """Concatenate forget + retain probability blocks (n_data x n_classes)."""
    return torch.cat([out["forget"]["probs"], out["retain"]["probs"]], dim=0)


def compute_kl(retrain_out, unlearned_out):
    retrain_all = all_probs(retrain_out)
    unlearned_all = all_probs(unlearned_out)
    return kl_divergence(retrain_all, unlearned_all).mean().item()


def find_retrain_out(seed_dir):
    matches = glob.glob(os.path.join(seed_dir, "retrain", "retrain_run_1_*_out.pth"))
    if not matches:
        return None
    return sorted(matches)[0]


def write_metric(results_json_path, kl):
    with open(results_json_path) as f:
        results = json.load(f)

    node = results.setdefault("outputs", {}).setdefault("retrain_vs_unlearned", {})
    node.setdefault("full_train", {})["kl_divergence"] = kl

    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute and print but do not modify the results jsons",
    )
    args = parser.parse_args()

    seed_dir = os.path.join(RESULTS_DIR, f"seed_{args.seed}")
    unlearn_dir = os.path.join(seed_dir, "unlearn")
    if not os.path.isdir(unlearn_dir):
        raise SystemExit(f"no unlearn directory at {unlearn_dir}")

    retrain_path = find_retrain_out(seed_dir)
    if retrain_path is None:
        raise SystemExit(f"no retrain_run_1 out file under {seed_dir}/retrain")
    print(f"reference: {retrain_path}\n")
    retrain_out = torch.load(retrain_path, weights_only=False)

    for run_name in sorted(os.listdir(unlearn_dir)):
        run_path = os.path.join(unlearn_dir, run_name)
        if not os.path.isdir(run_path):
            continue

        for method_name in sorted(os.listdir(run_path)):
            method_path = os.path.join(run_path, method_name)
            if not os.path.isdir(method_path):
                continue

            for epoch_name in sorted(os.listdir(method_path)):
                epoch_path = os.path.join(method_path, epoch_name)
                if not os.path.isdir(epoch_path):
                    continue

                out_files = glob.glob(os.path.join(epoch_path, "*_out.pth"))
                if not out_files:
                    print(f"[{run_name}/{method_name}/{epoch_name}] no *_out.pth, skipping")
                    continue

                out_file = sorted(out_files)[0]
                results_json_path = out_file[: -len("_out.pth")] + ".json"
                if not os.path.isfile(results_json_path):
                    print(f"[{run_name}/{method_name}/{epoch_name}] no {results_json_path}, skipping")
                    continue

                unlearned_out = torch.load(out_file, weights_only=False)
                kl = compute_kl(retrain_out, unlearned_out)

                tag = "" if args.dry_run else " -> written"
                print(f"{run_name}/{method_name}/{epoch_name}: kl_divergence = {kl:.6f}{tag}")

                if not args.dry_run:
                    write_metric(results_json_path, kl)


if __name__ == "__main__":
    main()
