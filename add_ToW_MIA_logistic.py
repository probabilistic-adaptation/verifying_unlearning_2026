"""
Add a "ToW_MIA_logistic" metric to every unlearned checkpoint's results json.

This is exactly the ToW_MIA metric computed in
evaluation/utils.py::measure_solo_and_comparison_metrics, but with the MIA efficacy
term taken from the *logistic-regression* MIA instead of the entropy-threshold MIA:

    da_retain = |acc_retain(unlearned) - acc_retain(retrain)| / 100
    da_test   = |acc_test(unlearned)   - acc_test(retrain)|   / 100
    dm        = |efficacy(unlearned)   - efficacy(retrain)|      # logistic_regression_MIA
    ToW_MIA_logistic = (1 - dm) * (1 - da_retain) * (1 - da_test)

Every unlearned checkpoint is compared against retrain_run_1 of the same seed (the
same reference measure_solo_and_comparison_metrics uses for the stored ToW_MIA).
Everything needed is already in the results jsons, so no models / *_out.pth are loaded.

The result is written back into each unlearned checkpoint's own results json under
the top-level "ToW_MIA_logistic" key, next to the existing "ToW" / "ToW_MIA".

Usage:
    python add_ToW_MIA_logistic.py                     # every results/seed_* folder
    python add_ToW_MIA_logistic.py --seed 4            # just one seed
    python add_ToW_MIA_logistic.py --retrain-run 1     # reference retrain run (default 1)
    python add_ToW_MIA_logistic.py --dry-run           # compute + print, don't write
"""

import argparse
import glob
import json
import os

RESULTS_DIR = "results"


def compute_tow_mia_logistic(unlearned, retrain):
    """`unlearned` / `retrain` are parsed results-json dicts."""
    da_retain = abs(unlearned["acc"]["retain"] - retrain["acc"]["retain"]) / 100
    da_test = abs(unlearned["acc"]["test"] - retrain["acc"]["test"]) / 100
    dm = abs(
        unlearned["logistic_regression_MIA"]["efficacy"]
        - retrain["logistic_regression_MIA"]["efficacy"]
    )
    return (1 - dm) * (1 - da_retain) * (1 - da_test)


def find_retrain_json(seed_dir, retrain_run):
    matches = glob.glob(os.path.join(seed_dir, "retrain", f"retrain_run_{retrain_run}_*.json"))
    matches = [m for m in matches if not m.endswith("_out.pth")]
    return sorted(matches)[0] if matches else None


def iter_unlearn_jsons(seed_dir):
    unlearn_dir = os.path.join(seed_dir, "unlearn")
    if not os.path.isdir(unlearn_dir):
        return
    for path in sorted(glob.glob(os.path.join(unlearn_dir, "run_*", "*", "epoch_*", "*.json"))):
        yield path


def process_seed(seed_dir, retrain_run, dry_run):
    retrain_path = find_retrain_json(seed_dir, retrain_run)
    if retrain_path is None:
        print(f"[{seed_dir}] no retrain_run_{retrain_run} json, skipping seed")
        return
    with open(retrain_path) as f:
        retrain = json.load(f)
    print(f"[{seed_dir}] reference: {retrain_path}")

    for json_path in iter_unlearn_jsons(seed_dir):
        with open(json_path) as f:
            results = json.load(f)

        if "acc" not in results or "logistic_regression_MIA" not in results:
            print(f"  {json_path}: missing acc / logistic_regression_MIA, skipping")
            continue

        tow = compute_tow_mia_logistic(results, retrain)
        rel = os.path.relpath(json_path, seed_dir)
        tag = "" if dry_run else " -> written"
        print(f"  {rel}: ToW_MIA_logistic = {tow:.6f} (ToW_MIA = {results.get('ToW_MIA', float('nan')):.6f}){tag}")

        if not dry_run:
            results["ToW_MIA_logistic"] = tow
            with open(json_path, "w") as f:
                json.dump(results, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="one seed (default: all results/seed_*)")
    parser.add_argument("--retrain-run", type=int, default=1, help="retrain run used as reference")
    parser.add_argument("--dry-run", action="store_true", help="compute and print but don't modify jsons")
    args = parser.parse_args()

    if args.seed is not None:
        seed_dirs = [os.path.join(RESULTS_DIR, f"seed_{args.seed}")]
    else:
        seed_dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "seed_*")))

    for seed_dir in seed_dirs:
        if not os.path.isdir(seed_dir):
            print(f"[{seed_dir}] not a directory, skipping")
            continue
        process_seed(seed_dir, args.retrain_run, args.dry_run)


if __name__ == "__main__":
    main()
