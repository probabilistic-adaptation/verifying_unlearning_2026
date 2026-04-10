import subprocess
import os
# from unlearning_metrics.evaluate.config import parameters
from config import parameters


# -------------------------------------------------------- #
# Get list of random seeds with a default if not specified #
# -------------------------------------------------------- #

if "random_seeds" not in parameters:
    print("No random seeds specified in config, using default: [42]")
    seeds = [42]
else:
    seeds = parameters["random_seeds"]

# For each seed ...
for seed in seeds:
    print(f"\n=== Starting experiment with seed {seed} ===\n")

    # ... create seed-specific directories
    seed_dir = f"results_seed{seed}"
    if not os.path.exists(seed_dir):
        os.makedirs(seed_dir)

    # ... train a new classifier on a given dataset
    train_command = [
        "python", "../Classification/main_train.py",
        "--save_dir", seed_dir,
        "--arch", parameters["arch"],
        "--dataset", parameters["dataset"],
        "--lr", str(parameters["lr"]),
        "--epochs", str(parameters["epochs_train"]),
        "--seed", str(seed),
        "--train_seed", str(seed)
    ]

    print(f"Running training with seed {seed}: {' '.join(train_command)}")
    subprocess.run(train_command, check=True)

    # ... and update model path for the current seed.
    model_name = parameters["origin_model_path"]  # Now just the filename
    seed_model_path = os.path.join(seed_dir, model_name)  # Full path including seed directory


    # Now, apply a handful of unlearning methods to the model you just trained,
    unlearning_commands = {
        "retrain": [
            "python", "../Classification/main_forget.py",
            #for class forgeting
            "--save_dir", f"{seed_dir}/retrain_{parameters['class_to_replace']}",
            #for amt forgetting
            # "--save_dir", f"{seed_dir}/retrain_{parameters['forgetting_data_amount']}",
            "--model_path", seed_model_path,  # Using full path with seed directory
            "--unlearn", "retrain",
            "--unlearn_epochs", str(parameters["epochs_for_unlearning"]["retrain"]),
            "--unlearn_lr", str(parameters["learning_rate_for_unlearning"]["retrain"]),
            "--seed", str(seed)
        ],
        "GA": [
            "python", "../Classification/main_forget.py",
            #class
            "--save_dir", f"{seed_dir}/ga_{parameters['class_to_replace']}",
            # amt to forget save
            # "--save_dir", f"{seed_dir}/ga_{parameters['forgetting_data_amount']}",
            "--model_path", seed_model_path,  # Using full path with seed directory
            "--unlearn", "ga",
            "--unlearn_epochs", str(parameters["epochs_for_unlearning"]["GA"]),
            "--unlearn_lr", str(parameters["learning_rate_for_unlearning"]["GA"]),
            "--seed", str(seed),
            "--retrain_epochs",str(parameters["epochs_for_unlearning"]["retrain"]),
        ],
        "wfisher": [
            "python", "-u", "../Classification/main_forget.py",
            # save when forgetting class
            "--save_dir", f"{seed_dir}/wfisher_{parameters['class_to_replace']}",
            # save when forgetting amount
            # "--save_dir", f"{seed_dir}/wfisher_{parameters['forgetting_data_amount']}",
            "--model_path", seed_model_path,  # Using full path with seed directory
            "--unlearn", "wfisher",
            "--alpha", str(parameters["alpha"]),
            "--seed", str(seed),
            "--retrain_epochs",str(parameters["epochs_for_unlearning"]["retrain"]),
        ],
        "FT": [
            "python", "-u", "../Classification/main_forget.py",
            # save when forgeting class
            "--save_dir", f"{seed_dir}/ft_{parameters['class_to_replace']}",
            # save when forgetting an amount
            # "--save_dir", f"{seed_dir}/ft_{parameters['forgetting_data_amount']}",
            "--model_path", seed_model_path,  # Using full path with seed directory
            "--unlearn", "ft",
            "--unlearn_epochs", str(parameters["epochs_for_unlearning"]["FT"]),
            "--unlearn_lr", str(parameters["learning_rate_for_unlearning"]["FT"]),
            "--seed", str(seed),
            "--retrain_epochs",str(parameters["epochs_for_unlearning"]["retrain"]),
        ]
    }

    # ... save new unlearning parameters, 
    for key, command in unlearning_commands.items():
        if "forgetting_data_amount" in parameters:
            command.extend(["--num_indexes_to_replace", str(parameters["forgetting_data_amount"])])
        if "class_to_replace" in parameters:
            command.extend(["--class_to_replace", str(parameters["class_to_replace"])])

    # ... create method-specific sub-directories
    for method in parameters["unlearn_methods"]:
        method_dir = f"{seed_dir}/{method.lower()}"
        if not os.path.exists(method_dir):
            os.makedirs(method_dir)

    for method in parameters["unlearn_methods"]:
        if method in unlearning_commands:
            unlearn_command = unlearning_commands[method]
            print(f"Running {method} unlearning with seed {seed}: {' '.join(unlearn_command)}")
            subprocess.run(unlearn_command, check=True)

    # ... and creating a table of results for all methods and metrics.
    create_table_command = [
        "python", "create_table.py",
        "--results_dir", seed_dir
    ]
    print(f"Creating table for seed {seed}: {' '.join(create_table_command)}")
    subprocess.run(create_table_command, check=True)

    print(f"\n=== Completed experiment with seed {seed} ===\n")

print("All experiments completed successfully!")