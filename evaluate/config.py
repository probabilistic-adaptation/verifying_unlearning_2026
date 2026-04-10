parameters = {
    # Random Seeds
    "random_seeds": [19, 12, 25],
    # Model and Dataset Configuration
    "arch": "resnet18",
    "dataset": "cifar10",
    "origin_model_path": "0model_SA_best.pth.tar",  # Changed to just filename without path

    # Training Parameters
    "lr": 0.1,
    # "epochs_train": 20,
    "epochs_train": 4, # use to test config changes b/c shorter run time

    # Unlearning Configuration
    # "unlearn_methods": ["retrain", "FT", "GA", "wfisher"],
    "unlearn_methods": ["retrain", "GA"],
    "class_to_replace": 0,
    # "forgetting_data_amount": 20250, # 5% is 2250, 10% is 4500, 15% is 6750, 30% is 13500, 45% is 20250

    # Method-specific Parameters
    "epochs_for_unlearning": {
        # "FT": 20,
        # "GA": 20,
        # "retrain": 20
        # "FT": 4, # use to test config changes b/c shorter run time
        "GA": 4, # use to test config changes b/c shorter run time
        "retrain": 4 # use to test config changes b/c shorter run time
    },
    "learning_rate_for_unlearning": {
        "FT": 0.01,
        "GA": 0.0001,
        "retrain": 0.01
    },

    # Additional Parameters
    "alpha": 0.5
}