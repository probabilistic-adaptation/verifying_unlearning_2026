import copy

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from data.utils import setup_seed
from models.archs.utils import init_model
from trainer.utils import train
from trainer.val import naive_validate

MAX_EPOCHS = 100
NUM_RUNS = 3
THRESHOLD = 0.05


def relearn_time(model, dataloaders, base_out, model_class, num_classes, device, seed, training_hp, print_freq=100):
    """
    "Relearn time" (Golatkar et al. 2020a): how many epochs of retraining on the FULL training set
    (forget + retain) it takes an unlearned model's forget-set loss to come back within THRESHOLD
    (relative) of the original, pre-unlearning model's forget-set loss -- i.e.
    `base_out["forget"]["avg_loss"]`.

    Uses the same training protocol as the retrain-from-scratch models (Adam, same lr/weight_decay,
    read from `training_hp`), but WITHOUT the cosine annealing schedule those use -- just a
    constant, small learning rate, per the original relearn-time formulation.

    Repeated NUM_RUNS times, each restarting from `model`'s own weights (so relearning never
    compounds across runs). Returns the average number of epochs across runs; a run that never
    reaches the threshold within MAX_EPOCHS is counted as MAX_EPOCHS (right-censored -- see
    "converged_per_run").
    """

    target_loss = base_out["forget"]["avg_loss"]

    forget_loader = dataloaders["forget"]
    full_train_dataset = ConcatDataset([forget_loader.dataset, dataloaders["retain"].dataset])
    full_train_loader = DataLoader(
        full_train_dataset,
        batch_size=dataloaders["retain"].batch_size,
        shuffle=True,
        num_workers=dataloaders["retain"].num_workers,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    base_state_dict = copy.deepcopy(model.state_dict())

    epochs_per_run = []
    converged_per_run = []

    for run in range(1, NUM_RUNS + 1):

        run_seed = seed * 1000 + run
        setup_seed(run_seed)

        relearn_model = init_model(model_class=model_class, num_classes=num_classes, checkpoint_path=None).to(device)
        relearn_model.load_state_dict(base_state_dict)

        opt = torch.optim.Adam(relearn_model.parameters(), lr=training_hp["learning_rate"], weight_decay=training_hp.get("weight_decay", 0))

        # a model that's already within THRESHOLD of the original forget-set loss needs 0 epochs of relearning
        current_loss = naive_validate(forget_loader, relearn_model, criterion=criterion, device=device)
        converged = abs(current_loss - target_loss) / target_loss <= THRESHOLD
        epoch = 0

        while not converged and epoch < MAX_EPOCHS:
            epoch += 1

            print(f"[relearn_time] run {run}/{NUM_RUNS}, epoch {epoch}/{MAX_EPOCHS}\n")
            relearn_model, *_ = train(full_train_loader, relearn_model, criterion, opt, epoch=epoch, print_freq=print_freq, device=device, w_and_b=False)

            current_loss = naive_validate(forget_loader, relearn_model, criterion=criterion, device=device)
            converged = abs(current_loss - target_loss) / target_loss <= THRESHOLD

        epochs_per_run.append(epoch)
        converged_per_run.append(converged)

        if not converged:
            print(
                f"[relearn_time] run {run}: did NOT reach within {THRESHOLD:.0%} of the original forget loss "
                f"({target_loss:.4f}) within {MAX_EPOCHS} epochs (final loss {current_loss:.4f}); "
                f"counting as {MAX_EPOCHS} epochs.\n"
            )

    return {
        "epochs_per_run": epochs_per_run,
        "converged_per_run": converged_per_run,
        "avg_epochs": sum(epochs_per_run) / len(epochs_per_run),
    }
