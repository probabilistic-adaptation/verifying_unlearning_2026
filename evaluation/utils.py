# from evaluation.accuracy import check_accuracy
from evaluation.MIA import MIA
from trainer.val import validate
import torch.nn as nn


def measure_unlearning_metrics(model, dataloaders, device):

    # we no longer require the user to specify what metrics they want - we just give them all, all the time.
    
    # fill in base information
    results = {}
    criterion= nn.CrossEntropyLoss()
    
    # add in metrics, as you'd like

    # we deliberately set print_freq very high so we dont actually print anything

    print("Evaluating forget set metrics...\n")
    forget_loss, forget_acc, forget_entr, forget_m_entr, forget_losses = validate(dataloaders["forget"], model, criterion = criterion, print_freq = 100_000, device = device, w_and_b=False)
    results.update(
        {
            "forget_loss": forget_loss,
            "forget_acc": forget_acc,
            "forget_entr": forget_entr,
            "forget_m_entr": forget_m_entr,
        }
    )

    print("Evaluating retain set metrics...\n")
    retain_loss, retain_acc, retain_entr, retain_m_entr, retain_losses = validate(dataloaders["retain"], model, criterion = criterion, print_freq = 100_000, device = device, w_and_b=False)
    results.update(
        {
            "retain_loss": retain_loss,
            "retain_acc": retain_acc,
            "retain_entr": retain_entr,
            "retain_m_entr": retain_m_entr,
        }
    )

    print("Evaluating test set metrics...\n")
    test_loss, test_acc, test_entr, test_m_entr, test_losses = validate(dataloaders["test"], model, criterion = criterion, print_freq = 100_000, device = device, w_and_b=False)
    results.update(
        {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_entr": test_entr,
            "test_m_entr": test_m_entr,
        }
    )

    print("Performing MIA attack...\n")
    attack_results = MIA(
        retain_loader_train = dataloaders["retain_one"], 
        test_loader = dataloaders["test"],
        retain_loader_test = dataloaders["retain_two"],
        forget_loader = dataloaders["forget"],
        model = model,
        device = device
        )
    results["MIA"] = attack_results

    return results
    

    


