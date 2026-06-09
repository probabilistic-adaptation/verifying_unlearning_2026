import torch.nn as nn

# forget, retain, and test accuracy + entropy + m_entropy
from trainer.val import validate

# MIA (threshold function + SVC)
from evaluation.MIA import MIA
from evaluation.SVC_MIA import SVC_MIA




def measure_unlearning_metrics(model, dataloaders, device):

    # fill in base information
    results = {}
    criterion= nn.CrossEntropyLoss()
    
    # add in metrics, as you'd like

    # we deliberately set `print_freq` very high so we dont actually print anything

    print("Evaluating forget set metrics...\n")
    forget_loss, forget_acc, forget_entr, forget_m_entr, forget_losses, forget_probs, forget_labels = validate(dataloaders["forget"], model, criterion = criterion, print_freq = 100_000, device = device, w_and_b=False)
    results.update(
        {
            "forget_loss": forget_loss,
            "forget_acc": forget_acc,
            "forget_entr": forget_entr,
            "forget_m_entr": forget_m_entr,
        }
    )

    print("Evaluating retain set metrics...\n")
    retain_loss, retain_acc, retain_entr, retain_m_entr, retain_losses, retain_probs, retain_labels = validate(dataloaders["retain"], model, criterion = criterion, print_freq = 100_000, device = device, w_and_b=False)
    results.update(
        {
            "retain_loss": retain_loss,
            "retain_acc": retain_acc,
            "retain_entr": retain_entr,
            "retain_m_entr": retain_m_entr,
        }
    )

    print("Evaluating test set metrics...\n")
    test_loss, test_acc, test_entr, test_m_entr, test_losses, test_probs, test_labels = validate(dataloaders["test"], model, criterion = criterion, print_freq = 100_000, device = device, w_and_b=False)
    results.update(
        {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_entr": test_entr,
            "test_m_entr": test_m_entr,
        }
    )

    # now need to gather output dists and labels for retain_one and retain_two, for the MIAs
    # Both MIAs recompute entropy and m-entropy, which is a bit of wasted computation,
    # could refactor that later
    # at least this way, they do not have to pass through the data loader for forget and test again
    print("Running through `retain_one` and `retain_two` for MIAs...\n")
    _, _, _, _, _, retain_one_probs, retain_one_labels = validate(dataloaders["retain_one"], model, criterion = criterion, print_freq = 100_000, device = device, w_and_b=False)
    _, _, _, _, _, retain_two_probs, retain_two_labels = validate(dataloaders["retain_two"], model, criterion = criterion, print_freq = 100_000, device = device, w_and_b=False)
    
    print("Performing threshold MIA attack...\n")
    attack_results = MIA(
        shadow_train_inputs = (retain_one_probs.numpy(), retain_one_labels.numpy()),
        shadow_test_inputs = (test_probs.numpy(), test_labels.numpy()),
        target_train_inputs = (retain_two_probs.numpy(), retain_two_labels.numpy()),
        target_test_inputs = (forget_probs.numpy(), forget_labels.numpy()),
        model = model,
        device = device
        )
    
    results["MIA"] = attack_results

    # # COMMENTING OUT SVC FOR NOW, TAKES TOO LONG

    # print("Performing SVC MIA attack...\n")
    # # `device` is inferred by whatever the device the model is on - dont explicitly pass it (might be smart to do this in other areas too, so you dont have to carry `device` around)
    # # need the (probs, labels) outputs for each data loader we care about - some have already been computed through other passes through forget and test

    # # we do not need to convert this output using ".numpy()", since the SVC operations are done using torch
    # attack_results = SVC_MIA(
    #     shadow_train_inputs = (retain_one_probs, retain_one_labels), 
    #     shadow_test_inputs = (test_probs, test_labels),
    #     target_train_inputs = (retain_two_probs, retain_two_labels),
    #     target_test_inputs = (forget_probs, forget_labels),
    #     model = model
    #     )
    # results["SVC_MIA"] = attack_results


    return results
    

    


