from evaluation.accuracy import check_accuracy


def measure_unlearning_metrics(name, epoch, metrics, model, dataloaders, device):

    # fill in base information
    results = {
        "name": name,
        "epoch": epoch
        }
    
    # add in metrics, as you'd like

    if "forget_acc" in metrics:
        forget_acc = check_accuracy(model, data_loader = dataloaders["forget_train"], device = device)
        results["forget_acc"] = forget_acc

    if "retain_acc" in metrics:
        retain_acc = check_accuracy(model, data_loader = dataloaders["retain_test"], device = device)
        results["retain_acc"] = retain_acc

    return results
    

    


