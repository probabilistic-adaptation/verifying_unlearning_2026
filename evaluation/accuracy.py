

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# def check_accuracy(model, data_loader, device):
    
#     model.eval()
#     num_correct = 0
#     for X, y in data_loader:
#         X, y = X.to(device), y.to(device)
#         with torch.no_grad():
#             y_hat = model(X).argmax(-1)
#         num_correct += (y_hat == y).sum().item()
#     return num_correct / len(data_loader.dataset)
