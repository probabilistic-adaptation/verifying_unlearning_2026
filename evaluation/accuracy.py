import torch

def check_accuracy(model, data_loader, device):
    
    model.eval()
    num_correct = 0
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            y_hat = model(X).argmax(-1)
        num_correct += (y_hat == y).sum().item()
    return num_correct / len(data_loader.dataset)
