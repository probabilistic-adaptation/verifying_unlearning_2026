import torch
import psutil
import os
import matplotlib.pyplot as plt
from datetime import datetime


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

def warmup_lr(epoch, step, optimizer, one_epoch_step, args):
    overall_steps = args["warmup"] * one_epoch_step
    current_steps = epoch * one_epoch_step + step

    lr = args["lr"] * current_steps / overall_steps
    lr = min(lr, args["lr"])

    for p in optimizer.param_groups:
        p["lr"] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_model_weight_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.requires_grad:
            # L2 Norm: square the values, sum them, and take the square root
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def get_memory_footprint():
    """Returns the current System RAM and GPU VRAM usage."""
    
    # System RAM
    process = psutil.Process(os.getpid())
    ram_gb = process.memory_info().rss / ( 1024**3 )
    
    # GPU VRAM usage
    vram_gb = 0.0
    if torch.cuda.is_available():
        # memory_allocated is the actual tensor storage
        # memory_reserved includes the cache held by PyTorch
        vram_gb = torch.cuda.memory_reserved() / (1024**3)
    elif torch.backends.mps.is_available():
        # Using driver_allocated because it reflects the actual 
        # hardware pressure on the Unified Memory pool.
        vram_gb = torch.mps.driver_allocated_memory() / (1024**3)
        
    return ram_gb, vram_gb


def evaluate(model, data_loader, criterion, device, print_predictions = False):
    
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        
        # for each batch...
        for data, target in data_loader:
            
            # ... send to the right device,
            data, target = data.to(device), target.to(device)

            # ... send data through the model,
            outputs = model(data)

            # ... calculate and update loss,
            total_loss += criterion( outputs , target ).item()

            # ... predict most likely class, 
            _, preds = torch.max(outputs, 1)
            
            # ... and tally accuracy.
            correct += (preds == target).sum().item()
            total += target.size(0)
            if print_predictions:
                print(f"Sample Preds: {preds[:5].cpu().numpy() + 1}")

    return total_loss / len(data_loader), correct / total

def training_regimen(model, train_loader, val_loader, opt, criterion, scheduler, device, num_epochs = 10, best_val_loss = torch.inf, model_path = f"model.pth", print_freq = 100):
    
    # For each epoch ...
    for k in range(1, num_epochs + 1):

        print(f" ----- EPOCH {k} ----- \n")
        # ... grab the current learning rate, 
        current_lr = opt.param_groups[0]["lr"]

        # ... train one epoch
        for i, (X, y) in enumerate(train_loader):
            
            X, y = X.to(device), y.to(device)
            
            opt.zero_grad() # zero gradients
            y_hat = model(X) # forward pass
            loss = criterion(y_hat, y)
            loss.backward()
            opt.step()

            if i % print_freq == 0:
                print(f"Batch {i}: Loss = {loss.cpu().item():.4f}")
 
        # ... eval on validation set
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # ... step scheduler
        scheduler.step(val_loss)

        # ... and message if learning rate changed for the next epoch
        new_lr = opt.param_groups[0]['lr']
        if new_lr < current_lr:
            print(f"\n[LR REDUCTION] Learning rate dropped from {current_lr:.1e} to {new_lr:.1e}")

        if new_lr < 1e-6:
            print(f"\n[EARLY STOPPING] LR {new_lr:.1e} is below threshold. Training halted.")
            break

        # ... track metrics
        # ... track model metrics,
        ram, vram = get_memory_footprint()
        current_norm = get_model_weight_norm(model)

        # ... message how we're doing
        print(f"Epoch {k} | LR: {current_lr:.1e} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f} | "
              f"RAM: {ram:.2f}GB | VRAM: {vram:.2f}GB | Weight Norm: {current_norm:.3f}")

        # ... and, if this is the best model so far, save it.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': k,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }
            torch.save(checkpoint, model_path, _use_new_zipfile_serialization = False)
            print(f"--- Epoch {k}: New best model saved! ---\n")
        
    return model, opt, scheduler



from sklearn.metrics import confusion_matrix
import seaborn as sns



# More comprehensive model evaluation (no transformations)
def full_evaluate(model, data_loader, device):
    
    model.eval()
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_times = []

    print("\n")

    with torch.no_grad():
        # for each batch, ...
        for idx, (X, labels) in enumerate(data_loader):
            
            # ... grab the data
            X, labels = X.to(device), labels.to(device)

            outputs = model(X)
            
            # ... make predictions
            _, preds = torch.max(outputs, 1)
            

            # ... evalute if we're correct or not
            for i in range(labels.size(0)):
                batch_idx = idx * data_loader.batch_size + i
                pred = preds[i].item()
                true_label = labels[i].item()
                is_correct = "✓" if pred == true_label else "✗"
                print(f"{is_correct}  pred={pred}  true={true_label}  |  {i}")

            # ... tally our results
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Now, generate a confusion matrix of all our incorrect predictions, ...
    unique_labels = sorted(list(set(all_labels) | set(all_preds)))
    cm = confusion_matrix(all_labels, all_preds, labels=unique_labels)
    plot_confusion_matrix(cm, unique_labels)

    # ... and tally our final accuracy
    accuracy = correct / total
    return accuracy, all_preds, all_labels, all_times

def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix: Classifications')
    plt.show()
    
def run_inference(model, data_loader, device):

    model = model.to(device)

    # Create dataloader

    print(f"\nRunning inference on {len(data_loader)} batches...")
    accuracy, preds, labels, times = full_evaluate(model, data_loader, device)

    # Summary
    num_correct = sum(p == l for p, l in zip(preds, labels))
    num_wrong = len(preds) - num_correct

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total items:         {len(preds)}")
    print(f"Correct:              {num_correct}")
    print(f"Incorrect:                {num_wrong}")
    print(f"")
    print(f"ACCURACY:             {accuracy*100:.2f}%")
    print(f"")
    print("="*50)
    return accuracy, preds, labels
