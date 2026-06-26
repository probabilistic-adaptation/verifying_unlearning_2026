
import sys
import time

import torch
import torch.nn.functional as F
import numpy as np
import wandb

from trainer.utils import AverageMeter
from evaluation.accuracy import accuracy

class UNSIR_noise(torch.nn.Module):

    """
    A class containing our noise training samples
    """
    def __init__(self, *dim):
        super().__init__()
        self.noise = torch.nn.Parameter(torch.randn(*dim), requires_grad = True)
        
    def forward(self):
        return self.noise
    
    
def UNSIR_noise_train(noise, model, forget_class_label, num_epochs = 5, noise_batch_size = 256, device='cuda'):

    # their github repo indeed hardcodes Adam here, with lr = 0.1 (specifically for learning the noise)
    opt = torch.optim.Adam(noise.parameters(), lr = 0.1)

    # eval mode: we're only optimising the noise, not the model; train mode would let
    # BN running stats drift toward the noise distribution before unlearning starts
    was_training = model.training
    model.eval()

    # their implementation did 5 epochs
    for epoch in range(num_epochs):
        total_loss = []
        inputs = noise()
        labels = torch.zeros(noise_batch_size).to(device)+forget_class_label
        outputs = model(inputs)
        
        # the original paper and github implementation uses .1 as lambda
        loss = -F.cross_entropy(outputs, labels.long()) + 0.1*torch.mean(torch.sum(inputs**2, [1, 2, 3]))

        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss.append(loss.cpu().detach().numpy())
        print(f"Epoch: [{epoch+1}] \t Loss: {np.mean(total_loss)}")

    model.train(was_training)
    return noise


def UNSIR_create_noisy_loader(noise, forget_class_label, retain_samples, batch_size, num_noise_batches=80, 
                              num_workers = 4, pin_memory = True):
    
    noisy_data = []
    for _ in range(num_noise_batches):
        batch = noise.forward()
        for j in range(batch.size(0)):
            noisy_data.append((batch[j].detach().cpu(), torch.tensor(forget_class_label)))
    
    num_noise_samples = len(noisy_data)

    other_samples = []
    for i in range(len(retain_samples)):
        other_samples.append((retain_samples[i][0].cpu(), torch.tensor(retain_samples[i][1])))
    noisy_data += other_samples
    noisy_loader = torch.utils.data.DataLoader(
        noisy_data, 
        batch_size=batch_size, 
        shuffle = True, 
        num_workers = num_workers, 
        pin_memory=pin_memory
        )
    
    print(f"Total samples: {len(noisy_data)}")
    print(f"Noise samples: {num_noise_samples}")
    print(f"Retain samples: {len(other_samples)}\n")
    
    return noisy_loader

def unsir_train_iter(epoch, loader, model, criterion, optimizer, split, print_freq, device='cuda', w_and_b=True):
    """
    One pass through `loader` under the given split (both yield 2-tuples: image, label).
    Returns (top1_avg, loss_avg) for "repair" and loss_avg for "impair".
    """
    losses_meter = AverageMeter()
    top1_meter = AverageMeter()

    start = time.time()
    for i, (image, target) in enumerate(loader):

        image = image.float().to(device)
        target = target.to(device)

        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1 = accuracy(output.data, target)[0]
        losses_meter.update(loss.item(), image.size(0))
        top1_meter.update(prec1.item(), image.size(0))

        if i % print_freq == 0:
            end = time.time()
            print(
                f"Epoch: [{epoch}][{i}/{len(loader)}]\t"
                f"[{split}]\t"
                f"Loss {losses_meter.val:.4f} ({losses_meter.avg:.4f})\t"
                f"Accuracy {top1_meter.val:.3f} ({top1_meter.avg:.3f})\t"
                f"Time {end - start:.2f}"
            )
            sys.stdout.flush()

            if w_and_b:
                if split == "repair":
                    wandb.log({
                        "train_loss": losses_meter.val,
                        "train_loss_avg": losses_meter.avg,
                        "train_acc": top1_meter.val,
                        "train_acc_avg": top1_meter.avg,
                    })
                else:
                    wandb.log({
                        "impair_loss": losses_meter.val,
                        "impair_loss_avg": losses_meter.avg,
                    })

            start = time.time()

    if split == "repair":
        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1_meter))
        return top1_meter.avg, losses_meter.avg
    else:
        return losses_meter.avg


def UNSIR(dataloaders, model, criterion, optimizer, epoch,
          print_freq, impair_epochs=1, lr_impair=.02, lr_repair=.01,
          device='cuda', w_and_b=True, **kwargs):
    """
    One round of UNSIR: an impair step on noisy data (for the first `impair_epochs` epochs)
    followed by a repair step on retain data every epoch.

    `num_epochs` (controlled by do_unlearning's loop) may exceed `impair_epochs`, in which
    case the remaining epochs are repair-only — allowing extra fine-tuning after impairing.

    Each phase uses its own learning rate (lr_impair / lr_repair), applied by mutating the
    optimizer's param groups before each step.
    """
    retain_loader = dataloaders['retain']
    noisy_loader = dataloaders['UNSIR_noisy_loader']

    model.train()
    # model.eval()

    impair_loss = 0
    if epoch <= impair_epochs:
        print("Performing impair step...\n")
        for pg in optimizer.param_groups:
            pg['lr'] = lr_impair
        impair_loss = unsir_train_iter(
            epoch, noisy_loader, model, criterion, optimizer,
            split="impair", print_freq=print_freq, device=device, w_and_b=w_and_b
        )
        optimizer.state.clear()

    print("Performing repair step...\n")
    for pg in optimizer.param_groups:
        pg['lr'] = lr_repair
    train_acc, repair_loss = unsir_train_iter(
        epoch, retain_loader, model, criterion, optimizer,
        split="repair", print_freq=print_freq, device=device, w_and_b=w_and_b
    )

    print("impair loss: {:.4f}\t repair loss: {:.4f}\t train_acc: {:.3f}".format(
        impair_loss, repair_loss, train_acc))

    return model, train_acc

