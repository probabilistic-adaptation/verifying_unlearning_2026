import sys
import time

import torch
from trainer.utils import AverageMeter
from evaluation.accuracy import accuracy
import wandb

from itertools import cycle

sys.path.append(".")


def train_negrad(dataloaders, model, criterion, optimizer, epoch, print_freq, device, w_and_b = True, alpha = .99):
    """
    Training loop for "NegGrad+", which ammounts to contrastive fine-tuning on retain set vs. gradient ascent on forget set

    This implementation is directly from Kurmanji et al 2023 codebase, with some variables renamed and arguments added/removed
    to mimic other methods in our codebase
    """
    # model.train()

    retain_loader = dataloaders["retain"]
    forget_loader = dataloaders["forget"]

    r_losses_meter = AverageMeter()
    f_losses_meter = AverageMeter()
    losses_meter = AverageMeter()
    top1_meter = AverageMeter()
    start = time.time()

    for idx, ((retain_input, target), (forget_input, forget_target)) in enumerate(zip(retain_loader, cycle(forget_loader))):
        
        
        # retain_input = retain_input.float()
        # forget_input = forget_input.float()
        retain_input, target, forget_input, forget_target = retain_input.to(device), target.to(device), forget_input.to(device), forget_target.to(device)

        # ===================forward=====================
        output = model(retain_input)
        del_output = model(forget_input)
        r_loss = criterion(output, target)
        f_loss = criterion(del_output, forget_target)
        # clamp GA loss at chance level for stability
        # chance_loss = torch.tensor(-torch.log(torch.tensor(1.0 / 10)), device=device)

        # # chance level is too high - just need to prevent the model from calamitously failing
        # f_loss = torch.clamp(criterion(del_output, forget_target), max=torch.tensor(-10, device=device))
        
        
        loss = alpha * r_loss - (1 - alpha) * f_loss
        
        prec1 = accuracy(output.data, target)[0]
        r_losses_meter.update(r_loss.item(), retain_input.size(0))
        f_losses_meter.update(f_loss.item(), retain_input.size(0))
        losses_meter.update(loss.item(), retain_input.size(0))
        top1_meter.update(prec1.item(), retain_input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        
        # print info
        if (idx + 1) % print_freq == 0:
            end = time.time()
            print(
                f"Epoch: [{epoch}][{idx}/{len(retain_loader)}]\t"
                f"Loss {losses_meter.val:.4f} ({losses_meter.avg:.4f})\t"
                f"R-loss {r_losses_meter.val:.4f} ({r_losses_meter.avg:.4f})\t"
                f"F-loss {f_losses_meter.val:.4f} ({f_losses_meter.avg:.4f})\t"
                f"Accuracy {top1_meter.val:.3f} ({top1_meter.avg:.3f})\t"
                f"Time {end - start:.2f}"
            )

            if w_and_b:
                wandb.log(
                    {
                        "train_loss": losses_meter.val,
                        "train_loss_avg": losses_meter.avg,
                        "train_acc": top1_meter.val,
                        "train_acc_avg": top1_meter.avg,
                    }
                )

            start = time.time()

    return model, top1_meter.avg

def NegGrad_plus(dataloaders, model, criterion, optimizer, epoch, print_freq, device, w_and_b, alpha = .99, **kwargs):
    return train_negrad(dataloaders, model, criterion, optimizer, epoch, print_freq, device, w_and_b, alpha)
