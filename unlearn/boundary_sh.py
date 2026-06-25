import copy
import time

import torch
import torch.nn as nn
from trainer.utils import AverageMeter
from evaluation.accuracy import accuracy
import wandb
# from .impl import iterative_unlearn


def discretize(x):
    return torch.round(x * 255) / 255


def FGSM_perturb(x, y, device, model=None, bound=None, criterion=None):
    model.zero_grad()
    x_adv = x.detach().clone().requires_grad_(True).to(device)

    pred = model(x_adv)
    loss = criterion(pred, y)
    loss.backward()

    # grad_sign = x_adv.grad.data.detach().sign()
    grad_sign = x_adv.grad.detach().sign()
    x_adv = x_adv + grad_sign * bound
    x_adv = discretize(torch.clamp(x_adv, 0.0, 1.0)) # this is to ensure the RGB values are valid integers

    return x_adv.detach()


# @iterative_unlearn
def boundary_shrink_iter(
    dataloaders, model, criterion, optimizer, epoch, print_freq, device, w_and_b=True, test_model=None, mask=None, bound = .1
):
    assert test_model is not None
    forget_loader = dataloaders["forget"]
    losses_meter = AverageMeter()
    top1_meter = AverageMeter()

    for i, (image, target) in enumerate(forget_loader):
        # if epoch < args.warmup:
        #     utils.warmup_lr(
        #         epoch, i + 1, optimizer, one_epoch_step=len(forget_loader), args=args
        #     )

        image, target = image.to(device), target.to(device)

        test_model.eval()
        image_adv = FGSM_perturb(
            image, target, device=device, model=test_model, bound=bound, criterion=criterion
        )

        adv_outputs = test_model(image_adv)
        adv_label = torch.argmax(adv_outputs, dim=1)

        # compute output
        output_clean = model(image)
        loss = criterion(output_clean, adv_label)

        optimizer.zero_grad()
        loss.backward()

        if mask:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name]
                    
        optimizer.step()
        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        # update trackers
        losses_meter.update(loss.item(), image.size(0))
        top1_meter.update(prec1.item(), image.size(0))

        if (i + 1) % print_freq == 0:
            print(
                f"Epoch: [{epoch}][{i}/{len(forget_loader)}]\t"
                f"Loss {losses_meter.val:.4f} ({losses_meter.avg:.4f})\t"
                f"Accuracy {top1_meter.val:.3f} ({top1_meter.avg:.3f})\t"
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

    # print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return model, top1_meter.avg


def boundary_shrink(dataloaders, model, criterion, optimizer, epoch, print_freq, device, w_and_b=True, mask=None, bound = .1, **kwargs):
    
    # the `test` model is always the current model immediately before the unlearning iteration
    # I'm worried this is meant to be the original model always, even after more epochs of this
    test_model = copy.deepcopy(model).to(device) 
    return boundary_shrink_iter(
        dataloaders, model, criterion, optimizer, epoch, print_freq, device, w_and_b, test_model=test_model, mask=mask, bound = bound
    )
