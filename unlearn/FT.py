import sys
import time

import torch
# import utils

# from .impl import iterative_unlearn
from trainer.utils import AverageMeter
from evaluation.accuracy import accuracy
import wandb

sys.path.append(".")
# from imagenet import get_x_y_from_data_dict


def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def FT_iter(
    dataloaders, model, criterion, optimizer, epoch, print_freq, device, w_and_b = True, mask=None, with_l1=False
):
    retain_loader = dataloaders["retain"]

    losses_meter = AverageMeter()
    top1_meter = AverageMeter()

    # switch to train mode
    # model.train()
    # print(f"[FT_iter] model.training = {model.training}")

    # start = time.time()
    # DO NOT NEED TO CLOCKING TIME INSIDE EACH FUNCTION - THAT IS HANDLED BY DO_UNLEARNING

    for i, (image, target) in enumerate(retain_loader):
        # if epoch < args.warmup:
        #     utils.warmup_lr(
        #         epoch, i + 1, optimizer, one_epoch_step=len(retain_loader), args=args
        #     )

        image, target = image.to(device), target.to(device)
        # if epoch < args.unlearn_epochs - args.no_l1_epochs:
        #     current_alpha = args.alpha * (
        #         1 - epoch / (args.unlearn_epochs - args.no_l1_epochs)
        #     )
        # else:
        #     current_alpha = 0
        # compute output
        
        output_clean = model(image)
        loss = criterion(output_clean, target)

        if with_l1:
            
            # hard coding this for now
            current_alpha = .1
            loss += current_alpha * l1_regularization(model)

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
            # end = time.time()
            print(
                f"Epoch: [{epoch}][{i}/{len(retain_loader)}]\t"
                f"Loss {losses_meter.val:.4f} ({losses_meter.avg:.4f})\t"
                f"Accuracy {top1_meter.val:.3f} ({top1_meter.avg:.3f})\t"
                # f"Time {end - start:.2f}"
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

            # start = time.time()

    # print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return model, top1_meter.avg


# @iterative_unlearn
def FT(dataloaders, model, criterion, optimizer, epoch, print_freq, device, w_and_b = True, mask=None, with_l1=False):
    return FT_iter(dataloaders, model, criterion, optimizer, epoch, print_freq, device, w_and_b, mask, with_l1)


# @iterative_unlearn
def FT_l1(dataloaders, model, criterion, optimizer, epoch, print_freq, device, w_and_b = True, mask=None, with_l1=True):
    return FT_iter(dataloaders, model, criterion, optimizer, epoch, print_freq, device, w_and_b, mask, with_l1 = True)
