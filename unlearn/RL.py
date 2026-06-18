import time
from copy import deepcopy

import numpy as np
import torch
from data.dataloaders import get_targets
from .FT import l1_regularization
from trainer.utils import AverageMeter
from evaluation.accuracy import accuracy

import wandb


# The original paper (Graves et al 2020) propose fine-tuning just on the relabled forget set
# Golatkar et al 2020 (Eternal sunshine spotless net). Huang et al 2024, and Triantafillou et al 2024 all fine-tune on FULL relabeled_forget + retain
# ---- this should intuitively lead to better maintanance of model utility.
# Chen et al 2023 supposedly only fine-tune on relabeled forget set, as was originally proposed

# I'm leaning towards fine-tuning on full,
# the original implementation given here implies "full" on some datasets but only fine-tuning with relabeled-forget on others, which seems weird


def setup_RL_loader(dataloaders):

    # ------------------------------------------------------------------------------------------------------------- #
    # --- We build the "random label" dataset by concatenating forget and retain, aRLer amending forget labels ---- #
    # ------------------------------------------------------------------------------------------------------------- #

    forget_loader = dataloaders["forget"]
    retain_loader = dataloaders["retain"]
    forget_dataset = deepcopy(forget_loader.dataset)
    targets = get_targets(forget_dataset)
    num_classes = np.max(targets) + 1 # we assume 0 indexing for first class
    
    # assign random labels
    try:
        forget_dataset.targets = np.random.randint(0, num_classes, forget_dataset.targets.shape)
    except:
        forget_dataset.dataset.targets = np.random.randint(0, num_classes, len(forget_dataset.dataset.targets))

    retain_dataset = retain_loader.dataset
    train_dataset = torch.utils.data.ConcatDataset([forget_dataset,retain_dataset])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=retain_loader.batch_size, shuffle=True, num_workers=retain_loader.num_workers, pin_memory=retain_loader.pin_memory)

    return train_loader




def RL_iter(
    dataloaders, model, criterion, optimizer, epoch, print_freq, device, w_and_b = True, mask=None, with_l1=False
):
    train_loader = dataloaders["RL_train_loader"]

    losses_meter = AverageMeter()
    top1_meter = AverageMeter()
    start = time.time()
    # We track time here just as a convenience, not as the official measurement of the run time efficiency of the algo (that happens inside do_unlearning)

    for i, (image, target) in enumerate(train_loader):
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
            end = time.time()
            print(
                f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                f"Loss {losses_meter.val:.4f} ({losses_meter.avg:.4f})\t"
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



def RL(dataloaders, model, criterion, optimizer, epoch, print_freq, device, w_and_b = True, mask=None, with_l1=False):
    return RL_iter(dataloaders, model, criterion, optimizer, epoch, print_freq, device, w_and_b, mask, with_l1)


def RL_l1(dataloaders, model, criterion, optimizer, epoch, print_freq, device, w_and_b = True, mask=None, with_l1=True):
    return RL_iter(dataloaders, model, criterion, optimizer, epoch, print_freq, device, w_and_b, mask, with_l1 = True)
