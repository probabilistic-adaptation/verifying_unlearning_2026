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
    start = time.time()
    # We track time here just as a convenience, not as the official measurement of the run time efficiency of the algo (that happens inside do_unlearning)

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
        losses_meter.update(loss.item(), image.size(0))
        top1_meter.update(prec1.item(), image.size(0))

        if (i + 1) % print_freq == 0:
            end = time.time()
            print(
                f"Epoch: [{epoch}][{i}/{len(retain_loader)}]\t"
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


# @iterative_unlearn
def FT(dataloaders, model, criterion, optimizer, epoch, print_freq, device, w_and_b = True, mask=None, with_l1=False, **kwargs):
    return FT_iter(dataloaders, model, criterion, optimizer, epoch, print_freq, device, w_and_b, mask, with_l1)


# @iterative_unlearn
def FT_l1(dataloaders, model, criterion, optimizer, epoch, print_freq, device, w_and_b = True, mask=None, with_l1=True, **kwargs):
    return FT_iter(dataloaders, model, criterion, optimizer, epoch, print_freq, device, w_and_b, mask, with_l1 = True)


# import copy
# import pruner
# import trainer


# def FT_prune(data_loaders, model, criterion, args, mask=None):
#     test_loader = data_loaders["test"]

#     # save checkpoint
#     initialization = copy.deepcopy(model.state_dict())

#     # unlearn
#     FT_l1(data_loaders, model, criterion, args, mask)

#     # val
#     pruner.check_sparsity(model)
#     trainer.validate(test_loader, model, criterion, args)

#     return model


# import pruner

# from .FT import FT_iter
# from .impl import iterative_unlearn

# prune_step = 2


# @iterative_unlearn
# def FT_prune_bi(data_loaders, model, criterion, optimizer, epoch, args):
#     # switch to train mode
#     model.train()

#     # prune
#     prune_rate = 1 - (1 - args.rate) ** (
#         1 / ((args.unlearn_epochs - 1) // prune_step + 1)
#     )

#     if (args.unlearn_epochs - epoch) % prune_step == 0:
#         if args.random_prune:
#             print("random pruning")
#             pruner.pruning_model_random(model, prune_rate)
#         else:
#             print("L1 pruning")
#             pruner.pruning_model(model, prune_rate)

#     pruner.check_sparsity(model)

#     return FT_iter(data_loaders, model, criterion, optimizer, epoch, args)
