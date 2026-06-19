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





# -------- moved RL proximal down here ----- #

# def RL_proximal(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
#     forget_loader = data_loaders["forget"]
#     retain_loader = data_loaders["retain"]
#     forget_dataset = deepcopy(forget_loader.dataset)
#     mask_ratio = args.mask_ratio
    
#     # concat all params
#     init_params = torch.concat([param.view(-1) for param in model.parameters()], dim=0)
#     n_params = init_params.numel()        
#     total_steps = args.unlearn_epochs * (len(forget_loader) + len(retain_loader))
    
#     if args.dataset == "cifar10" or args.dataset == "cifar100" or args.dataset == "TinyImagenet":
#         forget_dataset.targets = np.random.randint(0, args.num_classes, forget_dataset.targets.shape)
    
#         retain_dataset = retain_loader.dataset
#         train_dataset = torch.utils.data.ConcatDataset([forget_dataset,retain_dataset])
#         train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
#         losses = utils.AverageMeter()
#         top1 = utils.AverageMeter()
      
#         # switch to train mode
#         model.train()
      
#         start = time.time()
#         loader_len = len(forget_loader) + len(retain_loader)
      
#         if epoch < args.warmup:
#             utils.warmup_lr(epoch, i+1, optimizer,
#                             one_epoch_step=loader_len, args=args)
      
#         for it, (image, target) in enumerate(train_loader):
#             i = it + len(forget_loader)
#             image = image.cuda()
#             target = target.cuda()
#             # compute output
#             output_clean = model(image)
#             loss = criterion(output_clean, target)
      
#             optimizer.zero_grad()
#             loss.backward()

            
#             optimizer.step()
                  
#             ratio = int(mask_ratio * ((total_steps - (epoch * (len(forget_loader) + len(retain_loader)) + 1)) / total_steps * n_params))           
#             params = torch.concat([param.view(-1) for param in model.parameters()], dim=0)
#             diff_params = params - init_params
#             threshold = -torch.topk(-diff_params.abs(), ratio)[0][-1]
#             params = torch.where(diff_params > threshold, params - threshold, 
#                                         torch.where(diff_params < -threshold, params + threshold, init_params))
#             # update params
#             for name, param in model.named_parameters():
#                 param.data = params[:param.numel()].view(param.shape)
#                 params = params[param.numel():]
      
#             output = output_clean.float()
#             loss = loss.float()
#             # measure accuracy and record loss
#             prec1 = utils.accuracy(output.data, target)[0]
      
#             losses.update(loss.item(), image.size(0))
#             top1.update(prec1.item(), image.size(0))
      
#             if (i + 1) % args.print_freq == 0:
#                 end = time.time()
#                 print('Epoch: [{0}][{1}/{2}]\t'
#                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                       'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
#                       'Time {3:.2f}'.format(
#                           epoch, i, loader_len, end-start, loss=losses, top1=top1))
#                 start = time.time()
      
#     elif args.dataset == "svhn":
#         losses = utils.AverageMeter()
#         top1 = utils.AverageMeter()
      
#         # switch to train mode
#         model.train()
      
#         start = time.time()
#         loader_len = len(forget_loader) + len(retain_loader)
      
#         if epoch < args.warmup:
#             utils.warmup_lr(epoch, i+1, optimizer,
#                             one_epoch_step=loader_len, args=args)
        
#         for i, (image, target) in enumerate(forget_loader):
#             image = image.cuda()
#             target = torch.randint(0, args.num_classes, target.shape).cuda()
            
#             # compute output
#             output_clean = model(image)
#             loss = criterion(output_clean, target)
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             ratio = int(mask_ratio * ((total_steps - (epoch * (len(forget_loader) + len(retain_loader)) + 1)) / total_steps * n_params))           
#             params = torch.concat([param.view(-1) for param in model.parameters()], dim=0)
#             diff_params = params - init_params
#             threshold = -torch.topk(-diff_params.abs(), ratio)[0][-1]
#             params = torch.where(diff_params > threshold, params - threshold, 
#                                         torch.where(diff_params < -threshold, params + threshold, init_params))
#             # update params
#             for name, param in model.named_parameters():
#                 param.data = params[:param.numel()].view(param.shape)
#                 params = params[param.numel():]
            
#         for i, (image, target) in enumerate(retain_loader):
#             image = image.cuda()
#             target = target.cuda()
            
#             # compute output
#             output_clean = model(image)
#             loss = criterion(output_clean, target)
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             ratio = int(mask_ratio * ((total_steps - (epoch * (len(forget_loader) + len(retain_loader)) + i + 1)) / total_steps * n_params))           
#             params = torch.concat([param.view(-1) for param in model.parameters()], dim=0)
#             diff_params = params - init_params
#             threshold = -torch.topk(-diff_params.abs(), ratio)[0][-1]
#             params = torch.where(diff_params > threshold, params - threshold, 
#                                         torch.where(diff_params < -threshold, params + threshold, init_params))
#             # update params
#             for name, param in model.named_parameters():
#                 param.data = params[:param.numel()].view(param.shape)
#                 params = params[param.numel():]
            
#             output = output_clean.float()
#             loss = loss.float()
#             # measure accuracy and record loss
#             prec1 = utils.accuracy(output.data, target)[0]
            
#             losses.update(loss.item(), image.size(0))
#             top1.update(prec1.item(), image.size(0))
            
#             if (i + 1) % args.print_freq == 0:
#                end = time.time()
#                print('Epoch: [{0}][{1}/{2}]\t'
#                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                      'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
#                      'Time {3:.2f}'.format(
#                          epoch, i, loader_len, end-start, loss=losses, top1=top1))
#                start = time.time()
               
#     return top1.avg