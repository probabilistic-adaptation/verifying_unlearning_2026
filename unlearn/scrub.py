import sys
import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from evaluation.accuracy import accuracy
from trainer.utils import AverageMeter
import wandb


class DistillKL(nn.Module):
    """Average KL divergence loss between teacher and student models"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T ** 2) / y_s.shape[0]
        return loss


# def param_dist(model, swa_model, p):
#     # https://github.com/ojus1/SmoothedGradientDescentAscent/blob/main/SGDA.py
#     dist = 0.
#     for p1, p2 in zip(model.parameters(), swa_model.parameters()):
#         dist += torch.norm(p1 - p2, p='fro')
#     return p * dist


def train_distill(epoch, train_loader, model_s, model_t, criterion_cls, criterion_div,
                  optimizer, split, gamma, alpha, print_freq,
                  device='cpu', w_and_b = True):

    # maximize: eval keeps BN stats from drifting toward the forget set distribution
    # minimize: train mode lets BN running stats update alongside the weights as the
    #           model recovers on retain data — without this, frozen BN stats mismatch
    #           the post-maximize weights and produce near-random outputs
    if split == "minimize":
        model_s.train()
    else:
        model_s.eval()
    model_t.eval()

    losses_meter = AverageMeter()
    kd_losses_meter = AverageMeter()
    top1_meter = AverageMeter()

    start = time.time()
    for i, data in enumerate(train_loader):
        
        input, target = data
        input = input.float().to(device)
        target = target.to(device)

        # -------- forward pass -------- #
        logit_s = model_s(input)
        with torch.no_grad():
            logit_t = model_t(input)


        # -------- calculate losses -------- #
        loss_div = criterion_div(logit_s, logit_t)
        if split == "minimize":
            loss_cls = criterion_cls(logit_s, target)
            loss = gamma * loss_cls + alpha * loss_div
        elif split == "maximize":
            loss = -loss_div

        # loss = loss + param_dist(model_s, swa_model, smoothing)

        if split == "minimize":
            acc1 = accuracy(logit_s, target, topk=(1,))
            losses_meter.update(loss.item(), input.size(0))
            kd_losses_meter.update(loss_div.item(), input.size(0))
            top1_meter.update(acc1[0], input.size(0))
        elif split == "maximize":
            kd_losses_meter.update(loss.item(), input.size(0))


        # -------- backward pass -------- #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end = time.time()

        if i % print_freq == 0:
            end = time.time()
            if split == "minimize":
                print(
                    f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                    f"Loss {losses_meter.val:.4f} ({losses_meter.avg:.4f})\t"
                    f"Accuracy {top1_meter.val:.3f} ({top1_meter.avg:.3f})\t"
                    f"Time {end - start:.2f}"
                )
            else:
                print(
                    f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                    f"KD Loss {kd_losses_meter.val:.4f} ({kd_losses_meter.avg:.4f})\t"
                    f"Time {end - start:.2f}"
                )
            sys.stdout.flush()

            if w_and_b:
                if split == "minimize":
                    wandb.log(
                        {
                            "train_loss": losses_meter.val,
                            "train_loss_avg": losses_meter.avg,
                            "train_acc": top1_meter.val,
                            "train_acc_avg": top1_meter.avg,
                            "kd_loss": kd_losses_meter.val,
                            "kd_loss_avg": kd_losses_meter.avg,
                        }
                    )
                else:
                    wandb.log(
                        {
                            "kd_loss": kd_losses_meter.val,
                            "kd_loss_avg": kd_losses_meter.avg,
                        }
                    )

            start = time.time()

    if split == "minimize":
        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1_meter))
        return top1_meter.avg, losses_meter.avg
    else:
        return kd_losses_meter.avg


def scrub(dataloaders, model_s, model_t, criterion_cls, criterion_div, optimizer, scheduler, print_freq, device, epoch, w_and_b = True,
          gamma=1, # was .99 in original implementation
          alpha=0.5, # was .001 in original implementation
          # beta=0,          # weight for swa averaging
          # smoothing=0.0,   # weight for swa anchor regularisation (param_dist)
          # sstart=10,       # epoch at which swa updates begin
          maxsteps=2,
          num_epochs=3,
          **kwargs):
    
    """
    This executes one "round" of SCRUB, i.e. some combination of max-epochs and min-epochs
    In the paper, the numbers for ResNet on CIFAR10 is 2 max-steps, 3 min-steps, which we leave as the default params for this function
        This would in theory arrive at the "optimal" solution, according ot the paper, but we are going to push it further to eval metrics
        in theory we can just lower the learning rate if we move too fast

    """

    retain_trainloader = dataloaders['retain']
    forget_trainloader = dataloaders['forget']


    # def avg_fn(averaged_model_parameter, model_parameter, _num_averaged):
    #     return (1 - beta) * averaged_model_parameter + beta * model_parameter
    # swa_model = torch.optim.swa_utils.AveragedModel(model_s, avg_fn=avg_fn)

    # criterion_cls = nn.CrossEntropyLoss()
    # criterion_div = DistillKL(kd_T)

    # swa_model.to(device)

    if str(device).startswith('cuda'):
        cudnn.benchmark = True

    # for epoch in range(1, num_epochs + 1):

    maximize_loss = 0
    if epoch <= maxsteps:
        print("Performing max step...\n")
        maximize_loss = train_distill(
            epoch, forget_trainloader, model_s, model_t, criterion_cls, criterion_div,
            optimizer, "maximize", gamma, alpha, print_freq, device, w_and_b = w_and_b)

    print("Performing min step...\n")
    train_acc, train_loss = train_distill(
        epoch, retain_trainloader, model_s, model_t, criterion_cls, criterion_div,
        optimizer, "minimize", gamma, alpha, print_freq, device, w_and_b = w_and_b)

    # if epoch >= sstart:
    #     swa_model.update_parameters(model_s)

    print("maximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}".format(
        maximize_loss, train_loss, train_acc))

    scheduler.step()

    return model_s, train_acc
