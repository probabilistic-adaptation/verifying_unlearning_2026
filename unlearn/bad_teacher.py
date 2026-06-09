import torch
import time
from torch.nn import functional as F
from torch.utils.data import DataLoader
# from dataset import UnLearningData
import numpy as np
# from utils import *

from data.utils import split_random
from trainer.utils import AverageMeter
from evaluation.accuracy import accuracy
import wandb




def UnlearnerLoss(output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature):
    
    labels = torch.unsqueeze(labels, dim=1)
    
    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)

    # label 1 means forget sample
    # label 0 means retain sample
    overall_teacher_out = labels * u_teacher_out + (1-labels)*f_teacher_out
    student_out = F.log_softmax(output / KL_temperature, dim=1)
    return F.kl_div(student_out, overall_teacher_out, reduction='batchmean')


# OLD
# def bad_teacher_iter(unlearning_loader, model, unlearning_teacher, full_trained_teacher, optimizer, epoch, print_freq, device, w_and_b = True, KL_temperature = 1):
    
#     losses_meter = AverageMeter()
#     top1_meter = AverageMeter()

#     # losses = []
#     for i, batch in enumerate(unlearning_loader):
#         x, y = batch
#         x, y = x.to(device), y.to(device)

#         # gather teacher logits
#         with torch.no_grad():
#             full_teacher_logits = full_trained_teacher(x)
#             unlearn_teacher_logits = unlearning_teacher(x)

#         # compute our own output and loss
#         output = model(x)
#         optimizer.zero_grad()
#         loss = UnlearnerLoss(output = output, labels=y, full_teacher_logits=full_teacher_logits, 
#                 unlearn_teacher_logits=unlearn_teacher_logits, KL_temperature=KL_temperature)
#         loss.backward()
#         optimizer.step()

#         # measure stuff
#         output = output.float()
#         loss = loss.float()
#         prec1 = accuracy(output, y)[0]
#         # update trackers
#         losses_meter.update(loss.item(), x.size(0))
#         top1_meter.update(prec1.item(), x.size(0))

#         if (i + 1) % print_freq == 0:
#             print(
#                 f"Epoch: [{epoch}][{i}/{len(unlearning_loader)}]\t"
#                 f"Loss {losses_meter.val:.4f} ({losses_meter.avg:.4f})\t"
#                 f"Accuracy {top1_meter.val:.3f} ({top1_meter.avg:.3f})\t"
#             )

#             if w_and_b:
#                 wandb.log(
#                     {
#                         "train_loss": losses_meter.val,
#                         "train_loss_avg": losses_meter.avg,
#                         "train_acc": top1_meter.val,
#                         "train_acc_avg": top1_meter.avg,
#                     }
#                 )
        
#     return model, top1_meter.avg


def bad_teacher_iter(unlearning_loader, model, unlearning_teacher, full_trained_teacher, optimizer, epoch, print_freq, device, w_and_b=True, KL_temperature=1):
    
    losses_meter = AverageMeter()
    forget_agree_meter = AverageMeter()   # student vs. unlearn teacher on forget samples
    retain_agree_meter = AverageMeter()   # student vs. full teacher on retain samples
    start = time.time()

    for i, batch in enumerate(unlearning_loader):
        x, y = batch
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            full_teacher_logits = full_trained_teacher(x)
            unlearn_teacher_logits = unlearning_teacher(x)

        output = model(x)
        optimizer.zero_grad()
        loss = UnlearnerLoss(output=output, labels=y, full_teacher_logits=full_teacher_logits,
                unlearn_teacher_logits=unlearn_teacher_logits, KL_temperature=KL_temperature)
        loss.backward()
        optimizer.step()

        # split batch by forget/retain flag
        forget_mask = (y == 1)
        retain_mask = (y == 0)
        student_preds = output.detach().argmax(dim=1)

        if forget_mask.any():
            unlearn_preds = unlearn_teacher_logits.argmax(dim=1)
            forget_agree = (student_preds[forget_mask] == unlearn_preds[forget_mask]).float().mean()
            forget_agree_meter.update(forget_agree.item(), forget_mask.sum().item())

        if retain_mask.any():
            full_preds = full_teacher_logits.argmax(dim=1)
            retain_agree = (student_preds[retain_mask] == full_preds[retain_mask]).float().mean()
            retain_agree_meter.update(retain_agree.item(), retain_mask.sum().item())

        losses_meter.update(loss.float().item(), x.size(0))

        if (i + 1) % print_freq == 0:
            end = time.time()
            print(
                f"Epoch: [{epoch}][{i}/{len(unlearning_loader)}]\t"
                f"Loss {losses_meter.val:.4f} ({losses_meter.avg:.4f})\t"
                f"Forget→UnlearnT {forget_agree_meter.val:.3f} ({forget_agree_meter.avg:.3f})\t"
                f"Retain→FullT {retain_agree_meter.val:.3f} ({retain_agree_meter.avg:.3f})\t"
                f"Time {end - start:.2f}"
            )
            if w_and_b:
                wandb.log({
                    "train_loss": losses_meter.val,
                    "forget_teacher_agreement": forget_agree_meter.val,
                    "retain_teacher_agreement": retain_agree_meter.val,
                })
            start = time.time()

    return model, losses_meter.avg


# def fit_one_unlearning_cycle(epochs,  model, train_loader, val_loader, lr, device):
#     history = []
    
#     optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    
#     for epoch in range(epochs): 
#         model.train()
#         train_losses = []
#         lrs = []
#         for batch in train_loader:
#             loss = training_step(model, batch, device)
#             loss.backward()
#             train_losses.append(loss.detach().cpu())
            
#             optimizer.step()
#             optimizer.zero_grad()
            
#             lrs.append(get_lr(optimizer))
            
#         result = evaluate(model, val_loader, device)
#         result['train_loss'] = torch.stack(train_losses).mean()
#         result['lrs'] = lrs
#         epoch_end(model, epoch, result)
#         history.append(result)
#     return history

# def blindspot_unlearner(model, unlearning_teacher, full_trained_teacher, retain_data, forget_data, epochs = 10,
#                 optimizer = 'adam', lr = 0.01, batch_size = 256, num_workers = 32, 
#                 device = 'cuda', KL_temperature = 1):
#     # creating the unlearning dataset.
#     unlearning_data = UnLearningData(forget_data=forget_data, retain_data=retain_data)
#     unlearning_loader = DataLoader(unlearning_data, batch_size = batch_size, shuffle=True, 
#                             num_workers=num_workers, pin_memory=True)

#     unlearning_teacher.eval()
#     full_trained_teacher.eval()
#     optimizer = optimizer
#     if optimizer == 'adam':
#         optimizer = torch.optim.Adam(model.parameters(), lr = lr)
#     else:
#         # if optimizer is not a valid string, then assuming it as a function to return optimizer
#         optimizer = optimizer#(model.parameters())

#     for epoch in range(epochs):
#         loss = unlearning_step(model = model, unlearning_teacher= unlearning_teacher, 
#                         full_trained_teacher=full_trained_teacher, unlearning_loader=unlearning_loader, 
#                         optimizer=optimizer, device=device, KL_temperature=KL_temperature)
#         print("Epoch {} Unlearning Loss {}".format(epoch+1, loss))
        
        
# They used Adam as the optimizer for Bad_T

from torch.utils.data import Dataset
import copy

# I'm not convinced this __getitem__ indexing is correct
class BadTeacherUnLearningData(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len
    
    def __getitem__(self, index):
        if(index < self.forget_len):
            x = self.forget_data[index][0]
            y = 1
            return x,y
        else:
            x = self.retain_data[index - self.forget_len][0]
            y = 0
            return x,y

def bad_teacher(unlearning_loader, model, unlearning_teacher, full_trained_teacher, optimizer, epoch, print_freq, device, w_and_b = True, KL_temperature = 1):
    return bad_teacher_iter(unlearning_loader, model, unlearning_teacher, full_trained_teacher, optimizer, epoch, print_freq, device, w_and_b, KL_temperature)
