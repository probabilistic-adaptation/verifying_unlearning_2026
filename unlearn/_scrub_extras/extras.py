import sys
import copy
from trainer.utils import AverageMeter
from evaluation.accuracy import accuracy
import time
import torch


import torch.nn as nn
import torch.nn.functional as F
import  torch.optim as optim

# Their original class for storing DistillKL 

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


# Their original training loop

def train_distill(epoch, train_loader, module_list, swa_model, criterion_list, optimizer, opt, split, quiet=False):
    """One epoch distillation"""
    
    
    # by default, I am deleting any code which was already commented out when i grabbed it
    
    
    
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()


    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    # they pass the full criterion list, including `criterion_kd`, but it is never referenced, commented out
    # criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    kd_losses = AverageMeter()
    top1 = AverageMeter()


    end = time.time()
    for idx, data in enumerate(train_loader):
        # opt.distill is always 'kd', commenting this out and moving input, target = data up one level
        # if opt.distill in ['crd']:
        #     input, target, index, contrast_idx = data
        # else:
        input, target = data
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            # same as above - distill is always 'kd'
            # if opt.distill in ['crd']:
            #     contrast_idx = contrast_idx.cuda()
            #     index = index.cuda()

        # ===================forward=====================
        logit_s = model_s(input)
        with torch.no_grad():
            logit_t = model_t(input)


        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        # opt.distill is ALWAYS "kd", removing all others
        
        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        # elif opt.distill == 'hint':
        #     f_s = module_list[1](feat_s[opt.hint_layer])
        #     f_t = feat_t[opt.hint_layer]
        #     loss_kd = criterion_kd(f_s, f_t)
        # elif opt.distill == 'crd':
        #     f_s = feat_s[-1]
        #     f_t = feat_t[-1]
        #     loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        # elif opt.distill == 'attention':
        #     g_s = feat_s[1:-1]
        #     g_t = feat_t[1:-1]
        #     loss_group = criterion_kd(g_s, g_t)
        #     loss_kd = sum(loss_group)
        # elif opt.distill == 'nst':
        #     g_s = feat_s[1:-1]
        #     g_t = feat_t[1:-1]
        #     loss_group = criterion_kd(g_s, g_t)
        #     loss_kd = sum(loss_group)
        # elif opt.distill == 'similarity':
        #     g_s = [feat_s[-2]]
        #     g_t = [feat_t[-2]]
        #     loss_group = criterion_kd(g_s, g_t)
        #     loss_kd = sum(loss_group)
        # elif opt.distill == 'rkd':
        #     f_s = feat_s[-1]
        #     f_t = feat_t[-1]
        #     loss_kd = criterion_kd(f_s, f_t)
        # elif opt.distill == 'pkt':
        #     f_s = feat_s[-1]
        #     f_t = feat_t[-1]
        #     loss_kd = criterion_kd(f_s, f_t)
        # elif opt.distill == 'kdsvd':
        #     g_s = feat_s[1:-1]
        #     g_t = feat_t[1:-1]
        #     loss_group = criterion_kd(g_s, g_t)
        #     loss_kd = sum(loss_group)
        # elif opt.distill == 'correlation':
        #     f_s = module_list[1](feat_s[-1])
        #     f_t = module_list[2](feat_t[-1])
        #     loss_kd = criterion_kd(f_s, f_t)
        # elif opt.distill == 'vid':
        #     g_s = feat_s[1:-1]
        #     g_t = feat_t[1:-1]
        #     loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
        #     loss_kd = sum(loss_group)

        # else:
        #     raise NotImplementedError(opt.distill)

        # given opt.distill is always 'kd', loss_kd is hard-coded as zero, so the beta value is useless
        if split == "minimize":
            loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
        elif split == "maximize":
            loss = -loss_div

        loss = loss

        if split == "minimize" and not quiet:
            acc1, _ = accuracy(logit_s, target, topk=(1,1))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
        elif split == "maximize" and not quiet:
            kd_losses.update(loss.item(), input.size(0))
        
        # in their training loop, split is always 'minimize' or 'maximize', never 'linear', commenting out
        # elif split == "linear" and not quiet:
        #     acc1, _ = accuracy(logit_s, target, topk=(1, 1))
        #     losses.update(loss.item(), input.size(0))
        #     top1.update(acc1[0], input.size(0))
        #     kd_losses.update(loss.item(), input.size(0))


        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_value_(model_s.parameters(), clip)
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        if not quiet:
            if split == "mainimize":
                if idx % opt.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch, idx, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1))
                    sys.stdout.flush()

    
    if split == "minimize":
        if not quiet:
            print(' * Acc@1 {top1.avg:.3f} '
                  .format(top1=top1))

        return top1.avg, losses.avg
    else:
        return kd_losses.avg
    







# Their full args and implementation of the loop on small CIFAR-5 (i think)
args = {}

args.optim = 'adam'
args.gamma = 1
args.alpha = 0.5
# args.beta = 0 - never used
args.smoothing = 0.5
args.msteps = 3
# args.clip = 0.2 - the only reference to this was commented out when i got this
args.sstart = 10
args.kd_T = 2
args.distill = 'kd'

args.sgda_epochs = 10
args.sgda_learning_rate = 0.0005
args.lr_decay_epochs = [5,8,9]
args.lr_decay_rate = 0.1
args.sgda_weight_decay = 0.1#5e-4
args.sgda_momentum = 0.9

model_t = copy.deepcopy(teacher)
model_s = copy.deepcopy(student)

module_list = nn.ModuleList([])
module_list.append(model_s)
trainable_list = nn.ModuleList([])
trainable_list.append(model_s)

criterion_cls = nn.CrossEntropyLoss()
criterion_div = DistillKL(args.kd_T)
criterion_kd = DistillKL(args.kd_T)


criterion_list = nn.ModuleList([])
criterion_list.append(criterion_cls)    # classification loss
criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
criterion_list.append(criterion_kd)     # other knowledge distillation loss

# optimizer
if args.optim == "sgd":
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=args.sgda_learning_rate,
                          momentum=args.sgda_momentum,
                          weight_decay=args.sgda_weight_decay)
elif args.optim == "adam": 
    optimizer = optim.Adam(trainable_list.parameters(),
                          lr=args.sgda_learning_rate,
                          weight_decay=args.sgda_weight_decay)
elif args.optim == "rmsp":
    optimizer = optim.RMSprop(trainable_list.parameters(),
                          lr=args.sgda_learning_rate,
                          momentum=args.sgda_momentum,
                          weight_decay=args.sgda_weight_decay)

module_list.append(model_t)

if torch.cuda.is_available():
    module_list.cuda()
    criterion_list.cuda()
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    swa_model.cuda()

acc_rs = []
acc_fs = []
acc_ts = []
for epoch in range(1, args.sgda_epochs + 1):

    lr = sgda_adjust_learning_rate(epoch, args, optimizer)

    print("==> scrub unlearning ...")

    acc_r, acc5_r, loss_r = validate(retain_loader, model_s, criterion_cls, args, True)
    acc_f, acc5_f, loss_f = validate(forget_loader, model_s, criterion_cls, args, True)
    acc_rs.append(100-acc_r.item())
    acc_fs.append(100-acc_f.item())

    maximize_loss = 0
    if epoch <= args.msteps:
        maximize_loss = train_distill(epoch, forget_loader, module_list, None, criterion_list, optimizer, args, "maximize")
    train_acc, train_loss = train_distill(epoch, retain_loader, module_list, None, criterion_list, optimizer, args, "minimize",)

    print ("maximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}".format(maximize_loss, train_loss, train_acc))



