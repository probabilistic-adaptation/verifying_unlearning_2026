import torch
# import utils
from trainer.utils import AverageMeter
from evaluation.entropy import entropy, m_entropy
from evaluation.accuracy import accuracy
import wandb
import torch.nn.functional as F

def validate(val_loader, model, criterion, print_freq, device, w_and_b = True):
    """
    Run evaluation
    """
    losses = []
    probs = []
    targets = []
    losses_meter = AverageMeter()
    top1_meter = AverageMeter()
    entropy_meter = AverageMeter()
    m_entropy_meter = AverageMeter()


    # switch to evaluate mode
    model.eval()
    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # print(f"[validate] model.training = {model.training}")

    for i, (image, target) in enumerate(val_loader):
        image = image.to(device)
        target = target.to(device)

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        
        prob_dist = F.softmax(output, dim=-1)
        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.detach(), target)[0]

        # update trackers
        losses_meter.update(loss.item(), image.size(0))
        top1_meter.update(prec1.item(), image.size(0))
        entropy_meter.update( torch.mean(entropy(prob_dist)).item(), image.size(0) )
        m_entropy_meter.update( torch.mean(m_entropy(prob_dist, target)).item(), image.size(0) )

        # append loss, prob dist, and targets
        losses.append( loss.item() )
        probs.append( prob_dist.cpu() )
        targets.append( target.cpu() )


        if i % print_freq == 0:
            print(
                f"[{i}/{len(val_loader)}]\t"
                f"Loss {losses_meter.val:.4f} ({losses_meter.avg:.4f})\t"
                f"Accuracy {top1_meter.val:.3f} ({top1_meter.avg:.3f})\t"
                f"Entropy {entropy_meter.val:.4f} ({entropy_meter.avg:.4f})\t"
                f"M-Entropy {m_entropy_meter.val:.4f} ({m_entropy_meter.avg:.4f})\t"
            )

            if w_and_b and (i + 1) % print_freq == 0:
                wandb.log({"val_loss (batch)": losses_meter.val, "val_acc (batch)": top1_meter.val, "val_entropy (batch)": entropy_meter.val, "val_m_entropy (batch)": m_entropy_meter.val})

    print("val_accuracy {top1.avg:.3f}\n".format(top1=top1_meter))

    return losses_meter.avg, top1_meter.avg, entropy_meter.avg, m_entropy_meter.avg, losses, torch.cat(probs), torch.cat(targets)
