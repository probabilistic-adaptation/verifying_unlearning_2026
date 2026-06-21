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
    entropies = []
    m_entropies = []
    losses_meter = AverageMeter()
    top1_meter = AverageMeter()
    entropy_meter = AverageMeter()
    m_entropy_meter = AverageMeter()


    # switch to evaluate mode
    model.eval()

    for i, (image, target) in enumerate(val_loader):
        image = image.to(device)
        target = target.to(device)

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target) # this is the actual loss we would normally use
            per_sample_losses = F.cross_entropy(output, target, reduction='none') # we gather this just for measurements later


        prob_dist = F.softmax(output, dim=-1)
        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.detach(), target)[0]

        # update trackers
        losses_meter.update(loss.item(), image.size(0))
        top1_meter.update(prec1.item(), image.size(0))

        entr = entropy(prob_dist)
        entropy_meter.update( torch.mean(entr).item(), image.size(0))

        m_entr = m_entropy(prob_dist, target)
        m_entropy_meter.update( torch.mean( m_entr ).item(), image.size(0) )

        # append loss, prob dist, and targets
        losses.append( per_sample_losses.cpu() )
        probs.append( prob_dist.cpu() )
        targets.append( target.cpu() )
        entropies.append( entr.cpu() )
        m_entropies.append( m_entr.cpu() )



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

    out = {
        "avg_loss": losses_meter.avg,
        "avg_acc": top1_meter.avg,
        "avg_entr": entropy_meter.avg,
        "avg_m_entr": m_entropy_meter.avg,
        "losses": torch.cat(losses),
        "probs": torch.cat(probs),
        "targets": torch.cat(targets),
        "entropies": torch.cat(entropies),
        "m_entropies": torch.cat(m_entropies)
    }

    return out
