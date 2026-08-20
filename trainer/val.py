import torch
# import utils
from trainer.utils import AverageMeter
from evaluation.entropy import entropy, m_entropy
from evaluation.accuracy import accuracy
import wandb
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad

def naive_validate(val_loader, model, criterion, device):
    """
    Bare-bones version of `validate`: just the average loss over `val_loader`, without collecting
    probs/entropies/fisher info. For call sites (e.g. relearn_time, which re-evaluates the forget
    loss every epoch) that only need that one number and would otherwise pay for a lot of unused
    bookkeeping.
    """
    losses_meter = AverageMeter()

    model.eval()

    with torch.no_grad():
        for image, target in val_loader:
            image = image.to(device)
            target = target.to(device)

            output = model(image)
            loss = criterion(output, target)

            losses_meter.update(loss.item(), image.size(0))

    return losses_meter.avg


def validate(val_loader, model, criterion, print_freq, device, w_and_b = True, compute_fisher = False, fisher_chunk_size = 128):
    """
    Run evaluation

    If compute_fisher=True, also accumulate over val_loader (in the same pass, no extra loader
    iteration needed):
      - "fisher_diag": the diagonal empirical Fisher Information Matrix, per parameter element
        (the same quantity unlearn.fisher.fisher_information_matrix returns).
      - "grad_norm": the L2 norm of the gradient of the mean cross-entropy loss over the whole
        val_loader (Becker & Liebig, "Evaluating Machine Unlearning via Epistemic Uncertainty",
        Eq. 10-11), i.e. what evaluation.residual_information.efficacy_upper_bound needs.
    Both come from the same per-sample gradients of log p(y|x) w.r.t. the parameters: the FIM is
    their sum of squares, the loss gradient is (the negative of) their sum -- so getting both costs
    only one extra (vmap'd) forward/backward per batch, on top of the no_grad one already done for
    accuracy/entropy. This is off by default since most call sites (e.g. per-epoch training
    validation) don't need it and it isn't free.
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

    if compute_fisher:
        params = {k: v.detach() for k, v in model.named_parameters()}
        buffers = {k: v.detach() for k, v in model.named_buffers()}
        fisher_diag = {k: torch.zeros_like(v) for k, v in params.items()}
        grad_sum = {k: torch.zeros_like(v) for k, v in params.items()}
        total_fisher = 0

        def compute_log_prob(params, buffers, sample, label):
            # functional_call expects a batch dim, so add one back for a single sample
            sample = sample.unsqueeze(0)
            logits = functional_call(model, (params, buffers), (sample,))
            log_probs = torch.log_softmax(logits, dim=-1)
            # indexing log_probs[0, label] directly has no vmap batching rule when label is
            # itself a vmap+grad-transformed tensor -- nll_loss does, and is equivalent here
            # since it's a batch of 1 (mean reduction over one element = that element)
            return -torch.nn.functional.nll_loss(log_probs, label.unsqueeze(0))

        # grad w.r.t. the first arg (params); vmap batches over dim 0 of sample/label,
        # broadcasting params/buffers (in_dims=None) across the batch.
        # NOTE: vmap's own `chunk_size` kwarg only bounds *peak* memory while it computes the
        # per-sample grads -- the tensor it hands back is still [len(val_loader batch), *param
        # shape] for every parameter, i.e. one full gradient copy per sample in the loader's
        # batch (not per chunk). For a batch of 512 through an 11M-param ResNet that's ~23GB
        # regardless of chunk_size, which is what actually OOMs. So instead we slice the batch
        # into sub-batches of fisher_chunk_size ourselves below and accumulate incrementally,
        # which bounds the per-sample-grad tensor itself to [fisher_chunk_size, *param shape].
        per_sample_grad = vmap(grad(compute_log_prob), in_dims=(None, None, 0, 0))

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

        if compute_fisher:
            # torch.func.grad always runs with create_graph=True internally (needed for vmap
            # to work), so without no_grad here the returned per-sample grads would stay
            # attached to the ambient autograd graph instead of being freed once accumulated.
            with torch.no_grad():
                for sub_image, sub_target in zip(image.split(fisher_chunk_size), target.split(fisher_chunk_size)):
                    sub_grads = per_sample_grad(params, buffers, sub_image, sub_target)
                    for k in fisher_diag:
                        fisher_diag[k] += sub_grads[k].pow(2).sum(dim=0)
                        grad_sum[k] += sub_grads[k].sum(dim=0)
            total_fisher += image.shape[0]

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

    if compute_fisher:
        epsilon = 1e-8
        named_params = list(model.named_parameters())

        out["fisher_diag"] = [fisher_diag[name] / (total_fisher + epsilon) for name, _ in named_params]
        out["grad_norm"] = torch.linalg.norm(
            torch.cat([(grad_sum[name] / (total_fisher + epsilon)).flatten() for name, _ in named_params])
        ).item()

    return out
