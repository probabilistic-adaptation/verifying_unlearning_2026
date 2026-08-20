import copy

import torch
# from torch.autograd import grad
from tqdm import tqdm
from torch.func import functional_call, vmap, grad


# def fisher_information_matrix(model, train_dl, device):
#     """
#     Calculate the fisher information matrix for the given model and training data loader.

#     This is an appropriximation of the FIM, i.e. the Diagonal Empirical FIM
#     We do not calculate the covariance terms between parameters - we just calculate the variance of each param individually
#     (i.e. only the diagonal of the matrix)

#     The original authors (Golatkar et al 2020a) suggest this approx is good enough for the purpose of adding noise.

#     The FIM is computed over the retain set, since its purpose is to approx the Hessian over the retain set, 
#     used in the optimal scrubbing step.

#     The original authors find that F^-{1/2} approx's fisher noise better than F^-{1/4}
#     """
#     model.eval()
#     fisher_approximation = []
#     for parameter in model.parameters():
#         fisher_approximation.append(torch.zeros_like(parameter).to(device))
#     total = 0
#     for i, (data, label) in enumerate(tqdm(train_dl)):
#         data = data.to(device)
#         label = label.to(device)
#         predictions = torch.log_softmax(model(data), dim=-1)
#         real_batch = data.shape[0]

#         epsilon = 1e-8
#         for i in range(real_batch):
#             label_i = label[i]
#             prediction = predictions[i][label_i]
#             gradient = grad(
#                 prediction, model.parameters(), retain_graph=True, create_graph=False
#             )
#             for j, derivative in enumerate(gradient):
#                 fisher_approximation[j] += (derivative + epsilon) ** 2
#         total += real_batch
#     for i, parameter in enumerate(model.parameters()):
#         fisher_approximation[i] = fisher_approximation[i] / total

#     return fisher_approximation





def fisher_information_matrix(model, train_dl, device, fisher_chunk_size = 128):
    model.eval()

    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    fisher_approximation = {k: torch.zeros_like(v) for k, v in params.items()}
    total = 0

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
    # per-sample grads -- the tensor it hands back is still [len(train_dl batch), *param shape]
    # for every parameter, i.e. one full gradient copy per sample in the loader's batch (not per
    # chunk). For a large batch through a large model that's tens of GB regardless of chunk_size,
    # which is what actually OOMs. So instead we slice the batch into sub-batches of
    # fisher_chunk_size ourselves below and accumulate incrementally, which bounds the
    # per-sample-grad tensor itself to [fisher_chunk_size, *param shape].
    per_sample_grad = vmap(grad(compute_log_prob), in_dims=(None, None, 0, 0))

    for data, label in tqdm(train_dl):
        data = data.to(device)
        label = label.to(device)
        real_batch = data.shape[0]

        # torch.func.grad always runs with create_graph=True internally (needed for vmap to
        # work), so without no_grad here the returned per-sample grads would stay attached to
        # the ambient autograd graph instead of being freed once accumulated.
        with torch.no_grad():
            for sub_data, sub_label in zip(data.split(fisher_chunk_size), label.split(fisher_chunk_size)):
                sub_grads = per_sample_grad(params, buffers, sub_data, sub_label)
                for k in fisher_approximation:
                    fisher_approximation[k] += sub_grads[k].pow(2).sum(dim=0)

        total += real_batch

    for k in fisher_approximation:
        fisher_approximation[k] /= total + 1e-8 # to prevent underflow

    # return in the same order/type as model.parameters(), if you need a list
    return [fisher_approximation[name] for name, _ in model.named_parameters()]


def fisher(dataloaders, model, criterion, opt, epoch, device, w_and_b=True, **kwargs):
    """
    Add Fisher noise to a model's parameters
    Based on retain

    we leave a bunch of stale args in the function def_n for now, so as to maintain the pipeline structure

    Also returns a `bound_info` dict with everything needed to later compute the information-remaining
    bound of Golatkar et al. 2020a ("Eternal Sunshine of the Spotless Net"), Corollary 1 / Eq. 5,
    specialized to the Fisher-forgetting scrubbing function of Eq. 8 (h(w) = w, i.e. no Newton step):

        bound = (1/2) * sum_i (w0_i - w'_i)^2 / Sigma_i

    where w0 are these pre-scrub weights, w' are the weights of a model trained from scratch on the
    retain set, and Sigma is the per-parameter variance of the noise actually injected below (i.e. after
    the clamp). See evaluation/information_bound.py for the function that consumes this dict.
    """
    lam = kwargs['lambda']
    retain_loader = dataloaders["retain"]

    fisher_approximation = fisher_information_matrix(model, retain_loader, device)
    pre_scrub_state_dict = {name: param.detach().clone() for name, param in model.named_parameters()}
    noise_variance = {}

    with torch.no_grad():
        for i, (name, parameter) in enumerate(model.named_parameters()):

            # the original authors do not clamp the value of the noise, revisit this later
            # i have found the clamp is actually necessary, otherwise the noise destroys the model,
            # can we just make it larger?
            noise_std = torch.sqrt(lam / fisher_approximation[i]).clamp(max=2e-2) # was 1e-3
            # removing clamp to see what happens
            # noise_std = torch.sqrt(lam / fisher_approximation[i])
            noise_variance[name] = noise_std.pow(2)

            noise = noise_std * torch.empty_like(parameter).normal_(0, 1)
            # i think we need to add this multiplication back in?
            # noise = noise * 10 if parameter.shape[-1] == 10 else noise
            # noise = noise * 10 if name in ("fc.weight", "fc.bias") else noise
            parameter.add_(noise)

    bound_info = {
        "pre_scrub_state_dict": pre_scrub_state_dict,
        "noise_variance": noise_variance,
        "fisher_diag": {name: fisher_approximation[i] for i, (name, _) in enumerate(model.named_parameters())},
        "lambda": lam,
    }

    return model, 0, bound_info


# def hessian(dataset, model, loss_fn, args):
#     """
#     I think this calculates the magnitude of the second derivative of the loss for each param
#     """
#     model.eval()
#     device = f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu"
#     loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
#     train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

#     for p in model.parameters():
#         p.grad_acc = 0
#         p.grad2_acc = 0

#     for data, orig_target in tqdm(train_loader):
#         data, orig_target = data.to(device), orig_target.to(device)
#         output = model(data)
#         prob = torch.nn.functional.softmax(output, dim=-1).data

#         for y in range(output.shape[1]):
#             target = torch.empty_like(orig_target).fill_(y)
#             loss = loss_fn(output, target)
#             model.zero_grad()
#             loss.backward(retain_graph=True)
#             for p in model.parameters():
#                 if p.requires_grad:
#                     p.grad2_acc += torch.mean(prob[:, y]) * p.grad.data.pow(2)

#     for p in model.parameters():
#         p.grad2_acc /= len(train_loader)


# def get_mean_var(p, args, is_base_dist=False):
#     """
#     Get the mean and variance for a parameter based on its accumulated second derivative of the loss
#     """
#     var = copy.deepcopy(1.0 / (p.grad2_acc + 1e-8))
#     var = var.clamp(max=1e3)
#     if p.shape[0] == args.num_classes:
#         var = var.clamp(max=1e2)
#     var = args.alpha * var
#     if p.ndim > 1:
#         var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
#     if not is_base_dist:
#         mu = copy.deepcopy(p.data0.clone())
#     else:
#         mu = copy.deepcopy(p.data0.clone())

#     if p.shape[0] == args.num_classes and (
#         (args.num_indexes_to_replace == 4500 and args.dataset == "cifar10")
#         or (args.num_indexes_to_replace == 450 and args.dataset == "cifar100")
#     ):
#         mu[args.class_to_replace] = 0
#         var[args.class_to_replace] = 0.0001
    
#     # why are we adding more noise arbitrarily like this?
#     if p.shape[0] == args.num_classes:
#         # Last layer
#         var *= 10
#     elif p.ndim == 1:
#         # BatchNorm
#         var *= 10
#     return mu, var


# def fisher_new(data_loaders, model, criterion, args):
#     """
#     Add Fisher noise to a model's parameters using the FULL second derivative of the loss, instead of the fisher information matrix approximation
#     Based on retain
#     """
#     retain_loader = data_loaders["retain"]
#     dataset = retain_loader.dataset
#     for p in model.parameters():
#         p.data0 = copy.deepcopy(p.data.clone())
#     hessian(dataset, model, criterion, args)
#     for i, p in enumerate(model.parameters()):
#         mu, var = get_mean_var(p, args, False)
#         p.data = mu + var.sqrt() * torch.empty_like(p.data).normal_()
#     return model