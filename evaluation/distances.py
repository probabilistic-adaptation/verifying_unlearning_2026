
import torch
from torch.nn import functional as F
from tqdm import tqdm

def l2_distance(p, q):
    """
    p and q are presumed to be concatenated softmaxed-outputs from two models
    size: n_data x n_classes
    """
    
    distances = torch.sqrt(torch.sum(torch.square(p - q), dim=1))
    return distances.mean().item()





# def get_wasserstein_distance(tensor1, tensor2):
#     dists = []
#     for i in range(len(tensor1)):
#         dists.append(wasserstein_distance(tensor1[i].cpu().numpy(), tensor2[i].cpu().numpy()))
#     return sum(dists) / len(tensor1)





def JSDiv(p, q):
    m = (p + q) / 2
    return (0.5 * F.kl_div(torch.log(m), p, reduction='batchmean') + 0.5 * F.kl_div(torch.log(m), q, reduction='batchmean')).item()


def ZRFScore(tmodel, gold_model, forget_dl, device):
    """
    ZRF/UnLearningScore
    """
    model_preds = []
    gold_model_preds = []
    with torch.no_grad():
        for batch in tqdm(forget_dl):
            # x, y, cy = batch
            x, _ = batch
            x = x.to(device)
            model_output = tmodel(x)
            gold_model_output = gold_model(x)
            model_preds.append(F.softmax(model_output, dim=1).detach().cpu())
            gold_model_preds.append(F.softmax(gold_model_output, dim=1).detach().cpu())

    model_preds = torch.cat(model_preds, axis=0)
    gold_model_preds = torch.cat(gold_model_preds, axis=0)
    return 1 - JSDiv(model_preds, gold_model_preds)


def absolute_distance(p, q):
    """
    p and q presumed to be blocks of tensors of output probabilities (after softmax)
    size: n_data x n_classes
    """
    return torch.mean(torch.sum(torch.abs((p - q)), dim = 1)).item()


def kl_divergence(p, q):
    p_log = torch.log(p + 1e-20)  
    q_log = torch.log(q + 1e-20) 
    kl_div = torch.sum(p * (p_log - q_log), dim=1)
    return kl_div

def kl_div_metric(forget_probs_one, forget_probs_two, retain_probs_one, retain_probs_two):
    """
    in Chien et al, retrained goes first
    """ 
    kl_f = kl_divergence(forget_probs_one, forget_probs_two)
    kl_r = kl_divergence(retain_probs_one, retain_probs_two)
    return torch.mean( torch.cat([kl_f, kl_r], dim = 0) ).item()