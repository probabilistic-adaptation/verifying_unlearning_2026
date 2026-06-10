import torch
from torch.nn import functional as F
def l2_distance(p, q):
    """
    p and q are presumed to be concatenated softmaxed-outputs from two models
    size: n_data x n_classes
    """
    
    distances = torch.sqrt(torch.sum(torch.square(p - q), dim=1))
    return distances.mean().item()