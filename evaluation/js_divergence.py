from torch.nn import functional as F
import torch

def JSDiv(p, q):
    m = (p+q)/2
    return 0.5*F.kl_div(torch.log(p), m) + 0.5*F.kl_div(torch.log(q), m)