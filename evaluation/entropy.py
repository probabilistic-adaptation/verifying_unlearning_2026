import torch

def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)

# def m_entropy(p, labels, dim=-1, keepdim=False):
#     log_prob = torch.where(p > 0, p.log(), torch.tensor(1e-30).to(p.device).log())
#     reverse_prob = 1 - p
#     log_reverse_prob = torch.where(
#         p > 0, p.log(), torch.tensor(1e-30).to(p.device).log()
#     )
#     modified_probs = p.clone()
#     modified_probs[:, labels] = reverse_prob[:, labels]
#     modified_log_probs = log_reverse_prob.clone()
#     modified_log_probs[:, labels] = log_prob[:, labels]
#     return -torch.sum(modified_probs * modified_log_probs, dim=dim, keepdim=keepdim)


# add actual math equation underneath
def m_entropy(p, labels, dim=-1, keepdim=False):
    
    """
    Modified Entropy
    """
    
    epsilon = 1e-30
    
    # main prob
    log_prob = torch.where(p > 0, p.log(), torch.tensor(epsilon).to(p.device).log())
    
    # reverse prob
    reverse_prob = 1 - p
    log_reverse_prob = torch.where(reverse_prob > 0, reverse_prob.log(), torch.tensor(epsilon).to(p.device).log())
    
    # bring together
    idx = torch.arange(p.shape[0], device=p.device)
    modified_probs = p.clone()
    modified_probs[idx, labels] = reverse_prob[idx, labels]
    modified_log_probs = log_reverse_prob.clone()
    modified_log_probs[idx, labels] = log_prob[idx, labels]
    
    # output
    return -torch.sum(modified_probs * modified_log_probs, dim=dim, keepdim=keepdim)
