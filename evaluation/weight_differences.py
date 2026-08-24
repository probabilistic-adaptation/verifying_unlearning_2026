import torch


def weight_distance(unlearned_model, reference_model, type = "l2", layer_wise = False):
    
    if type == "l2":
        p = 2
    if type == "l1":
        p = 1

    params_a = torch.cat([p.flatten() for p in unlearned_model.parameters()])
    params_b = torch.cat([p.flatten() for p in reference_model.parameters()])
    
    if layer_wise:
        raise ValueError("layer-wise output is not implemented yet")
    else:    
        out = torch.norm(params_a - params_b, p=p)