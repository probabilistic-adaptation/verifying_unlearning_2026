from torch.nn import functional as F
import torch
import torch.nn as nn


# Use decorator to ensure no gradients are calculated
@torch.no_grad()
def get_activation_distance(model1, model2, dataloader, device='cuda'):
    # sftmx = nn.Softmax(dim=1)
    distances = []
    # Iterate through the dataloader
    for batch in dataloader:
        if len(batch) == 3:  # Ensure there are exactly 3 items
            x, _, _ = batch  # Unpack
        elif len(batch) == 2:  # If only 2 items are returned
            x, _ = batch
        else:
            raise ValueError("Unexpected batch size: expected 2 or 3 items")
        x = x.to(device)

        # Get the outputs of both models
        model1_out = model1(x)
        model2_out = model2(x)

        # Calculate softmax difference and then L2 norm (Euclidean distance)
        diff = torch.sqrt(torch.sum(torch.square(F.softmax(model1_out, dim=1)
                                                 - F.softmax(model2_out, dim=1)), axis=1))

        # Move the result to CPU and detach from the computation graph
        diff = diff.detach().cpu()
        distances.append(diff)

    # Concatenate all distances and return the mean
    distances = torch.cat(distances, axis=0)
    return distances.mean().item()
