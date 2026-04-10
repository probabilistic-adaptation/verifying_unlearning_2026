from scipy.stats import wasserstein_distance

def get_wasserstein_distance(tensor1, tensor2):
    dists = []
    for i in range(len(tensor1)):
        dists.append(wasserstein_distance(tensor1[i].cpu().numpy(), tensor2[i].cpu().numpy()))
    return sum(dists) / len(tensor1)
