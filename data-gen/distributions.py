import torch
import torch.distributions as dist

def get_distribution(dist_type, dist_params):
    if dist_type == "uniform":
        return dist.Uniform(**dist_params)
    elif dist_type == "bernoulli":
        return dist.Bernoulli(**dist_params)