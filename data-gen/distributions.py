import torch

def get_distribution(shape, dist_type, dist_params):
    if dist_type == "uniform":
        uni_max = dist_params["max"]
        uni_min = dist_params["min"]
        return (uni_max - uni_min) * torch.rand(shape) + uni_min