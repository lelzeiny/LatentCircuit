import torch
import torch.distributions as dist

def get_distribution(dist_type, dist_params):
    if dist_type == "uniform":
        return dist.Uniform(**dist_params)
    elif dist_type == "bernoulli":
        return dist.Bernoulli(**dist_params)
    elif dist_type == "cond_poisson":
        return ConditionalPoisson(**dist_params)
    elif dist_type == "cond_exp_bernoulli":
        return ConditionalExpBernoulli(**dist_params)
    elif dist_type == "clipped_exp":
        return ClippedExp(**dist_params)
    
class ConditionalPoisson:
    def __init__(self, scale):
        self.scale = scale
    
    def sample(self, cond, sample_shape = None):
        rate = self.scale * cond
        distribution = dist.Poisson(rate)
        sample = distribution.sample(sample_shape)
        return sample
    
class ConditionalExpBernoulli:
    def __init__(self, scale, prob_clip=0.5, prob_multiplier=0.1):
        # apply clip before multiplier
        self.scale = scale
        self.prob_clip = prob_clip
        self.prob_multiplier = prob_multiplier
    
    def sample(self, cond, sample_shape = torch.Size([])):
        rate = cond / self.scale # positive number
        prob = self.prob_multiplier * torch.clip(torch.exp(-rate), max=self.prob_clip)
        distribution = dist.Bernoulli(probs=prob)
        sample = distribution.sample(sample_shape)
        return sample
    
class ClippedExp:
    def __init__(self, scale, clip_min, clip_max):
        self.dist = dist.Exponential(rate = 1/scale)
        self.clip_min = clip_min
        self.clip_max = clip_max
    
    def sample(self, sample_shape = torch.Size([])):
        sample = self.dist.sample(sample_shape)
        return torch.clip(sample, self.clip_min, self.clip_max)