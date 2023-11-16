import pos_encoding
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tfd
import numpy as np
import networks
from omegaconf import open_dict

# TODO apply pos-encoding refactor for Res-MLP and ViT
# TODO create class/vector-conditional variant of diffusion model
class DiffusionModel(nn.Module):
    backbones = {"mlp": networks.ConditionalMLP, "res_mlp": networks.ResidualMLP, "unet": networks.UNet, "cond_unet": networks.CondUNet, "vit": networks.ViT}
    time_encodings = {"sinusoid": pos_encoding.get_positional_encodings, "none": pos_encoding.get_none_encodings}

    def __init__(self, backbone, backbone_params, in_channels, image_size, encoding_type, encoding_dim, max_diffusion_steps = 100, noise_schedule = "linear", device = "cpu", **kwargs):
        super().__init__()
        if backbone == "mlp" or backbone == "res_mlp":
            with open_dict(backbone_params):
                backbone_params.update({
                    "in_size": in_channels * image_size[0] * image_size[1],
                    "out_size": in_channels * image_size[0] * image_size[1],
                    "encoding_dim": encoding_dim,
                    "device": device,
                })
        elif backbone == "unet" or backbone == "cond_unet":
            with open_dict(backbone_params):
                backbone_params.update({
                    "in_channels": in_channels,
                    "out_channels": in_channels,
                    "image_shape": (image_size[0], image_size[1]),
                    "cond_dim": encoding_dim,
                    "device": device,
                })
        elif backbone == "vit":
            with open_dict(backbone_params):
                backbone_params.update({
                    "in_channels": in_channels,
                    "out_channels": in_channels,
                    "image_size": image_size[0],
                    "encoding_dim": encoding_dim,
                    "device": device,
                })
        if encoding_dim > 0:
            self.encoding = DiffusionModel.time_encodings[encoding_type](max_diffusion_steps, encoding_dim).to(device)
        self.encoding_dim = encoding_dim
        self._reverse_model = DiffusionModel.backbones[backbone](**backbone_params)
        self.in_channels = in_channels
        self.max_diffusion_steps = max_diffusion_steps
        self.image_size = image_size
        if noise_schedule == "linear":
            beta = get_linear_sched(max_diffusion_steps, kwargs["beta_1"], kwargs["beta_T"])
            self._alpha_bar = torch.tensor(compute_alpha(beta), device = device, dtype=torch.float32)
            self._beta = torch.tensor(beta, device = device, dtype=torch.float32)
        else:
            raise NotImplementedError
        
        self._loss = nn.MSELoss(reduction = "mean")

        # cache some variables:
        self._sqrt_alpha_bar = torch.sqrt(self._alpha_bar)
        self._sqrt_alpha_bar_complement = torch.sqrt(1 - self._alpha_bar)
        self._epsilon_dist = tfd.Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
        self._sigma = torch.sqrt(self._beta)

    def __call__(self, x, t):
        # input: x is (B, C, H, W) for images, t is (B)
        # output: epsilon predictions of model
        t_embed = self.compute_pos_encodings(t)
        return self._reverse_model(x, t_embed).view(*x.shape)
    
    def loss(self, x, t):
        B = x.shape[0] # x is (B, C, H, W) for images
        assert t.shape[0] == B and len(t.shape) == 1, "t has to have shape (B)"
        # sample epsilon and generate noisy images
        epsilon = self._epsilon_dist.sample(x.shape).squeeze(dim = -1) # (B, C, H, W)
        x = self._sqrt_alpha_bar[t-1].view(B, *([1]*len(x.shape[1:]))) * x + self._sqrt_alpha_bar_complement[t-1].view(B, *([1]*len(x.shape[1:]))) * epsilon
        x = self(x, t)
        metrics = {"epsilon_theta_mean": x.detach().mean().cpu().numpy(), "epsilon_theta_std": x.detach().std().cpu().numpy()}
        return self._loss(x, epsilon), metrics
    
    def forward_samples(self, x, intermediate_every = 0):
        intermediates = [x]
        for t in range(self.max_diffusion_steps):
            epsilon = self._epsilon_dist.sample(x.shape).squeeze(dim = -1)
            x_t = self._sqrt_alpha_bar[t] * x + self._sqrt_alpha_bar_complement[t] * epsilon
            if intermediate_every and t<(self.max_diffusion_steps-1) and t % intermediate_every == 0:
                intermediates.append(x_t)
        intermediates.append(x_t) # append final image
        return intermediates

    def reverse_samples(self, B, intermediate_every = 0):
        # B: batch size
        # intermediate_every: determines how often intermediate diffusion steps are saved and returned. 0 = no intermediates returned
        batch_shape = (B, self.in_channels, self.image_size[0], self.image_size[1])
        x = self._epsilon_dist.sample(batch_shape).squeeze(dim = -1) # (B, C, H, W)
        intermediates = [x]
        for t in range(self.max_diffusion_steps, 0 , -1):
            t_vec = torch.tensor(t, device=x.device).expand(B)
            z = self._epsilon_dist.sample(batch_shape).squeeze(dim = -1) if t>1 else torch.zeros_like(x)
            x = (1.0/torch.sqrt(1 - self._beta[t-1])) * (x - (self._beta[t-1]/self._sqrt_alpha_bar_complement[t-1]) * self(x, t_vec))
            if intermediate_every and t>1 and t % intermediate_every == 0:
                intermediates.append(x)
            x = x + self._sigma[t-1] * z
        intermediates.append(x) # append final image
        # normalize x
        x_max = torch.amax(x, dim=(1,2,3), keepdim=True)
        x_min = torch.amin(x, dim=(1,2,3), keepdim=True)
        x_normalized = (x - x_min)/(x_max - x_min)
        return x_normalized, intermediates
    
    def compute_pos_encodings(self, t):
        # t has shape (B,)
        if self.encoding_dim == 0:
            return None
        B = t.shape[0]
        encoding = self.encoding[t-1, :].view(B, self.encoding_dim)
        return encoding

class CondDiffusionModel(nn.Module):
    backbones = {"mlp": networks.ConditionalMLP, "res_mlp": networks.ResidualMLP, "unet": networks.UNet, "cond_unet": networks.CondUNet, "vit": networks.ViT, "attention_gnn": networks.AttentionGNN}
    time_encodings = {"sinusoid": pos_encoding.get_positional_encodings, "none": pos_encoding.get_none_encodings}
    # conditioning vec can be arbitrary
    # here we use a torch_geometry object
    def __init__(self, backbone, backbone_params, input_shape, encoding_type, encoding_dim, max_diffusion_steps = 100, noise_schedule = "linear", device = "cpu", **kwargs):
        super().__init__()
        if backbone == "mlp" or backbone == "res_mlp":
            with open_dict(backbone_params):
                backbone_params.update({
                    "in_size": np.prod(input_shape),
                    "out_size": np.prod(input_shape),
                    "encoding_dim": encoding_dim,
                    "device": device,
                })
        elif backbone == "unet" or backbone == "cond_unet":
            with open_dict(backbone_params):
                backbone_params.update({
                    "in_channels": input_shape[0],
                    "out_channels": input_shape[0],
                    "image_shape": (input_shape[1], input_shape[2]),
                    "cond_dim": encoding_dim,
                    "device": device,
                })
        elif backbone == "vit":
            with open_dict(backbone_params):
                backbone_params.update({
                    "in_channels": input_shape[0],
                    "out_channels": input_shape[0],
                    "image_size": input_shape[1],
                    "encoding_dim": encoding_dim,
                    "device": device,
                })
        elif backbone == "attention_gnn":
            with open_dict(backbone_params):
                backbone_params.update({
                    "in_node_channels": input_shape[0],
                    "out_node_channels": input_shape[0],
                    "encoding_dim": encoding_dim,
                    "device": device,
                })
        if encoding_dim > 0:
            self.encoding = DiffusionModel.time_encodings[encoding_type](max_diffusion_steps, encoding_dim).to(device)
        self.encoding_dim = encoding_dim
        self._reverse_model = DiffusionModel.backbones[backbone](**backbone_params)
        self.input_shape = input_shape
        self.max_diffusion_steps = max_diffusion_steps
        if noise_schedule == "linear":
            beta = get_linear_sched(max_diffusion_steps, kwargs["beta_1"], kwargs["beta_T"])
            self._alpha_bar = torch.tensor(compute_alpha(beta), device = device, dtype=torch.float32)
            self._beta = torch.tensor(beta, device = device, dtype=torch.float32)
        else:
            raise NotImplementedError
        
        self._loss = nn.MSELoss(reduction = "mean")

        # cache some variables:
        self._sqrt_alpha_bar = torch.sqrt(self._alpha_bar)
        self._sqrt_alpha_bar_complement = torch.sqrt(1 - self._alpha_bar)
        self._epsilon_dist = tfd.Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
        self._sigma = torch.sqrt(self._beta)

    def __call__(self, x, cond, t):
        # input: x is (B, C, H, W) for images, t is (B), cond is (1, x)
        # note: 1 graph at a time
        # output: epsilon predictions of model
        t_embed = self.compute_pos_encodings(t)
        return self._reverse_model(x, cond, t_embed).view(*x.shape)
    
    def loss(self, x, cond, t):
        B = x.shape[0] # x is (B, C, H, W) for images
        assert t.shape[0] == B and len(t.shape) == 1, "t has to have shape (B)"
        # sample epsilon and generate noisy images
        epsilon = self._epsilon_dist.sample(x.shape).squeeze(dim = -1) # (B, C, H, W)
        x = self._sqrt_alpha_bar[t-1].view(B, *([1]*len(x.shape[1:]))) * x + self._sqrt_alpha_bar_complement[t-1].view(B, *([1]*len(x.shape[1:]))) * epsilon
        x = self(x, cond, t)
        metrics = {"epsilon_theta_mean": x.detach().mean().cpu().numpy(), "epsilon_theta_std": x.detach().std().cpu().numpy()}
        return self._loss(x, epsilon), metrics
    
    def forward_samples(self, x, intermediate_every = 0):
        intermediates = [x]
        for t in range(self.max_diffusion_steps):
            epsilon = self._epsilon_dist.sample(x.shape).squeeze(dim = -1)
            x_t = self._sqrt_alpha_bar[t] * x + self._sqrt_alpha_bar_complement[t] * epsilon
            if intermediate_every and t<(self.max_diffusion_steps-1) and t % intermediate_every == 0:
                intermediates.append(x_t)
        intermediates.append(x_t) # append final image
        return intermediates

    def reverse_samples(self, B, cond, intermediate_every = 0):
        # B: batch size
        # intermediate_every: determines how often intermediate diffusion steps are saved and returned. 0 = no intermediates returned
        batch_shape = (B, *self.input_shape)
        x = self._epsilon_dist.sample(batch_shape).squeeze(dim = -1) # (B, C, H, W)
        intermediates = [x]
        for t in range(self.max_diffusion_steps, 0 , -1):
            t_vec = torch.tensor(t, device=x.device).expand(B)
            z = self._epsilon_dist.sample(batch_shape).squeeze(dim = -1) if t>1 else torch.zeros_like(x)
            x = (1.0/torch.sqrt(1 - self._beta[t-1])) * (x - (self._beta[t-1]/self._sqrt_alpha_bar_complement[t-1]) * self(x, cond, t_vec))
            if intermediate_every and t>1 and t % intermediate_every == 0:
                intermediates.append(x)
            x = x + self._sigma[t-1] * z
        intermediates.append(x) # append final image
        # normalize x
        x_max = torch.amax(x, dim=(1,2,3), keepdim=True)
        x_min = torch.amin(x, dim=(1,2,3), keepdim=True)
        x_normalized = (x - x_min)/(x_max - x_min)
        return x_normalized, intermediates
    
    def compute_pos_encodings(self, t):
        # t has shape (B,)
        if self.encoding_dim == 0:
            return None
        B = t.shape[0]
        encoding = self.encoding[t-1, :].view(B, self.encoding_dim)
        return encoding


def get_linear_sched(T, beta_1, beta_T):
    # returns noise schedule beta as numpy array with shape (T)
    return np.linspace(beta_1, beta_T, T)

def compute_alpha(beta):
    # computes alpha^bar as numpy array with shape (T)
    # input: (T)
    alpha = 1-beta
    alpha_bar = np.multiply.accumulate(alpha)
    return alpha_bar

def debug_plot_img(x, name = "debug_img", autoscale = False):
    # x is (C, H, W) image, this function plots and saves to file
    # assumes images are [-1, 1]
    import matplotlib.pyplot as plt
    # scaling
    x = x.movedim(0,-1)
    x = (x + 1)/2 if not autoscale else (x-x.min())/(x.max()-x.min())
    plt.imshow(x.cpu().detach().numpy())
    plt.savefig(name)