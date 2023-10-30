import pos_encoding
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tfd
import numpy as np
from omegaconf import open_dict

class ConvNet(nn.Module):
    def __init__(self, in_channels, image_shape, filter_sizes, cnn_strides, cnn_depths, padding="valid", residual=True, dropout=0.0, norm=True, **kwargs):
        super().__init__()
        cnn_depths = [in_channels] + cnn_depths
        self.residual = residual
        self.norm = norm
        self._conv_layers = []
        self._nonlinear_layers = []
        self._norm_layers = []
        self.in_depths = cnn_depths[:-1]
        self.out_depths = cnn_depths[1:]
        for size, stride, in_depth, out_depth in zip(filter_sizes, cnn_strides, self.in_depths, self.out_depths):
            self._norm_layers.append(nn.GroupNorm(num_groups=1, num_channels=in_depth))
            self._conv_layers.append(nn.Conv2d(in_depth, out_depth, size, stride=stride, padding=padding))
            self._nonlinear_layers.append(nn.ReLU())
        self.out_shape = get_feature_shape((in_channels, *image_shape), self._conv_layers)
        self._norm_layers = nn.ModuleList(self._norm_layers)
        self._conv_layers = nn.ModuleList(self._conv_layers)
        self._nonlinear_layers = nn.ModuleList(self._nonlinear_layers)
        self._dropout = nn.Dropout(p = dropout)

    def __call__(self, x):
        # x is (B, C, H, W)
        B, C, H, W = x.shape
        for conv_layer, nonlinear, norm, in_depth, out_depth in zip(self._conv_layers, self._nonlinear_layers, self._norm_layers, self.in_depths, self.out_depths):
            x_original = x
            if self.norm:
                x = norm(x)
            x = conv_layer(x)
            x = nonlinear(x)
            if self.residual and in_depth == out_depth:
                x = x_original + x
        return self._dropout(x)

class UNet(nn.Module): # conditional UNet class used for diffusion models
    encodings = {"sinusoid": pos_encoding.get_positional_encodings, "none": pos_encoding.get_none_encodings}

    def __init__(self, in_channels, out_channels, image_shape, cnn_depths, layers_per_block, filter_size, pooling_factor, encoding_type, encoding_dim, max_diffusion_steps, device="cpu", **kwargs):
        # length of CNN_depths determines how many levels u-net has
        super().__init__()
        self._down_conv_blocks = []
        self._up_conv_blocks = []
        self._transpose_conv_layers = []
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_shape = image_shape
        self.cnn_depths = cnn_depths
        self.filter_size = filter_size
        self.pooling_factor = pooling_factor
        self.encoding_dim = encoding_dim
        self.max_diffusion_steps = max_diffusion_steps

        # positional encoding for t
        self.encoding = UNet.encodings[encoding_type](max_diffusion_steps, encoding_dim).to(device)

        # create downward branch
        level_in_shape = (in_channels, *image_shape)
        for i, cnn_depth in enumerate(cnn_depths):
            level_net = ConvNet(
                level_in_shape[0]+encoding_dim if i==(len(cnn_depths)-1) else level_in_shape[0], 
                level_in_shape[1:], 
                [filter_size] * layers_per_block,
                [1] * layers_per_block,
                [cnn_depth] * layers_per_block,
                padding = "same",
                **kwargs
                )
            level_out_shape = level_net.out_shape
            level_in_shape = (level_out_shape[0], level_out_shape[1]//pooling_factor, level_out_shape[2]//pooling_factor)
            self._down_conv_blocks.append(level_net)
            print(level_in_shape, level_out_shape)
        self._down_conv_blocks = nn.ModuleList(self._down_conv_blocks)

        # create upsampling branch
        for i in range(len(cnn_depths)-2, -1, -1):
            level_in_shape = (2 * cnn_depths[i], level_out_shape[1] * pooling_factor, level_out_shape[2] * pooling_factor)
            print(cnn_depths[i], level_in_shape)
            transpose_layer = nn.ConvTranspose2d(level_out_shape[0], cnn_depths[i], kernel_size=1, stride=pooling_factor, padding=0, output_padding=pooling_factor-1)
            level_net = ConvNet(
                level_in_shape[0], 
                level_in_shape[1:], 
                [filter_size] * layers_per_block,
                [1] * layers_per_block,
                [cnn_depths[i]] * layers_per_block,
                padding = "same",
                **kwargs,
                )
            level_out_shape = level_net.out_shape
            self._transpose_conv_layers.append(transpose_layer)
            self._up_conv_blocks.append(level_net)
        print(level_out_shape)
        self._up_conv_blocks = nn.ModuleList(self._up_conv_blocks)
        self._transpose_conv_layers = nn.ModuleList(self._transpose_conv_layers)

        # output layer
        self._output_conv = nn.Conv2d(level_out_shape[0], out_channels, kernel_size=1, stride=1, padding="same")

    def __call__(self, x, t):
        # x is (B, C, H, W)
        B, _, _, _ = x.shape
        assert t.shape[0] == B and len(t.shape) == 1, "t has to have shape (B)"

        # downward branch
        skip_images = []
        for down_block in self._down_conv_blocks[:-1]:
            x = down_block(x)
            skip_images.append(x)
            x = F.max_pool2d(x, self.pooling_factor)
        
        # embed t
        _, _, latent_h, latent_w = x.shape
        pos_embed = self.compute_pos_encodings((latent_h, latent_w), t)
        x = torch.cat((x, pos_embed), dim = 1)
        x = self._down_conv_blocks[-1](x)

        # upward branch
        for i, (transpose_layer, up_block) in enumerate(zip(self._transpose_conv_layers, self._up_conv_blocks)):
            x = transpose_layer(x)
            x = torch.cat((x, skip_images[-(i+1)]), dim = 1)
            x = up_block(x)

        # output layer
        x = self._output_conv(x)
        return x

    def compute_pos_encodings(self, latent_shape, t):
        # t has shape (B,)
        B = t.shape[0]
        encoding_flat = self.encoding[t-1, :].view(B, self.encoding_dim, 1, 1)
        encoding = encoding_flat.expand(-1, -1, *latent_shape)
        return encoding

def get_feature_size(input_shape, conv_layers):
    # returns flattened size
    x = torch.zeros(input_shape)
    for conv_layer in conv_layers:
        x = conv_layer(x)
    return torch.numel(x)

def get_feature_shape(input_shape, conv_layers):
    # returns flattened size
    x = torch.zeros(input_shape)
    for conv_layer in conv_layers:
        x = conv_layer(x)
    return x.shape

class MLP(nn.Module):
    def __init__(self, num_layers, model_width, in_size, out_size, skip = False, layernorm = False, **kwargs):
        super().__init__()
        assert (not skip) or (in_size == out_size), "input and output dimensions must be equal for skip connection"
        self.num_layers = num_layers
        self.model_width = model_width
        self.in_size = in_size
        self.out_size = out_size
        layers = []
        for i in range(num_layers):
            inputs = in_size if i == 0 else model_width
            outputs = out_size if i == num_layers-1 else model_width
            layers.append(nn.Linear(inputs, outputs))
            layers.append(nn.ReLU())
        layers = layers[:-1] # remove final activation layer
        self.skip = skip
        self.layernorm = layernorm
        self._nn = nn.Sequential(*layers)
        self._ln = nn.LayerNorm(in_size)
    
    def __call__(self, x):
        # x is (B, ...)
        B = x.shape[0]
        x = x.view(B, -1)
        output = self._nn(self._ln(x)) if self.layernorm else self._nn(x) # TODO add dropout
        return x + output if self.skip else output

class ConditionalMLP(nn.Module):
    encodings = {"sinusoid": pos_encoding.get_positional_encodings}

    def __init__(self, num_layers, model_width, in_size, out_size, encoding_type, encoding_dim, max_diffusion_steps, device = "cpu", **kwargs):
        super().__init__()
        self.encoding_type = encoding_type
        self.encoding = ConditionalMLP.encodings[encoding_type](max_diffusion_steps, encoding_dim).to(device)
        self.mlp = MLP(num_layers, model_width, in_size + encoding_dim, out_size, **kwargs)

    def __call__(self, x, t):
        # t is a vector with shape (B)
        B = x.shape[0]
        assert t.shape[0] == B and len(t.shape) == 1, "t has to have shape (B)"
        t_encoded = self.encoding[t-1]
        x_original = x.view(B, -1)
        x = torch.cat((x_original, t_encoded), dim = -1)
        return x_original + self.mlp(x)

class ResidualMLP(nn.Module):
    encodings = {"sinusoid": pos_encoding.get_positional_encodings}

    def __init__(self, num_blocks, layers_per_block, model_width, in_size, out_size, encoding_type, encoding_dim, max_diffusion_steps, device = "cpu", **kwargs):
        # several residual blocks, each an MLP
        super().__init__()
        self.encoding_type = encoding_type
        self.encoding = ResidualMLP.encodings[encoding_type](max_diffusion_steps, encoding_dim).to(device)
        self.num_blocks = num_blocks
        blocks = []
        self._encoder = nn.Linear(in_size + encoding_dim, model_width)
        self._decoder = nn.Linear(model_width, out_size)
        for i in range(num_blocks):
            new_block = MLP(layers_per_block, model_width, model_width, model_width, skip = True, layernorm = True, **kwargs)
            blocks.append(new_block)
        self._model = nn.Sequential(*blocks)

    def __call__(self, x, t):
        # t is a vector with shape (B)
        B = x.shape[0]
        assert t.shape[0] == B and len(t.shape) == 1, "t has to have shape (B)"
        t_encoded = self.encoding[t-1]
        x = x.view(B, -1)
        x = torch.cat((x, t_encoded), dim = -1)
        x_original = self._encoder(x)
        x = self._model(x_original)
        x = self._decoder(x_original + x)
        return x

class DiffusionModel(nn.Module):
    backbones = {"mlp": ConditionalMLP, "res_mlp": ResidualMLP, "unet": UNet}
    
    def __init__(self, backbone, backbone_params, in_channels, image_size, max_diffusion_steps = 100, noise_schedule = "linear", device = "cpu", **kwargs):
        super().__init__()
        if backbone == "mlp" or backbone == "res_mlp":
            with open_dict(backbone_params):
                backbone_params.update({
                    "in_size": in_channels * image_size[0] * image_size[1],
                    "out_size": in_channels * image_size[0] * image_size[1],
                    "max_diffusion_steps": max_diffusion_steps,
                    "device": device,
                })
        elif backbone == "unet":
            with open_dict(backbone_params):
                backbone_params.update({
                    "in_channels": in_channels,
                    "out_channels": in_channels,
                    "image_shape": (image_size[0], image_size[1]),
                    "max_diffusion_steps": max_diffusion_steps,
                    "device": device,
                })
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
        # input: x is (B, C, H, W), t is (B)
        # output: epsilon predictions of model
        return self._reverse_model(x, t).view(*x.shape)
    
    def loss(self, x, t):
        B, C, H, W  = x.shape
        assert t.shape[0] == B and len(t.shape) == 1, "t has to have shape (B)"
        # sample epsilon and generate noisy images
        epsilon = self._epsilon_dist.sample((B, C, H, W)).squeeze(dim = -1) # (B, C, H, W)
        x = self._sqrt_alpha_bar[t-1].view(B, 1, 1, 1) * x + self._sqrt_alpha_bar_complement[t-1].view(B, 1, 1, 1) * epsilon
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