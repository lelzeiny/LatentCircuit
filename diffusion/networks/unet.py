import pos_encoding
import torch
import torch.nn as nn
import torch.nn.functional as F
import networks

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
        if encoding_dim > 0:
            self.encoding = UNet.encodings[encoding_type](max_diffusion_steps, encoding_dim).to(device)

        # create downward branch
        level_in_shape = (in_channels, *image_shape)
        for i, cnn_depth in enumerate(cnn_depths):
            level_net = networks.ConvNet(
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
        self._down_conv_blocks = nn.ModuleList(self._down_conv_blocks)

        # create upsampling branch
        for i in range(len(cnn_depths)-2, -1, -1):
            level_in_shape = (2 * cnn_depths[i], level_out_shape[1] * pooling_factor, level_out_shape[2] * pooling_factor)
            transpose_layer = nn.ConvTranspose2d(level_out_shape[0], cnn_depths[i], kernel_size=1, stride=pooling_factor, padding=0, output_padding=pooling_factor-1)
            level_net = networks.ConvNet(
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
        
        # embed t TODO use a better conditioning method
        if self.encoding_dim > 0:
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