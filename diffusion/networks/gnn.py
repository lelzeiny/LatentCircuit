import torch
import torch_geometric as tg
import torch_geometric.nn as tgn
import torch.nn as nn
import torch.functional as F
from .mlp import FiLM, MLP
from .vit import AttentionBlock
import networks.layers as layers

def get_conv_layer(layer_type, in_channels, out_channels, **layer_kwargs):
    layer_fns = {
        "gcn": tgn.GCNConv,
        "sage": tgn.SAGEConv,
        "gin": tgn.GINConv,
        "transformer": layers.TransformerConv,
        "gated": layers.GatedGraphConv,
        "gat": tgn.GATv2Conv,
        # "gine": tgn.GINEConv,
    }
    if layer_type == "gin":
        layer_params = {
            "nn": MLP(
                num_layers = layer_kwargs.get("nn_num_layer", 2), 
                model_width = layer_kwargs.get("nn_hidden_width", out_channels),
                in_size = in_channels, 
                out_size = out_channels,
            ),
            **layer_kwargs
        }
    elif layer_type == "gated":
        assert in_channels <= out_channels
        layer_params = {
            "out_channels": out_channels,
            **layer_kwargs
        }
    elif layer_type == "gat":
        if layer_kwargs.get("concat", True):
            assert (out_channels % layer_kwargs.get("heads", 1) == 0), "out channels must be divisible by number of heads in GAT"
            channel_divisor = layer_kwargs.get("heads", 1)
        else:
            channel_divisor = 1
        layer_params = {
            "in_channels": in_channels,
            "out_channels": out_channels // channel_divisor,
            **layer_kwargs
        }
    else:
        layer_params = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            **layer_kwargs
        }
    layer = layer_fns[layer_type](**layer_params)
    if layer_type == "gat":
        # use batch wrapper since tgn does not support batched dim
        layer = layers.BatchWrapper(layer)
    return layer

class GConvLayer(nn.Module):
    # TODO allow configuration of conv layer type
    def __init__(self, in_node_features, out_node_features):
        super().__init__()
        self._layer = tgn.GCNConv(in_node_features, out_node_features)
    
    def forward(self, x_in):
        x, data, t_embed = x_in
        edge_index = data.edge_index
        return self._layer(x, edge_index), data, t_embed

class ResGNNBlock(nn.Module):
    def __init__(self, in_node_features, out_node_features, hidden_node_features, cond_node_features, edge_features, num_layers, encoding_dim, residual=True, norm=True, dropout=0.0, conv_params={"layer_type": "gcn"}, device="cpu", **kwargs):
        super().__init__()
        self.in_node_features = in_node_features
        self.out_node_features = out_node_features
        self.hidden_node_features = hidden_node_features
        self.edge_features = edge_features
        self.residual = residual
        if residual:
            assert in_node_features == out_node_features, "input and output features must be equal to perform residual connection"
        self._gconv_layers = []
        self._linear_layers = []

        self._cond_layer = FiLM(encoding_dim, hidden_node_features, channel_axis=-1) if encoding_dim>0 else None
        for i in range(num_layers):
            in_features = in_node_features + cond_node_features if i==0 else hidden_node_features
            out_features = hidden_node_features if i<(num_layers-1) else out_node_features
            self._gconv_layers.append(get_conv_layer(
                in_channels=in_features, 
                out_channels=hidden_node_features,
                **conv_params
                ))
            self._linear_layers.append(nn.Linear(hidden_node_features, out_features))
        
        self._gconv_layers = nn.ModuleList(self._gconv_layers)
        self._linear_layers = nn.ModuleList(self._linear_layers)
        # self.linear = nn.Linear(self.hidden_node_features, self.out_node_features)
        if norm:
            self._norm = nn.GroupNorm(1, hidden_node_features)
        else:
            self._norm = None
        self._nonlinear = nn.ReLU()
        self._dropout = nn.Dropout(p = dropout)

    def forward(self, x, data, t): # data is conditioning info
        B, V, F = x.shape
        cond_x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        cond_x = cond_x.view(1, *cond_x.shape).expand(B, -1, -1)
        # TODO replace with GATConv or better, and make use of edge attributes
        x_skip = x
        x = torch.cat((x, cond_x), dim=-1)
        for i, (linear, conv) in enumerate(zip(self._linear_layers[:-1], self._gconv_layers[:-1])):
            if self._norm is not None and x.shape[-1] == self.hidden_node_features:
                x = torch.movedim(x, -1, 1)
                x = self._norm(x)
                x = torch.movedim(x, 1, -1)
            x = conv(x, edge_index)
            x = self._nonlinear(x)
            x = linear(x)
            x = self._nonlinear(x)
            x = self._dropout(x)
        x = self._gconv_layers[-1](x, edge_index)
        if (not self._cond_layer is None):
            x = self._cond_layer(x, t)
        x = self._nonlinear(x)
        x = self._linear_layers[-1](x)
        if self.residual:
            x = x + x_skip 
        return x 

class AttGNNBlock(nn.Module):
    def __init__(self, 
                 in_node_features, 
                 out_node_features, 
                 hidden_node_features, 
                 cond_node_features,
                 attention_extra_features, 
                 edge_features, 
                 num_layers, 
                 encoding_dim, 
                 residual=True, 
                 norm=True, 
                 dropout=0.0, 
                 conv_params={"layer_type": "gcn"}, 
                 device="cpu", 
                 **kwargs):
        super().__init__()
        self.in_node_features = in_node_features
        self.out_node_features = out_node_features
        self.hidden_node_features = hidden_node_features
        self.attention_extra_features = attention_extra_features
        self.edge_features = edge_features
        self.residual = residual
        if residual:
            assert in_node_features == out_node_features, "input and output features must be equal to perform residual connection"
        self._gconv_layers = []
        self._attention_layers = []
        self._linear_layers = []

        self._cond_layer = FiLM(encoding_dim, hidden_node_features, channel_axis=-1) if encoding_dim>0 else None
        for i in range(num_layers):
            in_features = in_node_features + cond_node_features if i==0 else hidden_node_features
            out_features = hidden_node_features if i<(num_layers-1) else out_node_features
            self._gconv_layers.append(get_conv_layer(
                in_channels=in_features, 
                out_channels=hidden_node_features,
                **conv_params
                ))
            att_model_size = hidden_node_features + cond_node_features + attention_extra_features
            self._attention_layers.append(AttentionBlock(kwargs["num_heads"], att_model_size, kwargs["ff_num_layers"], kwargs["ff_size_factor"], dropout))
            self._linear_layers.append(nn.Linear(att_model_size, out_features))
        
        self._gconv_layers = nn.ModuleList(self._gconv_layers)
        self._attention_layers = nn.ModuleList(self._attention_layers)
        self._linear_layers = nn.ModuleList(self._linear_layers)
        # self.linear = nn.Linear(self.hidden_node_features, self.out_node_features)
        if norm:
            self._norm = nn.GroupNorm(1, hidden_node_features)
        else:
            self._norm = None
        self._nonlinear = nn.ReLU()
        self._dropout = nn.Dropout(p = dropout)

    def forward(self, x, data, t, att_extra_input = None): # data is conditioning info
        B, V, F = x.shape
        assert att_extra_input is None or att_extra_input.shape[-1] == self.attention_extra_features, "extra attention features must have right shape"
        cond_x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        cond_x = cond_x.view(1, *cond_x.shape).expand(B, -1, -1)
        # TODO replace with GATConv or better, and make use of edge attributes
        x_skip = x
        x = torch.cat((x, cond_x), dim=-1)
        x_att_features = torch.cat((cond_x, att_extra_input), dim=-1) if att_extra_input is not None else cond_x
        for i, (linear, conv, attention) in enumerate(zip(self._linear_layers[:-1], self._gconv_layers[:-1], self._attention_layers[:-1])):
            if self._norm is not None and x.shape[-1] == self.hidden_node_features:
                x = torch.movedim(x, -1, 1)
                x = self._norm(x)
                x = torch.movedim(x, 1, -1)
            x = conv(x, edge_index)
            x = self._nonlinear(x)
            x = torch.cat((x, x_att_features), dim=-1)
            x = attention(x)
            x = linear(x)
            x = self._nonlinear(x)
            x = self._dropout(x)
        x = self._gconv_layers[-1](x, edge_index)
        if (not self._cond_layer is None):
            x = self._cond_layer(x, t)
        x = self._nonlinear(x)
        x = torch.cat((x, x_att_features), dim=-1)
        x = self._attention_layers[-1](x)
        x = self._linear_layers[-1](x)
        if self.residual:
            x = x + x_skip 
        return x

class ResGNN(nn.Module):
    blocks = {"res": ResGNNBlock, "att": AttGNNBlock}
    def __init__(self, in_node_features, out_node_features, hidden_size, hidden_node_features, cond_node_features, edge_features, layers_per_block, encoding_dim, dropout=0.0, device="cpu", block_type="res", **kwargs):
        super().__init__()
        self.in_node_features = in_node_features
        self.out_node_features = out_node_features
        self.hidden_node_features = hidden_node_features
        self.edge_features = edge_features

        self._gnn_blocks = []
        self.use_enc = not (hidden_size == in_node_features == out_node_features)
        if self.use_enc:
            self._gnn_blocks.append(GConvLayer(in_node_features, hidden_size))
        for i, hidden_node_size in enumerate(hidden_node_features):
            self._gnn_blocks.append(ResGNN.blocks[block_type](
                in_node_features=hidden_size, # note that this means after each block there is a large bottleneck
                out_node_features=hidden_size,
                hidden_node_features=hidden_node_size,
                cond_node_features=cond_node_features,
                edge_features=edge_features,
                num_layers=layers_per_block,
                encoding_dim=encoding_dim,
                residual=True,
                norm=True,
                dropout=dropout,
                device=device,
                **kwargs,
            ))
        if self.use_enc:
            self._gnn_blocks.append(GConvLayer(hidden_size, out_node_features))
        self._network = nn.Sequential(*self._gnn_blocks)
        print("ENCODER USED IN RESGNN", self.use_enc)

    def forward(self, x, cond, t_embed):
        x_skip = x
        x,_,_ = self._network((x, cond, t_embed))
        return (x + x_skip if self.use_enc else x)

class AttGNN(nn.Module):
    # Same as ResGNN, but with attention layer in between resBlocks
    def __init__(
            self, 
            in_node_features, 
            out_node_features, 
            hidden_size, 
            hidden_node_features,
            attention_node_features, 
            cond_node_features, 
            attention_extra_features,
            edge_features, 
            layers_per_block, 
            encoding_dim,
            conv_params,
            dropout=0.0,
            device="cpu",
            **kwargs, # should contain attention attention parameters
            ):
        super().__init__()
        self.in_node_features = in_node_features
        self.out_node_features = out_node_features
        self.hidden_node_features = hidden_node_features
        self.attention_node_features = attention_node_features
        self.attention_extra_features = attention_extra_features
        self.edge_features = edge_features

        gnn_blocks = []
        self.use_enc = not (hidden_size == in_node_features == out_node_features)
        if self.use_enc:
            gnn_blocks.append(GConvLayer(in_node_features, hidden_size))
        for i, (hidden_node_size, attention_node_size) in enumerate(zip(hidden_node_features, attention_node_features)):
            gnn_blocks.append(ResGNNBlock(
                in_node_features=hidden_size, # note that this means after each block there is a large bottleneck
                out_node_features=hidden_size,
                hidden_node_features=hidden_node_size,
                cond_node_features=cond_node_features,
                edge_features=edge_features,
                num_layers=layers_per_block,
                encoding_dim=encoding_dim,
                conv_params=conv_params,
                residual=True,
                norm=True,
                dropout=dropout,
                device=device,
            ))
            gnn_blocks.append(AttGNNBlock(
                in_node_features=hidden_size, # note that this means after each block there is a large bottleneck
                out_node_features=hidden_size,
                hidden_node_features=attention_node_size,
                cond_node_features=cond_node_features,
                attention_extra_features=attention_extra_features,
                edge_features=edge_features,
                num_layers=1,
                encoding_dim=encoding_dim,
                conv_params=conv_params,
                residual=True,
                norm=True,
                dropout=dropout,
                device=device,
                **kwargs,
            ))
        if self.use_enc:
            gnn_blocks.append(GConvLayer(hidden_size, out_node_features))
        self._gnn_blocks = nn.ModuleList(gnn_blocks)
        print("ENCODER USED IN ATTGNN", self.use_enc)

    def forward(self, x, cond, t_embed):
        x_skip = x
        for block in self._gnn_blocks:
            if isinstance(block, AttGNNBlock): # include attention conditioning
                x = block(x, cond, t_embed, att_extra_input=x) # x_skip
            else:
                x = block(x, cond, t_embed)
        return (x + x_skip if self.use_enc else x)

class GraphUNet(nn.Module):
    def __init__(
            self, 
            in_node_features, 
            out_node_features, 
            hidden_node_features, # list
            cond_node_features, 
            edge_features, 
            blocks_per_level, # list
            layers_per_block,
            level_block="res", 
            device="cpu", 
            **kwargs
        ):
        # length of CNN_depths determines how many levels u-net has
        super().__init__()
        self._down_conv_blocks = []
        self._up_conv_blocks = []
        self.in_node_features = in_node_features
        self.out_node_features = out_node_features
        self.hidden_node_features = hidden_node_features
        self.edge_features = edge_features
        self.cond_node_features = cond_node_features
        self.blocks_per_level = blocks_per_level
        self.layers_per_block = layers_per_block
        self.level_block = level_block
        self.device=device

        # create downward branch
        for i, (hidden_size, num_blocks) in enumerate(zip(hidden_node_features, blocks_per_level)):
            level_in_size = in_node_features if i==0 else hidden_node_features[i-1]
            if self.level_block == "res":
                level_in_layer = GConvLayer(level_in_size, hidden_size)
                level_blocks = [ResGNNBlock(
                    in_node_features = hidden_size, 
                    out_node_features = hidden_size, 
                    hidden_node_features = hidden_size, 
                    cond_node_features = cond_node_features, 
                    edge_features = edge_features, 
                    num_layers = layers_per_block,
                    device = device,
                    **kwargs
                    ) for _ in range(num_blocks)]
                if i == len(hidden_node_features)-1:
                    level_blocks.append(GConvLayer(hidden_size, level_in_size))
                level_net = nn.Sequential(level_in_layer, *level_blocks)
            else:
                raise NotImplementedError
            
            self._down_conv_blocks.append(level_net)
        self._down_conv_blocks = nn.ModuleList(self._down_conv_blocks)

        # create upsampling branch
        for i in range(len(hidden_node_features)-2, -1, -1):
            level_in_size = 2 * hidden_node_features[i]
            level_out_size = hidden_node_features[i-1] if i>0 else out_node_features
            hidden_size = hidden_node_features[i]
            num_blocks = blocks_per_level[i]
            if self.level_block == "res":
                level_in_layer = GConvLayer(level_in_size, hidden_size)
                level_blocks = [ResGNNBlock(
                    in_node_features = hidden_size, 
                    out_node_features = hidden_size, 
                    hidden_node_features = hidden_size, 
                    cond_node_features = cond_node_features, 
                    edge_features = edge_features, 
                    num_layers = layers_per_block,
                    device = device,
                    **kwargs
                    ) for _ in range(layers_per_block)]
                level_out_layer = GConvLayer(hidden_size, level_out_size)
                level_net = nn.Sequential(level_in_layer, *level_blocks, level_out_layer)
            else:
                raise NotImplementedError
            self._up_conv_blocks.append(level_net)
        self._up_conv_blocks = nn.ModuleList(self._up_conv_blocks)

    def __call__(self, x, data, t_enc):
        # x is (B, V, F)
        B, _, _ = x.shape
        assert t_enc.shape[0] == B and len(t_enc.shape) == 2, "t has to have shape (B, E)"
        x_skip = x

        # downward branch
        skip_images = []
        for down_block in self._down_conv_blocks[:-1]:
            x, _, _ = down_block((x, data, t_enc))
            skip_images.append(x)

        x, _, _ = self._down_conv_blocks[-1]((x, data, t_enc))

        # upward branch
        for i, up_block in enumerate(self._up_conv_blocks):
            x = torch.cat((x, skip_images[-(i+1)]), dim = -1)
            x, _, _ = up_block((x, data, t_enc))
        
        return x + x_skip