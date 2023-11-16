import torch
import torch_geometric as tg
import torch_geometric.nn as tgn
import torch.nn as nn
import torch.functional as F

class AttentionGNN(nn.Module):
    def __init__(self, in_node_features, out_node_features, hidden_node_features, edge_features, num_layers, dropout=0.0):
        super().__init__()
        self.in_node_features = in_node_features
        self.out_node_features = out_node_features
        self.hidden_node_features = hidden_node_features
        self.edge_features = edge_features
        self._gconv_layers = []
        for i in range(num_layers):
            in_features = in_node_features if i==0 else hidden_node_features
            out_features = hidden_node_features if i<(num_layers-1) else out_node_features
            self._gconv_layers.append(tgn.GATConv(in_features, out_features, edge_dim=edge_features))
        self.linear = nn.Linear(self.hidden_node_features, self.out_node_features)
        self._nonlinear = nn.ReLU()
        self._dropout = nn.Dropout(p = dropout)

    # TODO properly condition on t
    def forward(self, x, data, t): # data is conditioning info
        cond_x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        import ipdb; ipdb.set_trace()
        # TODO debug this
        x = torch.cat((x, cond_x), dim=-1)
        for conv in self._gconv_layers[:-1]:
            x = conv(x, edge_index, edge_attr=edge_attr) # adding edge features here!
            x = self._nonlinear(x)
            x = self._dropout(x)
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr) # edge features here as well
        x = self.linear(x)
        return x