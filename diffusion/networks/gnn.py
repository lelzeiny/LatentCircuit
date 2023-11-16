import torch
import torch_geometric as tg
import torch_geometric.nn as tgn
import torch.nn as nn
import torch.functional as F

NUM_EDGE_FEATURES = 4 # x1, y1, x2, y2
class GNN(nn.Module):
    def __init__(self, num_features=3, hidden_size=32, target_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.target_size = target_size
        self.convs = [tgn.GATConv(self.num_features, self.hidden_size, edge_dim = NUM_EDGE_FEATURES),
                      tgn.GATConv(self.hidden_size, self.hidden_size, edge_dim = NUM_EDGE_FEATURES)]
        self.linear = nn.Linear(self.hidden_size, self.target_size)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr=edge_attr) # adding edge features here!
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr) # edge features here as well
        x = self.linear(x)

        return F.relu(x) 