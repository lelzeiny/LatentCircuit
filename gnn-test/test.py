import torch_geometric as tg
import torch_geometric.nn as tgn
import torch
import torch.nn as nn

dataset = tg.datasets.Planetoid(root = "../datasets/graph/planetoid", name="Cora", transform=tg.transforms.NormalizeFeatures())

def print_dataset(dataset):
    print()
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('===========================================================================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

class GCN(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = tgn.GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = tgn.GCNConv(hidden_channels, hidden_channels)
        self.conv3 = tgn.GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index)
        return x

def update(model, data, optim, loss_fn):
    model.train()
    optim.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optim.step()
    return loss

def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    return test_acc

model = GCN(16)
data = dataset[0] # since only have 1 graph in dataset
optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss_fn = nn.CrossEntropyLoss()
for step_num in range(200):
    loss = update(model, data, optim, loss_fn)
    if step_num % 5 == 0:
        acc = test(model, data)
        print(step_num, loss.detach().cpu().item(), acc)