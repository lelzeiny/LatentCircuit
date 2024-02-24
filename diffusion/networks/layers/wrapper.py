import torch
import torch.nn as nn

class BatchWrapper(nn.Module):
    
    def __init__(self, net):
        super().__init__()
        self.net = net

    def __call__(self, x, edge_index, edge_attr=None, **kwargs):
        # process x, edge_index, edge_attr
        # x: (B, V, F)
        # edge_index: (2, E)
        # edge attributes: also batched
        B, V, F = x.shape
        _, E = edge_index.shape
        
        x_unbatched = x.view(B * V, F)

        edge_attr_unbatched = edge_attr.view(B * E, -1) if edge_attr is not None else None
        
        edge_index_unbatched = edge_index.movedim(-1, 0).unsqueeze(dim=0).expand(B, E, 2)
        edge_index_offset = torch.arange(0, V*B, V, device=edge_index.device, dtype=edge_index.dtype).view(B, 1, 1)
        edge_index_unbatched = edge_index_unbatched + edge_index_offset
        edge_index_unbatched = edge_index_unbatched.reshape(B * E, 2).movedim(0, -1)
        
        output_unbatched = self.net(x_unbatched, edge_index_unbatched, edge_attr=edge_attr_unbatched, **kwargs) # (B*V, F)
        
        # reshape output
        output = output_unbatched.view(B, V, -1)
        return output