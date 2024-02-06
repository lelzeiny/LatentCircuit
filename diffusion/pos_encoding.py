import torch
import numpy as np
import torch_geometric as tg

class SinusoidPosEncoding():
    def __init__(self, model_dim, num_heads): # give each head the full embedding
        self._pos_encoding = None
        self.model_dim = model_dim
        self.num_heads = num_heads
    
    def __call__(self, pos_idx):
        if self._pos_encoding is None or self._pos_encoding.shape[0] < pos_idx.shape[0]:
            self._pos_encoding = get_positional_encodings(2 * pos_idx.shape[0], self.model_dim // self.num_heads).repeat(1, self.num_heads)/np.sqrt(self.model_dim)
        return self._pos_encoding[:pos_idx.shape[0], :]

class NoneEncoding():
    def __init__(self, shape = [1]):
        self.shape = shape

    def __call__(self, pos_idx):
        return torch.zeros(self.shape)

class LaplacianEncoding():
    def __init__(self, encoding_dim):
        self.encoding_dim = encoding_dim
        self.encodings = tg.transforms.AddLaplacianEigenvectorPE(k=self.encoding_dim, is_undirected=True)

    def __call__(self, cond):
        cond_embedded = self.encodings(cond)
        return cond_embedded.laplacian_eigenvector_pe

def get_positional_encodings(num_pos, model_dim):
    # returns sin and cos positional encodings, each with model_dim/2 dimensions
    # pos: max number of positions
    # output: (num_pos, d_m)
    idx = (torch.arange(0, model_dim, 2)/model_dim).unsqueeze(0) # (1, d_m/2)
    pos_idx = torch.arange(num_pos).unsqueeze(-1) # (pos, 1)
    theta = pos_idx/torch.pow(10000.0, idx)
    embeddings = torch.cat((torch.sin(theta), torch.cos(theta)), dim = -1) # (pos, d_m)
    return embeddings[:, :model_dim]

def get_none_encodings(num_pos, model_dim):
    embeddings = torch.arange(num_pos, dtype=torch.float32).unsqueeze(-1).expand(-1, model_dim) # (pos, d_m)
    return embeddings