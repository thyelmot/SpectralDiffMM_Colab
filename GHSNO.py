import torch
import torch.nn as nn
import torch.nn.functional as F

class GHSNO(nn.Module):
    """
    Graph HyperSpectral Neural Operator layer
    Fuses multimodal features in the spectral domain across multiple bands
    """
    def __init__(self, in_features, out_features, num_bands):
        super(GHSNO, self).__init__()
        self.num_bands = num_bands
        
        # Spectral attention weight for each band (modality)
        self.band_weights = nn.Parameter(torch.ones(num_bands))
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, adj, embeds_list):
        """
        embeds_list: list of tensors of shape (N, D), length = num_bands, OR a single tensor (N, D)
        adj: Sparse adjacency matrix (N, N)
        """
        if isinstance(embeds_list, torch.Tensor):
            # Fallback to standard GCN behavior for individual modality processing in CL
            return torch.spmm(adj, embeds_list)

        # Stack modalities as bands
        x_bands = torch.stack(embeds_list, dim=1) # (N, num_bands, D)
        
        # Apply band attention
        weights = F.softmax(self.band_weights, dim=0).view(1, self.num_bands, 1)
        x_weighted = x_bands * weights
        
        # Fuse bands
        x_fused = x_weighted.sum(dim=1)
        
        # Graph convolution
        x_gcn = torch.spmm(adj, x_fused)
        
        # Transform
        out = self.linear(x_gcn)
        return self.activation(out)
