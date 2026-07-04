import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralBandMasking(nn.Module):
    def __init__(self, mask_ratio=0.3):
        super(SpectralBandMasking, self).__init__()
        self.mask_ratio = mask_ratio

    def mask_bands(self, x_bands):
        """
        x_bands: (N, num_bands, D)
        """
        if not self.training or self.mask_ratio <= 0:
            return x_bands
            
        N, num_bands, D = x_bands.shape
        # Create a mask for each band
        mask = torch.rand(num_bands, device=x_bands.device) > self.mask_ratio
        
        # Ensure at least one band is kept
        if not mask.any():
            mask[torch.randint(0, num_bands, (1,))] = True
            
        mask = mask.float().view(1, num_bands, 1)
        return x_bands * mask

    def consistency_loss(self, full_repr, masked_repr):
        """
        full_repr, masked_repr: (N, D) spatial embeddings
        Calculates MSE consistency loss
        """
        return F.mse_loss(full_repr, masked_repr)
