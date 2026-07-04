import torch
import torch.nn as nn
from Diffusion import GaussianDiffusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpectralDiffusion(GaussianDiffusion):
    def __init__(self, noise_scale, noise_min, noise_max, steps, num_bands, target_bands='high_mid'):
        super(SpectralDiffusion, self).__init__(noise_scale, noise_min, noise_max, steps)
        self.num_bands = num_bands
        self.target_bands = target_bands # 'high', 'high_mid', 'all'
        
    def _get_band_mask(self):
        mask = torch.zeros(self.num_bands, device=device)
        if self.target_bands == 'all':
            mask[:] = 1.0
        elif self.target_bands == 'high_mid':
            if self.num_bands >= 2:
                mask[self.num_bands//2:] = 1.0
            else:
                mask[:] = 1.0
        elif self.target_bands == 'high':
            mask[-1] = 1.0
        return mask
        
    def training_losses(self, model, x_start_bands, itmEmbeds, batch_index, model_feats):
        batch_size = x_start_bands.size(0)
        
        # Select timesteps
        ts = torch.randint(0, self.steps, (batch_size,)).long().to(device)
        noise = torch.randn_like(x_start_bands)
        
        # Band mask dictates which bands get noised and denoised
        band_mask = self._get_band_mask().view(1, self.num_bands, 1)
        
        if self.noise_scale != 0:
            # Apply forward diffusion process on target bands
            x_t_bands = self.q_sample(x_start_bands, ts, noise)
            # Combine noised target bands with clean non-target bands
            x_t_bands = x_t_bands * band_mask + x_start_bands * (1 - band_mask)
        else:
            x_t_bands = x_start_bands
            
        model_output_bands = model(x_t_bands, ts)
        
        # Calculate loss only on target bands
        mse = self.mean_flat((x_start_bands - model_output_bands) ** 2 * band_mask)
        
        weight = self.SNR(ts - 1) - self.SNR(ts)
        weight = torch.where((ts == 0), 1.0, weight)
        diff_loss = weight * mse
        
        # Calculate semantic/guidance loss using sum of bands
        x_start_pred = model_output_bands * band_mask + x_start_bands * (1 - band_mask)
        x_start_spatial = x_start_pred.sum(dim=1) # Reconstruct spatial signal
        
        usr_model_embeds = torch.mm(x_start_spatial, model_feats)
        usr_id_embeds = torch.mm(x_start_bands.sum(dim=1), itmEmbeds)
        gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)
        
        return diff_loss, gc_loss

    def p_sample_spectral(self, model, x_start_bands, steps):
        # We start with noise only on target bands
        band_mask = self._get_band_mask().view(1, self.num_bands, 1)
        
        if steps == 0:
            x_t_bands = x_start_bands
        else:
            t = torch.tensor([steps-1] * x_start_bands.shape[0]).to(device)
            x_t_bands = self.q_sample(x_start_bands, t)
            x_t_bands = x_t_bands * band_mask + x_start_bands * (1 - band_mask)
            
        indices = list(range(self.steps))[::-1]
        
        for i in indices:
            t = torch.tensor([i] * x_t_bands.shape[0]).to(device)
            model_mean, _ = self.p_mean_variance(model, x_t_bands, t)
            # Reconstruct bands
            x_t_bands = model_mean * band_mask + x_start_bands * (1 - band_mask)
            
        return x_t_bands.sum(dim=1) # Return spatial domain
