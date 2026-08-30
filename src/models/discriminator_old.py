import torch
import torch.nn as nn
from .layers import MLP

class DeepSetsDiscriminator(nn.Module):
    def __init__(self, 
                 feature_dim, 
                 parent_dim, 
                 phi_hidden_dims=[128, 256], 
                 rho_hidden_dims=[256, 128],
                 latent_dim=128):
        super().__init__()
        
        # Input dim for Phi is Daughter Features + Parent Features
        self.input_dim = feature_dim + parent_dim
        
        # 1. Phi Network (Per-Particle MLP)
        # Processes each particle independently.
        self.phi = MLP(self.input_dim, phi_hidden_dims, latent_dim)
        
        # 2. Rho Network (Global MLP)
        # Processes the summed representation of the set.
        self.rho = nn.Sequential(
            MLP(latent_dim, rho_hidden_dims, rho_hidden_dims[-1]),
            nn.Linear(rho_hidden_dims[-1], 1) 
            # Output is scalar (logit for Real/Fake)
        )

    def forward(self, daughters, parent, mask=None):
        """
        daughters: (Batch, N_max, feature_dim)
        parent: (Batch, parent_dim)
        mask: (Batch, N_max, 1) - Optional, used to zero out padding
        """
        batch_size, n_max, _ = daughters.size()
        
        # --- Local Conditioning ---
        # Expand parent to (Batch, N_max, parent_dim)
        parent_expanded = parent.unsqueeze(1).expand(batch_size, n_max, -1)
        
        # Concatenate: (Batch, N_max, feature_dim + parent_dim)
        x = torch.cat([daughters, parent_expanded], dim=2)
        
        # --- Apply Phi (Per Particle) ---
        # Flatten to (Batch * N_max, Input_Dim) for the MLP
        x_flat = x.view(-1, self.input_dim)
        
        # Latent representation for each particle
        latents = self.phi(x_flat)
        
        # Reshape back to (Batch, N_max, Latent_Dim)
        latents = latents.view(batch_size, n_max, -1)
        
        # --- Masking ---
        # Ensure padded particles contribute EXACTLY zero to the sum
        if mask is not None:
            latents = latents * mask
            
        # --- Sum Pooling (Permutation Invariant) ---
        # Sum over the N_max dimension
        global_latent = torch.sum(latents, dim=1)
        
        # --- Apply Rho (Global Classification) ---
        output = self.rho(global_latent)
        
        return output