import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class MuonGenerator(nn.Module):
    def __init__(self, noise_dim, parent_dim, max_daughters, kin_dim=8, pdg_dim=16, hidden_dim=256):
        super().__init__()
        self.max_daughters = max_daughters
        self.kin_dim = kin_dim
        self.pdg_dim = pdg_dim
        self.total_feature_dim = kin_dim + pdg_dim
        
        # 1. Global Trunk: Fuses Noise and Parent Context
        self.global_trunk = nn.Sequential(
            nn.Linear(noise_dim + parent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * max_daughters),
            nn.LeakyReLU(0.2)
        )
        
        # 2. Particle-Wise Network: Processes each slot independently
        self.particle_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2)
        )
        
        # 3. The Output Heads (Simultaneous Generation)
        # Kinematics: Tanh for [-1, 1] scaling
        self.kinematics_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, kin_dim),
            nn.Tanh()
        )
        
        # PDG Identity: Raw logits (Apply Gumbel-Softmax in training loop if needed)
        self.pdg_head = nn.Linear(hidden_dim // 2, pdg_dim)
        
        # Multiplicity Mask: Sigmoid for existence probability [0, 1]
        self.mask_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z, parent):
        batch_size = z.size(0)
        
        # Fuse and expand to [Batch, Max_Daughters, Hidden]
        x = torch.cat([z, parent], dim=-1)
        x = self.global_trunk(x)
        x = x.view(batch_size, self.max_daughters, -1)
        
        # Process particles
        particle_hidden = self.particle_net(x)
        
        # Generate all components at once
        kinematics = self.kinematics_head(particle_hidden)
        pdg_logits = self.pdg_head(particle_hidden)
        mask = self.mask_head(particle_hidden)
        
        # Concatenate features [Batch, Max_Daughters, 24]
        features = torch.cat([kinematics, pdg_logits], dim=-1)
        
        # Smoothly zero out non-existent particles
        features = features * mask
        
        # Return pred_n (sum of masks) just for your Aux MSE logging, if desired
        pred_n = torch.sum(mask, dim=1) 
        
        return features, mask, pred_n