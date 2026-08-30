import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# ==========================================
# STAGE 1: THE ORACLE
# ==========================================
class CategoricalOracle(nn.Module):
    def __init__(self, parent_dim=3, max_daughters=50, hidden_dim=128):
        super().__init__()
        self.max_daughters = max_daughters
        
        self.net = nn.Sequential(
            nn.Linear(parent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_daughters + 1) # Classes 0 to max_daughters
        )

    def forward(self, parent):
        return self.net(parent) # Returns raw logits


# ==========================================
# STAGE 2: THE CONDITIONED GAN
# ==========================================
class MuonGenerator(nn.Module):
    def __init__(self, noise_dim, parent_dim, max_daughters, kin_dim=8, pdg_dim=16, hidden_dim=256):
        super().__init__()
        self.max_daughters = max_daughters
        self.kin_dim = kin_dim
        self.pdg_dim = pdg_dim
        
        # CONDITIONING ADDED: We add +1 to the input for the target_N scalar
        self.global_trunk = nn.Sequential(
            nn.Linear(noise_dim + parent_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * max_daughters),
            nn.LeakyReLU(0.2)
        )
        
        self.particle_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2)
        )
        
        self.kinematics_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, kin_dim),
            nn.Tanh()
        )
        
        self.pdg_head = nn.Linear(hidden_dim // 2, pdg_dim)
        
        self.mask_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z, parent, target_n):
        batch_size = z.size(0)
        
        # Fuse Noise, Parent, and the Target N requested by the Oracle
        x = torch.cat([z, parent, target_n], dim=-1)
        x = self.global_trunk(x)
        x = x.view(batch_size, self.max_daughters, -1)
        
        particle_hidden = self.particle_net(x)
        
        kinematics = self.kinematics_head(particle_hidden)
        pdg_logits = self.pdg_head(particle_hidden)
        mask = self.mask_head(particle_hidden)
        
        features = torch.cat([kinematics, pdg_logits], dim=-1)
        features = features * mask
        
        return features, mask

class DeepSetsDiscriminator(nn.Module):
    def __init__(self, feature_dim, parent_dim, hidden_dim=128):
        super().__init__()
        
        self.phi = nn.Sequential(
            spectral_norm(nn.Linear(feature_dim, hidden_dim)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.LeakyReLU(0.2)
        )
        
        self.pool_norm = nn.LayerNorm(hidden_dim)
        
        # CONDITIONING ADDED: +1 to input for the N_condition
        self.rho = nn.Sequential(
            spectral_norm(nn.Linear(hidden_dim + parent_dim + 1, hidden_dim)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden_dim // 2, 1))
        )

    def forward(self, features, parent, mask, n_condition):
        features = features * mask
        h = self.phi(features)
        
        # Sum Pooling ensures the magnitude is proportional to particle count
        h_masked = h * mask
        h_agg = torch.sum(h_masked, dim=1)
        h_agg = self.pool_norm(h_agg)
        
        # Concatenate aggregate, parent, and the N_condition label
        global_repr = torch.cat([h_agg, parent, n_condition], dim=-1)
        score = self.rho(global_repr)
        
        return score