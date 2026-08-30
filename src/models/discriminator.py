import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class DeepSetsDiscriminator(nn.Module):
    def __init__(self, feature_dim, parent_dim, hidden_dim=128):
        super().__init__()
        
        self.phi = nn.Sequential(
            spectral_norm(nn.Linear(feature_dim, hidden_dim)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.LeakyReLU(0.2)
        )
        
        # ADDED: LayerNorm to stabilize massive variance from Sum pooling
        self.pool_norm = nn.LayerNorm(hidden_dim)
        
        self.rho = nn.Sequential(
            spectral_norm(nn.Linear(hidden_dim + parent_dim, hidden_dim)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden_dim // 2, 1))
        )

    def forward(self, features, parent, mask):
        features = features * mask
        h = self.phi(features)
        
        h_masked = h * mask
        h_agg = torch.sum(h_masked, dim=1)
        
        # ADDED: Stabilize the pooled vector before concatenating
        h_agg = self.pool_norm(h_agg)
        
        global_repr = torch.cat([h_agg, parent], dim=-1)
        score = self.rho(global_repr)
        
        return score