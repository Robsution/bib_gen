import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class CategoricalMultiplicity(nn.Module):
    """ 1. Pure Categorical Classification """
    def __init__(self, parent_dim=3, max_n=50, hidden_dim=128):
        super().__init__()
        self.max_n = max_n
        self.net = nn.Sequential(
            nn.Linear(parent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_n + 1) # Classes 0 through max_n
        )

    def forward(self, parent):
        return self.net(parent) # Returns raw logits

class NegBinomMultiplicity(nn.Module):
    """ 2. Negative Binomial (Count) Regression """
    def __init__(self, parent_dim=3, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(parent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.total_count_head = nn.Linear(hidden_dim, 1)
        self.logits_head = nn.Linear(hidden_dim, 1)

    def forward(self, parent):
        h = self.net(parent)
        # total_count must be strictly positive
        total_count = F.softplus(self.total_count_head(h)) + 1e-4
        logits = self.logits_head(h)
        return total_count, logits

class MDNMultiplicity(nn.Module):
    """ 3. Mixture Density Network (Continuous) """
    def __init__(self, parent_dim=3, num_gaussians=5, hidden_dim=128):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.net = nn.Sequential(
            nn.Linear(parent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.pi_head = nn.Linear(hidden_dim, num_gaussians)
        self.mu_head = nn.Linear(hidden_dim, num_gaussians)
        self.sigma_head = nn.Linear(hidden_dim, num_gaussians)

    def forward(self, parent):
        h = self.net(parent)
        pi = F.softmax(self.pi_head(h), dim=-1)
        mu = self.mu_head(h)
        # Sigma must be strictly positive
        sigma = F.elu(self.sigma_head(h)) + 1.0 + 1e-4
        return pi, mu, sigma

class DequantizedFlowMultiplicity(nn.Module):
    """ 4. 1D Conditional Flow with Uniform Dequantization """
    def __init__(self, parent_dim=3, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(parent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # For a basic 1D Affine Flow, we predict the shift (mu) and scale (sigma)
        self.shift = nn.Linear(hidden_dim, 1)
        self.scale = nn.Linear(hidden_dim, 1)

    def forward(self, parent):
        h = self.net(parent)
        mu = self.shift(h)
        sigma = F.softplus(self.scale(h)) + 1e-4
        return mu, sigma