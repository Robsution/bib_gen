"""Conditional models for the discrete daughter multiplicity."""

import torch
from torch import nn
from torch.nn import functional as F


class CategoricalMultiplicity(nn.Module):
    """Predict logits for every integer count from zero through ``max_n``."""

    def __init__(self, parent_dim: int = 3, max_n: int = 50, hidden_dim: int = 128) -> None:
        super().__init__()
        self.max_n = max_n
        self.net = nn.Sequential(
            nn.Linear(parent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_n + 1),
        )

    def forward(self, parent: torch.Tensor) -> torch.Tensor:
        return self.net(parent)


class NegativeBinomialMultiplicity(nn.Module):
    """Predict the parameters of a conditional negative-binomial distribution."""

    def __init__(self, parent_dim: int = 3, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = _conditional_trunk(parent_dim, hidden_dim)
        self.total_count_head = nn.Linear(hidden_dim, 1)
        self.logits_head = nn.Linear(hidden_dim, 1)

    def forward(self, parent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(parent)
        total_count = F.softplus(self.total_count_head(hidden)) + 1e-4
        return total_count, self.logits_head(hidden)


class GaussianMixtureMultiplicity(nn.Module):
    """Continuous Gaussian-mixture approximation to the count distribution."""

    def __init__(
        self,
        parent_dim: int = 3,
        components: int = 5,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.net = _conditional_trunk(parent_dim, hidden_dim)
        self.weight_head = nn.Linear(hidden_dim, components)
        self.mean_head = nn.Linear(hidden_dim, components)
        self.scale_head = nn.Linear(hidden_dim, components)

    def forward(
        self, parent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.net(parent)
        weights = F.softmax(self.weight_head(hidden), dim=-1)
        means = self.mean_head(hidden)
        scales = F.elu(self.scale_head(hidden)) + 1.0 + 1e-4
        return weights, means, scales


class DequantizedGaussianMultiplicity(nn.Module):
    """Model uniformly dequantized counts with one conditional Gaussian.

    This is intentionally named as a Gaussian baseline rather than a normalizing
    flow: the transformation is affine and has no learned sequence of flow layers.
    """

    def __init__(self, parent_dim: int = 3, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = _conditional_trunk(parent_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.scale_head = nn.Linear(hidden_dim, 1)

    def forward(self, parent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(parent)
        mean = self.mean_head(hidden)
        scale = F.softplus(self.scale_head(hidden)) + 1e-4
        return mean, scale


def _conditional_trunk(parent_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(parent_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
    )
