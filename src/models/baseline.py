"""Neural networks used by the conditioned SN-GAN baseline."""

import torch
from torch import nn
from torch.nn.utils import spectral_norm


class BIBGenerator(nn.Module):
    """Generate a padded daughter-particle family conditioned on its parent and size.

    The first ``kinematic_dim`` output features are bounded to ``[-1, 1]`` to
    match the processed dataset. The remaining features are unnormalized logits
    over particle species. A separate sigmoid head predicts which padded slots
    are active.
    """

    def __init__(
        self,
        noise_dim: int,
        parent_dim: int,
        max_daughters: int,
        kinematic_dim: int = 8,
        species_dim: int = 16,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.max_daughters = max_daughters
        self.kinematic_dim = kinematic_dim
        self.species_dim = species_dim

        self.global_trunk = nn.Sequential(
            nn.Linear(noise_dim + parent_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * max_daughters),
            nn.LeakyReLU(0.2),
        )
        self.particle_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
        )
        self.kinematics_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, kinematic_dim),
            nn.Tanh(),
        )
        # Keep this attribute name stable so existing checkpoints remain loadable.
        self.pdg_head = nn.Linear(hidden_dim // 2, species_dim)
        self.mask_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        noise: torch.Tensor,
        parent: torch.Tensor,
        normalized_n: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return generated features and soft slot masks.

        ``normalized_n`` must have shape ``(batch, 1)`` and contain the desired
        family size divided by ``max_daughters``.
        """

        batch_size = noise.shape[0]
        context = torch.cat((noise, parent, normalized_n), dim=-1)
        hidden = self.global_trunk(context)
        hidden = hidden.view(batch_size, self.max_daughters, -1)
        hidden = self.particle_net(hidden)

        kinematics = self.kinematics_head(hidden)
        species_logits = self.pdg_head(hidden)
        mask = self.mask_head(hidden)
        features = torch.cat((kinematics, species_logits), dim=-1) * mask
        return features, mask


class DeepSetsDiscriminator(nn.Module):
    """Score complete padded families with a permutation-invariant network."""

    def __init__(self, feature_dim: int, parent_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        # ``phi`` and ``rho`` are retained as checkpoint-stable DeepSets names.
        self.phi = nn.Sequential(
            spectral_norm(nn.Linear(feature_dim, hidden_dim)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.LeakyReLU(0.2),
        )
        self.pool_norm = nn.LayerNorm(hidden_dim)
        self.rho = nn.Sequential(
            spectral_norm(nn.Linear(hidden_dim + parent_dim + 1, hidden_dim)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden_dim // 2, 1)),
        )

    def forward(
        self,
        features: torch.Tensor,
        parent: torch.Tensor,
        mask: torch.Tensor,
        normalized_n: torch.Tensor,
    ) -> torch.Tensor:
        masked_features = features * mask
        particle_embeddings = self.phi(masked_features) * mask
        family_embedding = self.pool_norm(particle_embeddings.sum(dim=1))
        context = torch.cat((family_embedding, parent, normalized_n), dim=-1)
        return self.rho(context)
