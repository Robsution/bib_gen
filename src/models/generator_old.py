import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import MLP

class MuonGenerator(nn.Module):
    def __init__(self, 
                 noise_dim, 
                 parent_dim, 
                 max_daughters, 
                 feature_dim, 
                 hidden_dims=[256, 512, 1024]):
        super().__init__()
        self.noise_dim = noise_dim
        self.parent_dim = parent_dim
        self.max_daughters = max_daughters
        self.feature_dim = feature_dim # (Kinematics + PDG OneHot)
        
        # 1. The Common Trunk (processes Noise + Parent)
        self.input_dim = noise_dim + parent_dim
        self.trunk = MLP(self.input_dim, hidden_dims, hidden_dims[-1])
        
        # 2. Particle Feature Head
        # Outputs flat vector: N_max * feature_dim
        self.particle_head = nn.Linear(hidden_dims[-1], max_daughters * feature_dim)
        
        # 3. Multiplicity Head (Predicts N)
        # We predict a single continuous scalar representing the number of particles
        # We use Softplus to ensure N is positive
        self.count_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )

    def forward(self, noise, parent):
        """
        noise: (Batch, noise_dim)
        parent: (Batch, parent_dim)
        """
        # Concatenate inputs
        x = torch.cat([noise, parent], dim=1)
        
        # Pass through trunk
        embedding = self.trunk(x)
        
        # --- Branch A: Generate Raw Particles ---
        # Shape: (Batch, N_max * features)
        raw_flat = self.particle_head(embedding)
        # Reshape to (Batch, N_max, features)
        particles = raw_flat.view(-1, self.max_daughters, self.feature_dim)
        
        # Inside Generator forward() before the masking:
        # Assuming kinematics are the first 8 features after slicing out the mask (which we build separately)
        # feature_dim = 8 (kinematics) + num_pdgs
        
        kinematics = particles[:, :, :8]
        pdgs = particles[:, :, 8:]
        
        # Apply Tanh to kinematics [-1, 1]
        kinematics = torch.tanh(kinematics)
        
        # Apply Gumbel-Softmax to PDGs 
        # hard=True forces the output to be strictly 0s and 1s (closes the Critic loophole)
        # while still allowing gradients to flow back properly.
        pdgs = F.gumbel_softmax(pdgs, tau=1.0, hard=True, dim=-1)
        
        # Recombine
        particles = torch.cat([kinematics, pdgs], dim=2)
        
        # --- Branch B: Predict Multiplicity (N) ---
        # Shape: (Batch, 1)
        pred_N = self.count_head(embedding)
        
        # --- Branch B: Predict Multiplicity (N) ---
        # pred_N is 0.0 to 1.0 (thanks to the Sigmoid head)
        pred_N = self.count_head(embedding) 
        
        # --- Differentiable Masking ---
        # 1. Map the 0-1 decimal back to the "Particle Index" scale
        # If pred_N is 1.0 and max_daughters is 2, actual_n = 2.0
        # If pred_N is 1.0 and max_daughters is 500, actual_n = 500.0
        actual_n = pred_N * self.max_daughters 
        
        # 2. Create index list: [0, 1, 2... max_daughters-1]
        indices = torch.arange(self.max_daughters, device=noise.device).float()
        indices = indices.expand(noise.size(0), self.max_daughters) 
        
        # 3. The Shift: Subtraction of 0.5
        # This ensures that if actual_n is 1.0, the first particle (index 0) 
        # is kept and the second (index 1) is cut.
        temperature = 10.0 
        mask = torch.sigmoid((actual_n - indices - 0.5) * temperature)
        
        # Expand mask to match feature dim: (Batch, N_max, 1)
        mask = mask.unsqueeze(-1)
        
        # Apply mask to particles (Zero out "non-existent" daughters)
        masked_particles = particles * mask
        
        # We return the particles, the mask, and the predicted N (for auxiliary loss)
        return masked_particles, mask, pred_N