import torch
import torch.nn as nn

class MultiplicityPredictor(nn.Module):
    def __init__(self, parent_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(parent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # Outputs a normalized value between 0 and 1 (N / MAX_DAUGHTERS)
        )

    def forward(self, parent_kinematics):
        return self.net(parent_kinematics)