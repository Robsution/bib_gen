import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron block with Batch Norm and LeakyReLU.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.LeakyReLU(0.2)):
        super().__init__()
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(activation)
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)