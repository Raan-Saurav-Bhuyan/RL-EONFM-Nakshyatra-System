import torch
from torch import nn

class BaseEncoder(nn.Module):
    """
    Base Encoder (f_theta): A deep neural network that maps augmented OPM
    vectors to a representation vector h. This is the output used for
    downstream clustering.
    """
    def __init__(self, input_dim, representation_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, representation_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class ProjectionHead(nn.Module):
    """
    Projection Head (g_phi): A shallow MLP that maps the representation h
    to a latent space z where the contrastive loss is applied.
    """
    def __init__(self, representation_dim=32, projection_dim=16):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(representation_dim, representation_dim),
            nn.ReLU(True),
            nn.Linear(representation_dim, projection_dim)
        )

    def forward(self, h):
        return self.head(h)

class ContrastiveModel(nn.Module):
    """Combines the Base Encoder and Projection Head for end-to-end training."""
    def __init__(self, input_dim, representation_dim=32, projection_dim=16):
        super().__init__()
        self.base_encoder = BaseEncoder(input_dim, representation_dim)
        self.projection_head = ProjectionHead(representation_dim, projection_dim)

    def forward(self, x):
        """Passes input through encoder and projection head: x -> h -> z"""
        h = self.base_encoder(x)
        z = self.projection_head(h)
        return z
