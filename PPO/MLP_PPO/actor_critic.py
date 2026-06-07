# Import libraries: --->
# import torch
import torch.nn as nn
# import torch.optim as optim
# from torch.distributions import Categorical
# import numpy as np

class ActorCritic(nn.Module):
    """Lightweight Shared MLP for Policy and Value prediction."""
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # Shared backbone: --->
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Actor head (Policy): --->
        self.actor = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic head (Value): --->
        self.critic = nn.Sequential(
            nn.Linear(64, 1)
        )

    def forward(self, state):
        features = self.shared(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)

        return action_probs, state_value
