import torch
import torch.nn as nn
from torch.distributions import Categorical

class LightweightTemporalCNN(nn.Module):
    """
    2D CNN Backbone designed to process the Temporal State Window.
    Input Shape: (Batch, Channels=10_Years, Height=K_Clusters, Width=Features)
    """
    def __init__(self, in_channels=10, k_clusters=5, num_features=18):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten()
        )

        # Calculate flattened dimension: --->
        h_out = k_clusters // 2
        w_out = num_features // 2
        self.fc_input_dim = 32 * h_out * w_out

    def forward(self, x):
        return self.conv_block(x)

class ActorCriticCNN(nn.Module):
    """Actor-Critic architecture for PPO utilizing the shared CNN backbone."""
    def __init__(self, action_dim=2, in_channels=10, k_clusters=5, num_features=18):
        super().__init__()
        self.backbone = LightweightTemporalCNN(in_channels, k_clusters, num_features)

        # Actor: Policy Output (Classification): --->
        self.actor = nn.Sequential(
            nn.Linear(self.backbone.fc_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic: Value Output (Regression): --->
        self.critic = nn.Sequential(
            nn.Linear(self.backbone.fc_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError("Use act() or evaluate() instead of forward().")

    def act(self, state):
        features = self.backbone(state)
        action_probs = self.actor(features)
        dist = Categorical(action_probs)
        action = dist.sample()

        return action, dist.log_prob(action)

    def evaluate(self, state, action):
        features = self.backbone(state)
        action_probs = self.actor(features)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(features)

        return action_logprobs, state_value.squeeze(-1), dist_entropy
