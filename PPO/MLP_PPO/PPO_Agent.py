# Import libraries: --->
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# Import custom modules: --->
from .actor_critic import ActorCritic

# Device configuration: --->
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOAgent:
    """Proximal Policy Optimization Agent"""
    def __init__(self, state_dim, action_dim, lr = 1e-3, gamma = 0.99, eps_clip = 0.2, k_epochs = 4):
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.buffer = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': []}

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            action_probs, _ = self.policy(state)

        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['logprobs'].append(action_logprob)

        return action.item()

    def update(self):
        """Updates network parameters using the PPO Clipped Objective."""
        # Compute discounted rewards (Returns): --->
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer['rewards']), reversed(self.buffer['is_terminals'])):
            if is_terminal:
                discounted_reward = 0

            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalize returns: --->
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.stack(self.buffer['states']).to(device).detach()
        old_actions = torch.stack(self.buffer['actions']).to(device).detach()
        old_logprobs = torch.stack(self.buffer['logprobs']).to(device).detach()

        epoch_actor_loss = 0
        epoch_critic_loss = 0
        epoch_total_loss = 0

        # Optimize policy for K epochs: --->
        for _ in range(self.k_epochs):
            action_probs, state_values = self.policy(old_states)
            dist = Categorical(action_probs)

            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()

            # State values are squeezed to match reward tensor shape: --->
            state_values = torch.squeeze(state_values)

            # Generalized Advantage Estimation (simplified): --->
            advantages = rewards - state_values.detach()

            # Advantage Normalization (Crucial for Actor-Critic stability): --->
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO Ratio: (pi_theta / pi_theta__old): --->
            ratios = torch.exp(logprobs - old_logprobs)

            # Surrogate Loss: --->
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Separate losses for logging: --->
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * nn.MSELoss()(state_values, rewards)
            entropy_bonus = 0.01 * dist_entropy.mean()

            loss = actor_loss + critic_loss - entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Record the actor, critic and total losses epoch-wise: --->
            epoch_actor_loss += actor_loss.item()
            epoch_critic_loss += critic_loss.item()
            epoch_total_loss += loss.item()

        # Clear memory: --->
        for key in self.buffer:
            self.buffer[key].clear()

        return epoch_actor_loss / self.k_epochs, epoch_critic_loss / self.k_epochs, epoch_total_loss / self.k_epochs
