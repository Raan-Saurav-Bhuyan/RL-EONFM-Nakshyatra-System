import torch
import torch.nn as nn
import torch.optim as optim
from .gnn_model import ActorCriticGNN

class GNNRolloutBuffer:
    def __init__(self):
        self.states = []
        self.adjs = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        self.states.clear()
        self.adjs.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()

class PPOAgentGNN:
    """GNN-backed PPO Agent for explicit spatial fault localization."""
    def __init__(self, lr = 1e-3, gamma = 0.99, K_epochs = 100, eps_clip = 0.2):
        self.gamma, self.eps_clip, self.K_epochs = gamma, eps_clip, K_epochs
        self.buffer = GNNRolloutBuffer()

        self.policy = ActorCriticGNN()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCriticGNN()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, adj):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        adj_tensor = torch.FloatTensor(adj).unsqueeze(0)

        with torch.no_grad():
            action, logprob = self.policy_old.act(state_tensor, adj_tensor)

        self.buffer.states.append(state_tensor)
        self.buffer.adjs.append(adj_tensor)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(logprob)

        return action.item()

    def update(self):
        # Early exit if the buffer is completely empty: --->
        if len(self.buffer.states) == 0:
            return 0.0, 0.0, 0.0

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal: discounted_reward = 0

            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32)

        # Safe standard deviation calculation for batches of size 1: --->
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        else:
            rewards = rewards - rewards.mean()

        old_states = torch.cat(self.buffer.states).detach()
        old_adjs = torch.cat(self.buffer.adjs).detach()
        old_actions = torch.cat(self.buffer.actions).detach()
        old_logprobs = torch.cat(self.buffer.logprobs).detach()

        total_loss_val, actor_loss_val, critic_loss_val = 0.0, 0.0, 0.0

        # Actual PPO Optimization implementation: --->
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_adjs, old_actions)
            ratios = torch.exp(logprobs - old_logprobs)
            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            a_loss = -torch.min(surr1, surr2).mean()
            c_loss = 0.5 * self.MseLoss(state_values, rewards)
            loss = a_loss + c_loss - 0.01 * dist_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss_val += loss.item()
            actor_loss_val += a_loss.item()
            critic_loss_val += c_loss.item()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

        return actor_loss_val / self.K_epochs, critic_loss_val / self.K_epochs, total_loss_val / self.K_epochs
