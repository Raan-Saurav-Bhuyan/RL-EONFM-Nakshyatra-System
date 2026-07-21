import os
import torch
import torch.nn as nn
import torch.optim as optim
from .cnn_model import ActorCriticCNN

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()

class PPOAgentCNN:
    """CNN-backed Proximal Policy Optimization Agent with best-model checkpointing."""
    def __init__(
        self,
        action_dim = 2,
        in_channels = 5,
        lr_actor = 1e-4, lr_critic = 3e-4,
        gamma=0.99,
        K_epochs = 10,
        eps_clip = 0.2,
        save_dir = "models/CNN_PPO", checkpoint_name = "best_cnn_ppo.pt"):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()

        # Model checkpoint directory and best-loss tracker: --->
        self.save_dir = save_dir
        self.checkpoint_name = checkpoint_name
        os.makedirs(self.save_dir, exist_ok = True)
        self.best_total_loss = float('inf')

        self.policy = ActorCriticCNN(action_dim = action_dim, in_channels = in_channels)
        self.optimizer = optim.Adam([
            {'params': self.policy.backbone.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCriticCNN(action_dim = action_dim, in_channels = in_channels)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        # State shape from env: (5, 5, 18). Add batch dimension -> (1, 5, 5, 18): --->
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, action_logprob = self.policy_old.act(state_tensor)

        self.buffer.states.append(state_tensor)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.item()

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32)

        if rewards.shape[0] > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-7)

        old_states = torch.cat(self.buffer.states).detach()
        old_actions = torch.cat(self.buffer.actions).detach()
        old_logprobs = torch.cat(self.buffer.logprobs).detach()

        total_loss_val = 0.0
        actor_loss_val = 0.0
        critic_loss_val = 0.0

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs)
            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

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

        avg_total_loss = total_loss_val / self.K_epochs
        avg_actor_loss = actor_loss_val / self.K_epochs
        avg_critic_loss = critic_loss_val / self.K_epochs

        # Best-model checkpointing based on minimum total loss: --->
        if avg_total_loss < self.best_total_loss:
            prev_best = self.best_total_loss
            self.best_total_loss = avg_total_loss

            checkpoint_path = os.path.join(self.save_dir, self.checkpoint_name)
            torch.save({
                'policy_state_dict': self.policy.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_total_loss': self.best_total_loss,
                'actor_loss': avg_actor_loss,
                'critic_loss': avg_critic_loss,
                'total_loss': avg_total_loss,
            }, checkpoint_path)

            prev_str = f"{prev_best:.4f}" if prev_best != float('inf') else "inf"
            print(f"[CNN-PPO] New best model saved → {checkpoint_path}\n\t(total_loss: {avg_total_loss:.4f} < prev: {prev_str})")

        return avg_actor_loss, avg_critic_loss, avg_total_loss

    def load_best_model(self, path=None):
        """Load a previously saved best-model checkpoint."""
        if path is None:
            path = os.path.join(self.save_dir, self.checkpoint_name)

        if not os.path.isfile(path):
            print(f"[CNN-PPO] No checkpoint found at {path}")
            return

        checkpoint = torch.load(path, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_total_loss = checkpoint['best_total_loss']

        print(f"[CNN-PPO] Loaded best model from {path}\n\t(best_total_loss: {self.best_total_loss:.4f})")
