import os
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
        self.ground_truths = []  # Ground truth labels for auxiliary BCE loss

    def clear(self):
        self.states.clear()
        self.adjs.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        self.ground_truths.clear()

class PPOAgentGNN:
    """GNN-backed PPO Agent for full-graph fault classification with best-model checkpointing."""
    def __init__(
        self,
        lr = 3e-4,
        gamma = 0.99,
        K_epochs = 40,
        eps_clip = 0.2,
        lambda_cls = 0.1,
        use_auxiliary_cls_loss = True,
        save_dir = "models/GNN_PPO", checkpoint_name = "best_gnn_ppo.pt"):
        self.gamma, self.eps_clip, self.K_epochs = gamma, eps_clip, K_epochs
        self.lambda_cls = lambda_cls
        self.use_auxiliary_cls_loss = use_auxiliary_cls_loss
        self.buffer = GNNRolloutBuffer()

        # Model checkpoint directory and best-loss tracker: --->
        self.save_dir = save_dir
        self.checkpoint_name = checkpoint_name
        os.makedirs(self.save_dir, exist_ok = True)
        self.best_total_loss = float('inf')

        self.policy = ActorCriticGNN()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCriticGNN()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        self.BceLoss = nn.BCELoss()

    def select_action(self, state, adj):
        """
        Select per-node binary classifications for all components.

        Returns:
            actions: numpy array of shape [num_nodes] with 0/1 labels.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        adj_tensor = torch.FloatTensor(adj).unsqueeze(0)

        with torch.no_grad():
            actions, logprob = self.policy_old.act(state_tensor, adj_tensor)

        self.buffer.states.append(state_tensor)
        self.buffer.adjs.append(adj_tensor)
        self.buffer.actions.append(actions)       # (1, num_nodes)
        self.buffer.logprobs.append(logprob)       # (1,)

        return actions.squeeze(0).numpy()

    def store_ground_truth(self, ground_truth):
        """Store ground truth labels for auxiliary BCE loss computation."""
        gt_tensor = torch.FloatTensor(ground_truth).unsqueeze(0)
        self.buffer.ground_truths.append(gt_tensor)

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
        old_actions = torch.cat(self.buffer.actions).detach()    # (B, num_nodes)
        old_logprobs = torch.cat(self.buffer.logprobs).detach()  # (B,)

        # Prepare ground truth tensors for auxiliary BCE loss: --->
        has_ground_truth = (self.use_auxiliary_cls_loss and len(self.buffer.ground_truths) == len(self.buffer.states))
        if has_ground_truth:
            old_ground_truths = torch.cat(self.buffer.ground_truths).detach()  # (B, num_nodes)

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

            # Auxiliary BCE classification loss: --->
            if has_ground_truth:
                cls_probs = self.policy.get_classification_probs(old_states, old_adjs)
                
                # cls_probs: (B, num_nodes, 2) — use probability of faulty class (index 1): --->
                faulty_probs = cls_probs[:, :, 1]
                cls_loss = self.BceLoss(faulty_probs, old_ground_truths)
                loss = loss + self.lambda_cls * cls_loss

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
            print(f"[GNN-PPO] New best model saved → {checkpoint_path}\n\t(total_loss: {avg_total_loss:.4f} < prev: {prev_str})")

        return avg_actor_loss, avg_critic_loss, avg_total_loss

    def load_best_model(self, path = None):
        """Load a previously saved best-model checkpoint."""
        if path is None:
            path = os.path.join(self.save_dir, self.checkpoint_name)

        if not os.path.isfile(path):
            print(f"[GNN-PPO] No checkpoint found at {path}")
            return

        checkpoint = torch.load(path, weights_only = False)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_total_loss = checkpoint['best_total_loss']

        print(f"[GNN-PPO] Loaded best model from {path}\n\t(best_total_loss: {self.best_total_loss:.4f})")
