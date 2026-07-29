import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class GraphConvLayer(nn.Module):
    """Standard Graph Convolutional Layer for Message Passing."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        output = torch.bmm(adj, support)

        return output

class ActorCriticGNN(nn.Module):
    """
    Actor-Critic Network utilizing GNN to process spatial topology.

    The actor head performs per-node binary classification (healthy vs faulty)
    for every component in the augmented network graph simultaneously.
    The critic head outputs a scalar state value via mean-pooled node embeddings.
    """
    def __init__(self, num_features = 4, hidden_dim = 64):
        super().__init__()

        self.gcn1 = GraphConvLayer(num_features, hidden_dim)
        self.gcn2 = GraphConvLayer(hidden_dim, hidden_dim)

        # Actor: Per-node binary classification logits [healthy=0, faulty=1]: --->
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

        # Critic: Outputs value of the global network state: --->
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, adj):
        batch_size, num_nodes, _ = adj.shape
        eye = torch.eye(num_nodes, device = adj.device).unsqueeze(0).expand(batch_size, -1, -1)
        adj_hat = adj + eye

        # Node embeddings: --->
        h = F.relu(self.gcn1(x, adj_hat))
        h = F.relu(self.gcn2(h, adj_hat))

        # Actor Logits: (Batch, Num_Nodes, 2) — per-node binary classification: --->
        action_logits = self.actor(h)

        # Critic Global Value (Mean Pooling): (Batch, 1): --->
        global_h = torch.mean(h, dim=1)
        state_value = self.critic(global_h)

        return action_logits, state_value

    def act(self, x, adj):
        """
        Sample per-node binary classifications for all nodes simultaneously.

        Returns:
            actions:     (Batch, Num_Nodes) — 0 (healthy) or 1 (faulty) per node
            logprobs:    (Batch,) — summed log-probability across all node classifications
        """
        action_logits, _ = self.forward(x, adj)
        # action_logits: (Batch, Num_Nodes, 2): --->

        # Independent Categorical(2) per node: --->
        action_probs = F.softmax(action_logits, dim = -1)
        # action_probs: (Batch, Num_Nodes, 2): --->

        # Reshape for batched Categorical sampling: --->
        batch_size, num_nodes, _ = action_probs.shape
        flat_probs = action_probs.view(-1, 2)
        dist = Categorical(flat_probs)

        flat_actions = dist.sample()
        flat_logprobs = dist.log_prob(flat_actions)

        # Reshape back to (Batch, Num_Nodes): --->
        actions = flat_actions.view(batch_size, num_nodes)
        per_node_logprobs = flat_logprobs.view(batch_size, num_nodes)

        # Sum log-probs across nodes (joint log-probability): --->
        total_logprobs = per_node_logprobs.sum(dim = -1)

        return actions.detach(), total_logprobs.detach()

    def evaluate(self, x, adj, actions):
        """
        Evaluate log-probabilities and entropy for given per-node classification actions.

        Parameters:
            x:       (Batch, Num_Nodes, Num_Features)
            adj:     (Batch, Num_Nodes, Num_Nodes)
            actions: (Batch, Num_Nodes) — 0/1 per node

        Returns:
            logprobs:     (Batch,) — summed log-probability across all node classifications
            state_values: (Batch,) — critic state values
            entropy:      (Batch,) — mean entropy across all node classifications
        """
        action_logits, state_values = self.forward(x, adj)
        # action_logits: (Batch, Num_Nodes, 2): --->

        action_probs = F.softmax(action_logits, dim = -1)

        batch_size, num_nodes, _ = action_probs.shape
        flat_probs = action_probs.view(-1, 2)
        flat_actions = actions.view(-1)

        dist = Categorical(flat_probs)
        flat_logprobs = dist.log_prob(flat_actions)
        flat_entropy = dist.entropy()

        # Reshape and aggregate: --->
        per_node_logprobs = flat_logprobs.view(batch_size, num_nodes)
        per_node_entropy = flat_entropy.view(batch_size, num_nodes)

        # Sum log-probs, mean entropy across nodes: --->
        total_logprobs = per_node_logprobs.sum(dim = -1)
        mean_entropy = per_node_entropy.mean(dim = -1)

        return total_logprobs, state_values.squeeze(-1), mean_entropy

    def classify(self, x, adj):
        """
        Deterministic per-node classification for inference/evaluation.

        Returns:
            labels: (Batch, Num_Nodes) — 0 (healthy) or 1 (faulty)
            probs:  (Batch, Num_Nodes, 2) — softmax probabilities
        """
        with torch.no_grad():
            action_logits, _ = self.forward(x, adj)
            probs = F.softmax(action_logits, dim = -1)
            labels = probs.argmax(dim = -1)

        return labels, probs

    def get_classification_probs(self, x, adj):
        """
        Get per-node classification probabilities (differentiable, for auxiliary BCE loss).

        Returns:
            probs: (Batch, Num_Nodes, 2) — softmax probabilities [p_healthy, p_faulty]
        """
        action_logits, _ = self.forward(x, adj)
        probs = F.softmax(action_logits, dim = -1)

        return probs
