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
    """
    def __init__(self, num_features = 4, hidden_dim = 64):
        super().__init__()

        self.gcn1 = GraphConvLayer(num_features, hidden_dim)
        self.gcn2 = GraphConvLayer(hidden_dim, hidden_dim)

        # Actor: Outputs logits for selecting a component: --->
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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

        # Actor Logits: (Batch, Num_Nodes): --->
        action_logits = self.actor(h).squeeze(-1)

        # Critic Global Value (Mean Pooling): (Batch, 1): --->
        global_h = torch.mean(h, dim=1)
        state_value = self.critic(global_h)

        return action_logits, state_value

    def act(self, x, adj):
        action_logits, _ = self.forward(x, adj)
        action_probs = F.softmax(action_logits, dim = -1)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, x, adj, action):
        action_logits, state_values = self.forward(x, adj)
        action_probs = F.softmax(action_logits, dim = -1)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, state_values.squeeze(-1), dist_entropy
