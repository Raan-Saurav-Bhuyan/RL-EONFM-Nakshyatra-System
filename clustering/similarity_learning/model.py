import torch
from torch import nn
from torch.nn import Parameter

class Autoencoder(nn.Module):
    """
    Autoencoder architecture for dimensionality reduction and feature learning.
    Maps input OPM vectors to a latent space and reconstructs them.
    """
    def __init__(self, input_dim, latent_dim = 10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)

        return x_hat, z


class ClusteringLayer(nn.Module):
    """
    Clustering layer that computes soft assignments (q_ij) based on the similarity
    between latent embeddings and cluster centroids using a Student's t-distribution.
    """
    def __init__(self, n_clusters, n_features):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = 1.0

        # Trainable cluster centroids, initialized with Xavier uniform: --->
        self.centroids = Parameter(torch.Tensor(n_clusters, n_features))
        torch.nn.init.xavier_uniform_(self.centroids)

    def forward(self, z):
        """
        Calculates the soft assignment probability q_ij.
        """
        sum_sq = torch.sum((z.unsqueeze(1) - self.centroids)**2, 2)
        num = (1.0 + sum_sq / self.alpha)**(- (self.alpha + 1.0) / 2.0)
        q = num / torch.sum(num, dim = 1, keepdim = True)

        return q
