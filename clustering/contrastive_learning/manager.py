import numpy as np
from collections import defaultdict
import torch
from sklearn.cluster import KMeans

from ..LSH.preprocessing import StandardScaler
from .trainer import Trainer

class ContrastiveClusterManager:
    """
    Manages the Self-Supervised Contrastive Learning (SSL) clustering process.
    This involves training a contrastive encoder, generating representations,
    and then applying a standard clustering algorithm (K-Means).
    """
    def __init__(self, input_dim, n_clusters=5, representation_dim=32,
                 projection_dim=16, seed=42, **kwargs):
        """
        Args:
            input_dim (int): Number of OPM metrics.
            n_clusters (int): The number of clusters to form with K-Means.
            representation_dim (int): Dim of the encoder's output space (h).
            projection_dim (int): Dim of the projection head's output space (z).
            seed (int): Random seed for reproducibility.
            **kwargs: Additional arguments for the Trainer (e.g., epochs, lr).
        """
        self.scaler = StandardScaler()
        self.trainer = Trainer(
            input_dim=input_dim,
            representation_dim=representation_dim,
            projection_dim=projection_dim,
            seed=seed,
            **kwargs
        )
        self.n_clusters = n_clusters
        self.seed = seed
        self.trained = False

    def fit_predict(self, observations):
        """
        Normalizes data, trains the contrastive model, generates representations,
        and predicts clusters using K-Means.
        """
        norm_obs = self.scaler.fit_transform(observations)

        self.trainer.train(norm_obs)
        self.trained = True

        base_encoder = self.trainer.model.base_encoder
        base_encoder.eval()

        with torch.no_grad():
            x_tensor = torch.from_numpy(norm_obs).float().to(self.trainer.device)
            representations = base_encoder(x_tensor).cpu().numpy()

        kmeans = KMeans(n_clusters=self.n_clusters, n_init='auto', random_state=self.seed)
        labels = kmeans.fit_predict(representations)

        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[label].append(i)

        final_clusters = [clusters.get(i, []) for i in range(self.n_clusters)]

        return final_clusters

    def get_cluster_centroids(self, observations, clusters):
        """Calculates the mean OPM vector for each cluster."""
        if not self.trained:
            raise RuntimeError("The model has not been trained yet. Call fit_predict first.")

        centroids = []
        for cluster_indices in clusters:
            if not cluster_indices:
                centroids.append(np.full(observations.shape[1], np.nan))
                continue

            cluster_data = observations[cluster_indices]
            centroid = np.mean(cluster_data, axis=0)
            centroids.append(centroid)

        return np.array(centroids)
