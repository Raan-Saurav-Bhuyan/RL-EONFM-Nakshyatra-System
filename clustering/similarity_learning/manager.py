import numpy as np
from collections import defaultdict
from ..LSH.preprocessing import StandardScaler
from .trainer import Trainer

class SimilarityClusterManager:
    """
    Manages the Unsupervised Similarity Learning clustering process.

    This class orchestrates the pre-training and joint training of an autoencoder-based clustering model.
    """
    def __init__(self, input_dim, n_clusters = 5, latent_dim = 10, seed = 42, **kwargs):
        """
        Args:
            input_dim (int): Number of OPM metrics.
            n_clusters (int): The number of clusters to form.
            latent_dim (int): The dimensionality of the autoencoder's latent space.
            seed (int): Random seed for reproducibility.
            **kwargs: Additional arguments for the Trainer (e.g., epochs, lr).
        """
        self.scaler = StandardScaler()
        self.trainer = Trainer(
            input_dim=input_dim,
            n_clusters=n_clusters,
            latent_dim=latent_dim,
            seed=seed,
            **kwargs
        )
        self.n_clusters = n_clusters
        self.trained = False

    def fit_predict(self, observations):
        """
        Normalizes, trains the model, and predicts clusters for the observations.

        Args:
            observations (np.ndarray): Matrix of OPM metrics.

        Returns:
            list: A list of lists, where each inner list contains the indices of lightpaths belonging to a specific cluster.
        """
        # Normalize the data: --->
        norm_obs = self.scaler.fit_transform(observations)

        # Pre-train the autoencoder: --->
        self.trainer.pretrain_autoencoder(norm_obs)

        # Jointly train the model: --->
        self.trainer.train(norm_obs)
        self.trained = True

        # Predict cluster labels: --->
        labels = self.trainer.predict(norm_obs)

        # Format output to match LSH manager (list of lists of indices): --->
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[label].append(i)

        # Ensure all cluster keys from 0 to n_clusters-1 exist, even if empty: --->
        final_clusters = [clusters.get(i, []) for i in range(self.n_clusters)]

        return final_clusters

    def get_cluster_centroids(self, observations, clusters):
        """
        Calculates the mean OPM vector for each cluster.
        """
        if not self.trained:
            raise RuntimeError("The model has not been trained yet. Call fit_predict first.")

        centroids = []
        for cluster_indices in clusters:
            if not cluster_indices:                                         # <--- Handle empty clusters
                centroids.append(np.full(observations.shape[1], np.nan))
                continue

            cluster_data = observations[cluster_indices]
            centroid = np.mean(cluster_data, axis = 0)
            centroids.append(centroid)

        return np.array(centroids)
