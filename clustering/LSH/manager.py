import numpy as np
from .lsh import LSH
from .preprocessing import StandardScaler

class LSHClusterManager:
    """
    Manages the LSH clustering process for EON OPM data.
    Handles normalization and clustering execution.
    """
    def __init__(self, input_dim=4, num_functions_k=5, window_size_w=2.0, seed=42):
        """
        Args:
            input_dim (int): Number of OPM metrics (default 4: GSNR, OSNR, CD, PMD).
            num_functions_k (int): Number of hash functions.
            window_size_w (float): Window size for quantization.
            seed (int): Random seed.
        """
        self.scaler = StandardScaler()
        self.lsh = LSH(input_dim, num_functions_k, window_size_w, seed)

    def fit_predict(self, observations):
        """
        Clusters the lightpaths based on their OPM observations.

        Args:
            observations (np.ndarray): Matrix of shape (num_lightpaths, input_dim).

        Returns:
            list: A list of lists, where each inner list contains the indices
                  of lightpaths belonging to a specific cluster.
        """
        # Normalize the observations to ensure Euclidean distance is meaningful
        # across different units (dB, s/m^2, s).
        norm_obs = self.scaler.fit_transform(observations)

        # Perform LSH clustering
        cluster_map = self.lsh.cluster(norm_obs)

        # Extract just the groups of indices
        clusters = list(cluster_map.values())

        return clusters

    def get_cluster_centroids(self, observations, clusters):
        """
        Calculates the mean OPM vector for each cluster.
        Useful for creating the state representation for the RL agent.
        """
        centroids = []
        for cluster_indices in clusters:
            cluster_data = observations[cluster_indices]
            centroid = np.mean(cluster_data, axis=0)
            centroids.append(centroid)
        return np.array(centroids)
