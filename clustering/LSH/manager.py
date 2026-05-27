import numpy as np
from .lsh import LSH
from .preprocessing import StandardScaler
from eon_env import constants as const

class LSHClusterManager:
    """
    Manages the LSH clustering process for EON OPM data.
    Handles normalization and clustering execution.
    """
    def __init__(self, input_dim=4, num_functions_k=4, seed=42):
        """
        Args:
            input_dim (int): Number of OPM metrics (default 4: GSNR, OSNR, CD, PMD).
            num_functions_k (int): Number of hash functions.
            seed (int): Random seed.
        """
        self.scaler = StandardScaler()
        self.lsh = LSH(input_dim, num_functions_k, seed)
        self.fixed_signatures = []
        self.signature_to_id = {}

    def fit_predict(self, observations):
        """
        Clusters the lightpaths based on their OPM observations.

        Args:
            observations (np.ndarray): Matrix of shape (num_lightpaths, input_dim).

        Returns:
            list: A list of lists, where each inner list contains the indices
                  of lightpaths belonging to a specific cluster.
        """
        # Normalize the observations to ensure Euclidean distance is meaningful: --->
        # across different units (dB, s/m^2, s).
        norm_obs = self.scaler.fit_transform(observations)

        # Perform LSH clustering: --->
        cluster_map = self.lsh.cluster(norm_obs)

        # Extract just the groups of indices: --->
        clusters = list(cluster_map.values())

        return clusters

    def fit(self, observations):
        """
        Fits the LSH and establishes fixed cluster identities for RL state consistency.
        """
        norm_obs = self.scaler.fit_transform(observations)
        self.lsh.fit(norm_obs)

        hashes = self.lsh.compute_hashes(norm_obs)
        unique_hashes = list(set(tuple(h) for h in hashes))

        # Ensure deterministic ordering: --->
        self.fixed_signatures = sorted(unique_hashes)

        # Pad or truncate to ensure strictly N_CLUSTERS for the RL state shape: --->
        if len(self.fixed_signatures) > const.N_CLUSTERS:
            self.fixed_signatures = self.fixed_signatures[:const.N_CLUSTERS]
        else:
            while len(self.fixed_signatures) < const.N_CLUSTERS:
                # Create a dummy signature that won't match anything naturally: --->
                dummy_sig = tuple([-1] * self.lsh.k + [len(self.fixed_signatures)])
                self.fixed_signatures.append(dummy_sig)

        # Create a fast lookup map: --->
        self.signature_to_id = {sig: i for i, sig in enumerate(self.fixed_signatures)}

    def predict(self, observations):
        """
        Clusters observations mapping them to the fixed signatures.
        """
        if not getattr(self, 'fixed_signatures', None):
            self.fit(observations)

        norm_obs = self.scaler.transform(observations)
        hashes = self.lsh.compute_hashes(norm_obs)

        # Initialize fixed size cluster list: --->
        clusters = [[] for _ in range(const.N_CLUSTERS)]

        for idx, h in enumerate(hashes):
            sig = tuple(h)
            # Fast O(1) lookup instead of O(N) list search: --->
            c_idx = self.signature_to_id.get(sig)
            if c_idx is not None:
                clusters[c_idx].append(idx)
            else:
                # Fallback for drifted signatures: --->
                clusters[0].append(idx)

        return clusters

    def get_cluster_centroids(self, observations, clusters):
        """
        Calculates the mean OPM vector for each cluster.
        Useful for creating the state representation for the RL agent.
        """
        centroids = []
        for cluster_indices in clusters:
            cluster_data = observations[cluster_indices]
            centroid = np.mean(cluster_data, axis = 0)
            centroids.append(centroid)

        return np.array(centroids)
