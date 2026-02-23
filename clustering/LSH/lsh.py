import numpy as np
from collections import defaultdict

class LSH:
    """
    Locality Sensitive Hashing (LSH) implementation using p-stable distributions
    (Gaussian) for approximating Euclidean distance.
    """
    def __init__(self, input_dim, num_functions_k=10, window_size_w=4.0, seed=42):
        """
        Args:
            input_dim (int): Dimension of the input vectors (d).
            num_functions_k (int): Number of hash functions to concatenate (k).
            window_size_w (float): Quantization window size (w).
            seed (int): Random seed for reproducibility.
        """
        self.input_dim = input_dim
        self.k = num_functions_k
        self.w = window_size_w
        self.rng = np.random.default_rng(seed)

        # Generate random projection vectors 'a' from N(0, 1): --->
        # Shape: (k, d)
        self.a = self.rng.standard_normal((self.k, self.input_dim))

        # Generate random offsets 'b' from U[0, w]: --->
        # Shape: (k, 1)
        self.b = self.rng.uniform(0, self.w, (self.k, 1))

    def compute_hashes(self, X):
        """
        Computes the hash signatures for the input vectors X.

        Args:
            X (np.ndarray): Input matrix of shape (N, d).

        Returns:
            np.ndarray: Hash signatures matrix of shape (N, k).
        """
        # Projection: (N, d) . (d, k) -> (N, k): --->
        projections = np.dot(X, self.a.T)

        # Apply offset and windowing: floor((a.x + b) / w): --->
        # (Broadcasting: (N, k) + (1, k) -> (N, k))
        transformed = (projections + self.b.T) / self.w

        return np.floor(transformed).astype(int)

    def cluster(self, X):
        """
        Groups input vectors X into clusters based on hash collisions.

        Returns:
            dict: Mapping from hash signature (tuple) to list of indices.
        """
        hashes = self.compute_hashes(X)
        clusters = defaultdict(list)

        for idx, h_sig in enumerate(hashes):
            # Convert to tuple to use as dictionary key: --->
            sig_tuple = tuple(h_sig)
            clusters[sig_tuple].append(idx)

        return dict(clusters)
