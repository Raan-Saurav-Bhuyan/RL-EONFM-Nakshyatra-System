import numpy as np
from collections import defaultdict

class LSH:
    """
    Locality Sensitive Hashing (LSH) implementation using PCA-guided SimHash
    (Data-Dependent Sign Projections) for approximating Angular Similarity
    aligned to the axes of maximum data variance.
    """
    def __init__(self, input_dim, num_functions_k=4, seed=42):
        """
        Args:
            input_dim (int): Dimension of the input vectors (d).
            num_functions_k (int): Number of hash functions/hyperplanes (k).
            seed (int): Random seed for reproducibility.
        """
        self.input_dim = input_dim
        self.k = num_functions_k
        self.rng = np.random.default_rng(seed)
        self.a = None
        self.is_fitted = False          # <--- Check whether the clustering has run or not to keep the clusters static

    def fit(self, X):
        """
        Calculates the PCA-guided SimHash projection matrix based on baseline data.
        """
        # 1. PCA-Guided Hashing: Calculate Covariance Matrix: --->
        cov = np.cov(X, rowvar=False)

        # 2. Compute eigenvalues and eigenvectors: --->
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 3. Sort descending by variance captured (eigenvalues): --->
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 4. We can at most extract 'input_dim' principal components: --->
        actual_k = min(self.k, self.input_dim)
        top_evals = eigenvalues[:actual_k]
        top_evecs = eigenvectors[:, :actual_k]

        # 5. Scale eigenvectors: a_i = v_i / sqrt(lambda_i): --->
        scaling = 1.0 / np.sqrt(np.maximum(top_evals, 1e-10))
        a_pca = (top_evecs * scaling).T             # <--- Shape: (actual_k, input_dim)

        if self.k > self.input_dim:
            a_random = self.rng.standard_normal((self.k - self.input_dim, self.input_dim))
            self.a = np.vstack([a_pca, a_random])
        else:
            self.a = a_pca

        self.is_fitted = True

    def compute_hashes(self, X):
        """Projects inputs into binary hashes using the fitted matrix."""
        if not self.is_fitted:
            self.fit(X)

        # Projection: (N, d) . (d, k) -> (N, k): --->
        projections = np.dot(X, self.a.T)

        # SimHash mapping: positive projections become 1, negative become 0: --->
        # This converts the continuous space into a binary string for each vector.
        binary_hashes = (projections >= 0).astype(int)

        return binary_hashes

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
