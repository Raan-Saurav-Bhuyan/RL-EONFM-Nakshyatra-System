import numpy as np

class StandardScaler:
    """
    Standardizes features by removing the mean and scaling to unit variance.
    Essential for LSH based on Euclidean distance to treat all OPM metrics equally.
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Prevent division by zero for constant features
        self.std[self.std == 0] = 1.0

    def transform(self, X):
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
