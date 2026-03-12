import torch
from torch import nn
import numpy as np

class FixedConvAggregator(nn.Module):
    """
    Aggregates OPM metrics for a cluster of lightpaths into a single
    fixed-size feature vector using non-trainable 1D convolutions.

    This module applies a set of hand-crafted filters to each OPM metric
    independently (using grouped convolutions) and then uses max-pooling
    to extract the most salient features.
    """
    def __init__(self, num_metrics: int):
        """
        Initializes the fixed filters and layers.

        Args:
            num_metrics (int): The number of OPM metrics for each lightpath
                               (e.g., 4 for GSNR, OSNR, CD, PMD).
        """
        super().__init__()
        self.num_metrics = num_metrics

        # 1. Define a set of hand-crafted 1D filters (kernels): --->
        # These filters are designed to extract basic statistical properties.
        # Shape: (out_channels, in_channels/groups, kernel_size)

        # Filter to calculate a moving average (size 3): --->
        mean_filter = torch.tensor([1/3, 1/3, 1/3], dtype=torch.float32).view(1, 1, 3)

        # Filter to detect gradients (like a 1D Sobel operator): --->
        gradient_filter = torch.tensor([-1, 0, 1], dtype=torch.float32).view(1, 1, 3)

        # Filter to detect peaks/valleys (like a 1D Laplacian): --->
        laplacian_filter = torch.tensor([1, -2, 1], dtype=torch.float32).view(1, 1, 3)

        # Concatenate filters into a single weight tensor: --->
        # We have 3 filters to apply to each metric.
        self.filters = torch.cat([mean_filter, gradient_filter, laplacian_filter], dim=0)
        self.num_filters = self.filters.shape[0]

        # 2. Create the 1D Convolutional Layer: --->
        self.conv_layer = nn.Conv1d(
            in_channels = self.num_metrics,                                         # <--- in_channels: The number of OPM metrics.
            out_channels = self.num_metrics * self.num_filters,           # <--- out_channels: num_metrics * num_filters, because each metric gets its own set of filter responses.
            kernel_size = self.filters.shape[-1],                                      # <--- kernel_size: The width of our filters.
            groups = self.num_metrics,                                                # <--- groups: This is the key. Setting groups=num_metrics ensures that
                                                                                                        #        each metric is convolved only with the filters, without mixing
                                                                                                        #        information across metrics.
            padding = 'same',                                                               # <--- padding='same': Ensures output length is same as input, handling edge cases.
            bias = False                                                                         # <--- No bias needed for these filters
        )

        # 3. Manually set the weights and make them non-trainable: --->
        # The weight tensor must be repeated for each group.
        self.conv_layer.weight.data = self.filters.repeat(self.num_metrics, 1, 1)
        self.conv_layer.weight.requires_grad = False # This freezes the filters.

        # 4. Create the Aggregation Layer: --->
        # AdaptiveMaxPool1d(1) will find the maximum value for each of the
        # (num_metrics * num_filters) output channels, resulting in a
        # fixed-size output regardless of the number of lightpaths in the cluster.
        self.pool_layer = nn.AdaptiveMaxPool1d(1)

    def forward(self, cluster_opm_matrix: np.ndarray) -> np.ndarray:
        """
        Processes the OPM matrix of a single cluster.

        Args:
            cluster_opm_matrix (np.ndarray): A 2D numpy array of shape
                (num_lightpaths_in_cluster, num_metrics).

        Returns:
            np.ndarray: A 1D numpy array representing the aggregated
                feature vector for the cluster. The size will be
                (num_metrics * num_filters).
        """
        if cluster_opm_matrix.ndim != 2:
            raise ValueError("Input must be a 2D matrix (lightpaths, metrics).")

        if cluster_opm_matrix.shape[0] == 0:
            # Handle empty cluster by returning a zero vector: --->
            return np.zeros(self.num_metrics * self.num_filters)

        # Convert to PyTorch tensor: --->
        opm_tensor = torch.from_numpy(cluster_opm_matrix).float()

        # Reshape for Conv1d: (N, M) -> (1, N, M) -> (1, M, N): --->
        # (batch_size, channels, length)
        opm_tensor = opm_tensor.unsqueeze(0).permute(0, 2, 1)

        # Apply the fixed convolutional filters: --->
        # Input: (1, num_metrics, num_lightpaths)
        # Output: (1, num_metrics * num_filters, num_lightpaths)
        feature_maps = self.conv_layer(opm_tensor)

        # Aggregate features across all lightpaths using max pooling: --->
        # Input: (1, num_metrics * num_filters, num_lightpaths)
        # Output: (1, num_metrics * num_filters, 1)
        aggregated_features = self.pool_layer(feature_maps)

        # Remove unnecessary dimensions and convert back to numpy array: --->
        # Output: (num_metrics * num_filters,)
        final_vector = aggregated_features.squeeze().cpu().numpy()

        return final_vector
