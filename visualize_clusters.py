import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
# import argparse
import time
import warnings

import eon_env
from eon_env.v1 import constants as const

from clustering.LSH import LSHClusterManager, StandardScaler
from clustering.similarity_learning import SimilarityClusterManager
from clustering.contrastive_learning import ContrastiveClusterManager

from sklearn.decomposition import PCA

# Suppress warnings for cleaner output: --->
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=UserWarning, message="KMeans is known to have a memory leak on Windows with MKL")


def visualize_clusters_pca(observations, clusters, engine_name):
    """
    Visualizes the clusters in a 2D space using PCA.

    Args:
        observations (np.ndarray): The raw OPM data matrix.
        clusters (list): A list of lists, where each inner list contains the indices
                         of lightpaths in a cluster.
        engine_name (str): The name of the clustering engine used.
    """
    print("--- Reducing dimensionality with PCA for visualization ---")
    # Normalize the data, which is crucial for PCA: --->
    scaler = StandardScaler()
    norm_obs = scaler.fit_transform(observations)

    # Apply PCA to project the 4D data down to 2D: --->
    pca = PCA(n_components=2, random_state=42)
    principal_components = pca.fit_transform(norm_obs)

    # Create the scatter plot: --->
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use a colormap to assign a unique color to each cluster: --->
    num_clusters = len(clusters)
    colors = plt.cm.viridis(np.linspace(0, 1, num_clusters))

    for i, indices in enumerate(clusters):
        if not indices:
            continue            # <---  Skip empty clusters

        # Select the 2D points corresponding to the current cluster: --->
        cluster_pcs = principal_components[indices]

        ax.scatter(
            cluster_pcs[:, 0],
            cluster_pcs[:, 1],
            color = colors[i],
            label = f'Cluster {i+1} (Size: {len(indices)})',
            s = 60,                                                     # <--- Marker size
            alpha = 0.8,                                             # <--- Marker transparency
            edgecolors = 'k',                                       # <--- Marker edge color
            linewidth = 0.5
        )

    ax.set_title(f'Lightpath Clusters ({engine_name.upper()}) Visualized with PCA', fontsize=16)
    ax.set_xlabel('Principal Component 1', fontsize = 12)
    ax.set_ylabel('Principal Component 2', fontsize = 12)
    ax.legend(title="Clusters", loc='best')

    fig.tight_layout()
    # plt.show()

    plt.savefig(f"visualizations/clusters/{engine_name}_cluster_visualization.png")


if __name__ == '__main__':
    # 1. Run EON environment simulation to get OPM data: --->
    env = gym.make('EON-v0')
    observation, info = env.reset()
    print("--- Running EON Environment Simulation ---")
    for day in range(1, 101):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    print(f"Simulation finished after {info['step']} steps.\n")

    # 2. Initialize and run the selected clustering engine: --->
    input_dim = observation.shape[1]
    manager = None
    engine_name = 'lsh'

    print(f"--- Initializing {engine_name.capitalize()} Clustering Engine ---")
    start_time = time.time()

    if engine_name == 'lsh':
        manager = LSHClusterManager(
            input_dim = input_dim, num_functions_k = 8
        )
    elif engine_name == 'similarity':
        manager = SimilarityClusterManager(
            input_dim = input_dim, n_clusters = 5, latent_dim = 10,
            pretrain_epochs = 50, train_epochs = 100, verbose = True
        )
    elif engine_name == 'contrastive':
        manager = ContrastiveClusterManager(
            input_dim = input_dim, n_clusters = 5, representation_dim = 16,
            projection_dim = 8, epochs = 200, temperature = 0.1, verbose = True
        )

    print("\n--- Performing Clustering (this may take a moment) ---")
    clusters = manager.fit_predict(observation)
    end_time = time.time()
    print(f"--- Clustering Complete in {end_time - start_time:.2f} seconds ---\n")

    # Visualize the results: --->
    visualize_clusters_pca(observation, clusters, engine_name)

    env.close()
