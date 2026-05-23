import gymnasium as gym
import numpy as np
import time
import warnings
import torch

# Import custom modules: --->
import eon_env                                      # <--- Registers the EON-v0 environment
from eon_env import constants as const
from visualize_clusters import visualize_clusters_pca

# Import all clustering managers: --->
from clustering.LSH import LSHClusterManager, StandardScaler
from clustering.similarity_learning import SimilarityClusterManager
from clustering.contrastive_learning import ContrastiveClusterManager

# Import the new feature aggregator: --->
from cluster_aggregations import FixedConvAggregator

# Suppress Warnings for Cleaner Output: --->
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=UserWarning, message="KMeans is known to have a memory leak on Windows with MKL")

def get_user_choice() -> str:
    """
    Prompts the user to select a clustering engine and validates the input.

    Returns:
        str: The chosen engine name ('lsh', 'similarity', or 'contrastive').
    """
    while True:
        print("\n--- Please Select a Clustering Engine ---")
        print("1: Locality Sensitive Hashing (LSH)")
        print("2: Unsupervised Similarity Learning (Autoencoder)")
        print("3: Self-Supervised Contrastive Learning (SSL)")

        choice = input("Enter your choice (1, 2, or 3): ")

        if choice == '1':
            return 'lsh'
        elif choice == '2':
            return 'similarity'
        elif choice == '3':
            return 'contrastive'
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == '__main__':
    # EON Environment Simulation: --->
    # (Initialize the environment with the updated constants (800 lightpaths, higher failure rate))
    env = gym.make('EON-v0')
    observation, info = env.reset()

    print("--- Running EON Environment Simulation ---")
    # Simulate for enough steps to allow soft failures to develop: --->
    for day in range(1, 151):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    print(f"Simulation finished after {info['step']} steps with {const.NUM_LIGHTPATHS} lightpaths.\n")

    # User Selection of Clustering Engine: --->
    engine_name = get_user_choice()

    # Clustering Execution: --->
    input_dim = observation.shape[1]
    manager = None

    # Set common parameters: --->
    N_CLUSTERS = 10                 # <--- As per requirement, increased from 5 to 10

    print(f"\n--- Initializing {engine_name.upper()} Clustering Engine ---")
    start_time = time.time()

    if engine_name == 'lsh':
        # LSH determines the number of clusters automatically via PCA-guided SimHash collisions: --->
        # Using k = input_dim (4) functions creates a highly optimized 4-bit hash (max 16 possible clusters)
        manager = LSHClusterManager(
            input_dim = input_dim, num_functions_k = input_dim
        )
    elif engine_name == 'similarity':
        manager = SimilarityClusterManager(
            input_dim = input_dim, n_clusters = N_CLUSTERS, latent_dim = 10,
            pretrain_epochs = 50, train_epochs = 100, verbose = False # Verbose off for cleaner final output
        )
    elif engine_name == 'contrastive':
        manager = ContrastiveClusterManager(
            input_dim = input_dim, n_clusters = N_CLUSTERS, representation_dim = 16,
            projection_dim = 8, epochs = 200, temperature = 0.1, verbose = False # Verbose off
        )

    print("Performing clustering... (this may take a moment for deep learning models)")
    clusters = manager.fit_predict(observation)
    end_time = time.time()
    print(f"Clustering complete in {end_time - start_time:.2f} seconds.")
    print(f"Found {len([c for c in clusters if c])} non-empty clusters.\n")

    # Feature Aggregation using Fixed Convolutional Network: --->
    print("--- Performing Feature Aggregation on Clusters ---")

    # Initialize the aggregator. It's a non-trainable PyTorch module.: --->
    aggregator = FixedConvAggregator(num_metrics=input_dim)

    aggregated_cluster_features = []
    for i, cluster_indices in enumerate(clusters):
        if not cluster_indices:
            continue

        cluster_opm_matrix = observation[cluster_indices]
        feature_vector = aggregator(cluster_opm_matrix)
        aggregated_cluster_features.append(feature_vector)

        print(f"\nCluster {i+1} (Size: {len(cluster_indices)})")
        print(f"  - Aggregated Feature Vector (first 8 values): {np.round(feature_vector[:8], 4)}")
        print(f"  - Feature Vector Shape: {feature_vector.shape}")

    print("\n--- Simulation and Aggregation Complete ---")
    final_state_representation = np.array(aggregated_cluster_features)
    print(f"Final state representation for RL agent has shape: {final_state_representation.shape}")

    # Visualize the results: --->
    visualize_clusters_pca(observation, clusters, engine_name)

    env.close()
