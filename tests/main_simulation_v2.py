import sys
import os
import random
import time
import warnings
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Ensure project root directory is in sys.path for module resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eon_env.v2.environment import EONEnvV2
from eon_env.v2 import constants as const

# Import all clustering managers: --->
from clustering.LSH import LSHClusterManager
from clustering.similarity_learning import SimilarityClusterManager
from clustering.contrastive_learning import ContrastiveClusterManager

try:
    from tests.visualize_clusters_v2 import visualize_clusters_pca
except ImportError:
    try:
        from tests.visualize_clusters import visualize_clusters_pca
    except ImportError:
        from visualize_clusters import visualize_clusters_pca

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
        print("\n--- Please Select a Clustering Engine for V2 Digital Twin ---")
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


def run_v2_simulation_and_clustering(
    json_path = "nsfnet.json",
    target_services = const.NUM_LIGHTPATHS,
    sim_days = 730,
    engine_name: str = None
):
    """
    Provisions services using Flex-Grid & SDM constraints, advances temporal degradation,
    collects telemetry, and clusters OPM metrics via the selected clustering engine.
    """
    print("--- Initializing V2 EON Digital Twin Simulator ---")
    if not os.path.exists(json_path):
        root_json = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', json_path))
        if os.path.exists(root_json):
            json_path = root_json

    if not os.path.exists(json_path):
        print(f"Error: Could not find '{json_path}'. Please ensure the JSON topology exists.")
        return

    env = EONEnvV2(json_path)
    print(f"--- Provisioning up to {target_services} Random Services via Gym Env ---")
    observation, info = env.reset()

    # Initialize TensorBoard Writer with timestamped directory
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", f"runs/v2_simulation_{current_time}"))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"--- TensorBoard Logging Initialized in: {log_dir} ---")

    print(f"--- Simulating Network Hardware Degradation for {sim_days} days ---")
    for day in range(sim_days):
        observation, reward, terminated, truncated, info = env.step(0)

        active_services = info.get('active_services', 0)
        if active_services > 0:
            mean_gsnr = np.mean(observation[:active_services, 0])
            mean_ber = np.mean(observation[:active_services, 5])

            # ! Logging: --->
            writer.add_scalar('Metrics/Mean_GSNR_dB', mean_gsnr, day)
            writer.add_scalar('Metrics/Mean_Pre_FEC_BER', mean_ber, day)

    print("--- Collecting Granular OPM Telemetry Data ---")

    # Filter out empty padded rows to match active services: --->
    active_services = info.get('active_services', 0)
    observation = observation[:active_services]
    print(f"Collected Telemetry Matrix Shape: {observation.shape}")

    if observation.shape[0] == 0:
        print("No active services telemetry available to cluster.")
        return

    # User Selection of Clustering Engine if not explicitly passed: --->
    if engine_name is None:
        engine_name = get_user_choice()

    input_dim = observation.shape[1]
    manager = None
    N_CLUSTERS = 10

    print(f"\n--- Initializing {engine_name.upper()} Clustering Engine on Degraded Telemetry ---")
    start_time = time.time()

    if engine_name == 'lsh':
        manager = LSHClusterManager(
            input_dim = input_dim, num_functions_k = min(8, input_dim)
        )
    elif engine_name == 'similarity':
        manager = SimilarityClusterManager(
            input_dim = input_dim, n_clusters = N_CLUSTERS, latent_dim = 10,
            pretrain_epochs = 50, train_epochs = 100, verbose = False
        )
    elif engine_name == 'contrastive':
        manager = ContrastiveClusterManager(
            input_dim = input_dim, n_clusters = N_CLUSTERS, representation_dim = 16,
            projection_dim = 8, epochs = 200, temperature = 0.1, verbose = False
        )

    print("Performing clustering... (this may take a moment for deep learning models)")
    clusters = manager.fit_predict(observation)
    end_time = time.time()

    print(f"Clustering complete in {end_time - start_time:.2f} seconds.")
    active_clusters = [c for c in clusters if c]
    print(f"Isolated into {len(active_clusters)} distinct health/degradation profiles.")

    # Feature Aggregation using Fixed Convolutional Network: --->
    print("\n--- Performing Feature Aggregation on Clusters ---")

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
    print(f"Final state representation for RL agent has shape: {final_state_representation.shape}\n")

    # Pass to the PCA visualizer: --->
    visualize_clusters_pca(observation, clusters, f"v2_{engine_name}")

    writer.close()
    env.close()

if __name__ == '__main__':
    run_v2_simulation_and_clustering()
