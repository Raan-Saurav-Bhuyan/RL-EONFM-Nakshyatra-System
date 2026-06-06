import numpy as np
import random
import time
import os
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    raise ImportError("TensorBoard is required for logging but is not installed. Please run 'pip install tensorboard'.")

from eon_env.v2.environment import EONEnvV2
from eon_env.v2 import constants as const
from clustering.LSH import LSHClusterManager
from visualize_clusters import visualize_clusters_pca
from cluster_aggregations import FixedConvAggregator

def run_v2_simulation_and_clustering(json_path = "nsfnet.json", target_services = const.NUM_LIGHTPATHS, sim_days = 730):
    """
    Provisions services using Flex-Grid & SDM constraints, advances temporal degradation,
    collects telemetry, and clusters OPM metrics via LSH.
    """
    print("--- Initializing V2 EON Digital Twin Simulator ---")
    if not os.path.exists(json_path):
        print(f"Error: Could not find '{json_path}'. Please ensure the JSON topology exists.")
        return

    env = EONEnvV2(json_path)
    print(f"--- Provisioning up to {target_services} Random Services via Gym Env ---")
    observation, info = env.reset()

    # Initialize TensorBoard Writer with timestamped directory
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/v2_simulation_{current_time}"
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

    print("--- Performing LSH Clustering on Degraded Telemetry ---")
    input_dim = observation.shape[1]
    start_time = time.time()

    # Using the LSH backend to find Patient-0 anomalies dynamically
    lsh_manager = LSHClusterManager(input_dim=input_dim, num_functions_k=min(8, input_dim))
    clusters = lsh_manager.fit_predict(observation)
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

    # Pass to the existing PCA visualizer
    visualize_clusters_pca(observation, clusters, "v2_lsh")

    writer.close()

if __name__ == '__main__':
    run_v2_simulation_and_clustering()
