import sys
import os
import numpy as np
import warnings

# Ensure project root directory is in sys.path for module resolution: --->
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eon_env.v2.environment import EONEnvV2
from eon_env.v2 import constants as const
from clustering.LSH import LSHClusterManager


def analyze_cluster_purity(clusters, active_services):
    """
    Calculates the purity of the clustering result for V2 environment.

    Purity is a measure of the extent to which clusters contain a single class.
    Here, classes are 'HEALTHY' and 'FAILED' (or DEGRADED based on Pre-FEC BER).

    Args:
        clusters (list): A list of lists, where each inner list contains the indices of lightpaths in a cluster.
        active_services (list): List of ServiceDemand objects or dicts containing details for each lightpath.

    Returns:
        float: The overall purity score of the clustering.
    """
    num_lightpaths = len(active_services)
    if num_lightpaths == 0:
        return 0.0

    # Determine the "ground truth" class for each lightpath: --->
    ground_truth_labels = []
    for i in range(num_lightpaths):
        lp = active_services[i]
        if isinstance(lp, dict):
            opm = lp.get('opm_metrics', {})
        else:
            opm = getattr(lp, 'opm_metrics', {})

        ber = opm.get('pre_fec_ber', 0.0)
        # Pre-FEC BER < 1e-3 is considered HEALTHY (1), >= 1e-3 is FAILED/DEGRADED (0)
        status = 1 if ber < 1e-3 else 0
        ground_truth_labels.append(status)

    # Calculate purity: --->
    total_majority_members = 0
    for cluster_indices in clusters:
        if not cluster_indices:
            continue

        num_healthy = sum(1 for i in cluster_indices if ground_truth_labels[i] == 1)
        num_failed = len(cluster_indices) - num_healthy

        # The number of correctly assigned points in this cluster
        # is the size of the majority class: --->
        total_majority_members += max(num_healthy, num_failed)

    purity = total_majority_members / num_lightpaths
    return purity


if __name__ == '__main__':
    # Initialize the V2 EON environment: --->
    json_path = "nsfnet.json"
    if not os.path.exists(json_path):
        root_json = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', json_path))
        if os.path.exists(root_json):
            json_path = root_json

    env = EONEnvV2(network_json_path=json_path)
    observation, info = env.reset()

    # Run the simulation for a number of steps to generate diverse OPM data: --->
    print("--- Running V2 EON Environment Simulation ---")
    for day in range(1, 101):                   # <--- Simulate for 100 days
        observation, reward, terminated, truncated, info = env.step(0)

        if terminated or truncated:
            break

    active_services_count = info.get('active_services', len(env.simulator.active_services))
    observation = observation[:active_services_count]
    print(f"Simulation finished after {info['step']} steps with {active_services_count} active services.\n")

    # Initialize the LSH Cluster Manager: --->
    input_dim = observation.shape[1]
    lsh_manager = LSHClusterManager(
        input_dim = input_dim,
        num_functions_k = min(8, input_dim),     # <--- More functions -> finer-grained clusters
    )

    # Perform clustering on the final OPM observation matrix: --->
    print("--- Performing LSH Clustering (V2) ---")
    clusters = lsh_manager.fit_predict(observation)
    centroids = lsh_manager.get_cluster_centroids(observation, clusters)

    print(f"Found {len(clusters)} clusters from {active_services_count} active lightpaths.\n")

    # Qualitative Analysis: Print cluster details: --->
    print("--- Cluster Analysis ---")
    for i, (indices, centroid) in enumerate(zip(clusters, centroids)):
        print(f"Cluster {i+1} (Size: {len(indices)})")
        print(f"  Lightpath Indices: {indices}")
        print(f"  Centroid (GSNR, OSNR, CD, PMD, NLI, Pre-FEC BER): {np.round(centroid, 4)}")

    # Quantitative Analysis: Calculate and print cluster purity: --->
    active_services = env.simulator.active_services[:active_services_count]
    purity_score = analyze_cluster_purity(clusters, active_services)
    print("\n--- Quantitative Evaluation ---")
    print(f"Clustering Purity Score: {purity_score:.4f}")
    print("(Purity measures how well clusters separate 'HEALTHY' from 'FAILED' lightpaths. 1.0 is perfect.)")

    env.close()
