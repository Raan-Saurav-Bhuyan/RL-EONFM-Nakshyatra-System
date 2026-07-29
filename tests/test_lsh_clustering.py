import sys
import os
import gymnasium as gym
import numpy as np

# Ensure project root directory is in sys.path for module resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import eon_env                                                  # <--- Registers the EON-v0 environment
from clustering.LSH import LSHClusterManager
from eon_env.v1 import constants as const


def analyze_cluster_purity(clusters, lightpath_details):
    """
    Calculates the purity of the clustering result.

    Purity is a measure of the extent to which clusters contain a single class.
    Here, the classes are 'HEALTHY' and 'FAILED'.

    Args:
        clusters (list): A list of lists, where each inner list contains the indices of lightpaths in a cluster.
        lightpath_details (list): The 'info' dictionary from the environment, containing details for each lightpath.

    Returns:
        float: The overall purity score of the clustering.
    """
    num_lightpaths = len(lightpath_details)
    if num_lightpaths == 0:
        return 0.0

    # Determine the "ground truth" class for each lightpath: --->
    ground_truth_labels = []
    for i in range(num_lightpaths):
        lp = lightpath_details[i]
        req_gsnr = const.MODULATION_FORMATS[lp['modulation_format']]['req_gsnr_db']
        current_gsnr = lp['opm_metrics']['gsnr_db']
        status = 1 if current_gsnr >= req_gsnr else 0                   # <--- 1: HEALTHY, 0: FAILED
        ground_truth_labels.append(status)

    # Calculate purity: --->
    total_majority_members = 0
    for cluster_indices in clusters:
        if not cluster_indices:
            continue

        num_healthy = sum(1 for i in cluster_indices if ground_truth_labels[i] == 1)
        num_failed = len(cluster_indices) - num_healthy

        # The number of correctly assigned points in this cluster is the size of the majority class: --->
        total_majority_members += max(num_healthy, num_failed)

    purity = total_majority_members / num_lightpaths
    return purity


if __name__ == '__main__':
    # Initialize the EON environment: --->
    env = gym.make('EON-v0')
    observation, info = env.reset()

    # Run the simulation for a number of steps to generate diverse OPM data: --->
    print("--- Running EON Environment Simulation ---")
    for day in range(1, 101):                                                       # <--- Simulate for 100 days
        action = env.action_space.sample()                                 # <--- Action is trivial for now
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    print(f"Simulation finished after {info['step']} steps.\n")

    # Initialize the LSH Cluster Manager: --->
    lsh_manager = LSHClusterManager(
        input_dim = observation.shape[1],
        num_functions_k = 8,                    # <--- More functions -> finer-grained clusters
    )

    # Perform clustering on the final OPM observation matrix: --->
    print("--- Performing LSH Clustering ---")
    clusters = lsh_manager.fit_predict(observation)
    centroids = lsh_manager.get_cluster_centroids(observation, clusters)

    print(f"Found {len(clusters)} clusters from {const.NUM_LIGHTPATHS} lightpaths.\n")

    # Qualitative Analysis: Print cluster details: --->
    print("--- Cluster Analysis ---")
    for i, (indices, centroid) in enumerate(zip(clusters, centroids)):
        print(f"Cluster {i+1} (Size: {len(indices)})")
        print(f"  Lightpath Indices: {indices}")
        print(f"  Centroid (GSNR, OSNR, CD, PMD): {np.round(centroid, 2)}")

    # Quantitative Analysis: Calculate and print cluster purity: --->
    purity_score = analyze_cluster_purity(clusters, info['lightpath_details'])
    print("\n--- Quantitative Evaluation ---")
    print(f"Clustering Purity Score: {purity_score:.4f}")
    print("(Purity measures how well clusters separate 'HEALTHY' from 'FAILED' lightpaths. 1.0 is perfect.)")

    env.close()
