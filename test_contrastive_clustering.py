import gymnasium as gym
import numpy as np
import warnings

import eon_env
from clustering.contrastive_learning import ContrastiveClusterManager
from eon_env import constants as const

# Suppress RuntimeWarning from nan-handling in centroid calculation for empty clusters: --->
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

def analyze_cluster_purity(clusters, lightpath_details):
    """
    Calculates the purity of the clustering result.
    (Identical to the function in previous test scripts)
    """
    num_lightpaths = len(lightpath_details)
    if num_lightpaths == 0:
        return 0.0

    ground_truth_labels = []
    for i in range(num_lightpaths):
        lp = lightpath_details[i]
        req_gsnr = const.MODULATION_FORMATS[lp['modulation_format']]['req_gsnr_db']
        current_gsnr = lp['opm_metrics']['gsnr_db']
        status = 1 if current_gsnr >= req_gsnr else 0                           # <---: HEALTHY, 0: FAILED
        ground_truth_labels.append(status)

    total_majority_members = 0

    for cluster_indices in clusters:
        if not cluster_indices:
            continue

        num_healthy = sum(1 for i in cluster_indices if ground_truth_labels[i] == 1)
        num_failed = len(cluster_indices) - num_healthy
        total_majority_members += max(num_healthy, num_failed)

    purity = total_majority_members / num_lightpaths

    return purity


if __name__ == '__main__':
    # Initialize the EON environment and generate data: --->
    env = gym.make('EON-v0')
    observation, info = env.reset()

    print("--- Running EON Environment Simulation ---")
    for day in range(1, 101):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    print(f"Simulation finished after {info['step']} steps.\n")

    # Initialize the Contrastive Learning Cluster Manager: --->
    print("--- Initializing Self-Supervised Contrastive Learning ---")
    cluster_manager = ContrastiveClusterManager(
        input_dim = observation.shape[1],
        n_clusters = 5,
        representation_dim = 16,                                      # <--- Dimension of h
        projection_dim = 8,                                               # <--- Dimension of z
        epochs = 200,
        lr = 1e-3,
        temperature = 0.1,                                                # <--- Lower temp for harder negatives
        verbose = True
    )

    # 3. Perform clustering (trains the model, then runs K-Means): --->
    print("\n--- Performing Clustering (this may take a moment) ---")
    clusters = cluster_manager.fit_predict(observation)
    centroids = cluster_manager.get_cluster_centroids(observation, clusters)
    print("--- Clustering Complete ---\n")

    print(f"Found {len(clusters)} clusters from {const.NUM_LIGHTPATHS} lightpaths.\n")

    # Qualitative Analysis: Print cluster details: --->
    print("--- Cluster Analysis ---")
    for i, (indices, centroid) in enumerate(zip(clusters, centroids)):
        if not indices:
            print(f"Cluster {i+1} (Size: 0)")
            continue

        print(f"Cluster {i+1} (Size: {len(indices)})")
        print(f"  Lightpath Indices: {indices}")
        print(f"  Centroid (GSNR, OSNR, CD, PMD): {np.round(centroid, 2)}")

    # Quantitative Analysis: Calculate and print cluster purity: --->
    purity_score = analyze_cluster_purity(clusters, info['lightpath_details'])

    print("\n--- Quantitative Evaluation ---")
    print(f"Clustering Purity Score: {purity_score:.4f}")
    print("(Purity measures how well clusters separate 'HEALTHY' from 'FAILED' lightpaths. 1.0 is perfect.)")

    env.close()
