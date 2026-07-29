import sys
import os
import numpy as np
import warnings

# Ensure project root directory is in sys.path for module resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eon_env.v2.environment import EONEnvV2
from eon_env.v2 import constants as const
from clustering.similarity_learning import SimilarityClusterManager

# Suppress RuntimeWarning from nan-handling in centroid calculation for empty clusters: --->
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

def analyze_cluster_purity(clusters, active_services):
    """
    Calculates the purity of the clustering result for V2 environment.
    """
    num_lightpaths = len(active_services)
    if num_lightpaths == 0:
        return 0.0

    ground_truth_labels = []
    for i in range(num_lightpaths):
        lp = active_services[i]
        if isinstance(lp, dict):
            opm = lp.get('opm_metrics', {})
        else:
            opm = getattr(lp, 'opm_metrics', {})

        ber = opm.get('pre_fec_ber', 0.0)
        status = 1 if ber < 1e-3 else 0                       # <--- 1: HEALTHY, 0: FAILED/DEGRADED
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
    # Initialize the V2 EON environment and generate data: --->
    json_path = "nsfnet.json"
    if not os.path.exists(json_path):
        root_json = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', json_path))
        if os.path.exists(root_json):
            json_path = root_json

    env = EONEnvV2(network_json_path=json_path)
    observation, info = env.reset()

    print("--- Running V2 EON Environment Simulation ---")
    for day in range(1, 101):
        observation, reward, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            break

    active_services_count = info.get('active_services', len(env.simulator.active_services))
    observation = observation[:active_services_count]
    print(f"Simulation finished after {info['step']} steps with {active_services_count} active services.\n")

    # Initialize the Similarity Learning Cluster Manager: --->
    print("--- Initializing Similarity Learning Clustering (V2) ---")

    # Hyperparameters can be tuned for better performance: --->
    cluster_manager = SimilarityClusterManager(
        input_dim=observation.shape[1],
        n_clusters = 5,                                       # <--- A key hyperparameter, might need tuning
        latent_dim = 10,
        pretrain_epochs = 50,
        train_epochs = 100,
        lr = 1e-3,
        verbose = True                                       # <--- Show training progress
    )

    # Perform clustering (this will take some time due to training): --->
    print("\n--- Performing Clustering (this may take a moment) ---")
    clusters = cluster_manager.fit_predict(observation)
    centroids = cluster_manager.get_cluster_centroids(observation, clusters)
    print("--- Clustering Complete ---\n")

    print(f"Found {len(clusters)} clusters from {active_services_count} active lightpaths.\n")

    # Qualitative Analysis: Print cluster details: --->
    print("--- Cluster Analysis ---")
    for i, (indices, centroid) in enumerate(zip(clusters, centroids)):
        if not indices:
            print(f"Cluster {i+1} (Size: 0)")
            continue

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
