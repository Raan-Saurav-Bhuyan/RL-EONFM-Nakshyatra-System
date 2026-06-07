import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import math

from .simulator import EONSimulatorV2
from . import constants as const
from clustering.LSH import LSHClusterManager
from cluster_aggregations.fixed_conv_aggregator import FixedConvAggregator

class TemporalEONEnvV2(gym.Env):
    """
    Temporal Trend-Aware MDP (Predictive Maintenance) for the V2 Simulator.
    Simulates a decade of operation.
    State: (10_Years, Num_Clusters, Extracted_Features)
    Action: 0 (Monitor), 1 (Proactively Isolate/Maintain)
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, network_json_path = "nsfnet.json", k_clusters = 5):
        super().__init__()
        self.network_json_path = network_json_path

        # State Space dimensions: --->
        self.years_window = 10
        self.k_clusters = k_clusters
        self.num_metrics = 6
        self.features_per_cluster = self.num_metrics * 3 # 3 fixed conv filters

        # Tools: --->
        self.lsh_manager = LSHClusterManager(input_dim=self.num_metrics, num_functions_k=6)
        self.aggregator = FixedConvAggregator(num_metrics=self.num_metrics)

        # State Space: A 3D tensor -> (10, 5, 18): --->
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.years_window, self.k_clusters, self.features_per_cluster),
            dtype=np.float32
        )

        # Action Space: Binary (0 = Monitor, 1 = Isolate/Maintain): --->
        self.action_space = spaces.Discrete(2)

        self.simulator = None

        # Tracking variables for the complex reward function: --->
        self.initial_gsnr = 0.0
        self.final_gsnr = 0.0
        self.final_failed_lightpaths = 0

    def _get_active_telemetry(self) -> np.ndarray:
        """Extracts OPM telemetry for active services."""
        metrics_list = []
        for service in self.simulator.active_services:
            opm = service.opm_metrics

            metrics_list.append([
                opm.get('gsnr_db', 0.0), opm.get('osnr_db', 0.0),
                opm.get('cd', 0.0), opm.get('pmd', 0.0),
                opm.get('nli', 0.0), opm.get('pre_fec_ber', 0.0)
            ])

        return np.array(metrics_list, dtype=np.float32) if metrics_list else np.zeros((1, 6))

    def _process_yearly_clusters(self, telemetry: np.ndarray) -> np.ndarray:
        """
        Performs LSH clustering, Convolutional Aggregation, and
        sorts the top K clusters by severity (BER).
        """
        # Ensure enough variance for LSH: --->
        if telemetry.shape[0] < self.k_clusters:
            telemetry = np.pad(telemetry, ((0, self.k_clusters - telemetry.shape[0]), (0, 0)))

        clusters = self.lsh_manager.fit_predict(telemetry)

        cluster_summaries = []
        for indices in clusters:
            if not indices:
                continue

            cluster_opm = telemetry[indices]

            # Aggregate features using 1D Conv (18 dimensions): --->
            agg_features = self.aggregator(cluster_opm)

            # Calculate severity: Mean Pre-FEC BER (Index 5): --->
            severity = np.mean(cluster_opm[:, 5])
            cluster_summaries.append((severity, agg_features))

        # Sort descending by severity (worst degraded clusters first): --->
        cluster_summaries.sort(key=lambda x: x[0], reverse=True)

        # Select top K clusters and pad if necessary: --->
        fixed_representation = np.zeros((self.k_clusters, self.features_per_cluster), dtype=np.float32)
        for i in range(min(self.k_clusters, len(cluster_summaries))):
            fixed_representation[i] = cluster_summaries[i][1]

        return fixed_representation

    def reset(self, seed = None, options = None):
        """
        Runs a full decade (3650 days) of simulation to populate the sliding window.
        This establishes the MDP state for the RL agent.
        """
        super().reset(seed = seed)
        self.simulator = EONSimulatorV2(self.network_json_path)

        # Provision initial active grid: --->
        nodes = list(self.simulator.topology.graph.nodes())

        for _ in range(const.NUM_LIGHTPATHS * 3):
            src, dst = np.random.choice(nodes, 2, replace=False)
            self.simulator.provision_service(src, dst, 100.0)

            if len(self.simulator.active_services) >= const.NUM_LIGHTPATHS:
                break

        temporal_window = []

        telemetry = self._get_active_telemetry()
        self.initial_gsnr = np.mean(telemetry[:, 0]) if telemetry.shape[0] > 0 else 0.0

        # Simulate 10 years, snapshotting state every 365 days: --->
        for year in range(self.years_window):
            for day in range(365):
                self.simulator.step()

            telemetry = self._get_active_telemetry()
            yearly_features = self._process_yearly_clusters(telemetry)
            temporal_window.append(yearly_features)

        # Final health evaluation metrics: --->
        self.final_gsnr = np.mean(telemetry[:, 0]) if telemetry.shape[0] > 0 else 0.0

        # Hard failure threshold generally considered around Pre-FEC BER 1e-2 for generic models: --->
        self.final_failed_lightpaths = np.sum(telemetry[:, 5] > 0.01)

        self.current_state = np.stack(temporal_window) # Shape: (10, 5, 18)

        return self.current_state, {}

    def step(self, action):
        """
        Calculates the predictive maintenance cost-benefit reward.
        Because actions break continuity, this returns Done=True immediately.
        """
        degradation = self.initial_gsnr - self.final_gsnr # Drop in dB
        reward = 0.0

        if action == 0:                                                                                             # Passive Monitoring
            # Continuous penalty for ignored degradation and accumulated soft failures: --->
            reward -= (degradation * 2.0)
            reward -= (self.final_failed_lightpaths * 5.0)

            if self.final_failed_lightpaths > (const.NUM_LIGHTPATHS * 0.1):             # 10% network failure
                # Extreme penalty for ignoring an imminent hard failure: --->
                reward -= 100.0

        # Proactively Isolate / Maintenance: --->
        elif action == 1:
            maintenance_opex = -20.0
            if degradation < 1.5 and self.final_failed_lightpaths == 0:

                # False Positive: Wasted maintenance OpEx: --->
                reward = maintenance_opex
            else:
                # True Positive: Saved the network. Reward scales with the severity of the averted disaster: --->
                reward = maintenance_opex + (degradation * 10.0) + (self.final_failed_lightpaths * 15.0)

        return self.current_state, reward, True, False, {'degradation_db': degradation}
