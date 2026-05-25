import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

from eon_env import constants as const
from clustering.LSH import LSHClusterManager
from cluster_aggregations import FixedConvAggregator

class Surrogate_Reward_Wrapper(gym.ObservationWrapper):
    """
    Gym wrapper that converts raw OPM state into fixed-size aggregated features,
    maintains historical context, and computes gradient-based surrogate rewards.
    """
    def __init__(self, env):
        super().__init__(env)
        self.num_links = len(self.unwrapped.topology.edges_list)

        # Action Space: 0 = Monitor, 1..N = Isolate Link i-1
        self.action_space = spaces.Discrete(self.num_links + 1)

        # Initialize Feature Extraction Pipeline
        self.lsh_manager = LSHClusterManager(
            input_dim=4, num_functions_k=8
        )
        self.aggregator = FixedConvAggregator(num_metrics=4)

        # 4 metrics * 3 filters = 12 features per cluster
        self.features_per_cluster = 12
        self.single_state_dim = const.N_CLUSTERS * self.features_per_cluster
        self.history_dim = self.single_state_dim * const.HISTORY_WINDOW

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.history_dim,), dtype=np.float32
        )

        self.state_history = deque(maxlen=const.HISTORY_WINDOW)
        self.baseline_gsnr = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Fit LSH on the healthy baseline network state: --->
        self.lsh_manager.fit(obs)

        # Clear history and populate with initial states: --->
        agg_state = self._extract_features(obs)
        for _ in range(const.HISTORY_WINDOW):
            self.state_history.append(agg_state)

        self.baseline_gsnr = self._get_network_mean_gsnr(obs)

        return self._get_historical_state(), info

    def _extract_features(self, raw_obs):
        """Transforms raw OPM (800, 4) -> Aggregated Vector (120,)."""
        clusters = self.lsh_manager.predict(raw_obs)

        cluster_features = []
        for cluster_indices in clusters:
            if not cluster_indices:
                # Empty cluster padding: --->
                cluster_features.append(np.zeros(self.features_per_cluster))
            else:
                opm_matrix = raw_obs[cluster_indices]
                features = self.aggregator(opm_matrix)
                cluster_features.append(features)

        return np.concatenate(cluster_features)

    def _get_historical_state(self):
        """Flattens the history queue into a single 1D vector."""
        return np.concatenate(self.state_history)

    def _get_network_mean_gsnr(self, raw_obs):
        """Helper to calculate overall network health."""
        return np.mean(raw_obs[:, 0]) # Index 0 is GSNR

    def step(self, action):
        """Executes action, computes surrogate reward, and handles evaluation periods."""
        if action == 0:
            # Monitor: standard single step progression: --->
            raw_obs, reward, terminated, truncated, info = self.env.step(action)
            current_gsnr = self._get_network_mean_gsnr(raw_obs)

            # Minor penalty if network is actively degrading and we just wait: --->
            grad = current_gsnr - self.baseline_gsnr # Negative if degrading
            if grad < 0:                        # <--- Network is degrading
                reward = max(const.MAX_MONITOR_PENALTY, grad * const.MONITOR_PENALTY_FACTOR)
            else:
                reward = 0.0                 # <--- No degradation, no penalty for monitoring

            self.baseline_gsnr = current_gsnr
            self.state_history.append(self._extract_features(raw_obs))

            return self._get_historical_state(), reward, terminated, truncated, info

        else:
            # Reroute / Isolation: Evaluate the physical gradient impact: --->
            suspect_edge_idx = action - 1
            self.unwrapped.topology.isolate_link(suspect_edge_idx)

            # Get the (u,v) tuple for the suspected link: --->
            suspect_u, suspect_v = self.unwrapped.topology.edges_list[suspect_edge_idx]

            # 1. Store current degradation of the suspected link: --->
            original_suspect_degradation = self.unwrapped.topology._get_link_degradation(suspect_u, suspect_v)

            # 2. Calculate hypothetical GSNR if the suspected link was fixed (all other degradation remains): --->
            # (Temporarily set degradation of suspected link to 0)
            self.unwrapped.topology._set_link_degradation(suspect_u, suspect_v, 0.0)
            self.unwrapped._update_all_opm_metrics()
            hypothetical_obs = self.unwrapped._get_observation()
            hypothetical_gsnr_if_correct = self._get_network_mean_gsnr(hypothetical_obs)

            # Restore original degradation: --->
            self.unwrapped.topology._set_link_degradation(suspect_u, suspect_v, original_suspect_degradation)

            # Step environment forward by T_EVAL_STEPS to evaluate the physical gradient impact over time: --->
            for _ in range(const.T_EVAL_STEPS):
                eval_obs, _, env_terminated, env_truncated, _ = self.env.step(0)
                if env_terminated or env_truncated:
                    break

            eval_gsnr = self._get_network_mean_gsnr(eval_obs)

            # Surrogate Reward Calculation: Did isolating the link achieve the same GSNR as if it were perfectly fixed?
            # This decouples the reward from other background degradations. --->
            diff = abs(eval_gsnr - hypothetical_gsnr_if_correct)

            if diff < const.GRADIENT_EPSILON:
                reward = const.POS_REWARD  # Near-perfect localization
            elif diff > const.MAX_DIFFERENCE_FOR_PARTIAL_REWARD:
                reward = const.NEG_REWARD  # Significant mislocalization
            else:
                # Interpolate reward between POS_REWARD and NEG_REWARD
                # As diff increases from GRADIENT_EPSILON to MAX_DIFFERENCE_FOR_PARTIAL_REWARD,
                # reward decreases from POS_REWARD to NEG_REWARD.
                scaled_diff = (diff - const.GRADIENT_EPSILON) / (const.MAX_DIFFERENCE_FOR_PARTIAL_REWARD - const.GRADIENT_EPSILON)
                reward = const.POS_REWARD - scaled_diff * (const.POS_REWARD - const.NEG_REWARD)

            # Restore the link and metrics: --->
            self.unwrapped.topology.unisolate_all()
            self.unwrapped._update_all_opm_metrics()
            restored_obs = self.unwrapped._get_observation()
            self.state_history.append(self._extract_features(restored_obs))

            info = self.unwrapped._get_info()

            # Reroute actions always terminate the diagnostic episode: --->
            return self._get_historical_state(), reward, True, False, info
