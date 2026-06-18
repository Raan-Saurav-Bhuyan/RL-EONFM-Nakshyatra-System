import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import torch

class ComponentLocalizationEnv(gym.Env):
    """
    Micro-MDP for the lower-level GNN-DQN agent to explicitly localize
    soft failures at the ROADM or EDFA span level.
    """
    def __init__(self, simulator_v2):
        super().__init__()
        self.simulator = simulator_v2
        self.topology = self.simulator.topology

        # 1. Build the Augmented Graph (ROADMs + EDFAs): --->
        self.aug_graph = nx.Graph()

        # Maps index to (type, link_u, link_v, span_idx): --->
        self.component_mapping = []
        self._build_augmented_graph()

        self.num_components = len(self.component_mapping)

        # Action Space: Select any component to inspect: --->
        self.action_space = spaces.Discrete(self.num_components)

        # Observation Space: Node features for the GNN: --->
        # (Features: [Mean_GSNR, Mean_BER, Lightpath_Count, Checked_Flag])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_components, 4), dtype=np.float32
        )

        self.max_search_steps = 25
        self.current_step = 0
        self.checked_components = set()

    def _build_augmented_graph(self):
        """Expands logical links into sequential EDFA spans."""
        # Add ROADMs: --->
        for node in self.topology.graph.nodes():
            idx = len(self.component_mapping)
            self.aug_graph.add_node(idx, type='ROADM', id=node)
            self.component_mapping.append(('ROADM', node, None, None))

        # Add EDFAs as nodes and connect them: --->
        for u, v, data in self.topology.graph.edges(data=True):
            # Safely fetch num_spans, defaulting to length-based calculation or 1: --->
            num_spans = data.get('num_spans', max(1, int(data.get('weight', 80) // 80)))

            # Assuming 1-based indexing for ROADMs: --->
            prev_node_idx = int(u) - 1

            for span_idx in range(num_spans):
                idx = len(self.component_mapping)
                self.aug_graph.add_node(idx, type='EDFA')
                self.component_mapping.append(('EDFA', u, v, span_idx))

                # Connect previous node to this EDFA: --->
                self.aug_graph.add_edge(prev_node_idx, idx)
                prev_node_idx = idx

            # Connect final EDFA to target ROADM: --->
            target_node_idx = int(v) - 1
            self.aug_graph.add_edge(prev_node_idx, target_node_idx)

        self.adjacency_matrix = nx.to_numpy_array(self.aug_graph)

    def _get_node_features(self):
        """Maps lightpath telemetry to the components they traverse."""
        features = np.zeros((self.num_components, 4), dtype=np.float32)

        # Initialize lightpath counts: --->
        for lp in self.simulator.active_services:
            path = lp.path_nodes
            gsnr = lp.opm_metrics.get('gsnr_db', 0)
            ber = lp.opm_metrics.get('pre_fec_ber', 0)

            # Map telemetry to ROADMs and EDFAs on the path: --->
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]

                # Find the indices of the components in our augmented graph: --->
                for comp_idx, comp_info in enumerate(self.component_mapping):
                    c_type, c_u, c_v, c_span = comp_info

                    if (c_type == 'ROADM' and c_u in (u, v)) or \
                       (c_type == 'EDFA' and ((c_u == u and c_v == v) or (c_u == v and c_v == u))):

                        features[comp_idx, 0] += gsnr
                        features[comp_idx, 1] += ber
                        features[comp_idx, 2] += 1.0 # Lightpath count

        # Average the metrics: --->
        mask = features[:, 2] > 0
        features[mask, 0] /= features[mask, 2]
        features[mask, 1] /= features[mask, 2]

        # Apply checked flags: --->
        for idx in self.checked_components:
            features[idx, 3] = 1.0

        return features

    def reset(self, **kwargs):
        self.current_step = 0
        self.checked_components.clear()
        return self._get_node_features(), {'adjacency': self.adjacency_matrix}

    def step(self, action):
        self.current_step += 1
        self.checked_components.add(action)

        comp_type, u, v, span_idx = self.component_mapping[action]

        # Validate explicitly against the simulator's ground truth tracked failures: --->
        actual_degradation = 0.0
        is_faulty = False

        for f_type, f_u, f_v, f_span, f_sev in self.simulator.faulty_components:
            if f_type == comp_type and ((f_u == u and f_v == v) or (f_u == v and f_v == u)):
                is_faulty = True
                actual_degradation = f_sev

                break

        # Dynamic continuous rewards for the localization agent: --->
        if is_faulty:
            # Correct localization: Reward proportional to severity of identified fault --->
            reward = 50.0 + (actual_degradation * 10.0)
            terminated = True
        else:
            # Misclassification penalty. Scales with how bad the network currently is: --->
            network_severity = sum([f_sev for _, _, _, _, f_sev in self.simulator.faulty_components])
            reward = -2.0 - (network_severity * 2.0)
            terminated = False

        # Limit search to prevent infinite loops: --->
        truncated = self.current_step >= self.max_search_steps

        # Penalty for exhausting budget: --->
        if truncated and not terminated:
            reward = -10.0

        info = {'is_faulty': is_faulty, 'true_faults_count': len(self.simulator.faulty_components)}
        return self._get_node_features(), reward, terminated, truncated, info
