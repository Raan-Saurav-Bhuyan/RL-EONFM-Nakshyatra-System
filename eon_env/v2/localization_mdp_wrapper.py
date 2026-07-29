import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import torch

class ComponentLocalizationEnv(gym.Env):
    """
    Single-step classification MDP for the GNN-based PPO localization agent.

    The agent classifies ALL components in the augmented network graph
    simultaneously in a single forward pass, producing a binary label
    (healthy=0, faulty=1) for every node.

    The episode is exactly 1 step: classify all nodes → receive reward → done.
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

        # Action Space: Binary vector — classify each component as healthy (0) or faulty (1): --->
        self.action_space = spaces.MultiBinary(self.num_components)

        # Observation Space: Node features for the GNN: --->
        # (Features: [Mean_GSNR, Mean_BER, Lightpath_Count, Node_Degree])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_components, 4), dtype=np.float32
        )

        # Reward function coefficients (asymmetric weighting): --->
        self.r_tp = 10.0    # Reward for correctly identifying a faulty component
        self.r_tn = 0.5     # Small reward for correctly classifying a healthy component
        self.r_fp = 3.0     # Penalty for false alarm on a healthy component
        self.r_fn = 15.0    # Heavier penalty for missing a faulty component (safety-critical)
        self.lambda_f1 = 20.0  # Bonus scaled by episode F1-score

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

        # Pre-compute node degrees for the 4th feature: --->
        self._node_degrees = np.array(
            [self.aug_graph.degree(i) for i in range(len(self.component_mapping))],
            dtype=np.float32
        )

    def _get_ground_truth_labels(self):
        """
        Returns a binary vector [num_components] where 1 = faulty, 0 = healthy,
        and a severity vector [num_components] for reward scaling.
        Derived from simulator.faulty_components ground truth.
        """
        labels = np.zeros(self.num_components, dtype=np.float32)
        severities = np.zeros(self.num_components, dtype=np.float32)

        for f_type, f_u, f_v, f_span, f_sev in self.simulator.faulty_components:
            for comp_idx, comp_info in enumerate(self.component_mapping):
                c_type, c_u, c_v, c_span = comp_info

                if f_type == c_type and ((f_u == c_u and f_v == c_v) or (f_u == c_v and f_v == c_u)):
                    labels[comp_idx] = 1.0
                    severities[comp_idx] = max(severities[comp_idx], f_sev)

        return labels, severities

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

        # 4th feature: Node degree (structural connectivity): --->
        features[:, 3] = self._node_degrees

        return features

    def reset(self, **kwargs):
        return self._get_node_features(), {'adjacency': self.adjacency_matrix}

    def step(self, action):
        """
        Single-step full-graph classification.

        Parameters
        ----------
        action : np.ndarray or torch.Tensor of shape [num_components]
            Binary vector where 1 = predicted faulty, 0 = predicted healthy.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        # Convert action to numpy if tensor: --->
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        action = action.astype(np.float32)

        # Get ground truth: --->
        ground_truth, severities = self._get_ground_truth_labels()

        # Compute per-component confusion matrix: --->
        tp_mask = (action == 1) & (ground_truth == 1)
        fp_mask = (action == 1) & (ground_truth == 0)
        tn_mask = (action == 0) & (ground_truth == 0)
        fn_mask = (action == 0) & (ground_truth == 1)

        tp = int(tp_mask.sum())
        fp = int(fp_mask.sum())
        tn = int(tn_mask.sum())
        fn = int(fn_mask.sum())

        # ── Continuous composite reward: ──

        # Per-node rewards summed across all nodes: --->
        tp_reward = self.r_tp * (severities[tp_mask].sum() if tp > 0 else 0.0)
        tn_reward = self.r_tn * tn
        fp_penalty = self.r_fp * fp
        fn_penalty = self.r_fn * (severities[fn_mask].sum() if fn > 0 else 0.0)

        # F1-score bonus: --->
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_bonus = self.lambda_f1 * f1

        reward = tp_reward + tn_reward - fp_penalty - fn_penalty + f1_bonus

        # Episode always terminates immediately (single-step classification): --->
        terminated = True
        truncated = False

        info = {
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1': f1,
            'true_faults_count': int(ground_truth.sum()),
            'predictions': action.copy(),
            'ground_truth': ground_truth.copy(),
        }

        return self._get_node_features(), reward, terminated, truncated, info
