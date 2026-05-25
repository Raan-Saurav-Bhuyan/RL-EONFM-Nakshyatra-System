import networkx as nx
import numpy as np
from . import constants as const

class NetworkTopology:
    """
    Manages the network topology, including nodes, links, and soft failures.
    """
    def __init__(self):
        self.graph = self._create_nsfnet_topology()
        self.edges_list = list(self.graph.edges())
        self.isolated_link_idx = -1 # -1 means no link is isolated

    def _create_nsfnet_topology(self) -> nx.Graph:
        """
        Creates the NSFNET topology with 14 nodes and 21 links.
        Link attributes store physical properties.
        """
        G = nx.Graph()
        nodes = range(1, 15)
        G.add_nodes_from(nodes)

        # Define links with their lengths in km: --->
        edges = [
            (1, 2, 1100), (1, 3, 1700), (1, 4, 2900), (2, 3, 600), (2, 5, 900),
            (3, 6, 1100), (4, 7, 1000), (4, 9, 1400), (5, 6, 800), (5, 8, 1900),
            (6, 11, 2100), (7, 8, 800), (7, 9, 400), (8, 10, 900), (9, 10, 800),
            (9, 12, 1100), (10, 11, 800), (10, 13, 1000), (11, 14, 800),
            (12, 13, 300), (13, 14, 500)
        ]

        for u, v, length in edges:
            num_spans = int(np.ceil(length / const.SPAN_LENGTH_KM))

            G.add_edge(
                u, v,
                length_km=length,
                num_spans=num_spans,

                # This factor will be increased to simulate soft failures: --->
                degradation_factor_db = 0.0
            )

        return G

    def update_soft_failures(self):
        """
        Rarely introduces or worsens soft failures on specific static links.
        This simulates events like fiber bending or component aging.
        This restricts the degradation to a controlled set to test RL agent learning.
        """
        for u, v in self.graph.edges():
            # Restrict failures to specifically designated static edges
            if (u, v) in const.STATIC_FAILURE_EDGES or (v, u) in const.STATIC_FAILURE_EDGES:
                if np.random.rand() < const.FAILURE_PROBABILITY:
                    self.graph[u][v]['degradation_factor_db'] += const.DEGRADATION_PER_EVENT_DB
                    print(f"INFO: Soft failure degradation on static link ({u}-{v}) increased.")

    def isolate_link(self, edge_idx: int):
        """Simulates rerouting traffic away from a suspected link."""
        self.isolated_link_idx = edge_idx

    def unisolate_all(self):
        """Restores traffic to all links."""
        self.isolated_link_idx = -1

    def get_path_properties(self, path: list) -> dict:
        """Calculates total length and spans for a given path."""
        total_length_km = 0
        total_num_spans = 0
        total_degradation_db = 0

        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            total_length_km += self.graph[u][v]['length_km']
            total_num_spans += self.graph[u][v]['num_spans']

            # Only apply degradation if the link is NOT currently isolated (rerouted)
            current_edge = (u, v)
            if self.isolated_link_idx == -1 or current_edge not in [self.edges_list[self.isolated_link_idx], self.edges_list[self.isolated_link_idx][::-1]]:
                total_degradation_db += self.graph[u][v]['degradation_factor_db']

        return {
            'total_length_km': total_length_km,
            'total_num_spans': total_num_spans,
            'total_degradation_db': total_degradation_db
        }

    def get_shortest_path(self, source: int, target: int) -> list:
        """Finds the shortest path between two nodes."""
        return nx.shortest_path(self.graph, source=source, target=target, weight='length_km')

    def reset(self):
        """Resets all soft failures on the links."""
        for u, v in self.graph.edges():
            self.graph[u][v]['degradation_factor_db'] = 0.0
        self.isolated_link_idx = -1
