import networkx as nx
import json
from typing import List, Dict, Any
from . import constants as const

class FiberLayer:
    """Represents the physical fiber spans."""
    def __init__(self, length_km: float):
        self.length_km = length_km
        self.loss_coef = const.FIBER_ATTENUATION
        self.dispersion = const.FIBER_DISPERSION
        self.gamma = const.FIBER_GAMMA
        self.pmd_coef = const.FIBER_PMD_COEF
        self.macrobending_loss_db = 0.0 # Represents a localized fiber surge/bend

class EDFA:
    """Represents an intermediate Amplifier."""
    def __init__(self):
        self.nf = const.EDFA_NF
        self.gain = const.EDFA_TARGET_GAIN
        self.gain_ripple = 0.0 # Gain non-flatness
        self.pump_current_degradation_db = 0.0 # Tracks pump laser degradation
        self.thermal_drift_tilt = 0.0          # Gain non-flatness proxy

class FIFO:
    """Fan-in/Fan-out coupling device for Multi-Core/Multi-Mode Fibers."""
    def __init__(self):
        self.loss_db = const.FIFO_LOSS_DB

class ROADM:
    """Models programmable optical routing, WSS constraints, and core bypassing."""
    def __init__(self, uid: str, degree: int):
        self.uid = uid
        self.degree = degree
        self.wss_loss = const.WSS_LOSS_DB
        self.max_ports = const.WSS_MAX_PORTS
        self.wss_filter_shift_ghz = 0.0 # Tracks optical spectrum misalignment

        # Core-Bypassing Logic: If port requirements exceed max_ports, enable bypass: --->
        required_ports = self.degree * const.NUM_CORES
        self.core_bypass_enabled = required_ports > self.max_ports

class LinkLayer:
    """A sequence of Fibers and EDFAs connecting two ROADMs."""
    def __init__(self, source: str, target: str, total_length: float, span_length: float = 80.0):
        self.source = source
        self.target = target
        self.total_length = total_length

        # Build physical link components (spans and inline amplifiers): --->
        self.components = []
        spans = max(1, int(total_length // span_length))
        actual_span_len = total_length / spans

        for i in range(spans):
            fiber = FiberLayer(actual_span_len)
            self.components.append(fiber)
            if i < spans - 1:
                edfa = EDFA()
                # Transparent Span Configuration: Compensate exactly for preceding fiber loss:: --->
                # (Prevents signal power runaway and subsequent NLI explosion)
                edfa.gain = fiber.length_km * fiber.loss_coef
                self.components.append(edfa)

        # SDM Hardware Complexity: Fan-in and Fan-out devices at link edges: --->
        if const.SDM_MODE in ['MCF', 'MMF']:
            self.components.insert(0, FIFO())
            self.components.append(FIFO())

class NetworkTopology:
    """Graph-based foundation supporting nodes and multi-layer links."""
    def __init__(self):
        self.graph = nx.DiGraph() # Using DiGraph to represent directed fiber links
        self.edge_to_id = {}      # Maps (u, v) to an integer edge index for spectrum arrays
        self.id_to_edge = {}

    def load_from_json(self, network_json_path: str):
        """Loads the topology from an external JSON description (GNPy compatible)."""
        with open(network_json_path, 'r') as f:
            data = json.load(f)

        for element in data.get('elements', []):
            if element['type'] == 'ROADM':
                self.graph.add_node(element['uid'], type='ROADM')

        edge_idx = 0
        for connection in data.get('connections', []):
            src, dst = connection['from_node'], connection['to_node']
            length = connection.get('length', 80.0) # Default length if missing

            # Build structural layers: --->
            link = LinkLayer(src, dst, length)
            self.graph.add_edge(src, dst, link=link, weight=length)

            self.edge_to_id[(src, dst)] = edge_idx
            self.id_to_edge[edge_idx] = (src, dst)
            edge_idx += 1

        # Post-process to assign ROADM equipment with degree-aware core bypassing logic: --->
        for node in self.graph.nodes():
            degree = self.graph.degree(node)
            self.graph.nodes[node]['obj'] = ROADM(node, degree)

    def get_num_links(self) -> int:
        return self.graph.number_of_edges()

    def get_k_shortest_paths(self, src: str, dst: str, k: int = 3) -> List[List[str]]:
        """Yields K shortest structural routes."""
        try:
            paths = list(nx.shortest_simple_paths(self.graph, src, dst, weight='weight'))
            return paths[:k]
        except nx.NetworkXNoPath:
            return []

    def path_to_link_indices(self, path: List[str]) -> List[int]:
        """Converts a node-based path to a list of edge integer indices."""
        indices = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            indices.append(self.edge_to_id[(u, v)])

        return indices

    def extract_gnpy_components(self, path: List[str]) -> List[Any]:
        """
        Flattens the structural Path into an end-to-end sequence of physical components
        (ROADMs, FIFOs, Fibers, EDFAs) used by the PhysicalModel engine.
        """
        components = []
        for i in range(len(path)):
            node = path[i]
            components.append(self.graph.nodes[node]['obj']) # Add ROADM
            if i < len(path) - 1:
                u, v = path[i], path[i+1]
                link = self.graph[u][v]['link']
                components.extend(link.components)

        return components
