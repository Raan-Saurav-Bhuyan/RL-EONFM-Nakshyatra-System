import math
from typing import Dict, List, Any
from . import constants as const
from .topology import NetworkTopology
from .spectrum import SpectrumManager
from .topology import FiberLayer, EDFA
from .physical_model import GNPyPhysicalEngine
from .soft_failures import SoftFailureInjector

class ServiceDemand:
    """Represents an active connection in the network."""
    def __init__(self, s_id: int, src: str, dst: str, bitrate_gbps: float):
        self.s_id = s_id
        self.src = src
        self.dst = dst
        self.bitrate_gbps = bitrate_gbps

        # Superchannel properties mapped after provisioning: --->
        self.path_nodes = []
        self.core_idx = -1
        self.start_fsu = -1
        self.num_fsus = 0
        self.opm_metrics = {}

class EONSimulatorV2:
    """
    The core digital twin simulator tying Topology, Spectrum, Physics, and Degradations.
    """
    def __init__(self, network_json_path: str):
        self.topology = NetworkTopology()
        self.topology.load_from_json(network_json_path)

        self.spectrum_manager = SpectrumManager(self.topology.get_num_links())
        self.physics_engine = GNPyPhysicalEngine()
        self.failure_injector = SoftFailureInjector(self.topology)

        self.active_services: List[ServiceDemand] = []
        self.current_day = 0
        self.service_counter = 0

        # Tracks ground truth: (type, u, v, span_idx, degradation_severity): --->
        self.faulty_components = []

    def provision_service(self, src: str, dst: str, bitrate_gbps: float) -> bool:
        """Provisions a superchannel incorporating Flex-grid and SDM constraints."""
        # Calculate required FSUs based on Baud Rate and Spectral Efficiency: --->
        required_fsus = math.ceil((const.BAUD_RATE_GBDS * 1e9) / (const.FSU_RESOLUTION_GHZ * 1e9))

        demand = ServiceDemand(self.service_counter, src, dst, bitrate_gbps)

        # Try K-shortest paths: --->
        paths = self.topology.get_k_shortest_paths(src, dst)
        for path in paths:
            link_indices = self.topology.path_to_link_indices(path)
            allocation = self.spectrum_manager.find_superchannel_allocation(link_indices, required_fsus)

            if allocation:
                core_idx, start_fsu = allocation
                self.spectrum_manager.allocate(link_indices, core_idx, start_fsu, required_fsus)

                demand.path_nodes = path
                demand.core_idx = core_idx
                demand.start_fsu = start_fsu
                demand.num_fsus = required_fsus

                self.active_services.append(demand)
                self.service_counter += 1
                self._update_service_opm(demand)

                return True

        # Blocked due to resources or no path: --->
        return False

    def _update_service_opm(self, demand: ServiceDemand):
        """Updates physical OPM state utilizing GNPy underlying models."""
        components = self.topology.extract_gnpy_components(demand.path_nodes)
        metrics = self.physics_engine.evaluate_path_quality(
            components, const.LAUNCH_POWER_DBM, const.BAUD_RATE_GBDS, demand.core_idx
        )
        demand.opm_metrics = metrics

    def step(self):
        """Advances network temporal state."""
        self.current_day += 1
        self.failure_injector.step()

        # Update metrics after possible degradations: --->
        for demand in self.active_services:
            self._update_service_opm(demand)

        # Sync ground truth faulty components for the RL Localization Agent --->
        self._sync_faulty_components()

    def _sync_faulty_components(self):
        """Extracts the exact components experiencing soft failures for agent evaluation."""
        self.faulty_components.clear()

        # Scan physical components inside each link: --->
        for u, v, data in self.topology.graph.edges(data=True):
            link = data['link']
            for span_idx, comp in enumerate(link.components):
                if isinstance(comp, EDFA) and comp.nf > const.EDFA_NF:
                    severity = comp.nf - const.EDFA_NF
                    self.faulty_components.append(('EDFA', u, v, span_idx, severity))

                elif isinstance(comp, FiberLayer):
                    if comp.loss_coef > const.FIBER_ATTENUATION:
                        severity = comp.loss_coef - const.FIBER_ATTENUATION
                        self.faulty_components.append(('EDFA', u, v, span_idx, severity)) # Env treats fiber surges as link-level faults
