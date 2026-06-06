import random
from . import constants as const
from .topology import NetworkTopology, FiberLayer, EDFA, ROADM

class SoftFailureInjector:
    """
    Injects realistic equipment degradation over time (Subtask 2.2.6 & Simulator goal).
    Isolates fiber-related anomalies (loss, PMD) from equipment anomalies (EDFA NF).
    """
    def __init__(self, topology: NetworkTopology):
        self.topology = topology

    def step(self):
        """Advances time by one step (day) and randomly degrades components."""
        if random.random() > const.FAILURE_PROBABILITY_PER_DAY:
            return # No failure today

        edges = list(self.topology.graph.edges(data=True))
        if not edges: return

        # Pick a random link in the topology: --->
        u, v, data = random.choice(edges)
        link = data['link']

        # Pick a random physical component within the link layer: --->
        component = random.choice(link.components)

        if isinstance(component, FiberLayer):
            anomaly_type = random.choice(['loss', 'pmd'])
            if anomaly_type == 'loss':
                component.loss_coef += const.DEGRADATION_STEP_FIBER_LOSS
                print(f"ANOMALY: Increased Fiber Loss on link {u}->{v}")
            else:
                component.pmd_coef += const.DEGRADATION_STEP_PMD
                print(f"ANOMALY: Increased Fiber PMD on link {u}->{v}")
        elif isinstance(component, EDFA):
            component.nf += const.DEGRADATION_STEP_AMP_NF
            print(f"ANOMALY: Degraded EDFA Noise Figure on link {u}->{v}")
