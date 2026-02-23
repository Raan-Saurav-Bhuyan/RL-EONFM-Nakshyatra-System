from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Lightpath:
    """This class represents a single lightpath in the optical network."""
    lp_id: int
    source: int
    destination: int
    path: List[int]  # List of node IDs forming the path
    
    # Dynamically assigned network parameters: --->
    modulation_format: str
    bit_rate: float  # Gbps
    launch_power_dbm: float
    
    # OPM metrics collected from the network: --->
    opm_metrics: Dict[str, float] = field(default_factory = dict)

    def __repr__(self):
        return (f"LP(id = {self.lp_id}, {self.source}->{self.destination}, "
                f"mod = {self.modulation_format}, metrics = {self.opm_metrics})")
