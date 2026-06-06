import numpy as np
from typing import List, Tuple, Optional
from . import constants as const

class SpectrumManager:
    """
    Handles fine-grained spectral management for Flex-grid and SDM.
    """
    def __init__(self, num_links: int):
        # 3D Matrix: [Links, Cores, FSUs]: --->
        # False means available, True means occupied.
        self.allocation_grid = np.zeros(
            (num_links, const.NUM_CORES, const.NUM_FSUS), dtype=bool
        )

    def find_superchannel_allocation(self, link_indices: List[int], required_fsus: int) -> Optional[Tuple[int, int]]:
        """
        Searches for a block that satisfies:
        1. Spectral Continuity (Same FSUs along all links)
        2. Spectral Contiguity (Adjacent FSUs)
        3. Spatial Continuity (Same core end-to-end)
        """
        if not link_indices:
            return None

        # Extract the spectrum state for the required path: --->
        # Shape: (num_links_in_path, cores, fsus)
        path_grid = self.allocation_grid[link_indices, :, :]

        # Enforce Spectral & Spatial Continuity: collapse across links via logical OR: --->
        # Shape: (cores, fsus). If True, the slot is occupied somewhere on the path.
        collapsed_grid = np.any(path_grid, axis=0)

        # Search each core for contiguous FSUs: --->
        for core_idx in range(const.NUM_CORES):
            available_slots = ~collapsed_grid[core_idx] # True where available

            # Find contiguous blocks: --->
            count = 0
            for start_fsu in range(const.NUM_FSUS):
                if available_slots[start_fsu]:
                    count += 1
                    if count == required_fsus:
                        return (core_idx, start_fsu - required_fsus + 1)
                else:
                    count = 0

        return None # No allocation found (Blocking)

    def allocate(self, link_indices: List[int], core_idx: int, start_fsu: int, num_fsus: int):
        """Marks the FSUs as occupied."""
        self.allocation_grid[np.ix_(link_indices, [core_idx], range(start_fsu, start_fsu + num_fsus))] = True

    def deallocate(self, link_indices: List[int], core_idx: int, start_fsu: int, num_fsus: int):
        """Frees up the FSUs."""
        self.allocation_grid[np.ix_(link_indices, [core_idx], range(start_fsu, start_fsu + num_fsus))] = False
