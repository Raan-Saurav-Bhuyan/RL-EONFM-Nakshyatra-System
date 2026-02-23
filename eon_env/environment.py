import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from . import constants as const
from .topology import NetworkTopology
from .lightpath import Lightpath
from .physical_model import calculate_opm_metrics

class EONEnv(gym.Env):
    """
    Reinforcement Learning Environment for Elastic Optical Networks.

    The state represents the OPM metrics of all active lightpaths.
    The action space is currently a placeholder, as the primary goal is to
    simulate the network's state evolution over time.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super().__init__()
        self.topology = NetworkTopology()
        self.lightpaths = []
        self.current_step = 0

        # Define Observation Space: --->
        # (For each lightpath, we observe 4 OPM metrics: GSNR, OSNR, CD, PMD)
        # Shape: (Number of Lightpaths, Number of Metrics)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(const.NUM_LIGHTPATHS, 4),
            dtype=np.float32
        )

        # Define Action Space: --->
        # As the goal is to observe network degradation, we use a simple
        # discrete action space where action 0 means "continue simulation".
        # (This can be expanded later for an agent to take actions like rerouting.)
        self.action_space = spaces.Discrete(1)

    def _provision_initial_lightpaths(self):
        """Creates a set of random lightpaths at the start of an episode."""
        self.lightpaths = []
        nodes = list(self.topology.graph.nodes)

        for i in range(const.NUM_LIGHTPATHS):
            source, dest = random.sample(nodes, 2)
            path = self.topology.get_shortest_path(source, dest)

            # Assign random dynamic parameters: --->
            mod_format = random.choice(list(const.MODULATION_FORMATS.keys()))
            bit_rate = const.MODULATION_FORMATS[mod_format]['se'] * const.SYMBOL_RATE / 1e9 # Gbps
            launch_power = random.uniform(0.0, 2.0) # dBm

            lp = Lightpath(
                lp_id=i,
                source=source,
                destination=dest,
                path=path,
                modulation_format=mod_format,
                bit_rate=bit_rate,
                launch_power_dbm=launch_power
            )
            self.lightpaths.append(lp)

        print(f"Provisioned {len(self.lightpaths)} initial lightpaths.")

    def _update_all_opm_metrics(self):
        """Recalculates OPM metrics for every lightpath."""
        for lp in self.lightpaths:
            lp.opm_metrics = calculate_opm_metrics(lp, self.topology)

    def _get_observation(self) -> np.ndarray:
        """
        Constructs the state vector from the OPM metrics of all lightpaths.
        This vector is what the RL agent will receive.
        """
        obs = np.zeros((const.NUM_LIGHTPATHS, 4), dtype=np.float32)

        for i, lp in enumerate(self.lightpaths):
            metrics = lp.opm_metrics
            obs[i, 0] = metrics.get('gsnr_db', 0)
            obs[i, 1] = metrics.get('osnr_db', 0)
            obs[i, 2] = metrics.get('total_cd_s_m2', 0)
            obs[i, 3] = metrics.get('total_pmd_s', 0)

        return obs

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)

        self.current_step = 0
        self.topology.reset()
        self._provision_initial_lightpaths()
        self._update_all_opm_metrics()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        Executes one time step within the environment.
        This corresponds to one day of network operation.
        """
        self.current_step += 1

        # Simulate network evolution (soft failures): --->
        self.topology.update_soft_failures()

        # Recalculate OPMs for all lightpaths based on the new network state: --->
        self._update_all_opm_metrics()

        # Calculate reward: --->
        # (Reward is based on the number of lightpaths that meet their GSNR requirement)
        healthy_lightpaths = 0
        for lp in self.lightpaths:
            req_gsnr = const.MODULATION_FORMATS[lp.modulation_format]['req_gsnr_db']
            if lp.opm_metrics['gsnr_db'] >= req_gsnr:
                healthy_lightpaths += 1

        reward = healthy_lightpaths / len(self.lightpaths) # Normalize reward

        # 4. Check for termination conditions: --->
        terminated = reward < 0.5  # Episode ends if less than 50% of LPs are healthy
        truncated = self.current_step >= const.MAX_SIMULATION_STEPS

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_info(self):
        """Returns auxiliary diagnostic information."""
        return {
            "step": self.current_step,
            "lightpath_details": [lp.__dict__ for lp in self.lightpaths]
        }

    def render(self, mode='human'):
        """Prints a summary of the current environment state."""
        if mode == 'human':
            print(f"\n--- Step: {self.current_step} ---")
            for lp in self.lightpaths:
                req_gsnr = const.MODULATION_FORMATS[lp.modulation_format]['req_gsnr_db']
                gsnr = lp.opm_metrics['gsnr_db']
                status = "HEALTHY" if gsnr >= req_gsnr else "FAILED"
                print(f"  LP {lp.lp_id:2d} ({lp.source:2d}->{lp.destination:2d}): "
                      f"GSNR={gsnr:5.2f}dB (Req: {req_gsnr:4.1f}dB) - {status}")
