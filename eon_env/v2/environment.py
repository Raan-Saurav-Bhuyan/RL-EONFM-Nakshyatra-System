import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import os

from . import constants as const
from .simulator import EONSimulatorV2

class EONEnvV2(gym.Env):
    """
    Reinforcement Learning Environment for the Version 2.0 EON Digital Twin.
    Wraps the EONSimulatorV2 engine for Gym compatibility.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, network_json_path="nsfnet.json"):
        super().__init__()
        self.network_json_path = network_json_path

        self.simulator = None
        self.current_step = 0

        # Define Observation Space:
        # 6 OPM metrics: GSNR, OSNR, CD, PMD, NLI, Pre-FEC BER
        # Shape: (Number of Lightpaths, Number of Metrics)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(const.NUM_LIGHTPATHS, 6),
            dtype=np.float32
        )

        # Define Action Space (Discrete 1 for passive monitoring initially,
        # can be wrapped by RL Wrapper later for isolation actions)
        self.action_space = spaces.Discrete(1)

    def _provision_initial_lightpaths(self):
        """Creates a set of random superchannels at the start of an episode."""
        nodes = list(self.simulator.topology.graph.nodes())

        provisioned = 0
        max_attempts = const.NUM_LIGHTPATHS * 3

        for _ in range(max_attempts):
            src, dst = random.sample(nodes, 2)
            bitrate = random.choice([100.0, 200.0, 400.0]) # Gbps

            if self.simulator.provision_service(src, dst, bitrate):
                provisioned += 1

            if provisioned >= const.NUM_LIGHTPATHS:
                break

        print(f"Provisioned {provisioned}/{const.NUM_LIGHTPATHS} superchannels in V2 Environment.")

    def _get_observation(self) -> np.ndarray:
        """
        Constructs the state vector from the OPM metrics of all active services.
        Pads with zeros if some services were blocked due to spectrum limits.
        """
        obs = np.zeros((const.NUM_LIGHTPATHS, 6), dtype=np.float32)

        for i, service in enumerate(self.simulator.active_services):
            if i >= const.NUM_LIGHTPATHS:
                break
            metrics = service.opm_metrics
            obs[i, 0] = metrics.get('gsnr_db', 0.0)
            obs[i, 1] = metrics.get('osnr_db', 0.0)
            obs[i, 2] = metrics.get('cd', 0.0)
            obs[i, 3] = metrics.get('pmd', 0.0)
            obs[i, 4] = metrics.get('nli', 0.0)
            obs[i, 5] = metrics.get('pre_fec_ber', 0.0)

        return obs

    def reset(self, seed=None, options = None):
        """Resets the environment to an initial pristine state."""
        super().reset(seed=seed)

        self.current_step = 0
        self.simulator = EONSimulatorV2(self.network_json_path)
        self._provision_initial_lightpaths()

        return self._get_observation(), self._get_info()

    def step(self, action):
        """Executes one day of network operation."""
        self.current_step += 1
        self.simulator.step()

        return self._get_observation(), 0.0, False, self.current_step >= const.MAX_SIMULATION_DAYS, self._get_info()

    def _get_info(self):
        return {"step": self.current_step, "active_services": len(self.simulator.active_services)}
