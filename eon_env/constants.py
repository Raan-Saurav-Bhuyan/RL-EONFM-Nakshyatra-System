import numpy as np

# Physical Constants: --->
H_PLANCK = 6.62607015e-34  # Planck's constant (J·s)
C_LIGHT = 299792458  # Speed of light in vacuum (m/s)
C_LIGHT_FIBER = 2.05e8 # Speed of light in fiber (m/s)

# Fiber Parameters (Standard Single-Mode Fiber - SSMF): --->
FIBER_LOSS_DB = 0.2  # Attenuation coefficient (dB/km)
FIBER_LOSS_LINEAR = 10 ** (FIBER_LOSS_DB / 10)
FIBER_DISPERSION = 17e-6  # Chromatic dispersion (s/m^2 or ps/nm/km)
FIBER_PMD_COEFF = 0.1e-12 / np.sqrt(1e3) # PMD coefficient (s/sqrt(m))
FIBER_NONLINEAR_COEFF = 1.3e-3  # Nonlinear coefficient (1/W/m)

# Amplifier Parameters (EDFA): --->
AMP_GAIN_DB = 16.0  # Amplifier gain to compensate for 80km span loss (dB)
AMP_NOISE_FIGURE_DB = 5.0  # Noise Figure (dB)
AMP_NOISE_FIGURE_LINEAR = 10 ** (AMP_NOISE_FIGURE_DB / 10)

# Channel/Signal Parameters: --->
REFERENCE_BANDWIDTH = 12.5e9  # Reference bandwidth for OSNR (Hz)
SYMBOL_RATE = 32e9  # Baud rate (Baud)

# Modulation Formats: --->
# (Defines spectral efficiency (bits/symbol) and required GSNR (dB) for a BER of 1e-3)
MODULATION_FORMATS = {
    'BPSK': {'se': 1, 'req_gsnr_db': 6.8},
    'QPSK': {'se': 2, 'req_gsnr_db': 9.8},
    '8-QAM': {'se': 3, 'req_gsnr_db': 13.5},
    '16-QAM': {'se': 4, 'req_gsnr_db': 16.6},
}

# Simulation Parameters: --->
SPAN_LENGTH_KM = 80  # Length of a single fiber span (km)
NUM_LIGHTPATHS = 800  # Number of lightpaths to provision in the network
MAX_SIMULATION_STEPS = 365 # Corresponds to one year of daily monitoring

# Soft Failure Simulation: --->
FAILURE_PROBABILITY = 0.05  # Probability of a new soft failure event per step
DEGRADATION_PER_EVENT_DB = 0.5 # How much a link's quality degrades in one event (dB)
STATIC_FAILURE_EDGES = [(1, 2),  (7, 8), (12, 13), (5, 7), (3, 1), (1, 8), (2, 5), (11, 19)] # Static links to inject failures for RL testing

# RL and Surrogate Reward Parameters: --->
N_CLUSTERS = 22               # Fixed number of clusters for the state vector (roughly 1 per link)
HISTORY_WINDOW = 10            # Number of historical states to stack
T_EVAL_STEPS = 10              # No longer strictly needed with instant evaluation
GRADIENT_EPSILON = 0.01       # Minimum dB improvement required to consider stabilization successful
