"""Centralized Hyperparameters & Constants for EON Simulator v2"""

# 1. Flex-Grid & Spectrum Parameters: --->
FSU_RESOLUTION_GHZ = 12.5      # Frequency Slot Unit in GHz
NUM_FSUS = 320                          # 320 * 12.5 GHz = 4 THz (approx C-band)
START_FREQ_THZ = 191.3             # Standard ITU grid start
NUM_CORES = 7                           # Space Division Multiplexing (SDM) - e.g., 7-core MCF

# 2. Physical Layer / GNPy defaults: --->
BAUD_RATE_GBDS = 64.0               # Superchannel default baud rate
ROLLOFF = 0.15                              # Pulse shaping roll-off
LAUNCH_POWER_DBM = 0.0         # Channel launch power

# Default Fiber parameters (GNPy standards): --->
FIBER_ATTENUATION = 0.2               # dB/km
FIBER_DISPERSION = 16.7                # ps/(nm*km)
FIBER_GAMMA = 1.27                      # 1/W/km (Non-linear coefficient)
FIBER_PMD_COEF = 0.04                  # ps/sqrt(km)

# Default Amplifier Models (EDFA): --->
EDFA_NF = 5.5                                # Noise Figure in dB
EDFA_TARGET_GAIN = 20.0            # Target gain in dB

# 3. Simulation & Soft Failure Parameters: --->
MAX_SIMULATION_DAYS = 365                  # Number of days to simulate the network run
NUM_LIGHTPATHS = 800                            # Number of lightpaths to provision
FAILURE_PROBABILITY_PER_DAY = 0.02
DEGRADATION_STEP_FIBER_LOSS = 0.05    # dB/km increase per failure event
DEGRADATION_STEP_AMP_NF = 0.1           # dB increase per failure event
DEGRADATION_STEP_PMD = 0.01               # ps/sqrt(km) increase per event

# Path for GNPy equipment library: --->
JSON_EQPT_CONFIG_PATH = 'eqpt_config.json'

# 4. SDM and Hardware Complexity Parameters: --->
SDM_MODE = 'MCF'                             # Media Types: 'MCF' (Multi-Core), 'MMF' (Multi-Mode), 'SMFB' (Single-Mode Bundle)
MCF_COUPLING_COEFF = 4e-4            # 1/m (Coupling coefficient for IC-XT via Coupled Mode Theory)
BENDING_RADIUS = 0.05                     # m
PROPAGATION_CONSTANT = 4e6       # 1/m
MMF_COUPLING_COEFF = 1e-3          # 1/m (Inter-modal coupling strength)
FIFO_LOSS_DB = 1.5                            # Fan-in/Fan-out structural insertion loss (dB)
WSS_LOSS_DB = 6.0                            # High-port-count Wavelength Selective Switch loss (dB)
WSS_MAX_PORTS = 30                       # Practical WSS max ports before core-bypassing is triggered

# 5. Component-Level Soft Failure Simulation Parameters: --->
DEGRADATION_MACROBENDING_LOSS = 0.5             # Localized dB loss per fiber surge event
DEGRADATION_EDFA_PUMP = 0.2                               # dB drop in gain and increase in NF per pump failure event
DEGRADATION_WSS_FILTER_SHIFT = 0.1                     # GHz shift per event causing spectral distortion
WSS_FILTER_PENALTY_COEFF = 0.05                            # dB loss penalty per GHz shift
