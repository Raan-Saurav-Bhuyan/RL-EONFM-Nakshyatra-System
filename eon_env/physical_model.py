import numpy as np
from . import constants as const
from .lightpath import Lightpath
from .topology import NetworkTopology

def calculate_opm_metrics(lp: Lightpath, topology: NetworkTopology) -> dict:
    """
    Calculates all OPM metrics for a given lightpath using the current network state.
    This function simulates the data collected by OPM devices.
    """
    path_props = topology.get_path_properties(lp.path)
    num_spans = path_props['total_num_spans']
    total_length_m = path_props['total_length_km'] * 1000
    
    # Convert launch power from dBm to Watts: --->
    launch_power_w = 10**((lp.launch_power_dbm - 30) / 10)

    # 1. Calculate OSNR (Optical Signal-to-Noise Ratio): --->
    # ASE noise power = N_spans * h * f * B_ref * NF
    # (We work with linear values for calculation)
    signal_power = launch_power_w
    noise_power_ase = (
        num_spans * const.H_PLANCK * (const.C_LIGHT / 1550e-9) * 
        const.REFERENCE_BANDWIDTH * const.AMP_NOISE_FIGURE_LINEAR
    )
    osnr_linear = signal_power / noise_power_ase
    
    # Account for soft failure degradation: --->
    total_degradation_linear = 10**(path_props['total_degradation_db'] / 10)
    osnr_linear /= total_degradation_linear
    osnr_db = 10 * np.log10(osnr_linear)

    # 2. Calculate GSNR (Generalized Signal-to-Noise Ratio): --->
    # Using a simplified Gaussian Noise (GN) model for non-linear interference (NLI)
    # (NLI_power is proportional to P^3 * N_spans)
    nli_power = (
        (launch_power_w ** 3) * num_spans * 
        (const.FIBER_NONLINEAR_COEFF**2) / 
        (np.pi * const.FIBER_DISPERSION * (const.SYMBOL_RATE**2))
    )
    
    snr_nli_linear = signal_power / nli_power
    
    # 1/GSNR = 1/OSNR + 1/SNR_NLI: --->
    gsnr_linear = 1 / ( (1 / osnr_linear) + (1 / snr_nli_linear) )
    gsnr_db = 10 * np.log10(gsnr_linear)

    # 3. Calculate Chromatic Dispersion (CD): --->
    # (CD = D * L)
    total_cd = const.FIBER_DISPERSION * total_length_m

    # 4. Calculate Polarization Mode Dispersion (PMD): --->
    # (PMD accumulates with sqrt(L))
    total_pmd = const.FIBER_PMD_COEFF * np.sqrt(total_length_m)

    return {
        'gsnr_db': gsnr_db,
        'osnr_db': osnr_db,
        'total_cd_s_m2': total_cd,
        'total_pmd_s': total_pmd,
    }
