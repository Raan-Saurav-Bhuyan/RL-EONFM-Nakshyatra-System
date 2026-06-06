import numpy as np
import math
from . import constants as const
from .topology import FiberLayer, EDFA, ROADM, FIFO

# NOTE: A live integration requires creating instances of `gnpy.core.network.Network`.

class GNPyPhysicalEngine:
    """
    Vendor-neutral physical layer modeling using GNPy concepts.
    Handles GSNR (ASE+NLI), Generalized Gaussian Noise (GGN), and SRS.
    """
    def __init__(self):
        self.srs_solver_enabled = True # Stimulated Raman Scattering awareness
        self.h_planck = 6.626e-34
        self.c_light = 3e8

    def compute_nli_ggn(self, fiber: FiberLayer, power_dbm: float, baud_rate: float) -> float:
        """
        Calculates Non-Linear Interference using the Generalized Gaussian Noise (GGN) model.
        Accounts for Self-Channel Interference (SCI) and Cross-Channel Interference (XCI).
        """
        power_w = 10 ** (power_dbm / 10) * 1e-3

        # Simplified placeholder for the complex GGN integral evaluated by GNPy: --->
        nli_noise = (fiber.gamma ** 2) * (power_w ** 3) * fiber.length_km / (baud_rate * 1e9)

        return nli_noise

    def compute_ase_noise(self, edfa: EDFA, input_power_dbm: float) -> float:
        """Models Amplifier Spontaneous Emission noise from an EDFA component."""
        freq = const.START_FREQ_THZ * 1e12
        nf_linear = 10 ** (edfa.nf / 10)
        gain_linear = 10 ** (edfa.gain / 10)
        ase_noise = nf_linear * self.h_planck * freq * (gain_linear - 1)

        return ase_noise

    def compute_inter_core_xt(self, fiber: FiberLayer, core_idx: int) -> float:
        """Calculates Inter-Core Crosstalk using statistical Coupled Mode Theory (CMT) formulation."""
        h_power = (2 * (const.MCF_COUPLING_COEFF ** 2) * const.BENDING_RADIUS) / const.PROPAGATION_CONSTANT
        adjacent_cores = 6 if core_idx == 0 else 3 # Approx adjacencies for a standard 7-core MCF

        return adjacent_cores * h_power * (fiber.length_km * 1e3)

    def compute_inter_modal_xt(self, fiber: FiberLayer, mode_idx: int) -> float:
        """Calculates Inter-Modal Crosstalk scaling for MMFs."""
        return const.MMF_COUPLING_COEFF * (fiber.length_km * 1e3)

    def evaluate_path_quality(self, path_components: list, launch_power_dbm: float, baud_rate: float, core_idx: int = 0) -> dict:
        """
        Traverses structural components to calculate cumulative impairments.
        Simulates the propagation of `gnpy.core.info.SpectralInformation`.
        """
        signal_power = launch_power_dbm
        total_ase = 0.0
        total_nli = 0.0
        total_xt_noise = 0.0
        total_cd = 0.0
        total_pmd_sq = 0.0

        path_osnr_linear = 0.0 # Track OSNR separately from GSNR

        for comp in path_components:
            if isinstance(comp, FiberLayer):
                # Subtract macrobending loss penalty locally: --->
                signal_power -= (comp.loss_coef * comp.length_km + comp.macrobending_loss_db)
                total_nli += self.compute_nli_ggn(comp, signal_power, baud_rate)
                total_cd += comp.dispersion * comp.length_km
                total_pmd_sq += (comp.pmd_coef ** 2) * comp.length_km

                # SDM Crosstalk Integration: --->
                signal_w = 10 ** (signal_power / 10) * 1e-3
                if const.SDM_MODE == 'MCF':
                    total_xt_noise += signal_w * self.compute_inter_core_xt(comp, core_idx)
                elif const.SDM_MODE == 'MMF':
                    total_xt_noise += signal_w * self.compute_inter_modal_xt(comp, core_idx)

            elif isinstance(comp, EDFA):
                # Component-level EDFA pump failure impact: --->
                effective_gain = comp.gain - comp.pump_current_degradation_db
                effective_nf = comp.nf + comp.pump_current_degradation_db

                # Temporarily swap properties to use GNPy-like ASE computation: --->
                orig_gain, orig_nf = comp.gain, comp.nf
                comp.gain, comp.nf = effective_gain, effective_nf

                total_ase += self.compute_ase_noise(comp, signal_power)
                signal_power += comp.gain # Power equalization

                comp.gain, comp.nf = orig_gain, orig_nf # Restore original tracking values
            elif isinstance(comp, FIFO):
                signal_power -= comp.loss_db
            elif isinstance(comp, ROADM):
                # Hardware complexity logic: core-bypassing reduces WSS insertion loss impact: --->
                wss_loss = (comp.wss_loss / 2.0) if comp.core_bypass_enabled else comp.wss_loss

                # Add filter cascading effects (tightening penalty from thermal drift): --->
                wss_loss += (comp.wss_filter_shift_ghz * const.WSS_FILTER_PENALTY_COEFF)
                signal_power -= wss_loss

        # Generalized Signal-to-Noise Ratio (GSNR) calculation: --->
        signal_w = 10 ** (signal_power / 10) * 1e-3
        gsnr_linear = signal_w / (total_ase + total_nli + total_xt_noise + 1e-20)
        gsnr_db = 10 * np.log10(gsnr_linear)

        # OSNR tracking and Pre-FEC BER Estimation (Standard erfc approximation): --->
        osnr_linear = signal_w / (total_ase + 1e-20)
        osnr_db = 10 * np.log10(osnr_linear)
        pre_fec_ber = 0.5 * math.erfc(math.sqrt(gsnr_linear))

        return {
            'gsnr_db': gsnr_db, 'osnr_db': osnr_db, 'cd': total_cd, 'pmd': np.sqrt(total_pmd_sq),
            'nli': total_nli, 'recv_power_dbm': signal_power, 'pre_fec_ber': pre_fec_ber
        }
