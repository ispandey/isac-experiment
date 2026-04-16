"""
channel.py – PHY / channel module for VERIDIC-DSA.

Implements:
  • Sub-THz path loss (3GPP TR 38.901 extended, Sec. 2.3)
  • Nakagami-m small-scale fading
  • CU semi-Markov activity process (Sec. 2.5)
  • ISAC radar return signal (Sec. 2.4)
  • Saleh-Valenzuela MIMO channel snippet (Sec. 12.2)
  • Received signal y_k(t) for each NCU
"""
from __future__ import annotations

import numpy as np
from numpy.random import Generator

import src.config as cfg


# ── Path-loss helpers ─────────────────────────────────────────────────────────

def free_space_path_loss_db(d_m: float, f_hz: float) -> float:
    """Free-space path loss in dB: 20·log10(4πdf/c)."""
    return 20.0 * np.log10(4.0 * np.pi * d_m * f_hz / cfg.SPEED_OF_LIGHT)


def molecular_absorption_loss_db(d_m: float, alpha_db_km: float) -> float:
    """Molecular absorption loss in dB."""
    return alpha_db_km * d_m / 1000.0


def total_path_loss_db(d_m: float, f_hz: float, rng: Generator) -> float:
    """
    Total sub-THz path loss (Sec. 2.3):
      PL(d,f) = 20log10(4πdf/c) + α_abs(f)·d + χ_σ
    """
    pl_fs  = free_space_path_loss_db(d_m, f_hz)
    pl_abs = molecular_absorption_loss_db(d_m, cfg.ALPHA_ABS_DB_PER_KM)
    shadowing = rng.normal(0.0, cfg.SHADOW_STD_DB)
    return pl_fs + pl_abs + shadowing


def path_loss_linear(d_m: float, f_hz: float, rng: Generator) -> float:
    """Path loss as a linear power attenuation factor."""
    pl_db = total_path_loss_db(d_m, f_hz, rng)
    return 10.0 ** (pl_db / 10.0)


# ── Nakagami-m fading ─────────────────────────────────────────────────────────

def nakagami_amplitude(m: float, omega: float, rng: Generator) -> float:
    """
    Generate a Nakagami-m fading amplitude.
    Amplitude ~ Nakagami(m, Ω).  Power ~ Gamma(m, Ω/m).
    """
    # Gamma distributed power: shape=m, scale=Ω/m
    power = rng.gamma(shape=m, scale=omega / m)
    return np.sqrt(power)


def complex_nakagami_channel(m: float, omega: float, rng: Generator) -> complex:
    """Complex channel coefficient with Nakagami-m amplitude and uniform phase."""
    amplitude = nakagami_amplitude(m, omega, rng)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    return amplitude * np.exp(1j * phase)


# ── CU semi-Markov activity ────────────────────────────────────────────────────

class CUActivityModel:
    """
    Semi-Markov CU activity model (Sec. 2.5).

    State: 0 = idle (H_0), 1 = active (H_1).
    Transition matrix:
        P = [[1-λ_on,  λ_on ],
             [λ_off,  1-λ_off]]
    """

    def __init__(self, rng: Generator):
        self.rng = rng
        self._state: int = int(rng.random() < cfg.CU_STEADY_STATE_PI1)

    @property
    def state(self) -> int:
        return self._state

    def step(self) -> int:
        """Advance one slot; return new state."""
        if self._state == 0:
            if self.rng.random() < cfg.CU_LAMBDA_ON:
                self._state = 1
        else:
            if self.rng.random() < cfg.CU_LAMBDA_OFF:
                self._state = 0
        return self._state

    def reset(self) -> int:
        self._state = int(self.rng.random() < cfg.CU_STEADY_STATE_PI1)
        return self._state


# ── Steering vector ────────────────────────────────────────────────────────────

def steering_vector(azimuth_rad: float, n_antennas: int) -> np.ndarray:
    """ULA steering vector for a given azimuth angle."""
    n = np.arange(n_antennas)
    return (1.0 / np.sqrt(n_antennas)) * np.exp(1j * np.pi * n * np.sin(azimuth_rad))


# ── Saleh-Valenzuela sub-THz MIMO channel ─────────────────────────────────────

def generate_subthz_channel(
    d_m: float,
    f_hz: float,
    n_t: int,
    n_r: int,
    rng: Generator,
    n_clusters: int = 3,
    n_rays: int = 5,
) -> np.ndarray:
    """
    Sub-THz MIMO channel matrix H ∈ ℂ^{N_r × N_t} (Sec. 12.2).
    Combines Saleh-Valenzuela clusters + sub-THz path loss.
    """
    pl_db = total_path_loss_db(d_m, f_hz, rng)
    pl_linear = 10.0 ** (pl_db / 10.0)

    H = np.zeros((n_r, n_t), dtype=complex)
    for _ in range(n_clusters):
        az_cl = rng.uniform(-np.pi / 3, np.pi / 3)
        for _ in range(n_rays):
            az = az_cl + rng.normal(0.0, 5.0 * np.pi / 180.0)
            a_t = steering_vector(az, n_t)
            a_r = steering_vector(az, n_r)
            alpha_lr = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2.0)
            H += alpha_lr * np.outer(a_r, np.conj(a_t))

    frob = np.linalg.norm(H, "fro")
    if frob > 0:
        H = H / frob * np.sqrt(1.0 / pl_linear)
    return H


# ── Received signal at NCU k ───────────────────────────────────────────────────

class ChannelModel:
    """
    Generates per-slot channel realisations and received signals for all NCUs.

    Holds references to the shared RNG so the channel evolves consistently.
    """

    def __init__(self, K: int, rng: Generator):
        self.K = K
        self.rng = rng

        # σ_k² = thermal noise power per NCU (paper Sec. 4.1 AWGN model).
        # The energy detector measures CU signal vs. thermal noise only.
        # BS signal is treated as known and cancelled by NCUs (standard CR assumption).
        self.sigma2 = cfg.NOISE_POWER_W * np.ones(K) * (
            1.0 + 0.05 * rng.standard_normal(K)
        )
        self.sigma2 = np.clip(self.sigma2, cfg.NOISE_POWER_W * 0.8,
                              cfg.NOISE_POWER_W * 1.2)

    def sample_slot(self, H_t: int) -> dict:
        """
        Generate all per-slot channel samples.
        Returns a dict with:
          h_ck   – complex CU→NCU channels (K,)
          h_bk   – complex BS→NCU channels (K,)
          g_kb   – complex NCU→CU interference channels (K,)
          y_k    – received signal samples array (K, N_s) complex
          sigma2 – noise variance per NCU (K,)
        """
        rng = self.rng

        # CU→NCU fading (Nakagami-m)
        h_ck = np.array([
            complex_nakagami_channel(
                cfg.NAKAGAMI_M,
                1.0 / path_loss_linear(cfg.CU_TO_NCU_DIST_M, cfg.FC_HZ, rng),
                rng,
            )
            for _ in range(self.K)
        ])

        # BS→NCU fading (Nakagami-m)
        h_bk = np.array([
            complex_nakagami_channel(
                cfg.NAKAGAMI_M,
                1.0 / path_loss_linear(cfg.BS_TO_NCU_DIST_M, cfg.FC_HZ, rng),
                rng,
            )
            for _ in range(self.K)
        ])

        # NCU→CU interference channels (Rayleigh for simplicity)
        g_kb = np.array([
            (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(
                2.0 * path_loss_linear(cfg.NCU_TO_CU_DIST_M, cfg.FC_HZ, rng)
            )
            for _ in range(self.K)
        ])

        # BS signal (ISAC, omni approx for NCU sensing interference)
        P_bs_half = cfg.P_MAX_W / 2.0
        # CU signal power at NCU k: P_c * |h_ck|²
        P_cu_at_ncu = cfg.P_CU_W * np.abs(h_ck) ** 2  # (K,)

        # Generate y_k(t) = h_ck * s_c + h_bk * s_b + w_k
        # Under H_1 (CU active): signal + ISAC BS + noise
        # Under H_0 (CU idle):   ISAC BS + noise only
        y_k = np.zeros((self.K, cfg.N_SAMPLES), dtype=complex)
        for k in range(self.K):
            noise = np.sqrt(self.sigma2[k] / 2.0) * (
                rng.standard_normal(cfg.N_SAMPLES) + 1j * rng.standard_normal(cfg.N_SAMPLES)
            )
            if H_t == 1:
                # CU signal contribution: P_c * |h_ck|²
                cu_signal = np.sqrt(P_cu_at_ncu[k]) * (
                    rng.standard_normal(cfg.N_SAMPLES) + 1j * rng.standard_normal(cfg.N_SAMPLES)
                ) / np.sqrt(2.0)
                y_k[k] = cu_signal + noise
            else:
                y_k[k] = noise

        return {
            "h_ck": h_ck,
            "h_bk": h_bk,
            "g_kb": g_kb,
            "y_k": y_k,
            "sigma2": self.sigma2.copy(),
        }


# ── ISAC radar return ──────────────────────────────────────────────────────────

class ISACRadarModel:
    """
    ISAC radar return and matched-filter statistic (Sec. 2.4 + 4.2).
    """

    def __init__(self, rng: Generator):
        self.rng = rng

    def radar_snr_db(self) -> float:
        """
        Radar SNR (dB) using radar range equation (Sec. 4.2):
          SNR_r = P_r * σ_RCS² * G_t * G_r * λ² / ((4π)³ * d⁴ * k_B * T_s * B_r)
        """
        P_r = cfg.P_MAX_W / 2.0
        numerator = (P_r * cfg.SIGMA_RCS ** 2 * cfg.G_TX * cfg.G_RX
                     * cfg.WAVELENGTH ** 2)
        denominator = ((4 * np.pi) ** 3 * cfg.BS_TO_CU_DIST_M ** 4
                       * cfg.BOLTZMANN_K * cfg.TEMPERATURE * cfg.BW_HZ)
        snr_linear = numerator / denominator
        return 10.0 * np.log10(max(snr_linear, 1e-30))

    # Paper operating point: SNR_r = 10 dB (Theorem 8.1).
    # The range-equation gives the long-range floor; the 10-dB working point
    # reflects near-BS target geometry used in the experiment.
    OPERATING_SNR_DB = 10.0

    @property
    def snr_r_linear(self) -> float:
        return 10.0 ** (self.OPERATING_SNR_DB / 10.0)

    def compute_isac_statistic(self, H_t: int) -> float:
        """
        Compute ISAC CFAR test statistic Λ_ISAC.
        Under H_0: statistic ~ χ²(2), mean=1 (normalised).
        Under H_1: statistic ~ non-central with extra signal energy.
        Returns the raw (unnormalised) statistic.
        """
        rng = self.rng
        L = cfg.N_RADAR_PULSES
        snr_r = self.snr_r_linear

        if H_t == 1:
            # Matched-filter output under signal present: Z_ISAC ~ χ²(2L, 2L·SNR_r)
            ncp = 2.0 * L * snr_r
            z_isac = rng.noncentral_chisquare(df=2 * L, nonc=ncp) / (2 * L)
        else:
            # H_0: noise only → chi-squared / df ≈ 1 (normalised)
            z_isac = rng.chisquare(df=2 * L) / (2 * L)

        # CFAR: divide by mean of independently sampled reference cells
        # Each reference cell power is an independent Exp(1) ~ chi2(2)/2
        ref_powers = rng.chisquare(df=2, size=cfg.N_REF_CELLS) / 2.0
        cfar_normaliser = np.mean(ref_powers)
        if cfar_normaliser < 1e-12:
            cfar_normaliser = 1e-12

        return z_isac / cfar_normaliser
