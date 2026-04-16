"""
oracle.py – ISAC-backed Ground Truth Oracle (Sec. 8).

Implements:
  • CFAR test statistic computation Λ_ISAC (Sec. 4.2)
  • Oracle decision D_ISAC ∈ {0, 1}
  • Oracle confidence score C_ISAC ∈ [0, 1]
  • Oracle error probability bound (Theorem 8.1)
"""
from __future__ import annotations

import numpy as np
from numpy.random import Generator

import src.config as cfg
from src.channel import ISACRadarModel


class ISACOracle:
    """
    ISAC Base Station Oracle (Sec. 8).

    Computes the CFAR radar test statistic and makes the ground-truth
    CU activity decision, completely independently of NCU reports.
    """

    def __init__(self, rng: Generator):
        self.rng = rng
        self.radar = ISACRadarModel(rng)
        self._eta_cfar = self._compute_cfar_threshold()

    # ── CFAR threshold ─────────────────────────────────────────────────────────

    def _compute_cfar_threshold(self) -> float:
        """
        CA-CFAR threshold η for target false-alarm P_fa_ISAC.
        η = N_ref * (P_fa^{-1/N_ref} - 1)   [standard CFAR formula]
        """
        N = cfg.N_REF_CELLS
        pfa = cfg.P_FA_ISAC
        return N * (pfa ** (-1.0 / N) - 1.0)

    # ── Oracle decision ────────────────────────────────────────────────────────

    def observe(self, H_t: int) -> tuple[int, float, float]:
        """
        Run the ISAC radar for one slot and produce:
          D_ISAC   – binary oracle decision (int)
          C_ISAC   – confidence score ∈ [0, 1]
          Lambda   – raw CFAR statistic (float)

        Parameters
        ----------
        H_t : true CU state (0 or 1) — used only by the channel model
        """
        Lambda = self.radar.compute_isac_statistic(H_t)
        D_ISAC = int(Lambda >= self._eta_cfar)

        # Confidence score (Sec. 8.1): normalised relative to CFAR threshold
        raw_conf = (Lambda - self._eta_cfar) / max(self._eta_cfar, 1e-12)
        C_ISAC = float(1.0 / (1.0 + np.exp(-raw_conf)))   # sigmoid normalisation → [0,1]

        return D_ISAC, C_ISAC, float(Lambda)

    # ── Reliability bound ──────────────────────────────────────────────────────

    @staticmethod
    def oracle_error_bound(snr_r_db: float | None = None) -> float:
        """
        Theorem 8.1: P_e,ISAC ≤ exp(−N_t·L·SNR_r² / (2(1+SNR_r)²))

        Returns the upper bound on oracle error probability.
        """
        if snr_r_db is None:
            radar_tmp = ISACRadarModel.__new__(ISACRadarModel)
            radar_tmp.rng = None
            snr_r_db = ISACRadarModel(np.random.default_rng(0)).radar_snr_db()
        snr_r = 10.0 ** (snr_r_db / 10.0)
        exponent = (
            -cfg.N_ANTENNAS
            * cfg.N_RADAR_PULSES
            * snr_r ** 2
            / (2.0 * (1.0 + snr_r) ** 2)
        )
        return float(np.exp(exponent))

    # ── ROC curve for oracle ───────────────────────────────────────────────────

    def roc_curve(self, n_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """
        Analytical ROC for the ISAC CFAR detector (Sec. 4.2):
          P_d = P_fa^{1/(1+SNR_r·L)}

        Returns (P_fa_array, P_d_array).
        """
        snr_r = self.radar.snr_r_linear
        L = cfg.N_RADAR_PULSES
        pfa = np.linspace(0.0, 1.0, n_points)
        pd = pfa ** (1.0 / (1.0 + snr_r * L))
        return pfa, pd
