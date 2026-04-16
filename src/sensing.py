"""
sensing.py – Energy detection and per-NCU sensing module (Sec. 4.1).

Implements:
  • Energy test statistic T_k = (1/N_s) Σ|y_k[i]|²
  • Neyman-Pearson detection threshold λ_k (Eq. 4.1)
  • Binary decision d_k = 𝟙[T_k ≥ λ_k]
"""
from __future__ import annotations

import numpy as np
from numpy.random import Generator
from scipy.special import erfinv

import src.config as cfg


def _q_inv(p: float) -> float:
    """Inverse Q-function Q^{-1}(p) = √2 · erfinv(1 - 2p)."""
    return np.sqrt(2.0) * erfinv(1.0 - 2.0 * p)


def detection_threshold(sigma2_k: float) -> float:
    """
    Neyman-Pearson threshold under H_0 (Eq. 4.1):
      λ_k = σ_k² · (1 + √(2/N_s) · Q^{-1}(α_max))
    """
    return sigma2_k * (1.0 + np.sqrt(2.0 / cfg.N_SAMPLES) * _q_inv(cfg.ALPHA_MAX))


def energy_statistic(y_k: np.ndarray) -> float:
    """
    Energy test statistic T_k = (1/N_s) Σ_{i=1}^{N_s} |y_k[i]|²  (Sec. 4.1).
    y_k: complex received samples, shape (N_s,)
    """
    return float(np.mean(np.abs(y_k) ** 2))


def sensing_decision(T_k: float, lambda_k: float) -> int:
    """
    d_k = 𝟙[T_k ≥ λ_k]   (Sec. 4.1).
    Returns 1 (CU active detected) or 0 (CU idle).
    """
    return int(T_k >= lambda_k)


class SensingModule:
    """
    Manages per-NCU energy detection and decision-making.

    Parameters
    ----------
    K       : number of NCUs
    sigma2  : noise variance array of shape (K,)
    """

    def __init__(self, K: int, sigma2: np.ndarray):
        self.K = K
        self.sigma2 = sigma2
        self.lambda_k = np.array([detection_threshold(s2) for s2 in sigma2])
        # Track previous measurements for temporal smoothness constraint (ZKP)
        self.T_prev = np.zeros(K)

    def sense(self, y_k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute energy statistics and binary decisions for all K NCUs.

        Parameters
        ----------
        y_k : complex received signal, shape (K, N_s)

        Returns
        -------
        T  : energy statistics, shape (K,)
        d  : binary decisions,  shape (K,)
        """
        T = np.array([energy_statistic(y_k[k]) for k in range(self.K)])
        d = np.array([sensing_decision(T[k], self.lambda_k[k]) for k in range(self.K)])
        return T, d

    def update_prev(self, T: np.ndarray) -> None:
        """Update T_prev for temporal smoothness check."""
        self.T_prev = T.copy()

    def temporal_smoothness_ok(self, T: np.ndarray, delta_T_max: float | None = None) -> np.ndarray:
        """
        Check temporal smoothness: |T_k - T_k_prev| ≤ Δ_T  (ZKP constraint 5).
        If delta_T_max is None, use 5× the threshold as a default bound.
        Returns boolean array of shape (K,).
        """
        if delta_T_max is None:
            delta_T_max = 5.0 * np.mean(self.lambda_k)
        return np.abs(T - self.T_prev) <= delta_T_max
