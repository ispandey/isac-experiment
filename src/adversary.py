"""
adversary.py – Adversarial NCU behaviour models (Sec. 3.1).

Implements:
  • Type-I:   Selfish Falsifier (SF)        – always reports d̂=0
  • Type-II:  Strategic Threshold Falsifier (STF) – threshold-based
  • Type-III: Collusive Coalition Falsifier (CCF) – coordinated d̂=0
"""
from __future__ import annotations

from enum import Enum, auto

import numpy as np

import src.config as cfg


class AdversaryType(Enum):
    HONEST = auto()
    SF     = auto()   # Type-I
    STF    = auto()   # Type-II
    CCF    = auto()   # Type-III


class AdversaryModel:
    """
    Manages adversary role assignments and overrides NCU sensing decisions.
    """

    def __init__(self, K: int, adversary_fraction: float, adversary_type: AdversaryType, rng):
        """
        Parameters
        ----------
        K                  : total NCU count
        adversary_fraction : fraction of NCUs that are adversarial ∈ [0, 1)
        adversary_type     : which adversary class to assign
        rng                : numpy RNG
        """
        self.K = K
        self.rng = rng
        self.adversary_type = adversary_type

        n_adv = max(0, min(int(adversary_fraction * K), K - 1))
        adv_indices = rng.choice(K, size=n_adv, replace=False)
        self.is_adversary = np.zeros(K, dtype=bool)
        self.is_adversary[adv_indices] = True

        # STF: each adversary picks a private risk threshold γ_k
        self.stf_gamma = rng.uniform(0.3, 0.7, K)

    @property
    def n_adversaries(self) -> int:
        return int(np.sum(self.is_adversary))

    def apply(
        self,
        d_true: np.ndarray,
        T_k: np.ndarray,
        lambda_k: np.ndarray,
        rho: np.ndarray,
        q_drl: np.ndarray,
    ) -> np.ndarray:
        """
        Override honest decisions for adversarial NCUs.

        Parameters
        ----------
        d_true  : honest decisions from energy detection (K,)
        T_k     : energy statistics (K,)
        lambda_k: detection thresholds (K,)
        rho     : current reputation scores (K,)
        q_drl   : DRL base access prices (K,)

        Returns
        -------
        d_hat : reported decisions (K,) with adversary overrides
        """
        d_hat = d_true.copy()

        for k in range(self.K):
            if not self.is_adversary[k]:
                continue

            if self.adversary_type == AdversaryType.SF:
                # Type-I: always report idle (Sec. 3.1)
                d_hat[k] = 0

            elif self.adversary_type == AdversaryType.STF:
                # Type-II: compare perceived risk vs gain (Sec. 3.1)
                # Simplified: falsify if T_k / lambda_k < γ_k
                #   (low sensing SNR → risk of detection is low)
                ratio = T_k[k] / (lambda_k[k] + 1e-12)
                if ratio < self.stf_gamma[k] or (rho[k] > 0.3 and q_drl[k] > cfg.B_MIN * 0.5):
                    d_hat[k] = 0
                # Otherwise report truthfully

            elif self.adversary_type == AdversaryType.CCF:
                # Type-III: coordinate – all coalition members report idle
                d_hat[k] = 0

        return d_hat
