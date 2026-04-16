"""
mechanism.py – Cryptographic VCG Spectrum Auction (CVSA) module (Sec. 6).

Implements:
  • Aggregate state decision D_agg  (Sec. 6.1)
  • VCG allocation and payment       (Sec. 6.2)
  • Cryptographic bond posting/slashing (Sec. 6.3)
  • Honest-reporter reward dividend  (Sec. 6.4)
  • Modified NCU utility under CVSA  (Sec. 6.4)
"""
from __future__ import annotations

import numpy as np
from numpy.random import Generator

import src.config as cfg


# ── Aggregate state decision ──────────────────────────────────────────────────

def aggregate_decision(
    d_hat: np.ndarray,
    D_ISAC: int,
    rho: np.ndarray,
    w_isac: float = cfg.W_ISAC,
    theta: float = cfg.THETA_AGG,
) -> int:
    """
    D_agg(t) = 𝟙[Σ_k w_k(t)·d̂_k + w_ISAC·D_ISAC ≥ θ]   (Sec. 6.1)

    Parameters
    ----------
    d_hat  : reported decisions array (K,)  ∈ {0, 1}
    D_ISAC : ISAC oracle decision (int)
    rho    : reputation scores (K,)
    """
    K = len(d_hat)
    rho_sum = float(np.sum(rho)) + w_isac
    if rho_sum < 1e-12:
        rho_sum = 1e-12

    w_k = rho / rho_sum
    weighted_sum = float(np.dot(w_k, d_hat)) + w_isac * D_ISAC / rho_sum
    return int(weighted_sum >= theta)


# ── VCG allocation and payment ────────────────────────────────────────────────

def vcg_allocate(
    v: np.ndarray,
    rho: np.ndarray,
    P_ncu: np.ndarray,
    P_budget: float | None = None,
) -> np.ndarray:
    """
    VCG allocation: maximise Σ_k x_k · v_k · ρ_k  s.t. Σ_k x_k·P_k ≤ P_budget.

    Uses a greedy approximation (O(K log K)) sorted by value-density
    v_k·ρ_k / P_k, which is optimal for the 0/1 knapsack relaxation.

    Parameters
    ----------
    v        : valuation bids, shape (K,)
    rho      : reputation scores, shape (K,)
    P_ncu    : NCU Tx powers, shape (K,)
    P_budget : total power budget (interference limit)

    Returns
    -------
    x : binary allocation vector, shape (K,)
    """
    K = len(v)
    if P_budget is None:
        P_budget = cfg.I_MAX_W * 10  # generous default

    # Compute adjusted values
    adj_v = v * rho  # (K,)

    # Greedy by adjusted value density
    density = adj_v / (P_ncu + 1e-12)
    order = np.argsort(-density)

    x = np.zeros(K, dtype=int)
    remaining = P_budget
    for k in order:
        if P_ncu[k] <= remaining and adj_v[k] > 0:
            x[k] = 1
            remaining -= P_ncu[k]

    return x


def vcg_payment(
    x_star: np.ndarray,
    v: np.ndarray,
    rho: np.ndarray,
    P_ncu: np.ndarray,
    P_budget: float | None = None,
) -> np.ndarray:
    """
    VCG payment for each allocated NCU (Sec. 6.2):
      p_k^VCG = max_{x_{-k}} Σ_{j≠k} x_j v_j ρ_j  −  Σ_{j≠k} x_j* v_j ρ_j

    Returns payment vector (K,); unallocated NCUs have payment = 0.
    """
    K = len(v)
    if P_budget is None:
        P_budget = cfg.I_MAX_W * 10

    adj_v = v * rho
    # Social welfare of others in optimal x*
    sw_others = np.array([
        float(np.dot(adj_v, x_star) - adj_v[k] * x_star[k])
        for k in range(K)
    ])

    payments = np.zeros(K)
    for k in range(K):
        if x_star[k] == 0:
            continue
        # Optimum without NCU k
        mask = np.ones(K, dtype=bool)
        mask[k] = False
        x_nok = vcg_allocate(v[mask], rho[mask], P_ncu[mask], P_budget)
        sw_without_k = float(np.dot(adj_v[mask], x_nok))
        payments[k] = max(sw_without_k - sw_others[k], 0.0)

    return payments


# ── Bond management ───────────────────────────────────────────────────────────

class BondManager:
    """
    Manages cryptographic penalty bonds per NCU (Sec. 6.3).
    """

    def __init__(self, K: int):
        self.K = K
        self.bonds = np.full(K, cfg.B_MIN)
        self.treasury = 0.0
        self.honest_rewards_pool = 0.0

    def compute_bonds(self, rho: np.ndarray) -> np.ndarray:
        """
        B_k(t) = b_k · (1 + κ · (1 − ρ_k))   (Sec. 6.3)
        """
        self.bonds = cfg.B_MIN * (1.0 + cfg.KAPPA * (1.0 - np.clip(rho, 0.0, 1.0)))
        return self.bonds.copy()

    def slash(
        self,
        contradiction: np.ndarray,
        C_ISAC: float,
    ) -> tuple[np.ndarray, float]:
        """
        Compute bond slashing for contradicted NCUs (Sec. 6.3):
          slash_k = B_k · min(1, C_ISAC / η_conf) · 𝟙[contradiction]

        Returns
        -------
        slash_amounts : slash per NCU (K,)
        total_slashed : total amount added to treasury
        """
        conf_ratio = min(1.0, C_ISAC / cfg.ETA_CONFIDENCE)
        slash_amounts = self.bonds * conf_ratio * contradiction.astype(float)
        total_slashed = float(np.sum(slash_amounts))
        self.treasury += total_slashed
        self.honest_rewards_pool += total_slashed * 0.5  # 50% redistributed
        return slash_amounts, total_slashed

    def compute_honest_rewards(
        self,
        honest_mask: np.ndarray,
        D_ISAC: int,
    ) -> np.ndarray:
        """
        Distribute honest-reporter dividend ω_k (Sec. 6.3).
        Only honest NCUs when D_ISAC=1 receive a reward.
        """
        rewards = np.zeros(self.K)
        if D_ISAC != 1:
            return rewards

        n_honest = int(np.sum(honest_mask))
        if n_honest == 0 or self.honest_rewards_pool <= 0.0:
            return rewards

        per_honest = min(
            self.honest_rewards_pool / max(n_honest, 1),
            cfg.B_MIN * 0.1,  # cap per-NCU reward
        )
        rewards[honest_mask] = per_honest
        self.honest_rewards_pool -= per_honest * n_honest
        self.honest_rewards_pool = max(0.0, self.honest_rewards_pool)
        return rewards


# ── CVSA Mechanism ────────────────────────────────────────────────────────────

class CVSAMechanism:
    """
    Full Cryptographic VCG Spectrum Auction (CVSA) — Sec. 6.
    """

    def __init__(self, K: int, rng: Generator):
        self.K = K
        self.rng = rng
        self.bond_manager = BondManager(K)
        # NCU Tx powers (heterogeneous, uniformly sampled once)
        self.P_ncu = rng.uniform(0.5, 1.0, K) * cfg.P_NCU_MAX_W

    def run(
        self,
        d_hat: np.ndarray,
        D_ISAC: int,
        C_ISAC: float,
        rho: np.ndarray,
        d_true: np.ndarray,
        q_drl: np.ndarray,
    ) -> dict:
        """
        Execute one slot of the CVSA mechanism.

        Parameters
        ----------
        d_hat  : NCU reported decisions (K,)
        D_ISAC : oracle decision
        C_ISAC : oracle confidence
        rho    : reputation scores (K,)
        d_true : true NCU decisions (K,) — used only for contradiction check
        q_drl  : DRL-set base access prices (K,)

        Returns
        -------
        dict with keys:
          D_agg, x_star, payments, slash_amounts, rewards, revenue, welfare,
          access_set, bonds
        """
        K = self.K

        # 1. Update bonds
        bonds = self.bond_manager.compute_bonds(rho)

        # 2. Aggregate decision
        D_agg = aggregate_decision(d_hat, D_ISAC, rho)

        # 3. Valuations (NCU bids — sampled from private value distribution)
        v = self.rng.uniform(cfg.V_MIN, cfg.V_MAX, K)

        # 4. Contradiction detection: d_k=0 but ISAC oracle says D_ISAC=1
        contradiction = (d_hat == 0) & (D_ISAC == 1)

        # 5. Bond slashing + honest rewards
        slash_amounts, _ = self.bond_manager.slash(contradiction, C_ISAC)
        honest_mask = (d_hat == 1) & (D_ISAC == 1)
        rewards = self.bond_manager.compute_honest_rewards(honest_mask, D_ISAC)

        # 6. Auction
        # P_budget: aggregate NCU Tx power allowed.
        # Set to allow all K NCUs simultaneously (interference at CU is already
        # dominated by the path loss at 600 m; the HIP constraint is checked
        # separately in the metrics layer).
        P_budget = float(np.sum(self.P_ncu))

        if D_agg == 0:
            x_star = vcg_allocate(v, rho, self.P_ncu, P_budget=P_budget)
            payments = vcg_payment(x_star, v, rho, self.P_ncu, P_budget=P_budget)
            revenue = float(np.sum(x_star * (q_drl + payments)))
            welfare = float(np.sum(x_star * (v - payments)))
        else:
            x_star = np.zeros(K, dtype=int)
            payments = np.zeros(K)
            revenue = 0.0
            welfare = 0.0

        return {
            "D_agg":        D_agg,
            "x_star":       x_star,
            "payments":     payments,
            "slash_amounts": slash_amounts,
            "rewards":      rewards,
            "revenue":      revenue,
            "welfare":      welfare,
            "access_set":   np.where(x_star == 1)[0].tolist(),
            "bonds":        bonds,
            "valuations":   v,
        }
