"""
baselines.py – Baseline spectrum access systems (Sec. 13.2).

BL1: Naive Cooperative – standard energy detection + majority voting, no mechanism
BL2: OR-rule            – access if any NCU reports idle
BL3: AND-rule           – access only if all NCUs report idle
BL4: Weighted Reputation – reputation-based weighting, no crypto
BL5: Auction-only        – VCG auction without ZKP or bonds
BL6: Crypto-only         – ZKP + commitment without DRL pricing
BL7: Static Pricing      – VCG + ZKP + fixed prices (no PPO)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np
from numpy.random import Generator

import src.config as cfg
from src.adversary import AdversaryModel, AdversaryType
from src.channel import ChannelModel, CUActivityModel
from src.crypto import commit, prove, verify_proof
from src.mechanism import aggregate_decision, vcg_allocate, vcg_payment
from src.metrics import MetricStore, SlotRecord
from src.oracle import ISACOracle
from src.reputation import ReputationModel
from src.sensing import SensingModule


class BaselineType(Enum):
    BL1 = auto()  # Naive Cooperative
    BL2 = auto()  # OR-rule
    BL3 = auto()  # AND-rule
    BL4 = auto()  # Weighted Reputation
    BL5 = auto()  # Auction-only
    BL6 = auto()  # Crypto-only
    BL7 = auto()  # Static Pricing


class BaselineSystem:
    """
    Unified baseline simulation engine.
    Runs one of BL1-BL7 for comparison against VERIDIC-DSA.
    """

    def __init__(
        self,
        baseline: BaselineType,
        K: int = 10,
        adversary_fraction: float = 0.0,
        adversary_type: AdversaryType = AdversaryType.SF,
        seed: int = cfg.SEED,
    ):
        self.baseline = baseline
        self.K = K
        self.seed = seed
        self.rng: Generator = np.random.default_rng(seed)

        # Shared subsystems
        self.cu_model       = CUActivityModel(self.rng)
        self.channel_model  = ChannelModel(K, self.rng)
        self.sensing_module = SensingModule(K, self.channel_model.sigma2)
        self.oracle         = ISACOracle(self.rng)
        self.reputation     = ReputationModel(K)
        self.adversary      = AdversaryModel(K, adversary_fraction, adversary_type, self.rng)
        self.metrics        = MetricStore()
        self._P_ncu         = self.rng.uniform(0.5, 1.0, K) * cfg.P_NCU_MAX_W

        # Static price for BL6 / BL7
        self._static_price = (cfg.Q_MIN + cfg.Q_MAX) / 2.0
        self._slot_count = 0

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.cu_model.reset()
        self.reputation.reset()
        self.metrics.clear()
        self._slot_count = 0

    # ── Access rule helpers ───────────────────────────────────────────────────

    def _access_decision(self, d_hat: np.ndarray, D_ISAC: int, rho: np.ndarray) -> int:
        """
        Return 0 = grant access (spectrum idle), 1 = deny.
        Maps baseline rules to a single flag.
        """
        bl = self.baseline

        if bl == BaselineType.BL1:
            # Majority vote: access if majority report idle
            return int(np.mean(d_hat) >= 0.5)

        elif bl == BaselineType.BL2:
            # OR-rule: access if ANY NCU reports idle
            return int(np.any(d_hat == 0))

        elif bl == BaselineType.BL3:
            # AND-rule: access only if ALL NCUs report idle
            return int(np.all(d_hat == 0))

        elif bl == BaselineType.BL4:
            # Weighted reputation (no oracle, no crypto)
            w = rho / (rho.sum() + 1e-12)
            weighted = float(np.dot(w, 1 - d_hat))  # probability of idle
            return int(weighted >= 0.5)

        elif bl in (BaselineType.BL5, BaselineType.BL6, BaselineType.BL7):
            # All use CVSA aggregate decision (without ZKP enforcement for BL5)
            return 1 - aggregate_decision(d_hat, D_ISAC, rho)

        return 0

    def run_slot(self, slot: int) -> SlotRecord:
        """Execute one baseline slot."""
        K   = self.K
        rng = self.rng
        H_t = self.cu_model.step()
        ch  = self.channel_model.sample_slot(H_t)
        g_kb   = ch["g_kb"]
        sigma2 = ch["sigma2"]

        T_k, d_true = self.sensing_module.sense(ch["y_k"])
        lambda_k    = self.sensing_module.lambda_k
        T_prev      = self.sensing_module.T_prev.copy()

        # Adversary override
        q_static = np.full(K, self._static_price)
        d_hat = self.adversary.apply(d_true, T_k, lambda_k, self.reputation.rho, q_static)

        # ZKP (BL5 skips ZKP; BL1-BL4 have no crypto at all)
        zkp_valid = np.ones(K, dtype=bool)
        d_verified = d_hat.copy()
        prove_times = np.full(K, 0.0)

        if self.baseline in (BaselineType.BL6, BaselineType.BL7):
            for k in range(K):
                com_k, r_k = commit(d_hat[k], rng)
                is_adv = bool(self.adversary.is_adversary[k]) and d_hat[k] != d_true[k]
                zkp = prove(T_k[k], d_hat[k], r_k, com_k, lambda_k[k], T_prev[k],
                            adversarial=is_adv, rng=rng)
                ok = verify_proof(zkp, com_k, lambda_k[k])
                if not ok:
                    d_verified[k] = 0
                zkp_valid[k] = ok
                prove_times[k] = zkp.prove_time_ms

        # Oracle
        D_ISAC, C_ISAC, _ = self.oracle.observe(H_t)

        # Access decision
        rho_now = self.reputation.rho
        grant_access = self._access_decision(d_verified, D_ISAC, rho_now)
        D_agg = 1 - grant_access

        # Reputation update (BL4, BL5, BL6, BL7)
        contradiction = (d_verified == 0) & (D_ISAC == 1)
        falsification_detected = contradiction.copy()

        if self.baseline in (BaselineType.BL4, BaselineType.BL5,
                              BaselineType.BL6, BaselineType.BL7):
            self.reputation.update(contradiction, C_ISAC)

        self.sensing_module.update_prev(T_k)

        # Allocation (BL1-BL3: all or none)
        x_star   = np.zeros(K, dtype=int)
        payments = np.zeros(K)
        revenue  = 0.0
        welfare  = 0.0
        q_used   = np.full(K, self._static_price)

        if grant_access:
            if self.baseline in (BaselineType.BL1, BaselineType.BL2, BaselineType.BL3):
                # Grant all NCUs
                x_star[:] = 1
                revenue = float(np.sum(q_used))
                v = rng.uniform(cfg.V_MIN, cfg.V_MAX, K)
                welfare = float(np.sum(v - q_used))

            elif self.baseline in (BaselineType.BL4, BaselineType.BL5,
                                   BaselineType.BL6, BaselineType.BL7):
                v = rng.uniform(cfg.V_MIN, cfg.V_MAX, K)
                P_budget = float(np.sum(self._P_ncu))
                x_star  = vcg_allocate(v, rho_now, self._P_ncu, P_budget=P_budget)
                payments = vcg_payment(x_star, v, rho_now, self._P_ncu, P_budget=P_budget)
                revenue  = float(np.sum(x_star * (q_used + payments)))
                welfare  = float(np.sum(x_star * (v - payments)))

        interference_w = cfg.P_NCU_MAX_W * np.abs(g_kb) ** 2

        rec = SlotRecord(
            slot=slot,
            H_t=H_t,
            D_ISAC=D_ISAC,
            C_ISAC=C_ISAC,
            d_true=d_true.copy(),
            d_hat=d_verified.copy(),
            zkp_valid=zkp_valid.copy(),
            D_agg=D_agg,
            x_star=x_star.copy(),
            payments=payments.copy(),
            slash_amounts=np.zeros(K),
            rewards=np.zeros(K),
            q_drl=q_used.copy(),
            revenue=revenue,
            welfare=welfare,
            interference_w=interference_w.copy(),
            rho=rho_now.copy(),
            prove_times_ms=prove_times.copy(),
            falsification_detected=falsification_detected.copy(),
        )
        self.metrics.add(rec)
        self._slot_count += 1
        return rec

    def run(
        self,
        n_slots: int = cfg.N_EPISODES * cfg.EPISODE_LENGTH,
        verbose: bool = False,
        log_every: int = 5000,
    ) -> dict:
        """Run for n_slots total and return metric summary."""
        for slot in range(n_slots):
            self.run_slot(slot)
            if verbose and (slot + 1) % log_every == 0:
                print(f"  BL {self.baseline.name} slot {slot+1}/{n_slots}")
        return self.metrics.summary()
