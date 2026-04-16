"""
metrics.py – Performance metric computation (Sec. 13).

Implements:
  M1. DAF  – Detection Accuracy against Falsification
  M2. HIP  – Harmful Interference Probability
  M3. Rev  – Network Revenue
  M4. SW   – Social Welfare
  M5. SE   – Spectrum Efficiency
  M6. f_honest – Honest NCU fraction over time
  M7. ZKP overhead summary
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

import src.config as cfg


# ── Metric accumulator ────────────────────────────────────────────────────────

@dataclass
class SlotRecord:
    """All per-slot observable quantities needed for metric computation."""
    slot:            int
    H_t:             int                   # true CU state
    D_ISAC:          int                   # oracle decision
    C_ISAC:          float
    d_true:          np.ndarray            # (K,) true sensing decisions
    d_hat:           np.ndarray            # (K,) reported decisions
    zkp_valid:       np.ndarray            # (K,) bool
    D_agg:           int
    x_star:          np.ndarray            # (K,) allocation
    payments:        np.ndarray            # (K,)
    slash_amounts:   np.ndarray            # (K,)
    rewards:         np.ndarray            # (K,)
    q_drl:           np.ndarray            # (K,) prices
    revenue:         float
    welfare:         float
    interference_w:  np.ndarray            # (K,) interference power at CU
    rho:             np.ndarray            # (K,) reputation scores
    prove_times_ms:  np.ndarray            # (K,)
    falsification_detected: np.ndarray     # (K,) bool


@dataclass
class MetricStore:
    """Accumulates per-slot records and computes all metrics."""
    records: List[SlotRecord] = field(default_factory=list)

    def add(self, rec: SlotRecord) -> None:
        self.records.append(rec)

    def clear(self) -> None:
        self.records.clear()

    # ── M1: DAF ───────────────────────────────────────────────────────────────

    def daf(self) -> float:
        """
        Detection Accuracy against Falsification:
          DAF = (TP + TN) / (TP + TN + FP + FN)
        where events are NCU-slot pairs.
        """
        tp = tn = fp = fn = 0
        for rec in self.records:
            for k in range(len(rec.d_hat)):
                is_fals = int(rec.d_hat[k] != rec.d_true[k])
                detected = int(rec.falsification_detected[k])
                if is_fals and detected:
                    tp += 1
                elif not is_fals and not detected:
                    tn += 1
                elif not is_fals and detected:
                    fp += 1
                else:
                    fn += 1
        total = tp + tn + fp + fn
        return (tp + tn) / max(total, 1)

    # ── M2: HIP ───────────────────────────────────────────────────────────────

    def hip(self) -> float:
        """
        Harmful Interference Probability:
          HIP = Pr[Σ_{k∈A_t} P_k |g_kb|² > I_max | H_t=1]
        """
        active_cu_slots = [r for r in self.records if r.H_t == 1]
        if not active_cu_slots:
            return 0.0
        over_limit = sum(
            1 for r in active_cu_slots
            if np.sum(r.interference_w * r.x_star) > cfg.I_MAX_W
        )
        return over_limit / len(active_cu_slots)

    # ── M3: Revenue ───────────────────────────────────────────────────────────

    def total_revenue(self) -> float:
        return sum(r.revenue for r in self.records)

    def revenue_series(self) -> np.ndarray:
        return np.array([r.revenue for r in self.records])

    # ── M4: Social Welfare ────────────────────────────────────────────────────

    def total_welfare(self) -> float:
        return sum(r.welfare for r in self.records)

    # ── M5: Spectrum Efficiency ───────────────────────────────────────────────

    def spectrum_efficiency(self) -> float:
        """
        SE = useful throughput / bandwidth  (bits/s/Hz)

        Throughput per slot for allocated NCUs estimated via Shannon formula.
        """
        total_bits = 0.0
        for r in self.records:
            for k in np.where(r.x_star == 1)[0]:
                # SNR estimate: P_ncu / sigma2_k (rough)
                snr_k = cfg.P_NCU_MAX_W / cfg.NOISE_POWER_W
                cap_k = cfg.BW_HZ * np.log2(1.0 + snr_k)  # bits/s
                total_bits += cap_k * cfg.TAU_SEC
        total_time = len(self.records) * cfg.TAU_SEC
        if total_time <= 0:
            return 0.0
        return (total_bits / total_time) / cfg.BW_HZ  # bits/s/Hz

    # ── M6: Honest fraction ───────────────────────────────────────────────────

    def honest_fraction_series(self) -> np.ndarray:
        """Per-slot honest NCU fraction f_honest(t)."""
        series = []
        for r in self.records:
            K = len(r.d_hat)
            honest = int(np.sum(r.d_hat == r.d_true))
            series.append(honest / max(K, 1))
        return np.array(series)

    # ── M7: ZKP overhead ──────────────────────────────────────────────────────

    def zkp_overhead(self) -> dict:
        """Per-NCU average prover time, verifier time, and message size."""
        all_prove = [t for r in self.records for t in r.prove_times_ms]
        return {
            "mean_prove_ms":   float(np.mean(all_prove)) if all_prove else 0.0,
            "mean_verify_ms":  cfg.ZKP_VERIFY_TIME_MS,
            "total_bytes_per_slot_per_ncu": (
                cfg.COMMITMENT_SIZE_BYTES
                + cfg.ZKP_PROOF_SIZE_BYTES
                + cfg.BOND_CONFIRM_BYTES
                + cfg.REVEAL_BYTES
            ),
        }

    # ── Summary dict ──────────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "n_slots":    len(self.records),
            "DAF":        self.daf(),
            "HIP":        self.hip(),
            "Revenue":    self.total_revenue(),
            "Welfare":    self.total_welfare(),
            "SE":         self.spectrum_efficiency(),
            "f_honest":   float(np.mean(self.honest_fraction_series())),
            "ZKP":        self.zkp_overhead(),
        }
