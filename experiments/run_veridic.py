"""
run_veridic.py – Run full VERIDIC-DSA experiment and save results.

Usage:
    python experiments/run_veridic.py --K 10 --episodes 50000 --seed 42
    python experiments/run_veridic.py --K 10 --n_slots 10000 --seed 42  # quick test

The script runs:
  1. VERIDIC-DSA with varying adversary types and fractions
  2. Scalability sweep over K ∈ {10, 20, 30}
  3. Non-stationarity stress test
  4. Saves all results as JSON to results/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.adversary import AdversaryType
from src.experiment import VeridicDSA
import src.config as cfg


# ── Experiment configurations ─────────────────────────────────────────────────

ADV_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.33]    # A/K sweeps
ADV_TYPES     = [AdversaryType.SF, AdversaryType.STF, AdversaryType.CCF]


def run_veridic(
    K: int,
    n_slots: int,
    seed: int,
    adversary_fraction: float = 0.3,
    adversary_type: AdversaryType = AdversaryType.SF,
    nonstationarity: bool = False,
    verbose: bool = False,
) -> tuple[dict, list]:
    """Run VERIDIC-DSA for n_slots and return (summary, revenue_series)."""
    sim = VeridicDSA(
        K=K,
        adversary_fraction=adversary_fraction,
        adversary_type=adversary_type,
        seed=seed,
        nonstationarity=nonstationarity,
    )
    # Run episode by episode
    episode_length = min(200, n_slots)
    n_episodes = n_slots // episode_length
    revenues: list[float] = []
    for ep in range(n_episodes):
        ep_start = len(sim.metrics.records)
        for _ in range(episode_length):
            sim.run_slot(sim._slot_count)
        ep_recs = sim.metrics.records[ep_start:]
        revenues.append(sum(r.revenue for r in ep_recs))
        if verbose and (ep + 1) % 100 == 0:
            ep_summary = {
                "ep": ep + 1,
                "rev": revenues[-1],
                "rho_mean": float(np.mean(ep_recs[-1].rho)) if ep_recs else 0.0,
            }
            print(f"  Ep {ep+1}/{n_episodes} | Rev={ep_summary['rev']:.2f} "
                  f"| rho={ep_summary['rho_mean']:.3f}")

    summary = sim.metrics.summary()
    return summary, revenues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--K",        type=int,   default=10)
    parser.add_argument("--episodes", type=int,   default=50_000,
                        help="Total training episodes (each has EPISODE_LENGTH slots)")
    parser.add_argument("--n_slots",  type=int,   default=None,
                        help="Override: total slots (episodes * length)")
    parser.add_argument("--seed",     type=int,   default=cfg.SEED)
    parser.add_argument("--adv_frac", type=float, default=0.3)
    parser.add_argument("--adv_type", type=str,   default="SF",
                        choices=["SF", "STF", "CCF", "HONEST"])
    parser.add_argument("--no_sweep", action="store_true",
                        help="Skip K-sweep and adversary-fraction sweep")
    parser.add_argument("--nonstat",  action="store_true",
                        help="Enable non-stationarity shift at slot 500")
    parser.add_argument("--out_dir",  type=str,   default="results")
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)

    n_slots = args.n_slots if args.n_slots else args.episodes * cfg.EPISODE_LENGTH
    adv_type = AdversaryType[args.adv_type]

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict = {}

    # ── 1. Main VERIDIC-DSA run ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"VERIDIC-DSA Main Run  K={args.K}, adv_frac={args.adv_frac}, "
          f"adv_type={args.adv_type}, n_slots={n_slots}")
    print(f"{'='*60}")
    t0 = time.time()
    summary, revenues = run_veridic(
        K=args.K, n_slots=n_slots, seed=args.seed,
        adversary_fraction=args.adv_frac,
        adversary_type=adv_type,
        nonstationarity=args.nonstat,
        verbose=args.verbose,
    )
    elapsed = time.time() - t0
    summary["elapsed_s"]    = elapsed
    summary["revenues"]     = revenues[:500]   # store first 500 episodes
    all_results["veridic_main"] = summary
    print(f"DAF={summary['DAF']:.4f}  HIP={summary['HIP']:.6f}  "
          f"Rev={summary['Revenue']:.1f}  ({elapsed:.1f}s)")

    if not args.no_sweep:
        # ── 2. Adversary fraction sweep ─────────────────────────────────────
        print(f"\nAdversary fraction sweep (K={args.K}) …")
        frac_results = {}
        for frac in ADV_FRACTIONS:
            t0 = time.time()
            s, _ = run_veridic(
                K=args.K, n_slots=n_slots, seed=args.seed,
                adversary_fraction=frac, adversary_type=AdversaryType.SF,
            )
            s["elapsed_s"] = time.time() - t0
            frac_results[str(frac)] = s
            print(f"  frac={frac:.2f} | DAF={s['DAF']:.4f} HIP={s['HIP']:.6f}")
        all_results["adversary_fraction_sweep"] = frac_results

        # ── 3. K sweep ────────────────────────────────────────────────────────
        print(f"\nK sweep …")
        k_results = {}
        for K_val in cfg.K_VALUES:
            t0 = time.time()
            s, _ = run_veridic(
                K=K_val, n_slots=n_slots, seed=args.seed,
                adversary_fraction=args.adv_frac, adversary_type=adv_type,
            )
            s["elapsed_s"] = time.time() - t0
            k_results[str(K_val)] = s
            print(f"  K={K_val} | DAF={s['DAF']:.4f} HIP={s['HIP']:.6f}")
        all_results["K_sweep"] = k_results

        # ── 4. Adversary type sweep ───────────────────────────────────────────
        print(f"\nAdversary type sweep (K={args.K}) …")
        type_results = {}
        for atype in ADV_TYPES:
            t0 = time.time()
            s, _ = run_veridic(
                K=args.K, n_slots=n_slots, seed=args.seed,
                adversary_fraction=args.adv_frac, adversary_type=atype,
            )
            s["elapsed_s"] = time.time() - t0
            type_results[atype.name] = s
            print(f"  type={atype.name} | DAF={s['DAF']:.4f}")
        all_results["adversary_type_sweep"] = type_results

    # ── 5. Non-stationarity run ───────────────────────────────────────────────
    if args.nonstat:
        print(f"\nNon-stationarity stress test …")
        t0 = time.time()
        s_ns, rev_ns = run_veridic(
            K=args.K, n_slots=max(n_slots, 1200), seed=args.seed,
            adversary_fraction=args.adv_frac,
            adversary_type=adv_type,
            nonstationarity=True,
            verbose=args.verbose,
        )
        s_ns["elapsed_s"] = time.time() - t0
        s_ns["revenues"]  = rev_ns[:500]
        all_results["nonstationarity"] = s_ns
        print(f"  DAF={s_ns['DAF']:.4f} HIP={s_ns['HIP']:.6f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_file = out_dir / f"veridic_K{args.K}_s{args.seed}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
