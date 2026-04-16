"""
run_baseline.py – Run BL1-BL7 baseline experiments and save results.

Usage:
    python experiments/run_baseline.py --K 10 --n_slots 10000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.adversary import AdversaryType
from src.baselines import BaselineSystem, BaselineType
import src.config as cfg


def run_baselines(K: int, n_slots: int, seed: int, adversary_fraction: float = 0.3) -> dict:
    """Run all baselines and return a results dict."""
    results = {}
    for bl in BaselineType:
        print(f"  Running {bl.name} (K={K}, {n_slots} slots) …", end=" ", flush=True)
        t0 = time.time()
        sys_bl = BaselineSystem(
            baseline=bl,
            K=K,
            adversary_fraction=adversary_fraction,
            adversary_type=AdversaryType.SF,
            seed=seed,
        )
        summary = sys_bl.run(n_slots=n_slots)
        elapsed = time.time() - t0
        summary["elapsed_s"] = elapsed
        summary["K"]         = K
        summary["n_slots"]   = n_slots
        summary["seed"]      = seed
        results[bl.name] = summary
        print(f"DAF={summary['DAF']:.3f}  HIP={summary['HIP']:.4f}  "
              f"Rev={summary['Revenue']:.1f}  ({elapsed:.1f}s)")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--K",       type=int,   default=10,    help="Number of NCUs")
    parser.add_argument("--n_slots", type=int,   default=10_000, help="Total slots to simulate")
    parser.add_argument("--seed",    type=int,   default=cfg.SEED)
    parser.add_argument("--adv_frac", type=float, default=0.3,  help="Adversary fraction")
    parser.add_argument("--out_dir",  type=str,  default="results")
    args = parser.parse_args()

    np.random.seed(args.seed)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"VERIDIC-DSA Baseline Experiments")
    print(f"K={args.K}, n_slots={args.n_slots}, seed={args.seed}")
    print(f"{'='*60}")

    results = run_baselines(
        K=args.K,
        n_slots=args.n_slots,
        seed=args.seed,
        adversary_fraction=args.adv_frac,
    )

    out_file = out_dir / f"baselines_K{args.K}_s{args.seed}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
