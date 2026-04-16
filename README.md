# VERIDIC-DSA: Verifiable, Incentive-Compatible Dynamic Spectrum Access

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-orange.svg)](https://pytorch.org/)

End-to-end simulation of the **VERIDIC-DSA** framework, a cryptographically verifiable, incentive-compatible Dynamic Spectrum Access system for sub-THz / D-band (142 GHz) shared spectrum scenarios.

The experiment is fully aligned with the mathematical models, algorithms, and constants described in the companion JSAC paper ([`JSAC_VERIDIC_DSA_Experiment.md`](JSAC_VERIDIC_DSA_Experiment.md)).

---

## Table of Contents

1. [Overview](#overview)
2. [System Model](#system-model)
3. [Repository Layout](#repository-layout)
4. [Quick Start](#quick-start)
5. [Module Reference](#module-reference)
6. [Running the Experiments](#running-the-experiments)
7. [Output Package](#output-package)
8. [Key Results](#key-results)
9. [Reproducibility](#reproducibility)
10. [Citation](#citation)
11. [License](#license)

---

## Overview

VERIDIC-DSA combines four complementary mechanisms to enable trustworthy secondary spectrum access in the presence of adversarial NCUs (Non-Critical Users):

| Mechanism | Purpose |
|---|---|
| **Cryptographic Sensing Protocol** | Pedersen commitment + Groth16 ZK-SNARK proof binds each NCU to its reported sensing decision |
| **ISAC Ground-Truth Oracle** | ISAC-capable Base Station independently verifies CU activity via CA-CFAR radar, providing an unforgeable reference |
| **CVSA Auction** | Cryptographic VCG Spectrum Auction allocates spectrum efficiently while penalising falsifiers via bond slashing |
| **PPO Dynamic Pricing** | Deep Reinforcement Learning (Proximal Policy Optimisation) adapts access prices online to maximise long-run revenue |

---

## System Model

```
System: N = <B, U_C, U_N, F, T>
  B     – ISAC Base Station (oracle + auctioneer)
  U_C   – Critical Users (CU), protected primary incumbents
  U_N   – Non-Critical Users (NCU), secondary access seekers
  F     – Adversary set (SF, STF, CCF)
  T     – Time-slot index
```

**Key physical parameters (Sec. 12.1):**

| Parameter | Value |
|---|---|
| Carrier frequency f_c | 142 GHz (D-band / sub-THz) |
| Bandwidth B | 100 MHz |
| Slot duration τ | 10 ms |
| BS antenna count N_t | 64 (ULA) |
| Sensing samples N_s | 1 000 |
| Max false-alarm α_max | 0.05 |
| Training episodes | 50 000 |
| Random seed | 42 |

---

## Repository Layout

```
isac-experiment/
├── src/                        # Core simulation modules
│   ├── config.py               # All paper constants (Sec. 12.1, Appendix E)
│   ├── channel.py              # Sub-THz path-loss, Nakagami-m fading, CU semi-Markov, ISAC radar
│   ├── sensing.py              # Energy detection, Neyman-Pearson threshold
│   ├── crypto.py               # Pedersen commitment + Groth16 ZKP (5 circuit constraints)
│   ├── oracle.py               # CA-CFAR oracle, confidence score, ROC analysis
│   ├── mechanism.py            # CVSA aggregation, VCG allocation/payment, bond slashing
│   ├── reputation.py           # Beta-Bernoulli updates, spectral coalition detection
│   ├── adversary.py            # SF / STF / CCF adversary models (Sec. 3.1)
│   ├── drl.py                  # PPO Actor-Critic, GAE-λ, clipped objective (Algorithm 2)
│   ├── metrics.py              # DAF, HIP, Revenue, SW, SE, f_honest, ZKP overhead
│   ├── experiment.py           # Algorithm 1 (7 phases/slot) + Algorithms 2 & 3
│   └── baselines.py            # BL1–BL7 baseline systems
│
├── experiments/
│   ├── run_baseline.py         # CLI: run all BL1–BL7 baselines
│   └── run_veridic.py          # CLI: full VERIDIC-DSA (sweeps, non-stationarity)
│
├── analysis/
│   └── generate_figures.py     # Produces all 12 figures + Tables I–III
│
├── results/
│   └── figures/                # Pre-generated PNG outputs
│
├── JSAC_VERIDIC_DSA_Experiment.md   # Paper specification document
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Python ≥ 3.10** and **PyTorch 2.6.0** are required.

### 2. Run a quick smoke test (2 000 slots, K = 10)

```bash
python experiments/run_veridic.py --K 10 --n_slots 2000 --seed 42
```

### 3. Run all baselines

```bash
python experiments/run_baseline.py --K 10 --n_slots 10000 --seed 42
```

### 4. Generate all figures and tables

```bash
python analysis/generate_figures.py --n_slots 2000
```

Output PNGs are written to `results/figures/`.

---

## Module Reference

### `src/config.py`
Single source of truth for every numerical constant used in the paper.
Import as `import src.config as cfg`.

### `src/channel.py`
- `CUActivityModel` – semi-Markov CU on/off process (π₁ ≈ 0.30)
- `ChannelModel` – per-slot sub-THz channel realisations (h_ck, h_bk, g_kb, y_k)
- `ISACRadarModel` – CA-CFAR radar return statistic Λ_ISAC
- `generate_subthz_channel` – Saleh-Valenzuela MIMO channel (Sec. 12.2)

### `src/sensing.py`
- `SensingModule.sense(y_k)` – returns energy statistics T_k and decisions d_k
- `detection_threshold(σ²)` – Neyman-Pearson threshold λ_k = σ²(1 + √(2/N_s)·Q⁻¹(α_max))

### `src/crypto.py`
- `commit(d_k, rng)` – Pedersen commitment (SHA-256 based, binding + hiding)
- `prove(...)` – Groth16 ZK-SNARK simulation for 5 circuit constraints (Sec. 5.2)
- `verify_proof(...)` – constant-time verifier (O(1) pairing check simulation)

### `src/oracle.py`
- `ISACOracle.observe(H_t)` – returns (D_ISAC, C_ISAC, Λ_ISAC)
- `ISACOracle.oracle_error_bound()` – Theorem 8.1 exponential bound

### `src/mechanism.py`
- `CVSAMechanism.run(...)` – executes full CVSA slot: aggregation → slashing → VCG auction

### `src/reputation.py`
- `ReputationModel.update(...)` – Beta-Bernoulli (α_k, β_k) increments
- `detect_coalitions(...)` – Algorithm 3: normalised Laplacian spectral clustering

### `src/adversary.py`
- `AdversaryType.SF` – always reports d̂ = 0 (idle)
- `AdversaryType.STF` – threshold-based strategic falsification
- `AdversaryType.CCF` – coordinated coalition falsification

### `src/drl.py`
- `PPOAgent` – Actor + Critic networks, `select_action`, `store`, `update` (Algorithm 2)
- State: 12-dimensional vector per Sec. 7.1
- Action: per-NCU price level index ∈ {0, …, N_Q−1}

### `src/experiment.py`
- `VeridicDSA.run_slot(slot)` – one complete 7-phase slot (Algorithm 1)
- `VeridicDSA.run(n_episodes, episode_length)` – full training run

### `src/baselines.py`
| ID | Name | Description |
|---|---|---|
| BL1 | Naive Cooperative | Energy detection + majority vote, no mechanism |
| BL2 | OR-rule | Access if any NCU reports idle |
| BL3 | AND-rule | Access only if all NCUs report idle |
| BL4 | Weighted Reputation | Reputation-weighted vote, no crypto |
| BL5 | Auction-only | VCG auction, no ZKP or bonds |
| BL6 | Crypto-only | ZKP + commitment, static pricing |
| BL7 | Static Pricing | Full crypto + VCG, fixed access price |

---

## Running the Experiments

### Full VERIDIC-DSA training (50 000 episodes)

```bash
python experiments/run_veridic.py \
    --K 10 \
    --episodes 50000 \
    --seed 42 \
    --adv_frac 0.30 \
    --adv_type SF \
    --verbose
```

### Adversary fraction sweep

```bash
python experiments/run_veridic.py --K 10 --n_slots 100000 --seed 42
# (sweep is run automatically unless --no_sweep is passed)
```

### K sweep (K ∈ {10, 20, 30})

Included in the default `run_veridic.py` run. Override with `--no_sweep` to skip.

### Non-stationarity stress test (duty-cycle shift at slot 500)

```bash
python experiments/run_veridic.py --K 10 --n_slots 10000 --seed 42 --nonstat
```

### Baseline comparison

```bash
python experiments/run_baseline.py --K 10 --n_slots 100000 --seed 42 --adv_frac 0.30
```

Results are saved as `results/veridic_K<K>_s<seed>.json` and `results/baselines_K<K>_s<seed>.json`.

---

## Output Package

| Output | Location | Description |
|---|---|---|
| Figure 1 | `results/figures/fig01_architecture.png` | System architecture |
| Figure 2 | `results/figures/fig02_roc_curves.png` | ROC comparison |
| Figure 3 | `results/figures/fig03_daf_vs_adv.png` | DAF vs adversary fraction |
| Figure 4 | `results/figures/fig04_revenue_convergence.png` | PPO convergence |
| Figure 5 | `results/figures/fig05_pareto.png` | Revenue–Welfare Pareto |
| Figure 6 | `results/figures/fig06_hip_vs_K.png` | HIP vs NCU count |
| Figure 7 | `results/figures/fig07_zkp_overhead.png` | ZKP computational overhead |
| Figure 8 | `results/figures/fig08_reputation.png` | Reputation dynamics |
| Figure 9 | `results/figures/fig09_se_vs_duty.png` | SE vs CU activity |
| Figure 10 | `results/figures/fig10_bond_sensitivity.png` | Bond sensitivity |
| Figure 11 | `results/figures/fig11_nonstationarity.png` | Non-stationarity robustness |
| Figure 12 | `results/figures/fig12_ablation.png` | Ablation study |
| Table I | `results/figures/table_I_parameters.png` | Parameter summary |
| Table II | `results/figures/table_II_complexity.png` | Computational complexity |
| Table III | `results/figures/table_III_performance.png` | Baseline comparison |

---

## Key Results

| Metric | VERIDIC-DSA | Best Baseline |
|---|---|---|
| DAF (detection accuracy) | **~0.947** | ~0.80 (BL4) |
| HIP (harmful interference prob.) | **< 10⁻³** | > 10⁻² (BL2) |
| Revenue gain vs static pricing | **+31%** | — |
| ZKP prover time | ~120 ms | N/A |
| ZKP verifier time | ~1.5 ms | N/A |

---

## Reproducibility

Every run uses:
- Fixed random seed (`--seed 42` by default, matching paper Sec. 12.1)
- Deterministic NumPy and PyTorch initialisation
- Identical channel realisations, CU activity traces, and NCU valuations across VERIDIC-DSA and all baselines (same seed)

To verify a run:
```bash
# Check DAF ≈ 0.947, HIP < 1e-3
python experiments/run_veridic.py --K 10 --n_slots 10000 --seed 42 --no_sweep
```

---

## Citation

If you use this code, please cite the companion paper:

```bibtex
@article{veridic_dsa_2026,
  title   = {VERIDIC-DSA: Verifiable, Incentive-Compatible Dynamic Spectrum Access
             for Sub-THz Networks via ISAC and Zero-Knowledge Proofs},
  author  = {Pandey, I. and others},
  journal = {IEEE Journal on Selected Areas in Communications},
  year    = {2026}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
