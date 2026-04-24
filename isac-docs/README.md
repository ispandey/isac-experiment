# VERIDIC-DSA Documentation Index

This folder contains the complete technical documentation for the **VERIDIC-DSA** framework, organized to support journal writing for the companion IEEE JSAC paper.

---

## Document Index

| # | File | Description |
|---|------|-------------|
| 01 | [System Overview](01_system_overview.md) | Network model, problem formulation, novelty statement |
| 02 | [Sub-THz Channel Model](02_channel_model.md) | Path loss, Nakagami-m fading, ISAC radar, MIMO channel |
| 03 | [Cryptographic Sensing Protocol](03_cryptographic_sensing_protocol.md) | Pedersen commitment, Groth16 ZK-SNARK, circuit constraints |
| 04 | [ISAC Ground-Truth Oracle](04_isac_oracle.md) | CA-CFAR detector, confidence scoring, oracle error bounds |
| 05 | [Mechanism Design — CVSA](05_mechanism_design.md) | VCG auction, bond slashing, incentive-compatibility proofs |
| 06 | [Reputation & Trust Management](06_reputation_system.md) | Beta-Bernoulli model, spectral coalition detection |
| 07 | [DRL Dynamic Pricing — PPO](07_drl_pricing.md) | State/action/reward design, PPO algorithm, convergence |
| 08 | [Threat Model & Adversaries](08_threat_model_adversaries.md) | SF / STF / CCF adversary types, Byzantine tolerance |
| 09 | [Performance Metrics](09_performance_metrics.md) | DAF, HIP, Revenue, SW, SE, ZKP overhead definitions |
| 10 | [Simulation Parameters](10_simulation_parameters.md) | Full parameter table, reproducibility notes |
| 11 | [Theoretical Analysis](11_theoretical_analysis.md) | Key theorems, proofs, IC/IR/BB guarantees |
| 12 | [Google Colab Run Guide](12_colab_run_guide.md) | Hardware requirements, Colab setup, step-by-step instructions |

---

## How to Use These Documents

- **For journal writing:** Each document maps to one or more sections of the JSAC paper. Cross-references to paper sections (e.g., *Sec. 4.1*) are embedded throughout.
- **For code understanding:** Every mathematical formula is annotated with the implementing source file and function.
- **For reproducibility:** See [Simulation Parameters](10_simulation_parameters.md) and [Colab Run Guide](12_colab_run_guide.md).

---

## Quick Reference — Key Results

| Metric | VERIDIC-DSA | Best Baseline |
|--------|-------------|---------------|
| DAF (detection accuracy against falsification) | **≥ 0.947** | ~0.80 (BL4) |
| HIP (harmful interference probability) | **< 10⁻³** | > 10⁻² (BL2) |
| Revenue gain vs. static pricing | **+31%** | — |
| ZKP prover time | ~120 ms | N/A |
| ZKP verifier time | ~1.5 ms | N/A |
| Per-NCU protocol overhead | 576 bytes/slot | N/A |

---

## Citation

```bibtex
@article{veridic_dsa_2026,
  title   = {VERIDIC-DSA: Verifiable, Incentive-Compatible Dynamic Spectrum Access
             for Sub-THz Networks via ISAC and Zero-Knowledge Proofs},
  author  = {Pandey, I. and others},
  journal = {IEEE Journal on Selected Areas in Communications},
  year    = {2026}
}
```
