# 10 — Simulation Parameters & Reproducibility

> **Paper sections:** Sec. 12.1 (Parameters), Appendix E (DRL hyperparameters)  
> **Code:** `src/config.py`

---

## 10.1 Complete Parameter Table

### Physical Layer (Sec. 12.1)

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Carrier frequency | $f_c$ | 142 | GHz (D-band / sub-THz) |
| Bandwidth | $B_w$ | 100 | MHz |
| Speed of light | $c$ | $3\times10^8$ | m/s |
| Wavelength | $\lambda = c/f_c$ | ≈ 2.11 | mm |
| Slot duration | $\tau$ | 10 | ms |
| Sensing samples | $N_s$ | 1 000 | — |
| Temperature | $T$ | 290 | K |
| Boltzmann constant | $k_B$ | $1.38\times10^{-23}$ | J/K |
| Noise figure | NF | 7 | dB (linear: 5.01) |
| Thermal noise power | $\sigma^2 = k_B T B_w \cdot\text{NF}$ | $\approx 2.01\times10^{-12}$ | W |
| Molecular absorption | $\alpha_{\text{abs}}$ | 2.1 | dB/km |
| Shadowing std dev | $\sigma_\chi$ | 3.0 | dB |
| Nakagami-m parameter | $m$ | 2.0 | — |

### Critical User (CU)

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| CU Tx power | $P_c$ | 30 | dBm (≈ 1 W) |
| CU→NCU distance | $d_{ck}$ | 100 | m |
| CU activity rate (idle→active) | $\lambda_{\text{on}}$ | 0.10 | per slot |
| CU inactivity rate (active→idle) | $\lambda_{\text{off}}$ | 0.233 | per slot |
| Steady-state CU occupancy | $\pi_1$ | ≈ 0.30 | — |

### Non-Critical User (NCU)

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| NCU max Tx power | $P_k^{\max}$ | 23 | dBm (≈ 200 mW) |
| BS→NCU distance | $d_{bk}$ | 200 | m |
| NCU→CU distance (interference) | $d_{kb}$ | 600 | m |
| Max interference at CU | $I_{\max}$ | −80 | dBm ($10^{-11}$ W) |
| Max false alarm rate | $\alpha_{\max}$ | 0.05 | — |
| Valuation distribution | $v_k$ | $\mathcal{U}[0.5, 5.0]$ | revenue units |

### ISAC Base Station

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| BS max Tx power | $P_{\max}$ | 30 | dBm (≈ 1 W) |
| BS antenna count | $N_t$ | 64 | ULA |
| BS→CU distance (radar) | $d_{bc}$ | 1 000 | m |
| Radar integration pulses | $L$ | 128 | — |
| CFAR reference cells | $N_{\text{ref}}$ | 32 | — |
| CFAR guard cells | — | 4 | — |
| CFAR false alarm rate | $P_{\text{fa,ISAC}}$ | 0.01 | — |
| Radar cross section | $\sigma_{\text{RCS}}$ | 1.0 | m² |
| Radar SNR (operating point) | $\gamma_r$ | 10 | dB |
| Tx antenna gain | $G_t$ | 30 | linear (~15 dBi) |
| Rx antenna gain | $G_r$ | 30 | linear |

### Mechanism

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Oracle weight | $w_{\text{ISAC}}$ | 0.5 |
| Aggregation threshold | $\theta$ | 0.5 |
| Min bond | $B_{\min}$ | 10.0 |
| Penalty scaling | $\kappa$ | 2.0 |
| CFAR confidence normaliser | $\eta_{\text{conf}}$ | 2.0 |
| Co-falsification threshold | $\theta_{\text{collusion}}$ | 0.6 |
| Coalition suspicion threshold | $\theta_{\text{suspect}}$ | 0.5 |
| Coalition reputation amplification | $\xi_{\text{coal}}$ | 2.0 |
| Beta prior | $\alpha_0, \beta_0$ | 1.0 |

### DRL (PPO) Hyperparameters (Appendix E)

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Discount factor | $\gamma$ | 0.99 |
| GAE-λ | $\lambda$ | 0.95 |
| PPO clip | $\varepsilon$ | 0.2 |
| Actor learning rate | $\eta_\pi$ | $3\times10^{-4}$ |
| Critic learning rate | $\eta_V$ | $1\times10^{-3}$ |
| Entropy coefficient | $c_2$ | 0.01 |
| VF coefficient | $c_1$ | 0.5 |
| Mini-batch size | — | 128 |
| PPO epochs per update | $K_e$ | 4 |
| KL early-stop threshold | $\delta_{\text{KL}}$ | 0.05 |
| Update frequency | — | 200 slots |
| Rolling window size | $w$ | 10 |
| Replay window | $T_{\text{win}}$ | 500 |
| Price levels | $N_Q$ | 10 |
| Price range | $[Q_{\min}, Q_{\max}]$ | [0.1, 5.0] |
| Reward weight (welfare) | $\mu_1$ | 0.3 |
| Reward weight (interference) | $\mu_2$ | 10.0 |
| Reward weight (falsification) | $\mu_3$ | 5.0 |

### Simulation Scale

| Parameter | Value |
|-----------|-------|
| Training episodes | 50 000 |
| Slots per episode | 200 |
| Total training slots | $10^7$ |
| Random seed | 42 |
| NCU count sweep | $K \in \{10, 20, 30\}$ |
| Adversary fraction sweep | $\{0.0, 0.1, 0.2, 0.3, 0.33\}$ |
| Non-stationarity shift slot | 500 |

---

## 10.2 Reproducibility

All experiments use a **single shared random seed** (`SEED = 42`) with deterministic NumPy and PyTorch initialization. The same seed produces identical:
- CU activity traces
- Channel realisations
- NCU valuation sequences
- Adversary assignment

across VERIDIC-DSA and all baseline systems, enabling fair comparison.

```python
import numpy as np, torch
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

---

## 10.3 Computational Requirements

| Experiment | Slots | Approx. Wall-Clock Time |
|-----------|-------|------------------------|
| Smoke test ($K=10$, 2 000 slots) | 2 000 | ~30 s (CPU) / ~10 s (GPU) |
| Full VERIDIC-DSA ($K=10$, 50k ep) | $10^7$ | ~4–8 h (CPU) / ~45 min (T4 GPU) |
| All baselines ($K=10$, 100k slots) | 100 000 × 7 | ~2 h (CPU) |
| K-sweep ($K\in\{10,20,30\}$) | $3\times10^7$ | ~12 h (CPU) |
| Full paper experiment set | — | ~24 h (T4) / ~6 h (A100) |

> **Recommendation:** Use Google Colab A100 GPU for full paper experiments. See [Colab Run Guide](12_colab_run_guide.md).
