# 04 — ISAC Ground-Truth Oracle

> **Paper sections:** Sec. 8 (ISAC Oracle), Sec. 4.2 (CFAR), Theorem 8.1 (Error Bound)  
> **Code:** `src/oracle.py`, `src/channel.py :: ISACRadarModel`

---

## 4.1 Oracle Role and Motivation

The ISAC BS simultaneously transmits communication waveforms and performs **monostatic radar sensing** of the CU. This dual-function capability provides an **independent, unforgeable ground truth** for CU activity, which:

1. Cannot be manipulated by NCUs (they do not control the BS radar).
2. Enables the mechanism to distinguish NCU falsification from genuine sensing error.
3. Provides a cryptographic anchor: the oracle result is published after NCUs commit (preventing adaptive falsification).

---

## 4.2 CA-CFAR Detection (Sec. 4.2)

The BS applies **Cell-Averaging Constant False Alarm Rate (CA-CFAR)** detection to the matched-filter output.

### CFAR Threshold

$$\eta_{\text{CFAR}} = N_{\text{ref}} \left(P_{\text{fa,ISAC}}^{-1/N_{\text{ref}}} - 1\right)$$

| Parameter | Value |
|-----------|-------|
| $N_{\text{ref}}$ (reference cells) | 32 |
| $P_{\text{fa,ISAC}}$ (CFAR false alarm) | 0.01 |
| $\eta_{\text{CFAR}}$ (computed) | ≈ 4.95 |
| Guard cells | 4 |

> **Implementation:** `oracle.py :: ISACOracle._compute_cfar_threshold()`

### Oracle Decision

$$D_{\text{ISAC}} = \mathbf{1}[\Lambda_{\text{ISAC}} \geq \eta_{\text{CFAR}}]$$

### Oracle Confidence Score

The confidence score $C_{\text{ISAC}} \in [0,1]$ captures how strongly the statistic exceeds the threshold:

$$C_{\text{ISAC}} = \sigma\!\left(\frac{\Lambda_{\text{ISAC}} - \eta_{\text{CFAR}}}{\eta_{\text{CFAR}}}\right), \quad \sigma(x) = \frac{1}{1+e^{-x}}$$

This sigmoid normalisation maps the raw excess above the CFAR threshold to a calibrated probability, used to:
- Weight bond slashing: $\text{slash}_k \propto C_{\text{ISAC}}$
- Weight reputation updates: $\Delta_k = C_{\text{ISAC}} \cdot (1 + \xi_{\text{coal}} \cdot \mathbf{1}[\text{coalition}])$

> **Implementation:** `oracle.py :: ISACOracle.observe()`

---

## 4.3 Oracle Error Probability Bound — Theorem 8.1

**Theorem 8.1:** Under the non-central chi-squared model with $N_t$ BS antennas, $L$ integrated pulses, and per-antenna radar SNR $\gamma_r$:

$$P_{e,\text{ISAC}} \leq \exp\!\left(-\frac{N_t \cdot L \cdot \gamma_r^2}{2(1+\gamma_r)^2}\right)$$

**Numerical evaluation** at $N_t = 64$, $L = 128$, $\gamma_r = 10\,\text{dB}$ (linear: 10):

$$P_{e,\text{ISAC}} \leq \exp\!\left(-\frac{64 \cdot 128 \cdot 100}{2 \cdot (11)^2}\right) = \exp(-3287) \approx 0$$

The oracle error probability is effectively zero under these parameters, justifying treating $D_{\text{ISAC}}$ as cryptographic ground truth.

> **Implementation:** `oracle.py :: ISACOracle.oracle_error_bound()`

---

## 4.4 Analytical ROC Curve

The CFAR detector ROC curve (approximation for Swerling Type I target):

$$P_d = P_{\text{fa}}^{\frac{1}{1 + \gamma_r \cdot L}}$$

This simplification holds for large $L$ and moderate SNR; it gives:
- At $P_{\text{fa}} = 0.01$, $\gamma_r = 10$, $L = 128$: $P_d \approx 0.01^{1/1281} \approx 0.9965$

> **Implementation:** `oracle.py :: ISACOracle.roc_curve()`

---

## 4.5 Oracle Weight in Aggregation

The oracle contributes to the aggregate decision with weight $w_{\text{ISAC}} = 0.5$:

$$D_{\text{agg}} = \mathbf{1}\!\left[\frac{\sum_k \rho_k \hat{d}_k + w_{\text{ISAC}} D_{\text{ISAC}}}{\sum_k \rho_k + w_{\text{ISAC}}} \geq \theta\right]$$

Setting $w_{\text{ISAC}} = 0.5$ means the oracle carries equal weight to the sum of all NCU reputations when $\sum_k \rho_k = 0.5$ (low trust scenario), providing a strong bias toward the physically-grounded measurement.

---

## 4.6 Mathematical Validation Checklist

| Formula | Code | Status |
|---------|------|--------|
| $\eta = N_{\text{ref}}(P_{\text{fa}}^{-1/N_{\text{ref}}}-1)$ | `_compute_cfar_threshold()` | ✅ |
| $D_{\text{ISAC}} = \mathbf{1}[\Lambda \geq \eta]$ | `ISACOracle.observe()` | ✅ |
| $C_{\text{ISAC}} = \sigma((\Lambda-\eta)/\eta)$ | `ISACOracle.observe()` | ✅ |
| Theorem 8.1 exponent | `oracle_error_bound()` | ✅ ($\approx 0$) |
| $H_1$ statistic: $\chi^2_{\text{nc}}(2L, 2L\gamma_r)/(2L)$ | `ISACRadarModel.compute_isac_statistic()` | ✅ |
| $H_0$ statistic: $\chi^2(2L)/(2L)$ | `ISACRadarModel.compute_isac_statistic()` | ✅ |
| CFAR normalization: mean of $N_{\text{ref}}$ reference cells | `ISACRadarModel.compute_isac_statistic()` | ✅ |
