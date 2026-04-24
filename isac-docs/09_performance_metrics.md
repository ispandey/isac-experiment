# 09 — Performance Metrics

> **Paper sections:** Sec. 13 (Metrics), Sec. 13.1 (DAF), Sec. 13.2 (HIP), Sec. 13.3 (Revenue/Welfare), Sec. 13.4 (SE), Sec. 15.3 (ZKP overhead)  
> **Code:** `src/metrics.py`

---

## 9.1 M1 — Detection Accuracy against Falsification (DAF)

DAF measures the system's ability to correctly classify NCU behaviours at the NCU-slot level.

$$\text{DAF} = \frac{TP + TN}{TP + TN + FP + FN}$$

| Cell | Definition |
|------|-----------|
| TP | NCU falsified **and** was detected |
| TN | NCU was honest **and** was not flagged |
| FP | NCU was honest but was incorrectly flagged as falsifier |
| FN | NCU falsified but was not detected |

An NCU is deemed to have **falsified** if $\hat{d}_k \neq d_k^{\text{true}}$.  
An NCU is deemed **detected** if `falsification_detected[k]` is True (ZKP rejection + oracle contradiction).

**Target:** DAF $\geq 0.947$ (≈ 94.7% accuracy at 30% adversary fraction, SF type).

> **Implementation:** `metrics.py :: MetricStore.daf()`

---

## 9.2 M2 — Harmful Interference Probability (HIP)

HIP measures the probability that actively transmitting NCUs cause harmful interference at a CU receiver when the CU is actually present.

$$\text{HIP} = \Pr\!\left[\sum_{k \in \mathcal{A}_t} P_k |g_{kb}|^2 > I_{\max} \;\middle|\; H_t = 1\right]$$

Estimated empirically:

$$\widehat{\text{HIP}} = \frac{\#\text{slots where } H_t=1 \text{ and } \sum_{k} x_k^* P_k |g_{kb}|^2 > I_{\max}}{\#\text{slots where } H_t=1}$$

with $I_{\max} = -80\,\text{dBm} = 10^{-11}\,\text{W}$.

**Target:** HIP $< 10^{-3}$ ($< 0.1\%$).

> **Implementation:** `metrics.py :: MetricStore.hip()`

---

## 9.3 M3 — Network Revenue

$$R_{\text{total}} = \sum_{t=1}^{T} R(t) = \sum_{t=1}^{T} \sum_{k:\,x_k^*=1} \bigl(q_k^{\text{DRL}}(t) + p_k^{\text{VCG}}(t)\bigr)$$

Revenue is generated only when spectrum is idle ($D_{\text{agg}} = 0$) and NCUs win the auction.

**Target:** +31% over static pricing baseline (BL7) over 50 000 episodes.

> **Implementation:** `metrics.py :: MetricStore.total_revenue()`, `.revenue_series()`

---

## 9.4 M4 — Social Welfare

$$\text{SW}_{\text{total}} = \sum_{t=1}^{T} \sum_{k:\,x_k^*=1} (v_k(t) - p_k^{\text{VCG}}(t))$$

Social welfare is the sum of NCU consumer surpluses. The revenue–welfare Pareto frontier is traced by varying the reward weight $\mu_1$.

> **Implementation:** `metrics.py :: MetricStore.total_welfare()`

---

## 9.5 M5 — Spectrum Efficiency

$$\text{SE} = \frac{\sum_t \sum_{k:\,x_k^*=1} B_w \log_2(1 + \text{SNR}_k) \cdot \tau}{T \cdot \tau \cdot B_w} \quad \text{[bits/s/Hz]}$$

where $\text{SNR}_k \approx P_k / \sigma_k^2$, $B_w = 100\,\text{MHz}$, $\tau = 10\,\text{ms}$.

The simplification uses maximum NCU transmit power (upper bound on SE).

> **Implementation:** `metrics.py :: MetricStore.spectrum_efficiency()`

---

## 9.6 M6 — Honest NCU Fraction

$$f_{\text{honest}}(t) = \frac{1}{K}\sum_{k=1}^{K} \mathbf{1}[\hat{d}_k(t) = d_k^{\text{true}}(t)]$$

Tracks the fraction of NCUs behaving honestly each slot. Under VERIDIC-DSA, adversaries are deterred over time (reputation decay + bond slashing), so $f_{\text{honest}}$ should increase as a function of slot index.

> **Implementation:** `metrics.py :: MetricStore.honest_fraction_series()`

---

## 9.7 M7 — ZKP Computational Overhead (Sec. 15.3)

| Metric | Value |
|--------|-------|
| Mean prover time | ~120 ms |
| Mean verifier time | ~1.5 ms |
| Proof size | 192 bytes |
| Commitment size | 256 bytes |
| Bond confirmation | 64 bytes |
| Reveal message | 64 bytes |
| **Total overhead per NCU per slot** | **576 bytes** |

The verifier time is O(1) in circuit size (Groth16 constant-time pairing check), enabling the BS to verify $K = 30$ proofs in $30 \times 1.5 = 45\,\text{ms}$ — well within the 10 ms slot budget if parallelised.

> **Implementation:** `metrics.py :: MetricStore.zkp_overhead()`

---

## 9.8 Baseline Comparison Systems (BL1–BL7)

| ID | Name | ZKP | Oracle | Reputation | DRL Pricing |
|----|------|-----|--------|-----------|-------------|
| BL1 | Naive Cooperative | ✗ | ✗ | ✗ | ✗ |
| BL2 | OR-rule | ✗ | ✗ | ✗ | ✗ |
| BL3 | AND-rule | ✗ | ✗ | ✗ | ✗ |
| BL4 | Weighted Reputation | ✗ | ✓ | ✓ | ✗ |
| BL5 | Auction-only | ✗ | ✓ | ✓ | ✗ |
| BL6 | Crypto-only | ✓ | ✓ | ✓ | ✗ |
| BL7 | Static Pricing | ✓ | ✓ | ✓ | ✗ (fixed $q$) |
| **VERIDIC-DSA** | **Full system** | **✓** | **✓** | **✓** | **✓** |

**BL1 majority vote rule (corrected):** Access is granted when the *majority* of NCUs report CU idle ($\hat{d}_k = 0$):

$$\text{grant}_{\text{BL1}} = \mathbf{1}\!\left[\frac{1}{K}\sum_k \hat{d}_k < 0.5\right]$$

> **Bug fix:** The original code used `int(np.mean(d_hat) >= 0.5)` (inverted), which granted access when the majority detected CU presence. This has been corrected to `int(np.mean(d_hat) < 0.5)`.

---

## 9.9 Expected Results Summary

| Metric | VERIDIC-DSA | BL4 (best baseline) | BL2 (worst) |
|--------|-------------|---------------------|-------------|
| DAF | ≥ 0.947 | ~0.80 | ~0.50 |
| HIP | < 10⁻³ | > 10⁻² | > 10⁻¹ |
| Revenue (rel.) | 1.31× | 0.82× | 0.60× |
| ZKP overhead | 576 B/slot/NCU | N/A | N/A |
