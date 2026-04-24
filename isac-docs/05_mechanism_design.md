# 05 — Mechanism Design: Cryptographic VCG Spectrum Auction (CVSA)

> **Paper sections:** Sec. 6 (CVSA), Sec. 6.1 (Aggregation), Sec. 6.2 (VCG), Sec. 6.3 (Bonds), Sec. 6.4 (Utility)  
> **Code:** `src/mechanism.py`

---

## 5.1 Overview

The CVSA mechanism transforms the DSA resource allocation problem into a **mechanism design** problem where truthful reporting is the *dominant strategy* for rational NCUs, regardless of the strategies of other players.

The mechanism operates in four sub-phases each slot:

```
Phase 6a │ Update bond requirements B_k(t)
Phase 6b │ Compute aggregate access decision D_agg
Phase 6c │ Detect contradictions → slash bonds → reward honest reporters
Phase 6d │ VCG auction among non-contradicted NCUs → allocate x* → compute payments
```

---

## 5.2 Aggregate Access Decision (Sec. 6.1)

After ZKP verification, the BS forms a *reputation-weighted* aggregate decision incorporating the oracle:

$$D_{\text{agg}}(t) = \mathbf{1}\!\left[\frac{\sum_{k=1}^{K} \rho_k(t)\,\hat{d}_k + w_{\text{ISAC}}\,D_{\text{ISAC}}}{\sum_{k=1}^{K}\rho_k(t) + w_{\text{ISAC}}} \geq \theta\right]$$

| Parameter | Value |
|-----------|-------|
| $w_{\text{ISAC}}$ (oracle weight) | 0.5 |
| $\theta$ (aggregation threshold) | 0.5 |

- $D_{\text{agg}} = 0$: spectrum is idle → auction proceeds.
- $D_{\text{agg}} = 1$: spectrum is active → all access blocked.

> **Implementation:** `mechanism.py :: aggregate_decision()`

---

## 5.3 VCG Allocation — Welfare Maximisation (Sec. 6.2)

When $D_{\text{agg}} = 0$, the mechanism allocates spectrum to maximise *reputation-weighted social welfare*:

$$\mathbf{x}^*(t) = \arg\max_{\mathbf{x} \in \{0,1\}^K} \sum_{k=1}^{K} x_k\,v_k\,\rho_k \quad \text{s.t.} \quad \sum_{k=1}^{K} x_k P_k \leq P_{\text{budget}}$$

This is a 0/1 knapsack problem. The greedy approximation (optimal for the LP relaxation) sorts NCUs by value density:

$$\text{density}_k = \frac{v_k\,\rho_k}{P_k}$$

and allocates greedily in decreasing density order until the power budget is exhausted.

> **Implementation:** `mechanism.py :: vcg_allocate()`

---

## 5.4 VCG Payment — Truthfulness (Sec. 6.2)

The VCG payment for allocated NCU $k$ is:

$$p_k^{\text{VCG}} = \underbrace{\max_{\mathbf{x}_{-k}} \sum_{j \neq k} x_j v_j \rho_j}_{\text{optimal SW without }k} - \underbrace{\sum_{j \neq k} x_j^* v_j \rho_j}_{\text{others' SW in full solution}}$$

**Key property:** VCG payments make truthful bidding the dominant strategy — NCU $k$'s utility $u_k = v_k - p_k^{\text{VCG}}$ is maximised by reporting $\hat{v}_k = v_k$ regardless of other bids.

**Total revenue:**

$$R(t) = \sum_{k:\,x_k^*=1} \bigl(q_k^{\text{DRL}} + p_k^{\text{VCG}}\bigr)$$

where $q_k^{\text{DRL}}$ is the DRL-set base access price.

> **Implementation:** `mechanism.py :: vcg_payment()`

---

## 5.5 Cryptographic Penalty Bonds (Sec. 6.3)

Each NCU posts a bond before the slot begins. Bond requirements are dynamically computed from reputation:

$$B_k(t) = B_{\min}\,(1 + \kappa\,(1 - \rho_k(t)))$$

| Parameter | Value |
|-----------|-------|
| $B_{\min}$ | 10.0 (revenue units) |
| $\kappa$ (penalty scaling) | 2.0 |

### Bond Slashing

When NCU $k$'s committed report contradicts the oracle ($\hat{d}_k = 0$, $D_{\text{ISAC}} = 1$):

$$\text{slash}_k = B_k \cdot \min\!\left(1,\, \frac{C_{\text{ISAC}}}{\eta_{\text{conf}}}\right) \cdot \mathbf{1}[\text{contradiction}_k]$$

The confidence-weighted slash prevents over-penalisation when the oracle itself is uncertain ($C_{\text{ISAC}} < \eta_{\text{conf}} = 2.0$).

**Slashed funds allocation:**
- 50% → treasury (network operator revenue)
- 50% → honest-reporter reward pool

### Honest-Reporter Dividend

NCUs that correctly reported active CU ($\hat{d}_k = 1$ when $D_{\text{ISAC}} = 1$) receive:

$$\omega_k = \frac{\text{pool}}{N_{\text{honest}}}$$  (capped at $0.1 \cdot B_{\min}$ per NCU)

> **Implementation:** `mechanism.py :: BondManager`

---

## 5.6 NCU Utility (Sec. 6.4)

The modified NCU $k$ utility under CVSA:

$$u_k = x_k^*(v_k - q_k^{\text{DRL}} - p_k^{\text{VCG}}) + \omega_k - \text{slash}_k$$

**Individual Rationality (IR):** $u_k \geq 0$ for all truthful NCUs since $p_k^{\text{VCG}} \leq v_k$ (VCG property) and $\omega_k \geq 0$, $\text{slash}_k = 0$ for non-contradicted reporters.

---

## 5.7 Incentive Compatibility — Dominant Strategy Analysis

**Proposition 6.1:** Under CVSA, truthful reporting $\hat{d}_k = d_k$ is a weakly dominant strategy for each NCU.

*Proof sketch:*
1. If $\hat{d}_k = d_k$ (honest): NCU is not slashed; may receive $\omega_k > 0$; VCG payment ensures $u_k \geq 0$.
2. If $\hat{d}_k = 0 \neq d_k = 1$ (falsify active as idle): ZKP proof fails constraint C3 → proof rejected → report discarded. Bond slashed with probability proportional to $C_{\text{ISAC}}$. Reputation decreases.
3. If $\hat{d}_k = 1 \neq d_k = 0$ (falsify idle as active): ZKP proof fails constraint C4 → proof rejected. No gain from falsification (reduces own access probability).

In all falsification cases, expected utility is strictly lower than under truthful reporting. ∎

---

## 5.8 Mathematical Validation Checklist

| Formula | Code | Status |
|---------|------|--------|
| Aggregate decision (normalised weighted average) | `aggregate_decision()` | ✅ |
| VCG allocation (greedy knapsack) | `vcg_allocate()` | ✅ |
| VCG payment $= \max_{x_{-k}} SW_{-k} - SW_{-k}^*$ | `vcg_payment()` | ✅ |
| Bond $B_k = B_{\min}(1+\kappa(1-\rho_k))$ | `BondManager.compute_bonds()` | ✅ |
| Slash $= B_k \cdot \min(1, C/\eta) \cdot \mathbf{1}[\text{contradiction}]$ | `BondManager.slash()` | ✅ |
| Revenue $= \sum x_k^*(q_k^{\text{DRL}} + p_k^{\text{VCG}})$ | `CVSAMechanism.run()` | ✅ |
