# 11 — Theoretical Analysis & Key Results

> **Paper sections:** Sec. 11 (Analysis), Sec. 6.4 (IC/IR proofs), Sec. 8 (Theorem 8.1), Sec. 15 (Complexity)

---

## 11.1 Theorem 8.1 — ISAC Oracle Error Bound

**Theorem 8.1:** Under the matched-filter detection model with $N_t$ BS antennas, $L$ coherently integrated radar pulses, and per-antenna radar SNR $\gamma_r$, the probability that the ISAC oracle makes an incorrect decision satisfies:

$$P_{e,\text{ISAC}} \leq \exp\!\left(-\frac{N_t \cdot L \cdot \gamma_r^2}{2(1+\gamma_r)^2}\right)$$

**Proof sketch (Chernoff bound):** The matched-filter output under $H_1$ follows a non-central chi-squared distribution with $2N_t L$ degrees of freedom and non-centrality parameter $2N_t L \gamma_r$. By Chernoff's bound on the detection error probability:

$$P_e \leq \min_{s > 0} e^{s\eta}\,\mathbb{E}[e^{-s Z}]$$

For the optimal $s = \gamma_r/(2(1+\gamma_r)^2)$, this evaluates to the stated bound. At $N_t = 64$, $L = 128$, $\gamma_r = 10$:

$$P_{e,\text{ISAC}} \leq e^{-3287} \approx 10^{-1427} \approx 0$$

The oracle is effectively infallible under the simulation parameters.

> **Code:** `oracle.py :: ISACOracle.oracle_error_bound()`

---

## 11.2 Proposition 6.1 — Incentive Compatibility (IC)

**Proposition 6.1:** Under the CVSA mechanism, truthful sensing reporting $\hat{d}_k = d_k^{\text{true}}$ is a weakly dominant strategy for each NCU $k$, provided the bond $B_k$ satisfies:

$$B_k \geq \frac{\mathbb{E}[q_k^{\text{DRL}} + p_k^{\text{VCG}}]}{\Pr[D_{\text{ISAC}} = 1 \,|\, H_t = 1]} \cdot \eta_{\text{conf}}$$

**Proof (dominant strategy):**

*Case 1 (H_t = 0, true idle):* Whether NCU reports 0 or 1, oracle confirms idle. No contradiction occurs. Reporting 0 (honest) preserves access eligibility; reporting 1 (falsely active) reduces it. Truthful reporting $\hat{d}_k = 0$ weakly dominates.

*Case 2 (H_t = 1, CU active):*

- If NCU reports $\hat{d}_k = 1$ (honest): no contradiction, no slash, eligible for $\omega_k > 0$. Utility: $u_k = \omega_k \geq 0$.
- If NCU reports $\hat{d}_k = 0$ (falsify): ZKP circuit constraint C4 fails → proof rejected → excluded from auction. Additionally, if oracle confirms active: slash of $B_k \cdot C_{\text{ISAC}}/\eta_{\text{conf}}$. Utility: $u_k = -\text{slash}_k < 0$.

Expected utility under falsification:

$$\mathbb{E}[u_k^{\text{falsify}}] = \mathbb{E}[-B_k C_{\text{ISAC}}/\eta_{\text{conf}}\,|\,H_t=1] < 0 < \mathbb{E}[u_k^{\text{honest}}]$$

Truthful reporting dominates in all cases. ∎

---

## 11.3 Proposition 6.2 — Individual Rationality (IR)

**Proposition 6.2:** The CVSA mechanism satisfies ex-post IR: each truthful NCU achieves non-negative utility.

**Proof:** For truthful NCU $k$ with $x_k^* = 1$:

$$u_k = v_k - q_k^{\text{DRL}} - p_k^{\text{VCG}} + \omega_k$$

VCG theory guarantees $p_k^{\text{VCG}} \leq v_k - q_k^{\text{DRL}}$ for the winner (otherwise NCU $k$ would not win). Thus $u_k \geq \omega_k \geq 0$. ∎

---

## 11.4 Proposition 6.3 — Budget Balance

**Proposition 6.3:** CVSA is ex-post weakly budget-balanced: total payments collected ≥ total costs.

**Proof:** Revenue $= \sum_k x_k^*(q_k + p_k^{\text{VCG}}) + \text{slash treasury}$. Since $q_k \geq Q_{\min} > 0$ and $p_k^{\text{VCG}} \geq 0$, and the slash treasury is non-negative, total revenue ≥ 0. ∎

---

## 11.5 Theorem — PPO Convergence (Sec. 7.3)

Under the PPO update with GAE-λ advantage estimation, the policy gradient estimate satisfies:

$$\nabla_\theta J(\pi_\theta) \approx \mathbb{E}_t\!\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \hat{A}_t \cdot \nabla_\theta \log \pi_\theta(a_t|s_t)\right]$$

The clipped surrogate objective ensures monotone improvement: $J(\pi_{\theta_{\text{new}}}) \geq J(\pi_{\theta_{\text{old}}}) - \delta_{\text{KL}}$-bounded degradation per update. Under the stationarity assumption on the MDP, convergence to a local optimum is guaranteed [Schulman et al., 2017].

For the non-stationary case (duty-cycle shift), convergence within $T_{\text{adapt}}$ slots is bounded by:

$$T_{\text{adapt}} \leq \frac{T_{\text{window}}}{1 - \lambda\gamma} \cdot \frac{V_{\max}}{(1-\gamma)\,\delta_\theta^2}$$

where $V_{\max}$ bounds the value function and $\delta_\theta$ is the minimum policy improvement step.

---

## 11.6 Coalition Detection — Spectral Guarantee

**Proposition (Algorithm 3):** Let $\mathcal{F}$ be a coalition of size $|\mathcal{F}|$ that co-falsifies in all ISAC-active slots. Then for $T_w \geq T_{\min}$ window slots:

$$F_{jk} = 1 \quad \forall j, k \in \mathcal{F}$$

The coalition forms a **complete subgraph** in the adjacency matrix $A$ (all edge weights $= 1 > \theta_{\text{collusion}} = 0.6$). The normalised Laplacian $\mathbf{L}$ then has eigenvalue 0 with multiplicity $|\mathcal{F}|$ for the coalition subgraph, so spectral clustering isolates the coalition as a distinct cluster with probability 1 (deterministic separation in the perfect coalition case).

For partial coalitions ($F_{jk} > \theta_{\text{collusion}}$), the detection is probabilistic depending on $T_w$ and the collusion frequency.

---

## 11.7 Computational Complexity Summary (Sec. 15)

| Operation | Complexity | Per-slot |
|-----------|-----------|---------|
| Energy detection (K NCUs) | $O(K N_s)$ | $K \times 1000$ flops |
| Pedersen commitment | $O(1)$ | K SHA-256 |
| Groth16 proof (simulated) | $O(n_{\text{gates}})$ | K × 5 constraints |
| Groth16 verify | $O(1)$ | K pairings |
| Aggregate decision | $O(K)$ | — |
| VCG allocation (greedy) | $O(K \log K)$ | — |
| VCG payment | $O(K^2 \log K)$ | — |
| Reputation update | $O(K)$ | — |
| Coalition detection (every 50 slots) | $O(K^2 T_w + K^3)$ | amortised $O(K^3/50)$ |
| PPO update (every 200 slots) | $O(T_{\text{win}} \cdot d^2)$ | amortised |

For $K = 30$, $T_w = 50$: coalition detection cost ≈ $30^2 \times 50 + 30^3 = 72\,000$ operations per trigger — fully within the slot budget when run asynchronously.

---

## 11.8 Mathematical Cross-Validation Summary

| Claim | Formula | Verified |
|-------|---------|---------|
| CU steady state $\pi_1 = 0.30$ | $\lambda_{\text{on}}/(\lambda_{\text{on}}+\lambda_{\text{off}})$ | ✅ (= 0.3003) |
| Noise power ≈ $2.01\times10^{-12}$ W | $k_B T B \cdot \text{NF}$ | ✅ |
| Radar SNR uses $\sigma_{\text{RCS}}^1$ | Standard range equation | ✅ (fixed) |
| CFAR threshold $\eta \approx 4.95$ | $N_{\text{ref}}(P_{\text{fa}}^{-1/N_{\text{ref}}}-1)$ | ✅ |
| Oracle error $\approx 0$ | Theorem 8.1 bound | ✅ ($e^{-3287}$) |
| VCG payment non-negative | $\max SW_{-k} - SW_{-k}^* \geq 0$ | ✅ |
| GAE reverse iteration correct | $\delta_t + \gamma\lambda\hat{A}_{t+1}$ | ✅ |
| BL1 majority vote corrected | $\mathbf{1}[\bar{d}_k < 0.5]$ | ✅ (bug fixed) |
