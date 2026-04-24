# 07 — Deep Reinforcement Learning for Dynamic Pricing (PPO)

> **Paper sections:** Sec. 7 (DRL Pricing), Sec. 7.1 (State), Sec. 7.2 (Networks), Sec. 7.3 (Algorithm 2), Sec. 7.4 (Non-stationarity)  
> **Code:** `src/drl.py`

---

## 7.1 Problem Formulation

The dynamic pricing problem is formulated as a **Markov Decision Process (MDP)**:

$$\mathcal{M} = (\mathcal{S},\, \mathcal{A},\, P,\, r,\, \gamma)$$

| Component | Definition |
|-----------|-----------|
| State $s_t \in \mathcal{S} = \mathbb{R}^{12}$ | Observable network statistics |
| Action $a_t \in \mathcal{A} = \{0,\ldots,N_Q-1\}^K$ | Per-NCU price level indices |
| Transition $P(s_{t+1}\|s_t, a_t)$ | Determined by CU channel + NCU responses |
| Reward $r(s_t, a_t)$ | Revenue + welfare − penalties |
| Discount $\gamma$ | 0.99 |

---

## 7.2 State Space (Sec. 7.1)

The 12-dimensional state vector $s_t$ captures sufficient statistics of the spectrum access history:

$$s_t = \left[\bar{D}_{\text{ISAC}}^{(w)},\; \bar{v}^{(w)},\; \tilde{\sigma}_v^2,\; \bar{\rho}^{(w)},\; I_t^{\text{agg}},\; R_t^{\text{prev}},\; N_{\text{active}}(t),\; \Delta\hat{d}(t),\; \hat{\mu}_{\text{strat}},\; \bar{\rho}^{(w)},\; \bar{v}^{(w)},\; \bar{D}_{\text{ISAC}}^{(w)}\right]$$

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | $\bar{D}_{\text{ISAC}}^{(w)}$ | Mean ISAC oracle decision over window $w=10$ |
| 1 | $\bar{v}^{(w)}$ | Mean NCU valuation over window |
| 2 | $\tilde{\sigma}_v^2$ | Variance of NCU valuations over window |
| 3 | $\bar{\rho}^{(w)}$ | Mean NCU reputation over window |
| 4 | $I_t^{\text{agg}}$ | Last-slot aggregate interference |
| 5 | $R_t^{\text{prev}}$ | Last-slot revenue |
| 6 | $N_{\text{active}}(t)$ | Number of allocated NCUs last slot |
| 7 | $\Delta\hat{d}(t)$ | Last-slot falsification detection rate |
| 8 | $\hat{\mu}_{\text{strat}}$ | Window-mean falsification rate (strategy estimate) |
| 9–11 | (padding) | Repeated $\bar{\rho}$, $\bar{v}$, $\bar{D}$ for $|\mathcal{S}|=12$ |

State components are **Z-normalised** online using running mean and standard deviation.

> **Implementation:** `drl.py :: PPOAgent.build_state()`, `normalise_state()`

---

## 7.3 Action Space

The DRL agent outputs per-NCU **discrete price level indices**:

$$a_t = (a_1,\ldots,a_K), \quad a_k \in \{0, 1, \ldots, N_Q - 1\}$$

Price levels are uniformly discretised over $[Q_{\min}, Q_{\max}]$:

$$q_k^{\text{DRL}} = Q_{\min} + a_k \cdot \frac{Q_{\max} - Q_{\min}}{N_Q - 1}$$

| Parameter | Value |
|-----------|-------|
| $Q_{\min}$ | 0.1 |
| $Q_{\max}$ | 5.0 |
| $N_Q$ | 10 |
| Action space size | $N_Q^K = 10^K$ |

---

## 7.4 Reward Function (Eq. 7.3)

$$r_t = R_t + \mu_1\!\sum_{k:\,x_k^*=1}\!\log\!\left(1 + \frac{P_k}{\sigma_k^2}\right) - \mu_2\,\mathbf{1}[I_{\text{harm}} > I_{\max}] - \mu_3\!\sum_k \mathbf{1}[\text{falsification}_k]$$

| Term | Weight | Meaning |
|------|--------|---------|
| $R_t$ | 1.0 | Auction revenue |
| $\mu_1\,\text{SW}$ | 0.3 | Social welfare (log-sum Shannon capacity) |
| $-\mu_2\,\mathbf{1}[I > I_{\max}]$ | −10.0 | Hard interference penalty |
| $-\mu_3\,\sum_k \mathbf{1}[\text{fals}]$ | −5.0 | Falsification discouragement |

The joint optimisation of revenue and social welfare implements a tunable point on the Pareto frontier.

---

## 7.5 Network Architectures (Sec. 7.2)

### Actor Network $\pi_\theta(a|s)$

```
LayerNorm(12) → Linear(12→64, ReLU) → ResidualBlock(64) →
Linear(64→128, ReLU) → Linear(128→64, ReLU) → ResidualBlock(64) →
Linear(64 → K × N_Q)  →  reshape(K, N_Q)  →  Categorical per NCU
```

**Residual block:** $x \leftarrow x + \text{ReLU}(\text{Linear}(x))$

### Critic Network $V_\phi(s)$

```
Linear(12→128, ReLU) → Linear(128→64, ReLU) → Linear(64→1)
```

Both networks are trained on CUDA (or MPS / CPU) via Adam with gradient clipping at 0.5.

---

## 7.6 Algorithm 2 — PPO Update (Sec. 7.3)

**Input:** Replay buffer $\mathcal{B}$ of $T$ transitions $(s_t, a_t, r_t, s_{t+1}, \text{done}_t)$.

**Step 1:** Compute value estimates $V_\phi(s_t)$, $V_\phi(s_{t+1})$.

**Step 2 — GAE-λ Advantage Estimation:**

$$\hat{A}_t = \sum_{\ell=0}^{T-t-1} (\gamma\lambda)^\ell \delta_{t+\ell}, \quad \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

**Step 3:** Normalise advantages: $\hat{A}_t \leftarrow (\hat{A}_t - \mu_A)/(\sigma_A + \epsilon)$.

**Step 4:** Compute return targets: $\hat{G}_t = \hat{A}_t + V_\phi(s_t)$.

**Steps 5–10** (for $K_{\text{epochs}} = 4$ epochs, mini-batches of 128):

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

$$\mathcal{L}_{\text{CLIP}} = \mathbb{E}_t\!\left[\min\!\left(r_t \hat{A}_t,\; \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right]$$

$$\mathcal{L}_{\text{VF}} = \mathbb{E}_t\!\left[(V_\phi(s_t) - \hat{G}_t)^2\right]$$

$$\mathcal{L}_{\text{ENT}} = \mathbb{E}_t\!\left[\mathcal{H}[\pi_\theta(\cdot|s_t)]\right]$$

$$\mathcal{L}_{\text{total}} = -\mathcal{L}_{\text{CLIP}} + c_1 \mathcal{L}_{\text{VF}} - c_2 \mathcal{L}_{\text{ENT}}$$

**Step 11:** Early stopping if $\text{KL}(\pi_{\theta_{\text{old}}}, \pi_\theta) > \delta_{\text{KL}} = 0.05$.

| Hyperparameter | Symbol | Value |
|----------------|--------|-------|
| Discount factor | $\gamma$ | 0.99 |
| GAE-λ | $\lambda$ | 0.95 |
| Clip parameter | $\varepsilon$ | 0.2 |
| VF coefficient | $c_1$ | 0.5 |
| Entropy coefficient | $c_2$ | 0.01 |
| Actor learning rate | $\eta_\pi$ | 3×10⁻⁴ |
| Critic learning rate | $\eta_V$ | 1×10⁻³ |
| Update frequency | — | 200 slots |
| Mini-batch size | — | 128 |
| PPO epochs | $K_{\text{epochs}}$ | 4 |
| KL threshold | $\delta_{\text{KL}}$ | 0.05 |

> **Implementation:** `drl.py :: PPOAgent.update()`

---

## 7.7 Non-Stationarity Handling (Sec. 7.4)

CU duty-cycle can shift abruptly (e.g., radar activation). The PPO agent adapts via:

1. **Sliding window replay buffer** of size $T_{\text{window}} = 500$ slots. Old transitions are automatically discarded as newer data arrives.
2. **Frequent PPO updates** every 200 slots ensure rapid adaptation.
3. **State statistics re-centering** via online mean/std estimation with $\alpha = 0.01$ exponential smoothing.

The non-stationarity stress test shifts $\lambda_{\text{on}}: 0.10 \to 0.25$ and $\lambda_{\text{off}}: 0.233 \to 0.58$ at slot 500, doubling the CU duty cycle.

---

## 7.8 GPU / Mixed-Precision Support (Colab Optimization)

The PPO agent automatically uses the best available hardware:

```python
DEVICE = get_device()  # CUDA > MPS > CPU
```

On CUDA (Google Colab T4/A100), automatic mixed-precision (AMP) with `torch.amp.GradScaler` accelerates training by ~2–3× with negligible loss in numerical precision.

Checkpoints are saved/loaded via `PPOAgent.save_checkpoint()` / `load_checkpoint()` to support Colab session interruption recovery.

---

## 7.9 Mathematical Validation Checklist

| Formula | Code | Status |
|---------|------|--------|
| GAE: $\delta_t + \gamma\lambda\hat{A}_{t+1}$ (reverse iteration) | `PPOAgent.update()` | ✅ |
| PPO ratio $r_t = \exp(\log\pi_\theta - \log\pi_{\theta_{\text{old}}})$ | `PPOAgent.update()` | ✅ |
| CLIP objective $\min(r_t A, \text{clip}(r_t,1\pm\varepsilon)A)$ | `PPOAgent.update()` | ✅ |
| Total loss $-L_{\text{CLIP}} + c_1 L_{\text{VF}} - c_2 L_{\text{ENT}}$ | `PPOAgent.update()` | ✅ |
| Grad norm clipping at 0.5 | `PPOAgent.update()` | ✅ |
| Early stopping on KL > $\delta_{\text{KL}}$ | `PPOAgent.update()` | ✅ |
| Price grid $q_k = Q_{\min} + a_k(Q_{\max}-Q_{\min})/(N_Q-1)$ | `PPOAgent.__init__()` | ✅ |
