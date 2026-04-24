# 06 — Reputation & Trust Management System

> **Paper sections:** Sec. 9 (Reputation), Sec. 9.1 (Beta-Bernoulli), Sec. 9.2 (Coalition Detection), Sec. 9.3 (Long-term effects)  
> **Code:** `src/reputation.py`

---

## 6.1 Beta-Bernoulli Bayesian Reputation (Sec. 9.1)

Each NCU $k$ maintains a Bayesian reputation score derived from its behavioral history. We model the latent "honesty probability" $\rho_k$ as a Beta-distributed random variable:

$$\rho_k \sim \text{Beta}(\alpha_k, \beta_k)$$

**Posterior mean (reputation score):**

$$\hat{\rho}_k = \frac{\alpha_k}{\alpha_k + \beta_k} \in [0, 1]$$

**Prior:** $\alpha_k^{(0)} = \beta_k^{(0)} = 1$ (uniform prior — equal trust).

### Update Rule

At the end of each slot, given oracle output $D_{\text{ISAC}}$ and NCU report $\hat{d}_k$:

**Contradiction detected** ($\hat{d}_k = 0$, $D_{\text{ISAC}} = 1$):
$$\beta_k \leftarrow \beta_k + \Delta_k, \quad \Delta_k = C_{\text{ISAC}} \cdot \bigl(1 + \xi_{\text{coal}} \cdot \mathbf{1}[k \in \mathcal{C}]\bigr)$$

**Truthful** (no contradiction):
$$\alpha_k \leftarrow \alpha_k + 1$$

| Parameter | Value |
|-----------|-------|
| $\xi_{\text{coal}}$ (coalition amplification) | 2.0 |
| $\alpha_k^{(0)}, \beta_k^{(0)}$ | 1.0 |

The confidence-weighted penalty $\Delta_k \in [0, 3C_{\text{ISAC}}]$ ensures that uncertain oracle readings result in smaller reputation penalties, while confirmed active-CU detections ($C_{\text{ISAC}} \approx 1$) impose the full penalty.

> **Implementation:** `reputation.py :: ReputationModel.update()`

---

## 6.2 Long-Term Reputation Effects (Sec. 9.3)

**Recovery time:** If NCU $k$ accumulates excess $\beta$ from falsification, its estimated recovery time back to neutral reputation (assuming honest behavior from slot $t$ onward) is:

$$T_{\text{recovery},k} \approx \beta_k - \beta_k^{(0)}$$

because each honest slot increments $\alpha_k$ by 1, and the posterior mean converges back to 0.5 after approximately $\Delta\beta_k$ honest slots.

**Impact on auction:** Reputation directly enters the VCG allocation:
- Low $\rho_k$ → NCU's *effective* bid value is $v_k \rho_k$ → lower allocation priority.
- Low $\rho_k$ → higher bond requirement $B_k = B_{\min}(1 + \kappa(1-\rho_k))$.
- Combined effect: repeated falsification leads to exclusion from the auction.

---

## 6.3 Algorithm 3 — Spectral Coalition Detection (Sec. 9.2)

Coalition falsifiers (CCF adversaries) coordinate to all report idle simultaneously. Standard per-NCU detection cannot identify coordination. Algorithm 3 uses spectral graph clustering on the **co-falsification matrix**.

### Step-by-Step

**Input:** $T_w$-slot window of reported decisions $\{\hat{\mathbf{d}}(t)\}$ and oracle decisions $\{D_{\text{ISAC}}(t)\}$.

**Step 1 — Co-falsification matrix:**

$$F_{jk} = \frac{1}{|\{t : D_{\text{ISAC}}(t)=1\}|} \sum_{t:\,D_{\text{ISAC}}(t)=1} \mathbf{1}[\hat{d}_j(t) = 0]\,\mathbf{1}[\hat{d}_k(t) = 0]$$

$F_{jk}$ is the fraction of ISAC-active slots where both NCUs $j$ and $k$ simultaneously reported idle.

**Step 2 — Adjacency matrix:**

$$A_{jk} = \mathbf{1}[F_{jk} > \theta_{\text{collusion}}], \quad A_{jj} = 0$$

with $\theta_{\text{collusion}} = 0.6$.

**Step 3 — Normalised Graph Laplacian:**

$$\mathbf{L} = \mathbf{D}^{-1/2}(\mathbf{D} - \mathbf{A})\mathbf{D}^{-1/2}$$

where $\mathbf{D} = \text{diag}(\mathbf{A}\mathbf{1})$ is the degree matrix.

**Step 4 — Spectral embedding:**

Compute the $m$ eigenvectors corresponding to the $m$ smallest eigenvalues of $\mathbf{L}$, forming the matrix $\mathbf{U} \in \mathbb{R}^{K \times m}$.

**Step 5 — $k$-means clustering:**

Apply Lloyd's algorithm on the rows of $\mathbf{U}$ to cluster NCUs into $m$ groups.

**Step 6 — Suspicion scoring:**

For each cluster, compute the mean co-falsification rate among cluster members. If this rate exceeds $\theta_{\text{suspect}} = 0.5$, the cluster is flagged as a suspected coalition.

**Output:** Coalition mask $\mathcal{C} \subseteq \{1,\ldots,K\}$ (boolean array).

> **Implementation:** `reputation.py :: detect_coalitions()`, `_simple_kmeans()`  
> **Run frequency:** Every 50 slots (`COALITION_WINDOW = 50`)

---

## 6.4 Reputation Integration Across Modules

| Module | How reputation is used |
|--------|----------------------|
| `mechanism.py :: aggregate_decision()` | Reputation weights in vote $\sum \rho_k \hat{d}_k$ |
| `mechanism.py :: vcg_allocate()` | Adjusted bid value $= v_k \rho_k$ |
| `mechanism.py :: BondManager.compute_bonds()` | Bond $B_k \propto (1-\rho_k)$ |
| `drl.py :: PPOAgent.build_state()` | $\bar{\rho}(t)$ enters DRL state vector at index 3 |
| `adversary.py :: AdversaryModel.apply()` | STF adversary conditions on $\rho_k$ |

---

## 6.5 Mathematical Validation Checklist

| Formula | Code | Status |
|---------|------|--------|
| $\hat{\rho}_k = \alpha_k/(\alpha_k+\beta_k)$ | `ReputationModel.rho` property | ✅ |
| $\alpha_k \mathrel{+}= 1$ (truthful) | `ReputationModel.update()` | ✅ |
| $\beta_k \mathrel{+}= C_{\text{ISAC}}(1+\xi\mathbf{1}_{\text{coal}})$ (contradiction) | `ReputationModel.update()` | ✅ |
| $F_{jk}$ co-falsification matrix | `detect_coalitions()` | ✅ |
| Normalised Laplacian $\mathbf{L} = D^{-1/2}(D-A)D^{-1/2}$ | `detect_coalitions()` | ✅ |
| Spectral embedding (smallest $m$ eigenvectors of $\mathbf{L}$) | `detect_coalitions()` | ✅ |
| k-means++ initialisation | `_simple_kmeans()` | ✅ |
