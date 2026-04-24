# 08 — Threat Model & Adversary Types

> **Paper sections:** Sec. 3 (Threat Model), Sec. 3.1 (Adversary Types)  
> **Code:** `src/adversary.py`

---

## 8.1 Threat Model

### Attacker Goal

Adversarial NCUs aim to gain **unauthorised spectrum access** by falsely reporting CU absence when the CU is actually present. This causes:

1. Harmful interference to safety-critical CU receivers.
2. Unfair advantage over honest NCUs.
3. Revenue loss for the network operator.

### Attacker Capabilities

- **Network-layer access:** Can send arbitrary messages to the BS.
- **Local sensing:** Has access to its own (possibly modified) $T_k$ and $\lambda_k$.
- **No BS access:** Cannot tamper with ISAC radar, oracle, or other NCUs' sensing hardware.
- **No key access:** Cannot forge other NCUs' cryptographic proofs or commitments.

### Honest-But-Curious BS

The BS is assumed semi-honest: it follows the protocol but may try to infer private NCU valuations from auction bids. VCG payments are designed to be dominant-strategy truthful regardless of BS inference.

---

## 8.2 Adversary Types (Sec. 3.1)

### Type-I: Selfish Falsifier (SF)

$$\hat{d}_k^{\text{SF}} = 0 \quad \forall t$$

**Strategy:** Always report "CU idle" regardless of true measurement.

**Motivation:** Maximise personal throughput by always accessing the spectrum.

**Detectability:** High — consistent reporting of "idle" during confirmed ISAC-active slots triggers immediate contradiction detection.

---

### Type-II: Strategic Threshold Falsifier (STF)

$$\hat{d}_k^{\text{STF}} = \begin{cases} 0 & \text{if } T_k/\lambda_k < \gamma_k \text{ or } (\rho_k > 0.3 \text{ and } q_k > 0.5 B_{\min}) \\ d_k & \text{otherwise} \end{cases}$$

**Strategy:** Falsify only when the sensing SNR is low (hard to prove CU was present) or when the access price is high (incentive to cheat is larger). Each STF adversary has a private risk threshold $\gamma_k \sim \mathcal{U}[0.3, 0.7]$.

**Motivation:** Maximise expected utility by conditioning on detection risk.

**Detectability:** Medium — falsification is correlated with low-SNR slots, which overlap with genuine misdetection events. The ZKP circuit constraint C3 still catches the proof mismatch.

---

### Type-III: Collusive Coalition Falsifier (CCF)

$$\hat{d}_k^{\text{CCF}} = 0 \quad \forall k \in \mathcal{F}, \forall t$$

**Strategy:** All coalition members simultaneously report "CU idle" in every slot (same as SF, but coordinated).

**Motivation:** Coalition coverage of a large fraction of NCUs can overwhelm simple majority vote systems (without the ISAC oracle veto).

**Detectability:** 
- Per-NCU ZKP: individual proof rejection still catches each CCF member.
- Coalition-level: Algorithm 3 (spectral clustering on co-falsification matrix) detects the coordination pattern even if per-slot falsification rates are low.

---

## 8.3 Byzantine Fault Tolerance

**Claim:** VERIDIC-DSA is secure against up to $\lfloor(K-1)/3\rfloor$ Byzantine NCUs.

*Reasoning:* 
- ZKP enforcement limits each adversary to a single (invalid) false proof per slot.
- Oracle veto weight $w_{\text{ISAC}} = 0.5$ ensures that even if all $K$ NCUs falsify, the oracle's positive report ($D_{\text{ISAC}} = 1$) prevents aggregate access ($D_{\text{agg}} = 1$) whenever $D_{\text{ISAC}} = 1$.
- For the aggregation formula, the oracle weight $0.5$ ensures:

$$D_{\text{agg}} = \mathbf{1}\!\left[\frac{\text{sum\_NCU} + 0.5 \cdot D_{\text{ISAC}}}{\text{sum\_rho} + 0.5} \geq 0.5\right]$$

When $D_{\text{ISAC}} = 1$ and all NCUs falsify (report 0), the numerator = $0.5 \cdot 1 = 0.5$, and denominator $= K\bar{\rho} + 0.5$. For $K\bar{\rho} \leq 0.5$, $D_{\text{agg}} = 1$ (access blocked correctly).

---

## 8.4 Adversary Parameters in Simulation

| Parameter | Default | Range |
|-----------|---------|-------|
| Adversary fraction $|\mathcal{F}|/K$ | 0.30 | 0.0 – 0.33 |
| Adversary type | SF | SF, STF, CCF |
| STF threshold $\gamma_k$ | $\sim\mathcal{U}[0.3, 0.7]$ | — |

The experiment sweeps $|\mathcal{F}|/K \in \{0, 0.1, 0.2, 0.3, 0.33\}$ and all three adversary types.

> **Implementation:** `adversary.py :: AdversaryModel.apply()`

---

## 8.5 Adversary Impact on Each Module

| Module | Adversary Effect | VERIDIC-DSA Mitigation |
|--------|-----------------|----------------------|
| Sensing | False $\hat{d}_k = 0$ when CU active | ZKP rejects inconsistent proofs |
| Aggregation | Drive $D_{\text{agg}} \to 0$ when CU active | Oracle veto, reputation weighting |
| Auction | Increased access probability for adversaries | VCG payment + bond slash discourages gain |
| Reputation | Accumulation of false reports | $\beta_k$ penalty degrades future priority |
| Coalition | Coordinate to overcome threshold | Algorithm 3 spectral detection |
