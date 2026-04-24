# 01 — System Overview

> **Paper sections:** Sec. 1 (Introduction), Sec. 2 (System Model)  
> **Code:** `src/config.py`, `src/experiment.py`

---

## 1.1 The Core Problem

In 6G networks operating above 100 GHz (sub-THz / D-band), vast spectral resources are reserved for **Critical Users (CUs)** — entities such as air-traffic control (ATC) radars, military MIMO systems, and weather-sensing infrastructure. These users operate intermittently, leaving large spectral voids.

**Non-Critical Users (NCUs)** — mobile broadband subscribers, IoT platforms, V2X terminals — are permitted opportunistic secondary access when CUs are silent. This is the Dynamic Spectrum Access (DSA) paradigm.

### The Fundamental Vulnerability

The canonical DSA protocol asks NCUs to *self-report* their sensing observations to the network controller. An economically rational NCU faces a **strictly dominated strategy**: falsify "idle" regardless of true CU activity, since doing so maximises throughput and minimises access costs. The consequences if the network is deceived:

1. **Interference catastrophe** at safety-critical CU receivers.
2. **Revenue misallocation** under falsified demand signals.
3. **System fairness collapse** as dishonest NCUs outcompete honest ones.

---

## 1.2 Formal System Model

**Definition 2.1 (Network):**

$$\mathcal{N} = \langle \mathcal{B},\, \mathcal{U}_C,\, \mathcal{U}_N,\, \mathcal{F},\, \mathcal{T} \rangle$$

| Symbol | Meaning |
|--------|---------|
| $\mathcal{B}$ | ISAC Base Station (oracle + auctioneer) |
| $\mathcal{U}_C$ | Set of Critical Users (CU), protected primary incumbents |
| $\mathcal{U}_N = \{1,\ldots,K\}$ | Set of Non-Critical Users (NCU), $K$ secondary access seekers |
| $\mathcal{F} \subset \mathcal{U}_N$ | Adversary set (falsifiers) |
| $\mathcal{T} = \{0,1,2,\ldots\}$ | Discrete time-slot index |

---

## 1.3 CU Activity Model

The CU occupies the licensed band intermittently, modelled as a **semi-Markov two-state process** (Sec. 2.5):

$$H_t \in \{0, 1\}, \quad H_t = 1 \text{ (CU active)},\quad H_t = 0 \text{ (CU idle)}$$

State transition matrix per slot:

$$\mathbf{P} = \begin{pmatrix} 1-\lambda_{\text{on}} & \lambda_{\text{on}} \\ \lambda_{\text{off}} & 1-\lambda_{\text{off}} \end{pmatrix}$$

**Steady-state CU activity probability:**

$$\pi_1 = \frac{\lambda_{\text{on}}}{\lambda_{\text{on}} + \lambda_{\text{off}}} = \frac{0.10}{0.10 + 0.233} \approx 0.30$$

> **Implementation:** `src/channel.py :: CUActivityModel`  
> **Parameters:** `cfg.CU_LAMBDA_ON = 0.10`, `cfg.CU_LAMBDA_OFF = 0.233`

---

## 1.4 The VERIDIC-DSA Framework — Five Pillars

| Pillar | Innovation | Code Module |
|--------|-----------|-------------|
| **ISAC Oracle** | BS performs simultaneous radar sensing to obtain independent ground truth, cryptographically bound per slot | `src/oracle.py` |
| **Commit-Reveal Sensing Protocol** | NCUs submit sensing decisions as Pedersen Commitments *before* the oracle publishes, preventing post-hoc falsification | `src/crypto.py` |
| **ZKP-Verified Sensing** | NCUs prove via Groth16 zk-SNARKs that their committed decision was derived from a measurement above a certified noise floor | `src/crypto.py` |
| **CVSA Mechanism** | VCG-augmented second-price auction with cryptographic penalty bonds ensures truthful reporting is the unique dominant strategy | `src/mechanism.py` |
| **PPO Revenue Engine** | Proximal Policy Optimisation agent dynamically prices spectrum slots, jointly maximising operator revenue and social welfare | `src/drl.py` |

---

## 1.5 Claimed Contributions

**C1.** First formulation of spectrum sensing falsification as a **Bayesian Mechanism Design** problem with cryptographic enforcement in 6G ISAC networks.

**C2.** First application of **Pedersen Commitments + Groth16 zk-SNARKs** to physical-layer sensing reports, establishing a cryptographically binding audit trail without revealing I/Q measurements.

**C3.** Proof that the proposed auction mechanism is **incentive-compatible (IC), individually rational (IR), and ex-post budget-balanced** even when up to $\lfloor(K-1)/3\rfloor$ NCUs are Byzantine adversaries.

**C4.** A **PPO-based dynamic pricing agent** that achieves Pareto-optimal revenue–welfare tradeoff with convergence guarantees under non-stationary CU activity.

**C5.** End-to-end simulations demonstrating **≥ 94.7% detection accuracy** against falsification attacks, **31% revenue gain** over static pricing, and **< 0.1% harmful interference probability** to CUs.

---

## 1.6 Seven-Phase Per-Slot Protocol (Algorithm 1)

```
Each time slot t:
 Phase 1 │ NCU sensing + Pedersen commitment + Groth16 proof + bond posting
 Phase 2 │ BS verifies ZK proofs (O(1) per NCU)
 Phase 3 │ ISAC oracle publishes D_ISAC, C_ISAC, Λ_ISAC
 Phase 4 │ NCUs reveal (d_k, r_k); BS audits via commitment binding check
          │  → Contradiction detected → bond slashing + reputation penalty
 Phase 5 │ PPO DRL agent selects per-NCU access prices q_k
 Phase 6 │ CVSA: aggregate decision D_agg → VCG allocation x* → payments
 Phase 7 │ Allocated NCUs transmit; reward computed; PPO buffer updated
```

> **Implementation:** `src/experiment.py :: VeridicDSA.run_slot()`

---

## 1.7 Network Topology and Distances

| Link | Distance |
|------|----------|
| BS → NCU (communication) | 200 m |
| CU → NCU (sensing, signal source) | 100 m |
| NCU → CU receiver (interference) | 600 m |
| BS → CU (radar sensing target) | 1 000 m |

> These distances are set in `src/config.py` as `BS_TO_NCU_DIST_M`, `CU_TO_NCU_DIST_M`, `NCU_TO_CU_DIST_M`, `BS_TO_CU_DIST_M`.
