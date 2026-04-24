# 03 — Cryptographic Sensing Protocol (CSP)

> **Paper sections:** Sec. 5 (CSP), Sec. 5.1 (Pedersen Commitment), Sec. 5.2 (Groth16 ZKP), Sec. 12.3 (Implementation)  
> **Code:** `src/crypto.py`, `src/sensing.py`

---

## 3.1 Energy Detection (Sec. 4.1)

Each NCU $k$ independently performs spectrum sensing over $N_s = 1000$ complex samples per slot.

### Energy Test Statistic

$$T_k = \frac{1}{N_s} \sum_{i=1}^{N_s} |y_k(i)|^2$$

### Neyman-Pearson Detection Threshold

Under $H_0$ (CU idle), $T_k \cdot N_s / \sigma_k^2 \sim \chi^2(2N_s)$ (central). By the Neyman-Pearson Lemma, the threshold controlling false alarm to $\alpha_{\max} = 0.05$ is:

$$\lambda_k = \sigma_k^2 \left(1 + \sqrt{\frac{2}{N_s}} \cdot Q^{-1}(\alpha_{\max})\right)$$

where $Q^{-1}(p) = \sqrt{2}\,\text{erfinv}(1-2p)$.

**Honest binary sensing decision:**

$$d_k = \mathbf{1}[T_k \geq \lambda_k]$$

> **Implementation:** `sensing.py :: detection_threshold()`, `energy_statistic()`, `sensing_decision()`  
> **Parameters:** `cfg.N_SAMPLES = 1000`, `cfg.ALPHA_MAX = 0.05`

---

## 3.2 Pedersen Commitment (Sec. 5.1)

Before the ISAC oracle publishes $D_{\text{ISAC}}$, each NCU $k$ **commits** to its sensing decision $d_k$ using a Pedersen commitment scheme.

### Commitment Construction

$$\text{com}_k = g^{d_k} \cdot h^{r_k} \bmod p$$

In the simulation (256-bit safe-prime group proxy via SHA-256):

$$\text{com}_k = \text{SHA-256}(d_k \,\|\, r_k)$$

where $r_k \xleftarrow{\$} \{0,1\}^{256}$ is a uniformly random blinding factor.

### Security Properties

| Property | Guarantee |
|----------|-----------|
| **Binding** | Hash collision resistance $\Rightarrow$ NCU cannot change $d_k$ after commit |
| **Hiding** | $r_k$ is uniformly random $\Rightarrow$ $\text{com}_k$ is uniformly distributed (IND-CPA) |

> **Implementation:** `crypto.py :: PedersenCommitment.commit()`, `.verify()`  
> **Size:** 256 bytes per commitment (`cfg.COMMITMENT_SIZE_BYTES = 256`)

---

## 3.3 Groth16 ZK-SNARK — SensingVerification Circuit (Sec. 5.2)

After committing, NCU $k$ proves — **without revealing** $T_k$ or $r_k$ — that its commitment is *consistent* with a valid sensing measurement. This is the **SensingVerification** statement:

$$\exists\,(T_k, r_k) \text{ such that:}$$

| # | Constraint | Meaning |
|---|-----------|---------|
| C1 | $\text{com}_k = H(d_k, r_k)$ | Commitment is valid |
| C2 | $d_k \in \{0,1\}$ | Binary decision |
| C3 | $d_k = 1 \Rightarrow T_k \geq \lambda_k$ | Active detection is real |
| C4 | $d_k = 0 \Rightarrow T_k < \lambda_k$ | Idle detection is real |
| C5 | $|T_k - T_k^{\text{prev}}| \leq \Delta_T$ | Temporal smoothness (anti-replay) |

where $\Delta_T = 5\lambda_k$ is the temporal smoothness bound.

### Proof/Verify Complexity

| Operation | Time | Size |
|-----------|------|------|
| Groth16 Prover | ~120 ms | 192 bytes |
| Groth16 Verifier | ~1.5 ms | O(1) pairing check |

The verifier time is **constant** (2 pairing equations) regardless of circuit size — a key advantage for the BS which verifies $K$ proofs per slot.

> **Implementation:** `crypto.py :: Groth16Simulator.prove()`, `.verify()`  
> **Timing:** `cfg.ZKP_PROVE_TIME_MS = 120.0`, `cfg.ZKP_VERIFY_TIME_MS = 1.5`

---

## 3.4 Adversarial Proof Behaviour

When an adversarial NCU $k$ reports $\hat{d}_k \neq d_k^{\text{true}}$:
- It cannot satisfy constraint C3 or C4 (the measurement $T_k$ contradicts its report).
- It must fabricate a proof with random bytes.
- The BS verifier **always rejects** fabricated proofs (circuit constraints fail).
- Rejected NCU is treated as non-participating: $\hat{d}_k \leftarrow 0$.

---

## 3.5 Commit-Reveal Timeline

```
t₀  NCUs sense, compute T_k, d_k
t₁  NCUs commit: com_k = H(d_k, r_k), generate ZKP, post bond B_k
t₂  BS verifies ZK proofs (Phase 2)
t₃  ISAC oracle publishes D_ISAC, C_ISAC (Phase 3)
t₄  NCUs reveal (d_k, r_k) — cannot change after t₁ (binding property)
t₅  BS audits: checks com_k = H(d_k, r_k), flags contradictions (Phase 4)
```

The commit at $t_1$ *before* the oracle publishes at $t_3$ is the critical ordering that prevents post-hoc falsification: NCUs cannot change their report after seeing the oracle's verdict.

---

## 3.6 Per-NCU Communication Overhead (Sec. 15.3)

| Message | Size |
|---------|------|
| Pedersen commitment | 256 bytes |
| Groth16 proof | 192 bytes |
| Bond confirmation | 64 bytes |
| Reveal (d_k, r_k) | 64 bytes |
| **Total per NCU per slot** | **576 bytes** |

For $K = 10$ NCUs and $\tau = 10\,\text{ms}$ slots: aggregate overhead = 57.6 KB/s (negligible vs. 100 MHz bandwidth).

---

## 3.7 Production Deployment Notes (Sec. 12.3)

For a production deployment, replace the SHA-256 simulation with:

- **Elliptic curve group:** BN254 or BLS12-381 (128-bit security)
- **Circuit compiler:** circom2
- **Prover backend:** snarkjs (JavaScript) or bellman (Rust)
- **On-chain verification:** Ethereum-compatible Solidity verifier (2 pairing ops, ~300k gas)

The 2048-bit safe-prime Pedersen group is replaced by the BN254 elliptic curve group for production (`cfg.PEDERSEN_BITS` is a simulation speed proxy).

---

## 3.8 Mathematical Validation Checklist

| Formula | Code | Status |
|---------|------|--------|
| $T_k = (1/N_s)\sum|y_k(i)|^2$ | `energy_statistic()` | ✅ |
| $Q^{-1}(p) = \sqrt{2}\,\text{erfinv}(1-2p)$ | `_q_inv()` | ✅ |
| $\lambda_k = \sigma_k^2(1+\sqrt{2/N_s} Q^{-1}(\alpha))$ | `detection_threshold()` | ✅ |
| C3: $d_k=1 \Rightarrow T_k \geq \lambda_k$ | `Groth16Simulator.prove()` | ✅ |
| C4: $d_k=0 \Rightarrow T_k < \lambda_k$ | `Groth16Simulator.prove()` | ✅ |
| C5: $\|T_k - T_k^{\text{prev}}\| \leq 5\lambda_k$ | `Groth16Simulator.prove()` | ✅ |
| Commitment binding: $H(d_k, r_k) = \text{com}_k$ | `PedersenCommitment.verify()` | ✅ |
