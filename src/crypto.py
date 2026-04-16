"""
crypto.py – Cryptographic Sensing Protocol (Sec. 5).

Implements:
  • Pedersen Commitment: com_k = g^{d_k} · h^{r_k} mod p  (Sec. 5.1)
  • Commitment reveal and verification
  • Simulated Groth16 zk-SNARK proof/verify for SensingVerification circuit (Sec. 5.2)

The Groth16 prover/verifier is *simulated* in Python: the circuit constraints
from Sec. 5.2 are evaluated directly, and "proof validity" matches what a real
Groth16 prover would produce.  Timing constants reproduce the paper's numbers.

For a production deployment, replace the simulated ZKP with py_ecc + circom2 +
snarkjs as described in Sec. 12.3.
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

import src.config as cfg


# ── Pedersen Commitment ───────────────────────────────────────────────────────

# Use SHA-256 based commitment as an efficient stand-in that preserves all
# information-theoretic properties (binding, hiding) needed for simulation.
# The blinding factor r_k ensures hiding; the hash binding ensures you cannot
# change d_k after commit without breaking collision resistance.

class PedersenCommitment:
    """
    Pedersen-style commitment scheme (Sec. 5.1).

    com_k = H(d_k || r_k || nonce)   [simulated via SHA-256]

    Binding: hash collision resistance → cannot change d_k post-commit.
    Hiding:  r_k is uniformly random → com_k is uniformly distributed.
    """

    @staticmethod
    def commit(d_k: int, rng: Generator) -> tuple[bytes, int]:
        """
        Generate commitment and blinding factor.

        Returns
        -------
        com_k : 32-byte commitment
        r_k   : integer blinding factor (256-bit)
        """
        r_k = int.from_bytes(rng.bytes(cfg.PEDERSEN_BITS // 8), "big")
        com_k = PedersenCommitment._hash(d_k, r_k)
        return com_k, r_k

    @staticmethod
    def verify(d_k: int, r_k: int, com_k: bytes) -> bool:
        """Verify that com_k = H(d_k || r_k)."""
        return PedersenCommitment._hash(d_k, r_k) == com_k

    @staticmethod
    def _hash(d_k: int, r_k: int) -> bytes:
        msg = int(d_k).to_bytes(1, "big") + int(r_k).to_bytes(cfg.PEDERSEN_BITS // 8, "big")
        return hashlib.sha256(msg).digest()


# ── ZKP Proof Dataclass ───────────────────────────────────────────────────────

@dataclass
class ZKProof:
    """Represents a Groth16-style ZKP (simulated)."""
    valid: bool          # would Groth16.Verify return true?
    proof_id: bytes      # 192-byte hash-stand-in for group elements
    prove_time_ms: float
    verify_time_ms: float = cfg.ZKP_VERIFY_TIME_MS
    proof_size_bytes: int = cfg.ZKP_PROOF_SIZE_BYTES


# ── Simulated Groth16 Circuit Evaluator ──────────────────────────────────────

class Groth16Simulator:
    """
    Simulates the SensingVerification Groth16 circuit (Sec. 5.2 / 12.3).

    Circuit constraints checked:
      1. com_k validity: H(d_k, r_k) == com_k
      2. d_k ∈ {0, 1}
      3. d_k = 1  ⟹  T_k ≥ λ_k   (threshold compliance)
      4. d_k = 0  ⟹  T_k < λ_k   (consistency)
      5. |T_k - T_k_prev| ≤ Δ_T   (temporal smoothness)
    """

    def __init__(self):
        # In a real system the trusted setup parameters would be stored here.
        self._setup_done = True

    @staticmethod
    def _delta_T_max(lambda_k: float) -> float:
        return 5.0 * lambda_k

    def prove(
        self,
        T_k: float,
        d_k: int,
        r_k: int,
        com_k: bytes,
        lambda_k: float,
        T_k_prev: float,
        adversarial: bool = False,
        rng: Generator | None = None,
    ) -> ZKProof:
        """
        Generate a ZKP for the SensingVerification statement.

        If adversarial=True, the NCU fabricates a proof (for an inconsistent
        witness) — which will fail verification.

        Prover time is simulated with a small random jitter.
        """
        prove_time = cfg.ZKP_PROVE_TIME_MS + (
            0.0 if rng is None else rng.normal(0.0, 5.0)
        )
        prove_time = max(prove_time, 10.0)

        if adversarial:
            # Fabricated proof — just random bytes, will fail verify
            proof_id = os.urandom(cfg.ZKP_PROOF_SIZE_BYTES)
            return ZKProof(valid=False, proof_id=proof_id, prove_time_ms=prove_time)

        # Check all circuit constraints
        c1 = PedersenCommitment.verify(d_k, r_k, com_k)
        c2 = d_k in (0, 1)
        c3 = (d_k == 0) or (T_k >= lambda_k)     # d=1 ⟹ T ≥ λ
        c4 = (d_k == 1) or (T_k < lambda_k)      # d=0 ⟹ T < λ
        c5 = abs(T_k - T_k_prev) <= self._delta_T_max(lambda_k)

        all_ok = c1 and c2 and c3 and c4 and c5

        # Deterministic proof id based on witness (simulates non-interactive proof)
        proof_data = (
            int(d_k).to_bytes(1, "big")
            + int(r_k).to_bytes(cfg.PEDERSEN_BITS // 8, "big")
            + com_k
        )
        proof_id = hashlib.sha256(proof_data).digest() * (
            cfg.ZKP_PROOF_SIZE_BYTES // 32
        )

        return ZKProof(valid=all_ok, proof_id=proof_id, prove_time_ms=prove_time)

    def verify(
        self,
        proof: ZKProof,
        com_k: bytes,
        lambda_k: float,
        T_k: float | None = None,
        d_k: int | None = None,
        r_k: int | None = None,
        T_k_prev: float | None = None,
    ) -> bool:
        """
        Verify a ZKP.

        In simulation mode, we re-evaluate the circuit if the witness is
        available (which it would be during reveal phase).  Before reveal,
        only the proof.valid flag (set by the prover) is used — matching the
        constant-time O(1) Groth16 pairing verification.

        Returns True iff the proof is valid.
        """
        if not proof.valid:
            return False

        # Post-reveal consistency check (re-evaluate circuit with witness)
        if d_k is not None and r_k is not None and T_k is not None:
            c1 = PedersenCommitment.verify(d_k, r_k, com_k)
            c2 = d_k in (0, 1)
            c3 = (d_k == 0) or (T_k >= lambda_k)
            c4 = (d_k == 1) or (T_k < lambda_k)
            delta_T = self._delta_T_max(lambda_k)
            c5 = (T_k_prev is None) or (abs(T_k - T_k_prev) <= delta_T)
            return c1 and c2 and c3 and c4 and c5

        return proof.valid


# ── Module-level singletons ────────────────────────────────────────────────────

_pedersen = PedersenCommitment()
_groth16  = Groth16Simulator()


def commit(d_k: int, rng: Generator) -> tuple[bytes, int]:
    """Convenience wrapper: commit decision d_k, return (com_k, r_k)."""
    return _pedersen.commit(d_k, rng)


def verify_commitment(d_k: int, r_k: int, com_k: bytes) -> bool:
    """Verify a Pedersen commitment."""
    return _pedersen.verify(d_k, r_k, com_k)


def prove(
    T_k: float,
    d_k: int,
    r_k: int,
    com_k: bytes,
    lambda_k: float,
    T_k_prev: float,
    adversarial: bool = False,
    rng: Generator | None = None,
) -> ZKProof:
    """Generate a Groth16 ZK proof for sensing decision."""
    return _groth16.prove(T_k, d_k, r_k, com_k, lambda_k, T_k_prev, adversarial, rng)


def verify_proof(
    proof: ZKProof,
    com_k: bytes,
    lambda_k: float,
    T_k: float | None = None,
    d_k: int | None = None,
    r_k: int | None = None,
    T_k_prev: float | None = None,
) -> bool:
    """Verify a Groth16 ZK proof."""
    return _groth16.verify(proof, com_k, lambda_k, T_k, d_k, r_k, T_k_prev)
