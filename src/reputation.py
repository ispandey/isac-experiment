"""
reputation.py – Bayesian Reputation & Trust Management (Sec. 9).

Implements:
  • Beta-Bernoulli Bayesian reputation model  (Sec. 9.1)
  • Spectral coalition detection (Algorithm 3) (Sec. 9.2)
  • Long-term reputation effects              (Sec. 9.3)
"""
from __future__ import annotations

import numpy as np
from numpy.random import Generator

import src.config as cfg


# ── Bayesian Beta-Bernoulli Reputation ────────────────────────────────────────

class ReputationModel:
    """
    Per-NCU Beta-Bernoulli reputation (Sec. 9.1).

    Each NCU k has parameters (α_k, β_k) with:
      ρ_k = α_k / (α_k + β_k) ∈ [0, 1]

    Update rule:
      • Truthful (no contradiction):  α_k += 1
      • Contradiction detected:        β_k += Δ_k
    """

    def __init__(self, K: int):
        self.K = K
        self.alpha = np.full(K, cfg.ALPHA0)
        self.beta  = np.full(K, cfg.BETA0)

    @property
    def rho(self) -> np.ndarray:
        """Reputation scores ρ_k = α_k / (α_k + β_k), shape (K,)."""
        return self.alpha / (self.alpha + self.beta)

    def update(
        self,
        contradiction: np.ndarray,
        C_ISAC: float,
        coalition_members: np.ndarray | None = None,
    ) -> None:
        """
        Update reputations for one slot.

        Parameters
        ----------
        contradiction    : boolean array (K,) – True iff NCU k contradicted
        C_ISAC           : oracle confidence score (used as Δ weight)
        coalition_members: boolean array (K,) – True for coalition suspects
        """
        if coalition_members is None:
            coalition_members = np.zeros(self.K, dtype=bool)

        for k in range(self.K):
            if contradiction[k]:
                # Weighted penalty increment Δ_k (Sec. 9.1)
                delta_k = C_ISAC * (
                    1.0 + cfg.XI_COALITION * float(coalition_members[k])
                )
                self.beta[k] += delta_k
            else:
                # Truthful: α_k += 1
                self.alpha[k] += 1.0

    def reset(self) -> None:
        """Reset all reputations to prior."""
        self.alpha[:] = cfg.ALPHA0
        self.beta[:] = cfg.BETA0

    def recovery_time(self, k: int) -> float:
        """
        Estimated honest-recovery time (slots) for NCU k (Sec. 9.3):
          T_recovery = (β_k - β_0) / α_k_honest_rate
        """
        excess_beta = max(0.0, self.beta[k] - cfg.BETA0)
        # Honest rate ≈ 1 increment per slot
        return excess_beta


# ── Algorithm 3: Spectral Coalition Detection ─────────────────────────────────

def detect_coalitions(
    co_falsification_history: list[np.ndarray],
    D_isac_history: list[int],
    K: int,
    theta_collusion: float = cfg.THETA_COLLUSION,
    theta_suspect: float = cfg.THETA_SUSPECT,
    n_clusters: int = 3,
) -> tuple[np.ndarray, list[list[int]]]:
    """
    Algorithm 3: Spectral Coalition Detection (Sec. 9.2).

    Parameters
    ----------
    co_falsification_history : list of d_hat arrays (K,) over T_w slots
    D_isac_history           : list of D_ISAC values over T_w slots
    K                        : number of NCUs
    theta_collusion          : co-falsification threshold θ_collusion
    theta_suspect            : coalition suspicion threshold θ_suspect
    n_clusters               : number of spectral clusters

    Returns
    -------
    coalition_mask : boolean array (K,) – True for suspected coalition members
    coalition_sets : list of index lists
    """
    T_w = len(D_isac_history)
    if T_w == 0:
        return np.zeros(K, dtype=bool), []

    D_isac_arr = np.array(D_isac_history, dtype=int)   # (T_w,)
    n_isac_active = max(int(np.sum(D_isac_arr)), 1)

    # Step 1: Build co-falsification matrix F_{jk}
    F = np.zeros((K, K))
    for t_idx in range(T_w):
        if D_isac_arr[t_idx] == 1:
            d_hat = co_falsification_history[t_idx]
            for j in range(K):
                for k in range(K):
                    if d_hat[j] == 0 and d_hat[k] == 0:
                        F[j, k] += 1
    F /= n_isac_active

    # Step 2: Adjacency matrix
    A = (F > theta_collusion).astype(float)
    np.fill_diagonal(A, 0.0)

    # Step 3: Normalised Laplacian L = D^{-1/2}(D-A)D^{-1/2}
    degrees = A.sum(axis=1)
    D_inv_sqrt = np.diag(np.where(degrees > 0, 1.0 / np.sqrt(degrees + 1e-12), 0.0))
    L = D_inv_sqrt @ (np.diag(degrees) - A) @ D_inv_sqrt

    # Step 4: Top-m eigenvectors (smallest eigenvalues of L)
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return np.zeros(K, dtype=bool), []
    m = min(n_clusters, K)
    U = eigvecs[:, :m]  # (K, m)

    # Step 5: k-means on rows of U
    labels = _simple_kmeans(U, n_clusters=m, rng=None)

    # Step 6: Compute suspicion score per cluster
    coalition_mask = np.zeros(K, dtype=bool)
    coalition_sets: list[list[int]] = []

    for cluster_id in range(m):
        members = np.where(labels == cluster_id)[0]
        if len(members) == 0:
            continue
        # Suspicion score: fraction of slots co-falsified when ISAC=1
        scores = []
        for k in members:
            n_k = members[members != k]
            if len(n_k) == 0:
                scores.append(0.0)
                continue
            total = 0.0
            for t_idx in range(T_w):
                if D_isac_arr[t_idx] == 1:
                    d_hat = co_falsification_history[t_idx]
                    neighbor_falsify = np.any(d_hat[n_k] == 0)
                    total += float(neighbor_falsify)
            scores.append(total / n_isac_active)
        mean_score = np.mean(scores) if scores else 0.0
        if mean_score > theta_suspect:
            coalition_mask[members] = True
            coalition_sets.append(members.tolist())

    return coalition_mask, coalition_sets


def _simple_kmeans(
    X: np.ndarray,
    n_clusters: int,
    rng: Generator | None,
    max_iter: int = 50,
) -> np.ndarray:
    """
    Simple Lloyd k-means on rows of X.
    Initialises centres by k-means++ style random spread.
    """
    N = X.shape[0]
    if n_clusters >= N:
        return np.arange(N)

    rng_np = np.random.default_rng(cfg.SEED) if rng is None else rng

    # Initialise centres
    idx = rng_np.integers(0, N)
    centres = [X[idx]]
    for _ in range(1, n_clusters):
        dists = np.array([
            min(np.linalg.norm(x - c) ** 2 for c in centres)
            for x in X
        ])
        probs = dists / (dists.sum() + 1e-12)
        idx = rng_np.choice(N, p=probs)
        centres.append(X[idx])
    centres = np.stack(centres)

    labels = np.zeros(N, dtype=int)
    for _ in range(max_iter):
        # Assignment step
        dists = np.array([
            [np.linalg.norm(X[i] - centres[c]) for c in range(n_clusters)]
            for i in range(N)
        ])
        new_labels = np.argmin(dists, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        # Update step
        for c in range(n_clusters):
            members = X[labels == c]
            if len(members) > 0:
                centres[c] = members.mean(axis=0)

    return labels
