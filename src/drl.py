"""
drl.py – PPO Actor-Critic for Dynamic Spectrum Pricing (Sec. 7).

Implements:
  • PPO Actor network   π_θ(a|s)  (Sec. 7.2)
  • PPO Critic network  V_φ(s)    (Sec. 7.2)
  • GAE-λ advantage estimation    (Sec. 7.3)
  • PPO clipped objective + entropy bonus (Sec. 7.3)
  • Algorithm 2: PPO Update
  • Non-stationarity handling via sliding window replay (Sec. 7.4)
  • DRL state construction (Sec. 7.1)
  • GPU / mixed-precision (AMP) support for Google Colab
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import src.config as cfg

# ── Device selection (GPU if available, used by Colab) ───────────────────────

def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()


# ── Network architectures ─────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Simple residual connection around a dense layer."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.act    = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.linear(x))


class Actor(nn.Module):
    """
    PPO Actor π_θ(a|s) (Sec. 7.2):
      LayerNorm → Dense(64,ReLU)+Residual → Dense(128,ReLU) →
      Dense(64,ReLU)+Residual → Dense(K×|Q|, Softmax per NCU)
    """

    def __init__(self, state_dim: int, K: int, n_price_levels: int):
        super().__init__()
        self.K = K
        self.n_price_levels = n_price_levels

        self.norm   = nn.LayerNorm(state_dim)
        self.fc1    = nn.Linear(state_dim, 64)
        self.res1   = ResidualBlock(64)
        self.fc2    = nn.Linear(64, 128)
        self.fc3    = nn.Linear(128, 64)
        self.res2   = ResidualBlock(64)
        self.fc_out = nn.Linear(64, K * n_price_levels)

        self.act = nn.ReLU()

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Returns logits of shape (batch, K, n_price_levels).
        """
        x = self.norm(s)
        x = self.act(self.fc1(x))
        x = self.res1(x)
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.res2(x)
        x = self.fc_out(x)
        x = x.view(-1, self.K, self.n_price_levels)
        return x  # logits; apply Categorical per NCU

    def action_distribution(self, s: torch.Tensor):
        """Return a list of Categorical distributions, one per NCU."""
        logits = self.forward(s)   # (batch, K, Q)
        return [Categorical(logits=logits[:, k, :]) for k in range(self.K)]

    def sample_action(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample one price-level index per NCU.
        Returns action indices (K,) and log-probs (K,).
        """
        dists = self.action_distribution(s)
        actions   = torch.stack([d.sample()         for d in dists], dim=-1)  # (batch, K)
        log_probs = torch.stack([d.log_prob(actions[:, k]) for k, d in enumerate(dists)], dim=-1)
        return actions.squeeze(0), log_probs.squeeze(0)

    def log_prob(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Log-prob of action a under current policy. (batch, K)."""
        dists = self.action_distribution(s)
        return torch.stack(
            [dists[k].log_prob(a[:, k]) for k in range(self.K)], dim=-1
        )


class Critic(nn.Module):
    """
    PPO Critic V_φ(s) (Sec. 7.2):
      shared backbone (first 2 layers with actor) → Dense(128,ReLU) →
      Dense(64,ReLU) → Dense(1,Linear)
    Implemented as a standalone network; weights not shared in this version.
    """

    def __init__(self, state_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.act = nn.ReLU()

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(s))
        x = self.act(self.fc2(x))
        return self.fc3(x)


# ── PPO Agent ─────────────────────────────────────────────────────────────────

class PPOAgent:
    """
    PPO agent for dynamic spectrum pricing (Sec. 7).
    Implements Algorithm 2 (PPO Update).
    """

    def __init__(self, K: int):
        self.K = K
        self.n_price_levels = cfg.N_PRICE_LEVELS
        # Price grid
        self.price_grid = np.linspace(cfg.Q_MIN, cfg.Q_MAX, self.n_price_levels)

        state_dim = cfg.STATE_DIM
        self.device = DEVICE

        self.actor  = Actor(state_dim, K, self.n_price_levels).to(self.device)
        self.critic = Critic(state_dim).to(self.device)

        self.actor_old = Actor(state_dim, K, self.n_price_levels).to(self.device)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.actor_old.eval()

        self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=cfg.DRL_LR_ACTOR)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=cfg.DRL_LR_CRITIC)

        # Automatic Mixed Precision scaler (no-op on CPU)
        self._use_amp = self.device.type == "cuda"
        self._scaler  = torch.amp.GradScaler("cuda") if self._use_amp else None

        # Sliding window replay buffer (non-stationarity handling, Sec. 7.4)
        self.buffer: deque = deque(maxlen=cfg.REPLAY_WINDOW)

        # Running stats for state normalisation
        self._state_mean = np.zeros(state_dim)
        self._state_std  = np.ones(state_dim)
        self._n_updates  = 0

    # ── State construction (Sec. 7.1) ─────────────────────────────────────────

    @staticmethod
    def build_state(
        D_isac_history: list[int],
        v_history: list[np.ndarray],
        rho_history: list[np.ndarray],
        I_agg_history: list[float],
        revenue_history: list[float],
        n_active_history: list[int],
        falsification_history: list[float],
        K: int,
    ) -> np.ndarray:
        """
        Construct state s_t ∈ ℝ^12 (Sec. 7.1).

          s_t = [D̄_ISAC^(w), v̄^(w), σ̃_v², ρ̄^(w), I_t^agg,
                 R_t^prev, N_active(t), Δd̂(t),  +4 padding]
        """
        w = cfg.DRL_WINDOW_SIZE
        hist = lambda lst: lst[-w:] if len(lst) >= w else lst

        D_isac_w    = np.mean(hist(D_isac_history))         if D_isac_history    else 0.0
        v_vals      = [np.mean(v) for v in hist(v_history)] if v_history         else [0.0]
        v_bar       = np.mean(v_vals)
        v_var       = np.var(v_vals)
        rho_vals    = [np.mean(r) for r in hist(rho_history)] if rho_history     else [1.0]
        rho_bar     = np.mean(rho_vals)
        I_agg       = I_agg_history[-1]    if I_agg_history    else 0.0
        rev_prev    = revenue_history[-1]  if revenue_history   else 0.0
        n_active    = n_active_history[-1] if n_active_history  else K
        df_rate     = falsification_history[-1] if falsification_history else 0.0

        # Strategy profile estimate (last-W mean falsification)
        strategy_est = np.mean(hist(falsification_history)) if falsification_history else 0.0

        state = np.array([
            D_isac_w,
            v_bar, v_var,
            rho_bar,
            I_agg,
            rev_prev,
            n_active,
            df_rate,
            strategy_est,
            rho_bar,   # repeated for dim=12
            v_bar,
            D_isac_w,
        ], dtype=np.float32)
        return state

    def normalise_state(self, s: np.ndarray) -> np.ndarray:
        """Z-normalise state using running estimates."""
        return ((s - self._state_mean) / (self._state_std + 1e-8)).astype(np.float32)

    def _update_state_stats(self, s: np.ndarray) -> None:
        """Online update of running mean/std."""
        self._n_updates += 1
        alpha = min(0.01, 1.0 / self._n_updates)
        self._state_mean = (1 - alpha) * self._state_mean + alpha * s
        self._state_std  = (1 - alpha) * self._state_std  + alpha * np.abs(s - self._state_mean)

    # ── Action selection ──────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Select price actions.

        Returns
        -------
        q_prices   : price vector (K,) in [Q_MIN, Q_MAX]
        action_idx : discrete action indices (K,)
        value      : critic value estimate
        """
        self._update_state_stats(s)
        s_norm = self.normalise_state(s)
        s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(self.device)

        action_idx, _ = self.actor.sample_action(s_t)
        value          = self.critic(s_t).item()

        action_np  = action_idx.cpu().numpy().astype(int)
        q_prices   = self.price_grid[action_np]  # (K,)
        return q_prices, action_np, value

    # ── Store transition ──────────────────────────────────────────────────────

    def store(self, s, a, r, s_next, done):
        """Add (s, a, r, s_next, done) to replay buffer."""
        self.buffer.append((
            self.normalise_state(s),
            np.array(a, dtype=np.int64),
            float(r),
            self.normalise_state(s_next),
            float(done),
        ))

    # ── Algorithm 2: PPO Update ────────────────────────────────────────────────

    def update(self) -> dict:
        """
        Execute PPO training step (Algorithm 2) on current buffer.

        Returns dict with training stats.
        """
        if len(self.buffer) < 2:
            return {}

        # Unpack buffer
        states, actions, rewards, next_states, dones = zip(*self.buffer)
        S  = torch.FloatTensor(np.stack(states)).to(self.device)
        A  = torch.LongTensor(np.stack(actions)).to(self.device)    # (T, K)
        R  = torch.FloatTensor(rewards).to(self.device)
        NS = torch.FloatTensor(np.stack(next_states)).to(self.device)
        D  = torch.FloatTensor(dones).to(self.device)

        T = len(R)

        # Step 1–2: Compute returns and GAE-λ advantages
        with torch.no_grad():
            values      = self.critic(S).squeeze(-1)       # (T,)
            next_values = self.critic(NS).squeeze(-1)      # (T,)

        gamma   = cfg.DRL_GAMMA
        lam     = cfg.DRL_GAE_LAMBDA

        advantages = torch.zeros(T, device=self.device)
        last_gae   = 0.0
        for t in reversed(range(T)):
            mask   = 1.0 - D[t]
            delta  = R[t] + gamma * next_values[t] * mask - values[t]
            last_gae = float(delta) + gamma * lam * mask * last_gae
            advantages[t] = last_gae

        returns = advantages + values.detach()

        # Step 3: Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute old log probs
        with torch.no_grad():
            old_lp = self.actor_old.log_prob(S, A).sum(-1)  # (T,)

        clip_eps = cfg.DRL_CLIP_EPS
        stats = {"l_clip": 0.0, "l_vf": 0.0, "l_ent": 0.0, "kl": 0.0}
        n_iters = 0

        for _ in range(cfg.DRL_K_EPOCHS):
            # Shuffle mini-batches
            idx = torch.randperm(T, device=self.device)
            bs  = cfg.DRL_BATCH_SIZE
            for start in range(0, T, bs):
                batch = idx[start: start + bs]
                if len(batch) < 2:
                    continue

                amp_ctx = (
                    torch.amp.autocast("cuda")
                    if self._use_amp
                    else torch.amp.autocast("cpu", enabled=False)
                )
                with amp_ctx:
                    # Step 4: Ratio
                    new_lp = self.actor.log_prob(S[batch], A[batch]).sum(-1)
                    ratio  = torch.exp(new_lp - old_lp[batch].detach())

                    # Step 5: CLIP objective
                    adv_b = advantages[batch]
                    l_clip = torch.min(
                        ratio * adv_b,
                        torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_b,
                    ).mean()

                    # Step 6: Value function loss
                    v_pred = self.critic(S[batch]).squeeze(-1)
                    l_vf   = ((v_pred - returns[batch].detach()) ** 2).mean()

                    # Step 7: Entropy bonus
                    logits = self.actor(S[batch])  # (batch, K, Q)
                    l_ent  = 0.0
                    for k in range(self.K):
                        dist  = Categorical(logits=logits[:, k, :])
                        l_ent = l_ent + dist.entropy().mean()
                    l_ent = l_ent / self.K

                    # Step 8: Total loss
                    l_total = -l_clip + cfg.DRL_VF_C1 * l_vf - cfg.DRL_ENTROPY_C2 * l_ent

                # KL check (Step 11)
                with torch.no_grad():
                    kl = (old_lp[batch] - new_lp).mean().item()

                if abs(kl) > cfg.DRL_KL_MAX:
                    break  # early stopping

                # Step 9–10: Gradient step (AMP-aware)
                self.opt_actor.zero_grad()
                self.opt_critic.zero_grad()
                if self._use_amp and self._scaler is not None:
                    self._scaler.scale(l_total).backward()
                    self._scaler.unscale_(self.opt_actor)
                    self._scaler.unscale_(self.opt_critic)
                    nn.utils.clip_grad_norm_(self.actor.parameters(),  0.5)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self._scaler.step(self.opt_actor)
                    self._scaler.step(self.opt_critic)
                    self._scaler.update()
                else:
                    l_total.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(),  0.5)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.opt_actor.step()
                    self.opt_critic.step()

                stats["l_clip"] += l_clip.item()
                stats["l_vf"]   += l_vf.item()
                stats["l_ent"]  += l_ent if isinstance(l_ent, float) else l_ent.item()
                stats["kl"]     += kl
                n_iters += 1

        if n_iters > 0:
            for k in stats:
                stats[k] /= n_iters

        # Copy updated actor to old
        self.actor_old.load_state_dict(self.actor.state_dict())
        return stats

    # ── Checkpoint helpers (for Colab session persistence) ────────────────────

    def save_checkpoint(self, path: str) -> None:
        """Save agent state to disk (for Colab Drive persistence)."""
        torch.save({
            "actor":        self.actor.state_dict(),
            "actor_old":    self.actor_old.state_dict(),
            "critic":       self.critic.state_dict(),
            "opt_actor":    self.opt_actor.state_dict(),
            "opt_critic":   self.opt_critic.state_dict(),
            "state_mean":   self._state_mean,
            "state_std":    self._state_std,
            "n_updates":    self._n_updates,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load agent state from disk."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_old.load_state_dict(ckpt["actor_old"])
        self.critic.load_state_dict(ckpt["critic"])
        self.opt_actor.load_state_dict(ckpt["opt_actor"])
        self.opt_critic.load_state_dict(ckpt["opt_critic"])
        self._state_mean = ckpt["state_mean"]
        self._state_std  = ckpt["state_std"]
        self._n_updates  = ckpt["n_updates"]
