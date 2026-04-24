"""
experiment.py – VERIDIC-DSA Main Protocol (Algorithm 1, Sec. 10).

Orchestrates the full per-slot protocol:
  Phase 1 : NCU sensing, commitment, ZKP generation, bond post
  Phase 2 : ZKP verification
  Phase 3 : ISAC oracle publication
  Phase 4 : Reveal & audit (contradiction, slashing, rewards, reputation)
  Phase 5 : DRL pricing
  Phase 6 : CVSA auction
  Phase 7 : Transmission, feedback, PPO update

Also includes:
  • Algorithm 2 execution (via PPOAgent.update())
  • Algorithm 3 execution (via detect_coalitions, every COALITION_WINDOW slots)
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional

import numpy as np
from numpy.random import Generator

try:
    from tqdm.auto import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

import src.config as cfg
from src.adversary import AdversaryModel, AdversaryType
from src.channel import ChannelModel, CUActivityModel
from src.crypto import commit, prove, verify_commitment, verify_proof
from src.drl import PPOAgent
from src.mechanism import CVSAMechanism
from src.metrics import MetricStore, SlotRecord
from src.oracle import ISACOracle
from src.reputation import ReputationModel, detect_coalitions
from src.sensing import SensingModule


COALITION_WINDOW = 50    # run coalition detection every N slots


class VeridicDSA:
    """
    Full VERIDIC-DSA simulation engine.

    Parameters
    ----------
    K               : number of NCUs
    adversary_fraction : fraction of adversarial NCUs
    adversary_type  : AdversaryType (HONEST, SF, STF, CCF)
    seed            : random seed
    nonstationarity : if True, CU duty-cycle shifts at slot NONSTATIONARITY_SLOT
    """

    def __init__(
        self,
        K: int = 10,
        adversary_fraction: float = 0.0,
        adversary_type: AdversaryType = AdversaryType.HONEST,
        seed: int = cfg.SEED,
        nonstationarity: bool = False,
    ):
        self.K = K
        self.seed = seed
        self.nonstationarity = nonstationarity
        self.rng: Generator = np.random.default_rng(seed)

        # Subsystems
        self.cu_model      = CUActivityModel(self.rng)
        self.channel_model = ChannelModel(K, self.rng)
        self.sensing_module = SensingModule(K, self.channel_model.sigma2)
        self.oracle        = ISACOracle(self.rng)
        self.mechanism     = CVSAMechanism(K, self.rng)
        self.reputation    = ReputationModel(K)
        self.adversary     = AdversaryModel(K, adversary_fraction, adversary_type, self.rng)
        self.ppo_agent     = PPOAgent(K)
        self.metrics       = MetricStore()

        # Rolling histories for state construction and coalition detection
        self._D_isac_hist:     list[int]         = []
        self._v_hist:          list[np.ndarray]  = []
        self._rho_hist:        list[np.ndarray]  = []
        self._I_hist:          list[float]       = []
        self._rev_hist:        list[float]       = []
        self._nactive_hist:    list[int]         = []
        self._df_hist:         list[float]       = []

        self._cofals_hist:     list[np.ndarray]  = []   # Algorithm 3 input
        self._coalition_mask   = np.zeros(K, dtype=bool)

        self._prev_state:  Optional[np.ndarray] = None
        self._prev_action: Optional[np.ndarray] = None
        self._slot_count = 0

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.cu_model.reset()
        self.reputation.reset()
        self.metrics.clear()
        self._D_isac_hist.clear()
        self._v_hist.clear()
        self._rho_hist.clear()
        self._I_hist.clear()
        self._rev_hist.clear()
        self._nactive_hist.clear()
        self._df_hist.clear()
        self._cofals_hist.clear()
        self._coalition_mask = np.zeros(self.K, dtype=bool)
        self._prev_state  = None
        self._prev_action = None
        self._slot_count  = 0

    # ── Non-stationarity injection ────────────────────────────────────────────

    def _maybe_shift_duty_cycle(self, slot: int) -> None:
        """Shift CU duty cycle at slot NONSTATIONARITY_SLOT (Sec. 14.1 Fig 11)."""
        if self.nonstationarity and slot == cfg.NONSTATIONARITY_SLOT:
            # Double the on-rate → higher CU activity
            cfg.CU_LAMBDA_ON  = 0.25
            cfg.CU_LAMBDA_OFF = 0.58

    # ── Main slot execution (Algorithm 1) ─────────────────────────────────────

    def run_slot(self, slot: int) -> SlotRecord:
        """Execute one complete VERIDIC-DSA slot. Returns a SlotRecord."""
        K   = self.K
        rng = self.rng

        self._maybe_shift_duty_cycle(slot)

        # ── CU activity ────────────────────────────────────────────────────────
        H_t = self.cu_model.step()

        # ── Channel ────────────────────────────────────────────────────────────
        ch = self.channel_model.sample_slot(H_t)
        g_kb   = ch["g_kb"]     # (K,) NCU→CU interference channels
        sigma2 = ch["sigma2"]   # (K,) noise variance

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PHASE 1: SENSING & COMMITMENT
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        T_k, d_true = self.sensing_module.sense(ch["y_k"])
        lambda_k    = self.sensing_module.lambda_k
        T_prev      = self.sensing_module.T_prev.copy()

        # Apply adversary overrides to get d_hat
        q_prev = (
            np.full(K, (cfg.Q_MIN + cfg.Q_MAX) / 2)
            if self._prev_action is None
            else self.ppo_agent.price_grid[self._prev_action]
        )
        d_hat = self.adversary.apply(d_true, T_k, lambda_k, self.reputation.rho, q_prev)

        # Pedersen commitments + ZKP proofs
        com_list  = []
        r_list    = []
        zkp_list  = []
        prove_times = np.zeros(K)
        adv_mask  = self.adversary.is_adversary

        for k in range(K):
            is_adv = bool(adv_mask[k])
            # Adversary may try to commit d_hat[k] (which differs from T_k)
            # but will fail the ZKP circuit check
            com_k, r_k = commit(d_hat[k], rng)
            com_list.append(com_k)
            r_list.append(r_k)

            zkp = prove(
                T_k=T_k[k],
                d_k=d_hat[k],
                r_k=r_k,
                com_k=com_k,
                lambda_k=lambda_k[k],
                T_k_prev=T_prev[k],
                # Adversary fabricates proof only if d_hat differs from d_true
                adversarial=(is_adv and d_hat[k] != d_true[k]),
                rng=rng,
            )
            zkp_list.append(zkp)
            prove_times[k] = zkp.prove_time_ms

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PHASE 2: ZKP VERIFICATION
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        zkp_valid = np.zeros(K, dtype=bool)
        d_verified = d_hat.copy()
        for k in range(K):
            ok = verify_proof(
                proof=zkp_list[k],
                com_k=com_list[k],
                lambda_k=lambda_k[k],
            )
            if not ok:
                # Discard NCU k – treat as not participating (report idle absent)
                d_verified[k] = 0
            zkp_valid[k] = ok

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PHASE 3: ISAC ORACLE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        D_ISAC, C_ISAC, Lambda_ISAC = self.oracle.observe(H_t)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PHASE 4: REVEAL & AUDIT
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Commitment binding check
        binding_ok = np.array([
            verify_commitment(d_verified[k], r_list[k], com_list[k])
            for k in range(K)
        ])

        # Contradiction: NCU reports idle (0) but ISAC says active (1)
        contradiction = (d_verified == 0) & (D_ISAC == 1)

        # Falsification detected: contradiction AND commitment was valid
        falsification_detected = contradiction & binding_ok

        # Update reputations
        self.reputation.update(
            contradiction=falsification_detected,
            C_ISAC=C_ISAC,
            coalition_members=self._coalition_mask,
        )
        self.sensing_module.update_prev(T_k)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PHASE 5: DRL PRICING
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        df_rate = float(np.mean(d_verified != d_true))
        I_agg_prev = self._I_hist[-1] if self._I_hist else 0.0
        rev_prev   = self._rev_hist[-1] if self._rev_hist else 0.0
        rho_now    = self.reputation.rho

        state = PPOAgent.build_state(
            self._D_isac_hist, self._v_hist, self._rho_hist,
            self._I_hist, self._rev_hist, self._nactive_hist, self._df_hist,
            K,
        )
        q_drl, action_idx, _value = self.ppo_agent.select_action(state)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PHASE 6: CVSA AUCTION
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        mech_out = self.mechanism.run(
            d_hat=d_verified,
            D_ISAC=D_ISAC,
            C_ISAC=C_ISAC,
            rho=rho_now,
            d_true=d_true,
            q_drl=q_drl,
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PHASE 7: TRANSMISSION & FEEDBACK
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        x_star = mech_out["x_star"]

        # Interference at CU receiver: I_k = P_ncu_k * |g_kb|² per NCU
        interference_w = cfg.P_NCU_MAX_W * np.abs(g_kb) ** 2  # (K,)
        I_harm = float(np.sum(interference_w * x_star)) if H_t == 1 else 0.0

        # DRL reward (Eq. 7.3)
        social_welfare_term = cfg.MU1 * float(
            np.sum(np.log1p(cfg.P_NCU_MAX_W / cfg.NOISE_POWER_W) * x_star)
        )
        interference_penalty = cfg.MU2 * float(I_harm > cfg.I_MAX_W)
        falsif_penalty = cfg.MU3 * float(np.sum(falsification_detected))
        reward = mech_out["revenue"] + social_welfare_term - interference_penalty - falsif_penalty

        # Build next state for replay
        n_active = int(np.sum(x_star))
        self._D_isac_hist.append(D_ISAC)
        self._v_hist.append(mech_out["valuations"])
        self._rho_hist.append(rho_now.copy())
        self._I_hist.append(float(np.sum(interference_w)))
        self._rev_hist.append(mech_out["revenue"])
        self._nactive_hist.append(n_active)
        self._df_hist.append(df_rate)
        self._cofals_hist.append(d_verified.copy())

        # Keep histories at REPLAY_WINDOW length
        for lst in [self._D_isac_hist, self._v_hist, self._rho_hist,
                    self._I_hist, self._rev_hist, self._nactive_hist,
                    self._df_hist, self._cofals_hist]:
            while len(lst) > cfg.REPLAY_WINDOW:
                lst.pop(0)

        next_state = PPOAgent.build_state(
            self._D_isac_hist, self._v_hist, self._rho_hist,
            self._I_hist, self._rev_hist, self._nactive_hist, self._df_hist,
            K,
        )

        # Store DRL transition
        if self._prev_state is not None:
            self.ppo_agent.store(
                self._prev_state, self._prev_action, reward, state, done=0.0
            )

        self._prev_state  = state
        self._prev_action = action_idx

        # PPO update every DRL_UPDATE_FREQ steps
        if (slot + 1) % cfg.DRL_UPDATE_FREQ == 0:
            self.ppo_agent.update()

        # Algorithm 3: Coalition detection every COALITION_WINDOW slots
        if (slot + 1) % COALITION_WINDOW == 0 and len(self._cofals_hist) >= COALITION_WINDOW:
            self._coalition_mask, _ = detect_coalitions(
                co_falsification_history=list(self._cofals_hist[-COALITION_WINDOW:]),
                D_isac_history=list(self._D_isac_hist[-COALITION_WINDOW:]),
                K=K,
            )

        # Record slot
        rec = SlotRecord(
            slot=slot,
            H_t=H_t,
            D_ISAC=D_ISAC,
            C_ISAC=C_ISAC,
            d_true=d_true.copy(),
            d_hat=d_verified.copy(),
            zkp_valid=zkp_valid.copy(),
            D_agg=mech_out["D_agg"],
            x_star=x_star.copy(),
            payments=mech_out["payments"].copy(),
            slash_amounts=mech_out["slash_amounts"].copy(),
            rewards=mech_out["rewards"].copy(),
            q_drl=q_drl.copy(),
            revenue=mech_out["revenue"],
            welfare=mech_out["welfare"],
            interference_w=interference_w.copy(),
            rho=rho_now.copy(),
            prove_times_ms=prove_times.copy(),
            falsification_detected=falsification_detected.copy(),
        )
        self.metrics.add(rec)
        self._slot_count += 1
        return rec

    # ── Episode / full run ────────────────────────────────────────────────────

    def run_episode(self, episode_length: int = cfg.EPISODE_LENGTH) -> dict:
        """Run one episode (T slots) and return metric summary."""
        for slot in range(episode_length):
            self.run_slot(self._slot_count)
        return self.metrics.summary()

    def run(
        self,
        n_episodes: int = cfg.N_EPISODES,
        episode_length: int = cfg.EPISODE_LENGTH,
        verbose: bool = False,
        log_every: int = 500,
        checkpoint_path: str | None = None,
        checkpoint_every: int = 1000,
    ) -> tuple[dict, list]:
        """
        Full training run over N episodes.

        Parameters
        ----------
        n_episodes       : total number of training episodes
        episode_length   : slots per episode
        verbose          : print progress to stdout
        log_every        : print interval (episodes)
        checkpoint_path  : if set, save PPO agent checkpoint here periodically
        checkpoint_every : checkpoint save interval (episodes)

        Returns
        -------
        final_summary : metric summary dict over last episode
        episode_revenues : list of per-episode total revenues
        """
        episode_revenues = []

        ep_iter = range(n_episodes)
        if _HAS_TQDM and verbose:
            ep_iter = _tqdm(ep_iter, desc="VERIDIC-DSA Training", unit="ep", dynamic_ncols=True)

        for ep in ep_iter:
            ep_start_records = len(self.metrics.records)
            for _ in range(episode_length):
                self.run_slot(self._slot_count)
            ep_records = self.metrics.records[ep_start_records:]
            ep_rev = sum(r.revenue for r in ep_records)
            episode_revenues.append(ep_rev)

            if checkpoint_path and (ep + 1) % checkpoint_every == 0:
                self.ppo_agent.save_checkpoint(checkpoint_path)

            if verbose and (ep + 1) % log_every == 0:
                ep_daf = MetricStore(records=list(ep_records)).daf()
                ep_hip = MetricStore(records=list(ep_records)).hip()
                if _HAS_TQDM:
                    ep_iter.set_postfix(rev=f"{ep_rev:.1f}", DAF=f"{ep_daf:.3f}", HIP=f"{ep_hip:.4f}")
                else:
                    print(
                        f"Ep {ep+1}/{n_episodes} | Rev={ep_rev:.2f} "
                        f"| DAF={ep_daf:.3f} | HIP={ep_hip:.4f}"
                    )

        return self.metrics.summary(), episode_revenues
