"""
config.py – All simulation parameters from VERIDIC-DSA paper (Section 12.1 + Appendix E).
"""
import numpy as np

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ── Physical layer constants ──────────────────────────────────────────────────
SPEED_OF_LIGHT = 3e8          # m/s
BOLTZMANN_K    = 1.38e-23     # J/K
TEMPERATURE    = 290.0        # Kelvin (noise reference)

# ── Carrier / bandwidth ───────────────────────────────────────────────────────
FC_HZ    = 142e9              # 142 GHz (D-band)
BW_HZ    = 100e6              # 100 MHz
WAVELENGTH = SPEED_OF_LIGHT / FC_HZ  # ~2.11 mm

# ── Slot parameters ───────────────────────────────────────────────────────────
TAU_MS      = 10.0            # ms per slot
TAU_SEC     = TAU_MS / 1000.0
N_SAMPLES   = 1000            # sensing samples per slot (N_s)

# ── ISAC BS ───────────────────────────────────────────────────────────────────
N_ANTENNAS  = 64              # ULA antenna count (N_t)
P_MAX_DBM   = 30.0            # BS max Tx power dBm
P_MAX_W     = 10 ** (P_MAX_DBM / 10) / 1000.0   # ~1 W

# ── Critical User (CU) ────────────────────────────────────────────────────────
P_CU_DBM    = 30.0            # CU Tx power dBm (1 W — military/ATC radar transmitter)
P_CU_W      = 10 ** (P_CU_DBM / 10) / 1000.0   # ~1 W

# semi-Markov CU activity: π_1 = λ_on / (λ_on + λ_off) = 0.30
CU_LAMBDA_ON  = 0.10          # transition rate idle→active
CU_LAMBDA_OFF = 0.233         # transition rate active→idle (π_1 ≈ 0.30)
CU_STEADY_STATE_PI1 = CU_LAMBDA_ON / (CU_LAMBDA_ON + CU_LAMBDA_OFF)  # ≈ 0.30

# ── NCU parameters ────────────────────────────────────────────────────────────
P_NCU_MAX_DBM = 23.0          # 3GPP UE max Tx power dBm
P_NCU_MAX_W   = 10 ** (P_NCU_MAX_DBM / 10) / 1000.0  # ~200 mW
NOISE_FIGURE_DB = 7.0         # Noise figure (dB)
NOISE_FIGURE    = 10 ** (NOISE_FIGURE_DB / 10)

# Thermal noise power at receiver: N_0 = k_B * T * B * NF
NOISE_POWER_W = BOLTZMANN_K * TEMPERATURE * BW_HZ * NOISE_FIGURE   # ~3.96e-13 W

# ── Energy detection ──────────────────────────────────────────────────────────
ALPHA_MAX = 0.05              # max false alarm rate (5%)

# ── Sub-THz molecular absorption at 142 GHz (dB/km, Appendix B) ──────────────
ALPHA_ABS_DB_PER_KM = 2.1    # H₂O absorption at 142 GHz
SHADOW_STD_DB = 3.0          # Shadowing std deviation (dB) — reduced for short-range sub-THz

# ── Nakagami-m fading parameter ───────────────────────────────────────────────
NAKAGAMI_M = 2.0             # shape parameter for sub-THz small-scale fading

# ── ISAC radar ────────────────────────────────────────────────────────────────
N_RADAR_PULSES   = 128       # L – integration pulses
N_REF_CELLS      = 32        # CFAR reference cells
CFAR_GUARD_CELLS = 4
P_FA_ISAC        = 0.01      # ISAC false alarm rate target
SIGMA_RCS        = 1.0       # radar cross section (m²)
G_TX             = 30.0      # transmit antenna gain (linear, ~15 dBi)
G_RX             = 30.0      # receive antenna gain (linear)

# ── Interference ──────────────────────────────────────────────────────────────
I_MAX_DBM  = -80.0           # maximum allowable interference at CU receiver (dBm)
I_MAX_W    = 10 ** (I_MAX_DBM / 10) / 1000.0
EPSILON_INT = 1e-3           # outage probability bound for interference

# ── Bond / mechanism ─────────────────────────────────────────────────────────
B_MIN          = 10.0        # minimum bond (revenue units)
KAPPA          = 2.0         # penalty scaling factor
ETA_CONFIDENCE = 2.0         # CFAR confidence normalizer η_conf

# ── Auction ───────────────────────────────────────────────────────────────────
W_ISAC          = 0.5        # ISAC oracle weight in aggregate decision
THETA_AGG       = 0.5        # aggregation threshold θ
THETA_COLLUSION = 0.6        # co-falsification threshold θ_collusion
THETA_SUSPECT   = 0.5        # coalition suspicion threshold

# ── Valuation distribution ────────────────────────────────────────────────────
V_MIN = 0.5                  # NCU min willingness-to-pay
V_MAX = 5.0                  # NCU max willingness-to-pay

# ── DRL (PPO) hyperparameters (Appendix E best values) ──────────────────────
DRL_GAMMA       = 0.99       # discount factor γ_RL
DRL_GAE_LAMBDA  = 0.95       # GAE-λ
DRL_CLIP_EPS    = 0.2        # PPO clip parameter ε_clip
DRL_LR_ACTOR    = 3e-4       # actor learning rate
DRL_LR_CRITIC   = 1e-3       # critic learning rate
DRL_ENTROPY_C2  = 0.01       # entropy coefficient c₂
DRL_VF_C1       = 0.5        # value-function coefficient c₁
DRL_BATCH_SIZE  = 128        # mini-batch size
DRL_K_EPOCHS    = 4          # PPO update epochs per batch
DRL_KL_MAX      = 0.05       # early stopping KL threshold
DRL_UPDATE_FREQ = 200        # update every N steps (episode length)
DRL_WINDOW_SIZE = 10         # rolling-average window w
REPLAY_WINDOW   = 500        # sliding window T_window for non-stationarity

# ── DRL reward weights ───────────────────────────────────────────────────────
MU1 = 0.3                    # social welfare weight
MU2 = 10.0                   # interference penalty weight
MU3 = 5.0                    # falsification penalty weight

# ── Price levels ─────────────────────────────────────────────────────────────
Q_MIN = 0.1
Q_MAX = 5.0
N_PRICE_LEVELS = 10          # discretised price levels per NCU

# ── State dimension ──────────────────────────────────────────────────────────
STATE_DIM = 12               # |s_t| = 12 (Sec. 7.1)

# ── Simulation episodes ───────────────────────────────────────────────────────
N_EPISODES      = 50_000     # total training episodes
EPISODE_LENGTH  = 200        # slots per episode

# ── Reputation ────────────────────────────────────────────────────────────────
ALPHA0 = 1.0                 # Beta prior α₀
BETA0  = 1.0                 # Beta prior β₀
XI_COALITION = 2.0           # coalition reputation amplification ξ

# ── Pedersen commitment (2048-bit safe prime proxy; using 256-bit for speed) ──
# We use a 256-bit safe prime group for simulation speed.
# In production, 2048-bit primes are required.
PEDERSEN_BITS = 256

# ── ZKP timing constants (Sec. 5.2, production hardware) ─────────────────────
ZKP_PROVE_TIME_MS   = 120.0  # ms, simulated prover time
ZKP_VERIFY_TIME_MS  = 1.5    # ms, simulated verifier time
ZKP_PROOF_SIZE_BYTES = 192   # bytes per Groth16 proof
COMMITMENT_SIZE_BYTES = 256   # bytes per Pedersen commitment

# ── Communication overhead (Sec. 15.3) ───────────────────────────────────────
BOND_CONFIRM_BYTES = 64
REVEAL_BYTES = 64
# Total per NCU per slot = 256 + 192 + 64 + 64 = 576 bytes

# ── NCU count sweep ───────────────────────────────────────────────────────────
K_VALUES = [10, 20, 30]      # scalability study values

# ── Non-stationarity shift slot ──────────────────────────────────────────────
NONSTATIONARITY_SLOT = 500   # duty-cycle shift at this slot

# ── Topology / distance ──────────────────────────────────────────────────────
BS_TO_NCU_DIST_M = 200.0     # base station to NCU distance (m)
CU_TO_NCU_DIST_M = 100.0     # CU to NCU distance (m)  — 100m gives good sensing SNR at sub-THz
NCU_TO_CU_DIST_M = 600.0     # NCU to CU receiver distance (m)  (for interference)
BS_TO_CU_DIST_M  = 1000.0    # BS to CU distance (for radar)

# ── Derived noise spectral density ───────────────────────────────────────────
def noise_power_w(sigma2_offset_db: float = 0.0) -> float:
    """Return noise power in Watts, optionally with an additional offset (dB)."""
    return NOISE_POWER_W * 10 ** (sigma2_offset_db / 10)
