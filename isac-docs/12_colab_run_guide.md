# 12 — Google Colab Run Guide

> Complete instructions for running VERIDIC-DSA experiments on Google Colab,  
> including hardware requirements, setup steps, configuration, and tips.

---

## 12.1 Why Google Colab?

| Feature | Free Colab | Colab Pro | Colab Pro+ |
|---------|-----------|-----------|------------|
| GPU | T4 (15 GB) | T4 / V100 | A100 (40 GB) |
| RAM | 12.7 GB | 25.5 GB | 51 GB |
| Session limit | 12 h | 24 h | 24 h |
| Background execution | ✗ | ✗ | ✓ |
| **Recommended for** | Smoke test / BL sweep | Full K=10 run | Full paper experiments |

For a complete paper experiment (50 000 episodes × 200 slots = 10M slots with K=10):

| Hardware | Estimated wall-clock |
|----------|---------------------|
| CPU (Colab free) | ~6–10 h |
| T4 GPU (Colab free/Pro) | ~45–90 min |
| A100 GPU (Colab Pro+) | ~15–25 min |

---

## 12.2 Hardware Requirements

### Minimum (smoke test, 2 000 slots, K=5)
- **RAM:** 4 GB
- **GPU:** Not required
- **Disk:** 500 MB (model + deps)
- **Time:** < 2 min

### Recommended (full K=10 experiment, 50k episodes)
- **RAM:** 12 GB (T4 Colab)
- **GPU:** NVIDIA T4 (CUDA 12.x, 15 GB VRAM)
- **Disk:** 2 GB (results + checkpoints)
- **Time:** ~60–90 min (T4)

### Full Paper (K-sweep + baseline comparison)
- **RAM:** 25+ GB
- **GPU:** A100 40 GB (Colab Pro+) or V100 16 GB
- **Disk:** 5 GB
- **Time:** ~4–6 h (A100)

---

## 12.3 Step-by-Step Setup

### Step 1 — Set Runtime to GPU

In Colab:
```
Runtime → Change runtime type → Hardware accelerator → GPU (T4 or A100)
```

### Step 2 — Clone the Repository

```python
# Cell 1: Clone
!git clone https://github.com/ispandey/isac-experiment.git
%cd isac-experiment
```

### Step 3 — Run the One-Shot Setup Script

```python
# Cell 2: Setup (installs deps, detects GPU, sets seed)
exec(open('colab_setup.py').read())
```

Expected output:
```
Installing dependencies …
Dependencies installed.

==================================================
  VERIDIC-DSA Hardware Summary
==================================================
  python          : 3.10.12
  torch           : 2.6.0
  numpy           : 1.26.4
  device          : CUDA (GPU)
  gpu_name        : Tesla T4
  gpu_memory      : 15.0 GB
  cuda_version    : 12.2
==================================================

Global random seed set to 42

Package imports OK — ready to run experiments.
```

### Step 4 — (Optional) Mount Google Drive for Checkpoints

```python
# Cell 3: Mount Drive
from colab_setup import mount_drive
CHECKPOINT_DIR = mount_drive('veridic_checkpoints')
CHECKPOINT_PATH = f'{CHECKPOINT_DIR}/ppo_agent.pt'
```

This allows your PPO agent weights to persist across Colab sessions. If your session disconnects, you can resume training from the last checkpoint.

### Step 5 — Run Quick Smoke Test

```python
# Cell 4: 500-slot smoke test (< 1 min)
from src.experiment import VeridicDSA
from src.adversary import AdversaryType

sim = VeridicDSA(K=5, adversary_fraction=0.3,
                 adversary_type=AdversaryType.SF, seed=42)
for i in range(500):
    sim.run_slot(i)

s = sim.metrics.summary()
print(f"DAF={s['DAF']:.4f}  HIP={s['HIP']:.6f}  Rev={s['Revenue']:.2f}")
```

**Expected:** DAF > 0.94, HIP < 0.01.

---

## 12.4 Full Training Run (K=10, 50k Episodes)

```python
from src.experiment import VeridicDSA
from src.adversary import AdversaryType
import json, os

# Configuration
K           = 10
ADV_FRAC    = 0.30
N_EPISODES  = 50_000
SEED        = 42

sim = VeridicDSA(K=K, adversary_fraction=ADV_FRAC,
                 adversary_type=AdversaryType.SF, seed=SEED)

# Resume from checkpoint if available
if 'CHECKPOINT_PATH' in dir() and os.path.exists(CHECKPOINT_PATH):
    sim.ppo_agent.load_checkpoint(CHECKPOINT_PATH)
    print('Checkpoint loaded — resuming training')

summary, revenues = sim.run(
    n_episodes   = N_EPISODES,
    verbose      = True,          # shows tqdm progress bar
    log_every    = 500,
    checkpoint_path   = CHECKPOINT_PATH if 'CHECKPOINT_PATH' in dir() else None,
    checkpoint_every  = 1000,     # save every 1000 episodes
)

print(f"\n=== Final Results ===")
print(f"  DAF     : {summary['DAF']:.4f}")
print(f"  HIP     : {summary['HIP']:.6f}")
print(f"  Revenue : {summary['Revenue']:.2f}")
print(f"  SE      : {summary['SE']:.2f} b/s/Hz")

# Save results
os.makedirs('results', exist_ok=True)
with open(f'results/veridic_K{K}_s{SEED}.json', 'w') as f:
    json.dump({'summary': summary, 'revenues': revenues}, f, indent=2, default=str)
```

---

## 12.5 Running from the Command Line (Colab Terminal)

```bash
# Open Colab terminal: Tools → Terminal
cd /content/isac-experiment

# Quick run (2000 slots, no sweep)
python experiments/run_veridic.py \
    --K 10 --n_slots 2000 --seed 42 \
    --adv_frac 0.3 --adv_type SF \
    --no_sweep --verbose

# Full paper run (50k episodes × 200 slots)
python experiments/run_veridic.py \
    --K 10 --episodes 50000 --seed 42 \
    --adv_frac 0.3 --adv_type SF \
    --verbose

# With non-stationarity stress test
python experiments/run_veridic.py \
    --K 10 --episodes 10000 --seed 42 \
    --adv_frac 0.3 --adv_type CCF \
    --nonstat --verbose

# Generate all figures (after run_veridic.py)
python analysis/generate_figures.py --n_slots 2000
```

---

## 12.6 Using the Colab Notebook

Open the included notebook directly:

1. Upload `VERIDIC_DSA_Colab.ipynb` to Google Drive.
2. Open it with Google Colab.
3. Set runtime to GPU.
4. Run cells sequentially (Cells 1–9 map to each experiment section).

Or open directly from GitHub:

```
https://colab.research.google.com/github/ispandey/isac-experiment/blob/main/VERIDIC_DSA_Colab.ipynb
```

---

## 12.7 Memory Management Tips

The simulation is CPU-memory dominated (NumPy arrays for slot records). For $T = 10^7$ slots with $K = 10$ NCUs, the `MetricStore` holds ~$10^7$ `SlotRecord` objects.

**Tips to reduce RAM usage:**

```python
# Option 1: Clear metrics after each episode and only track summary stats
sim.metrics.clear()  # call periodically

# Option 2: Reduce episode length (use more, shorter episodes)
summary, revenues = sim.run(n_episodes=50000, episode_length=200)  # default

# Option 3: Run baselines separately, not simultaneously
```

For Colab free tier (12 GB RAM), the recommended configuration:
- K ≤ 20
- `episode_length = 200` (default)
- Clear `MetricStore` every 5 000 episodes using `sim.metrics.clear()`

---

## 12.8 GPU Utilisation

The PPO neural networks (Actor + Critic) are small (< 50k parameters total) and run on GPU. The per-slot simulation (channel, ZKP, mechanism) runs on CPU. GPU utilisation peaks during `PPOAgent.update()` calls (every 200 slots).

To monitor GPU usage in Colab:

```python
!nvidia-smi
```

Or continuously:

```python
import subprocess, time
for _ in range(10):
    print(subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                                   '--format=csv,noheader']).decode().strip())
    time.sleep(5)
```

**Typical GPU utilisation:** 5–15% (burst during PPO updates). The bottleneck is CPU-side simulation, not GPU computation. To improve GPU utilisation for larger K, increase `DRL_BATCH_SIZE` in `config.py`.

---

## 12.9 Requirements Summary

Full `requirements.txt`:

```
numpy==1.26.4
scipy==1.11.4
torch==2.6.0
matplotlib==3.9.0
seaborn==0.13.2
plotly==5.22.0
cvxpy==1.5.0
pandas==2.2.2
tqdm==4.66.4
```

These are pre-installed on Colab (except cvxpy and plotly). The `colab_setup.py` script installs all automatically.

**CUDA compatibility:** PyTorch 2.6.0 supports CUDA 11.8 / 12.x. Colab's T4 GPUs run CUDA 12.2, which is fully compatible.

---

## 12.10 Troubleshooting

| Problem | Solution |
|---------|---------|
| `ModuleNotFoundError: No module named 'src'` | Run `exec(open('colab_setup.py').read())` first, or add `import sys; sys.path.insert(0,'.')` |
| `CUDA out of memory` | Reduce `DRL_BATCH_SIZE` in `config.py` (try 64) |
| Session disconnected (Colab timeout) | Save checkpoint to Drive every 1000 episodes; reload with `sim.ppo_agent.load_checkpoint(path)` |
| `ImportError: libGL.so.1` (matplotlib) | Run `!apt-get install -y libgl1-mesa-glx` |
| Slow simulation (no GPU speedup) | Expected — simulation is CPU-bound; GPU only accelerates PPO updates |
| `tqdm` not showing progress bar | Install: `!pip install -q tqdm` |
| `torch.amp.GradScaler` warning on CPU | Harmless — AMP is a no-op on CPU |

---

## 12.11 End-to-End Colab Run Checklist

- [ ] Set runtime to GPU (T4 or A100)
- [ ] Clone repository
- [ ] Run `colab_setup.py`
- [ ] (Optional) Mount Google Drive for checkpoints
- [ ] Verify GPU detected (`device: CUDA (GPU)`)
- [ ] Run 500-slot smoke test (DAF > 0.94)
- [ ] Run full training (50k episodes)
- [ ] Save checkpoints to Drive periodically
- [ ] Run baseline comparison (BL1–BL7)
- [ ] Run adversary sweep
- [ ] Generate paper figures: `python analysis/generate_figures.py`
- [ ] Download results JSON from `/content/isac-experiment/results/`
