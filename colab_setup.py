"""
colab_setup.py – One-shot environment bootstrap for Google Colab.

Run this cell first in any Colab notebook:

    !git clone https://github.com/ispandey/isac-experiment.git
    %cd isac-experiment
    exec(open("colab_setup.py").read())

What this script does:
  1. Installs all Python dependencies (requirements.txt).
  2. Detects and reports GPU / TPU availability.
  3. Optionally mounts Google Drive for checkpoint persistence.
  4. Patches sys.path so `import src.*` works from any cell.
  5. Sets global random seeds for reproducibility.
  6. Prints a concise hardware summary.
"""
from __future__ import annotations

import os
import subprocess
import sys

# ── 1. Install dependencies ────────────────────────────────────────────────────
print("Installing dependencies …")
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"]
)
print("Dependencies installed.\n")

# ── 2. Path setup ──────────────────────────────────────────────────────────────
_repo_root = os.path.abspath(os.path.dirname(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# ── 3. Hardware detection ──────────────────────────────────────────────────────
import torch
import numpy as np

def _detect_hardware() -> dict:
    info: dict = {}
    info["python"] = sys.version.split()[0]
    info["torch"]  = torch.__version__
    info["numpy"]  = np.__version__

    if torch.cuda.is_available():
        info["device"]      = "CUDA (GPU)"
        info["gpu_name"]    = torch.cuda.get_device_name(0)
        info["gpu_memory"]  = (
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
        info["cuda_version"] = torch.version.cuda
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["device"] = "MPS (Apple Silicon)"
    else:
        info["device"] = "CPU"

    # Check for Colab TPU
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
        info["device"] = "TPU (XLA)"
    except ImportError:
        pass

    return info

hw = _detect_hardware()
print("=" * 50)
print("  VERIDIC-DSA Hardware Summary")
print("=" * 50)
for k, v in hw.items():
    print(f"  {k:<16}: {v}")
print("=" * 50)

# ── 4. Seed ─────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print(f"\nGlobal random seed set to {SEED}")

# ── 5. Google Drive mount (optional) ──────────────────────────────────────────
DRIVE_CHECKPOINT_DIR = None

def mount_drive(checkpoint_subdir: str = "veridic_checkpoints") -> str:
    """
    Mount Google Drive and return path to checkpoint directory.

    Call this function to enable persistent checkpoint saving:

        from colab_setup import mount_drive
        ckpt_dir = mount_drive()
    """
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
        ckpt_dir = f"/content/drive/MyDrive/{checkpoint_subdir}"
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Drive mounted. Checkpoints will be saved to: {ckpt_dir}")
        return ckpt_dir
    except Exception as exc:
        print(f"Drive mount skipped: {exc}")
        ckpt_dir = os.path.join(_repo_root, "results", "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Using local checkpoint dir: {ckpt_dir}")
        return ckpt_dir

# ── 6. Quick import check ──────────────────────────────────────────────────────
try:
    import src.config as cfg  # noqa: F401
    print("\nPackage imports OK — ready to run experiments.")
except ImportError as e:
    print(f"\nImport error: {e}. Make sure you are in the repo root.")

print(
    "\nQuick-start:\n"
    "  from src.experiment import VeridicDSA\n"
    "  from src.adversary  import AdversaryType\n"
    "  sim = VeridicDSA(K=10, adversary_fraction=0.3,\n"
    "                   adversary_type=AdversaryType.SF, seed=42)\n"
    "  summary, revenues = sim.run(n_episodes=500, verbose=True)\n"
    "  print(summary)\n"
)
