"""Env-var-driven wrapper around the self-contained 3D-TC exact-diagonalisation
reference (`Three_TC/tests/colab_exact_diag.py`) for NERSC batch jobs.

The reference script hardcodes its parameters in a `PARAMS` block. This wrapper
reads them from environment variables instead, so a single sbatch script + Slurm
job array can sweep a field (e.g. h_z) without editing source. Anything unset
falls back to the reference defaults.

Usage (inside a Slurm job):
    HX=0.3 HZ=0.5 L=2 OUT=$PSCRATCH/tc_ed/out.json python run_ed.py
"""
import os
import sys
from pathlib import Path

# Import `run` from the in-repo reference (lives in Three_TC/tests).
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Three_TC" / "tests"))
from colab_exact_diag import run  # noqa: E402


def _f(name, default):
    v = os.environ.get(name)
    return float(v) if v is not None else default


L = int(os.environ.get("L", 2))
params = {
    "Lx": L, "Ly": L, "Lz": L,
    "hx": _f("HX", 0.2), "hy": _f("HY", 0.0), "hz": _f("HZ", 0.2),
    "J": _f("J", 1.0),
    "k": int(os.environ.get("K", 2)),
    "out": os.environ.get("OUT"),
}

if __name__ == "__main__":
    print(f"[run_ed] params = {params}", flush=True)
    run(params)
