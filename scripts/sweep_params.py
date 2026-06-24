#!/usr/bin/env python
"""
scripts/sweep_params.py
─────────────────────────────────────────────────────────────────────────────
Enumerate the hyperparameter grid for the SLURM job array
(`nersc/submit_nqs_sweep.sh`). Edit the axis lists below to change the sweep —
this file is the single source of truth for what gets explored.

    python scripts/sweep_params.py --count   # grid size N (set --array=0-(N-1))
    python scripts/sweep_params.py 7         # CLI flags for combo #7

Each combo prints a flag string to append to `python -m Three_TC.train ...`,
covering the four axes chosen for the first sweep: learning rate (dt, lr_min),
diag_shift, pre-Wilson capacity (n_noninv, noninv_channels), and post-Wilson
width (inv_hidden). The array task id selects the combo, exactly like
`submit_ed_sweep.sh` derives HZ from SLURM_ARRAY_TASK_ID.
"""
import itertools
import sys

# --- sweep axes (edit me) ----------------------------------------------------
DT_LRMIN    = [(0.02, 0.002), (0.01, 0.001)]       # (dt, lr_min) pairs
DIAG_SHIFT  = [1e-4, 1e-3, 1e-2]                    # QGT/SR regularization
PRE_WILSON  = [(1, 1), (2, 1), (1, 4)]             # (n_noninv, noninv_channels)
POST_WILSON = [(8,), (16, 16)]                      # inv_hidden widths
# product size = 2 * 3 * 3 * 2 = 36 combos


def combos():
    return list(itertools.product(DT_LRMIN, DIAG_SHIFT, PRE_WILSON, POST_WILSON))


def flags(combo) -> str:
    (dt, lr_min), ds, (n_noninv, nc), inv_hidden = combo
    return " ".join([
        f"--dt {dt}", f"--lr_min {lr_min}", f"--diag_shift {ds}",
        f"--n_noninv {n_noninv}", f"--noninv_channels {nc}",
        "--inv_hidden " + " ".join(str(w) for w in inv_hidden),
    ])


def main(argv):
    cs = combos()
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    if argv[0] == "--count":
        print(len(cs))
        return
    idx = int(argv[0])
    if not 0 <= idx < len(cs):
        sys.exit(f"index {idx} out of range [0, {len(cs)})")
    print(flags(cs[idx]))


if __name__ == "__main__":
    main(sys.argv[1:])
