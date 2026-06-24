"""
Three_TC/optimize.py
─────────────────────────────────────────────────────────────────────────────
Hyperparameter-search driver for ToricCNN_full against exact diagonalisation.

Figure of merit (minimise):
    delta = |E_NQS - E_exact| / |E_exact|
at a chosen h_z taken from the precomputed reference `threed_bosonic.json`
(L=2 PBC bosonic, hx=0.2, J=1).  No local ED — E_exact is read from the file.

Also reported (reference-free) is
    Vscore = N · Var(H) / <H>^2,
which via   delta ≈ Var/(gap·|E0|)   diagnoses whether a run is variational-
limited (raise ansatz capacity) or noise/optimisation-limited (more
samples/steps).  `meas_floor = E_err/|E_exact|` is the smallest delta this run
could even resolve at its sample budget.

Construction + the optimisation loop are the shared `Three_TC.builders` ones, so
what we score here is exactly what `train.py` would train.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import numpy as np

from Three_TC.builders import build_state, run_loop, with_defaults
from Three_TC.validation import nqs_observables

REF_FILE = "threed_bosonic.json"
HX = 0.2          # the field that generated threed_bosonic.json
DEFAULT_IDX = 8   # h_z ≈ 0.316, gap ≈ 0.94 — the agreed figure-of-merit point


def load_ref(idx: int = DEFAULT_IDX, ref_file: str = REF_FILE) -> Dict[str, float]:
    d = json.load(open(ref_file))
    return {"hz": d["h_z"][idx], "E_exact": d["E0"][idx], "gap": d["gap"][idx],
            "B_exact": d["B"][idx], "Mz_exact": d["Mz"][idx], "idx": idx}


def evaluate(config: Dict[str, Any], *,
             hz_idx: int = DEFAULT_IDX,
             n_iter: int = 80, dt: float = 2e-2,
             lr_min: Optional[float] = 2e-3, diag_shift: float = 2e-4,
             n_samples: int = 2048, n_chains: int = 16, n_discard: int = 8,
             seed: int = 0, ref_file: str = REF_FILE,
             ledger: Optional[str] = "outputs/opt_ledger.jsonl",
             label: str = "", verbose: bool = True,
             probe_every: int = 0) -> Dict[str, Any]:
    """Build + train + score one ToricCNN_full config at one h_z. Appends a row
    to `ledger` (JSONL) and returns it.

    `config` keys (all optional): hidden, noninv_channels, n_noninv, inv_hidden.
    Training knobs come in as explicit args so a sweep can vary them freely.
    """
    ref = load_ref(hz_idx, ref_file)
    run_cfg = with_defaults({
        "L": 2, "bc": "PBC", "model": "bosonic",
        "hx": HX, "hy": 0.0, "hz": ref["hz"], "J": 1.0,
        "arch": "ToricCNN_full",
        "hidden": config.get("hidden", 8),
        "noninv_channels": config.get("noninv_channels", 1),
        "n_noninv": config.get("n_noninv", 1),
        "inv_hidden": tuple(config.get("inv_hidden", ()) or ()),
        "n_samples": n_samples, "n_chains": n_chains, "n_discard": n_discard,
        "seed": seed,
    })

    geo, hi, Ham, vs, xz = build_state(run_cfg)

    t0 = time.time()
    if probe_every:
        # Run in segments, scoring delta at each boundary — a cheap convergence
        # curve (no per-step vs.expect penalty). lr decay is applied per segment
        # so the global schedule still spans the full n_iter.
        Ex = ref["E_exact"]
        done = 0
        while done < n_iter:
            k = min(probe_every, n_iter - done)
            # piecewise cosine: reuse the same (dt, lr_min) each segment
            run_loop(vs, Ham, n_iter=k, dt=dt, diag_shift=diag_shift, lr_min=lr_min)
            done += k
            e = float(np.real(vs.expect(Ham).mean))
            print(f"    [{label}] {done:>3}/{n_iter}: delta={abs(e-Ex)/abs(Ex):.3e}"
                  f"  E={e:+.4f}", flush=True)
    else:
        run_loop(vs, Ham, n_iter=n_iter, dt=dt, diag_shift=diag_shift, lr_min=lr_min)
    runtime = time.time() - t0

    obs = nqs_observables(vs, Ham, geo, xz_stabs=xz)
    E = obs["E0"]
    delta = abs(E - ref["E_exact"]) / abs(ref["E_exact"])
    row = {
        "label": label, "hz": ref["hz"], "gap": ref["gap"],
        "delta": delta, "meas_floor": obs["E_err"] / abs(ref["E_exact"]),
        "Vscore": obs["Vscore"], "E_var": obs["E_var"],
        "E_nqs": E, "E_err": obs["E_err"], "E_exact": ref["E_exact"],
        "dB": abs(obs["B_p_mean"] - ref["B_exact"]),
        "dMz": abs(obs["sz_mean"] - ref["Mz_exact"]),
        "A_v_mean": obs["A_v_mean"], "B_p_mean": obs["B_p_mean"], "sz_mean": obs["sz_mean"],
        "n_params": int(vs.n_parameters), "runtime_s": runtime,
        "n_iter": n_iter, "n_samples": n_samples, "dt": dt, "lr_min": lr_min,
        "diag_shift": diag_shift, "seed": seed,
        "hidden": config.get("hidden", 8),
        "noninv_channels": config.get("noninv_channels", 1),
        "n_noninv": config.get("n_noninv", 1),
        "inv_hidden": list(config.get("inv_hidden", ()) or ()),
    }
    if ledger:
        os.makedirs(os.path.dirname(ledger), exist_ok=True)
        with open(ledger, "a") as f:
            f.write(json.dumps(row) + "\n")
    if verbose:
        print(f"[{label or 'run':>22}] delta={delta:.3e} (floor {row['meas_floor']:.1e})"
              f"  Vscore={obs['Vscore']:.2e}  E={E:+.4f} vs {ref['E_exact']:+.4f}"
              f"  dMz={row['dMz']:.3f}  np={row['n_params']}  {runtime:.0f}s", flush=True)
    return row


def _cli():
    import argparse
    p = argparse.ArgumentParser(
        description="Verify ToricCNN_full vs ED at one h_z from "
                    "threed_bosonic.json. Prints delta=|E_NQS-E_exact|/|E_exact| "
                    "(+ Vscore). SR uses the dense QGT automatically.")
    p.add_argument("--hz_idx", type=int, default=DEFAULT_IDX,
                   help="index into the 20-pt h_z grid (8 → h_z≈0.316, the FOM)")
    p.add_argument("--n_iter", type=int, default=250)
    p.add_argument("--n_samples", type=int, default=2048)
    p.add_argument("--dt", type=float, default=2e-2)
    p.add_argument("--lr_min", type=float, default=2e-3)
    p.add_argument("--diag_shift", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe_every", type=int, default=50,
                   help="print delta every K steps (0=off); 50 reproduces the "
                        "cosine-restart schedule of the reported run")
    p.add_argument("--hidden", type=int, default=8)
    p.add_argument("--noninv_channels", type=int, default=1)
    p.add_argument("--n_noninv", type=int, default=1)
    a = p.parse_args()
    evaluate({"hidden": a.hidden, "noninv_channels": a.noninv_channels,
              "n_noninv": a.n_noninv},
             hz_idx=a.hz_idx, n_iter=a.n_iter, n_samples=a.n_samples,
             dt=a.dt, lr_min=a.lr_min, diag_shift=a.diag_shift, seed=a.seed,
             probe_every=a.probe_every, label="verify")


if __name__ == "__main__":
    _cli()
