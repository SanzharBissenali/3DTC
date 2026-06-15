"""
Analyze a single training run from its JSON file.

Example Usage:
    .venv/bin/python scripts/analyze_run.py outputs/G-equiv_2_hx020.json
    .venv/bin/python scripts/analyze_run.py outputs/G-equiv_2_hx020.json --exact
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make the repo root importable regardless of where this script is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np


def _f(x):
    """Parse scalar values that may be stored as strings (real or complex)."""
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    try:
        return float(s)
    except ValueError:
        return float(complex(s).real)


def load_run(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def summarize(d: dict, path: str) -> dict:
    sp = d["sim_params"]
    energy = [_f(x) for x in d["energy"]]
    evar = [_f(x) for x in d["energy_var"]]
    eom = [_f(x) for x in d["energy_eom"]]
    n_iter = len(energy)
    acc = [_f(x) for x in d["MCMC_accepted"]]
    tot = [_f(x) for x in d["MCMC_total"]]
    acc_ratio = [a / t if t else float("nan") for a, t in zip(acc, tot)]

    Lx = sp["Lx"][0]
    bc = sp["BC"][0]
    hx, hy, hz = sp["hx"][0], sp["hy"][0], sp["hz"][0]

    # Final ~10% of iterations: stable estimate of E
    tail = max(1, n_iter // 10)
    E_final = np.mean(energy[-tail:])
    E_final_std = np.std(energy[-tail:])

    print(f"\n=== {os.path.basename(path)} ===")
    print(f"  System : Lx={Lx} {bc}, hx={hx} hy={hy} hz={hz}, dtype={sp['param_dtype'][0]}")
    print(f"  Network: arch={sp['architecture_type'][0]}, noninv={sp['n_chann_noninv']}, "
          f"inv={sp['n_chann_inv']}, k={sp['kernel_size_noninv'][0]}, n_params={sp['n_params'][0]}")
    print(f"  Training: {n_iter} iters, dt={sp['dt'][0]}, diag_shift={sp['diag_shift'][0]}")
    print(f"  Energy : initial={energy[0]:.4f} -> final(mean of last {tail})={E_final:.6f} "
          f"± {E_final_std:.1e}  (last Var(H)={evar[-1]:.2e})")
    print(f"  MCMC   : final acceptance={acc_ratio[-1]:.3f}, "
          f"min over run={min(acc_ratio):.3f}")
    return {
        "energy": energy, "energy_var": evar, "energy_eom": eom,
        "acc_ratio": acc_ratio, "Lx": Lx, "bc": bc,
        "hx": hx, "hy": hy, "hz": hz, "E_final": E_final, "E_final_std": E_final_std,
    }


def exact_E0(Lx, bc, hx, hy, hz, dtype="float64"):
    """Compute exact ground-state energy via Lanczos. Only feasible for small Lx."""
    import netket as nk
    from model.geometry import ToricCodeGeometry
    from model.hamiltonian import create_hamiltonian

    g = ToricCodeGeometry(Lx, Lx, bc)
    hi = nk.hilbert.Spin(s=0.5, N=g.N)
    if hy != 0:
        dtype = "complex"
    H = create_hamiltonian(hi, g.vertex_all, g.plaq_all, g.bonds,
                           hx=hx, hy=hy, hz=hz, dtype=dtype)
    evals = nk.exact.lanczos_ed(H, k=1, compute_eigenvectors=False)
    return float(np.real(np.asarray(evals).ravel()[0])), g.N


def plot(d: dict, summary: dict, out_path: str, exact_E0_val: float | None = None):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    iters = np.arange(len(summary["energy"]))

    # Energy trace
    ax = axes[0, 0]
    ax.plot(iters, summary["energy"], "-", lw=1, label="⟨H⟩")
    ax.fill_between(iters,
                    np.array(summary["energy"]) - np.array(summary["energy_eom"]),
                    np.array(summary["energy"]) + np.array(summary["energy_eom"]),
                    alpha=0.3)
    if exact_E0_val is not None:
        ax.axhline(exact_E0_val, ls="--", color="k", lw=1, label=f"exact E₀={exact_E0_val:.4f}")
    ax.set_xlabel("TDVP iteration")
    ax.set_ylabel("energy")
    ax.set_title("Energy convergence")
    ax.legend()

    # Var(H) trace, log scale
    ax = axes[0, 1]
    var = np.array(summary["energy_var"])
    var = np.clip(var, 1e-12, None)
    ax.semilogy(iters, var)
    ax.set_xlabel("TDVP iteration")
    ax.set_ylabel("Var(H)")
    ax.set_title("Variance ↘ 0 means exact eigenstate")

    # Acceptance ratio
    ax = axes[1, 0]
    ax.plot(iters, summary["acc_ratio"])
    ax.set_xlabel("TDVP iteration")
    ax.set_ylabel("MCMC acceptance")
    ax.set_ylim(0, max(0.1, max(summary["acc_ratio"]) * 1.1))
    ax.set_title("Acceptance (low = single-flip sampler struggling)")

    # Magnetizations
    ax = axes[1, 1]
    op = d.get("order_params", {})
    for key, color in [("magnetization_Xmean", "C0"),
                       ("magnetization_Ymean", "C1"),
                       ("magnetization_Zmean", "C2")]:
        series = op.get(key, [])
        if not series:
            continue
        means = [np.mean(s) for s in series]
        ax.plot(np.linspace(0, len(iters), len(means)), means,
                "o-", ms=3, color=color, label=key.split("_")[1])
    ax.set_xlabel("TDVP iteration (subsampled)")
    ax.set_ylabel("⟨σ⟩ averaged over qubits")
    ax.set_title("Magnetization growth")
    ax.legend()

    fig.suptitle(f"Lx={summary['Lx']} {summary['bc']}  hx={summary['hx']} hy={summary['hy']} hz={summary['hz']}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"  Saved figure -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", help="Path to a G-equiv_*.json file")
    ap.add_argument("--exact", action="store_true",
                    help="Also compute exact GS energy via Lanczos (small Lx only)")
    args = ap.parse_args()

    d = load_run(args.json_path)
    s = summarize(d, args.json_path)

    exact_val = None
    if args.exact:
        try:
            exact_val, N = exact_E0(s["Lx"], s["bc"], s["hx"], s["hy"], s["hz"])
            err = s["E_final"] - exact_val
            rel = abs(err) / abs(exact_val)
            print(f"  EXACT  : E₀={exact_val:.6f}  (N={N} qubits, 2^N={2**N})")
            print(f"  ERROR  : ΔE={err:+.6f}  rel={rel:.2e}")
        except Exception as e:
            print(f"  Exact diag skipped: {e}")

    fig_path = Path(args.json_path).with_suffix(".png")
    plot(d, s, str(fig_path), exact_val)


if __name__ == "__main__":
    main()
