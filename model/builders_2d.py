"""
model/builders_2d.py
─────────────────────────────────────────────────────────────────────────────
Single source of truth for turning a `config` dict into a runnable 2D VMC setup
for the transformer NQS — the 2D analogue of `Three_TC/builders.py`, kept
independent of the 3D package (per CLAUDE.md "2D and 3D are separate packages").

Reused, geometry-agnostic pieces are imported from the shared top-level modules:
the vertex-cluster sampler rules (`simulation/custom_sampler.py`), the 2D
geometry + Hamiltonian (`model/`). The generic optimization loop is small and is
kept here so this driver does not depend on `Three_TC`.

Config keys (all optional except L; see DEFAULTS):
    System      : L (req), bc
    Hamiltonian : hx, hy(=0 only), hz, J
    Architecture: arch (="factored_transformer"), d, n_heads, n_layers, mlp_ratio
    Sampling    : n_samples, n_chains, n_discard, chunk_size, n_sweeps, seed
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx

from simulation.custom_sampler import WeightedRule, MultiRule
from model.geometry import ToricCodeGeometry
from model.hamiltonian import create_hamiltonian
from model.transformer import FactoredAttentionWavefunction


DEFAULTS: Dict[str, Any] = {
    "bc": "OBC",
    "hx": 0.0, "hy": 0.0, "hz": 0.0, "J": 1.0,
    "arch": "factored_transformer",
    "d": 16, "n_heads": 4, "n_layers": 4, "mlp_ratio": 2,
    "n_samples": 4096, "n_chains": 16, "n_discard": 8,
    "chunk_size": None, "n_sweeps": None, "seed": 0,
}


def with_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of `config` with DEFAULTS filled in and `dtype` derived."""
    cfg = {**DEFAULTS, **config}
    if "L" not in cfg:
        raise KeyError("config must specify system size 'L'")
    if cfg["hy"] != 0.0:
        raise NotImplementedError(
            "hy != 0 (sign problem) needs a complex ansatz; not supported yet.")
    cfg.setdefault("dtype", "float64")
    return cfg


# =============================================================================
# Builders
# =============================================================================

def build_geometry(config: Dict[str, Any]):
    L = config["L"]
    return ToricCodeGeometry(Lx=L, Ly=L, bc=config.get("bc", "OBC"))


def build_hamiltonian(config: Dict[str, Any], geo, hi):
    return create_hamiltonian(
        hi=hi, vertex_all=geo.vertex_all, plaq_all=geo.plaq_all, bonds=geo.bonds,
        hx=config.get("hx", 0.0), hy=config.get("hy", 0.0),
        hz=config.get("hz", 0.0), J=config.get("J", 1.0),
        dtype=config.get("dtype", "float64"))


def build_model(config: Dict[str, Any], geo):
    """Instantiate the ansatz named by `config['arch']`."""
    arch = config.get("arch", "factored_transformer")
    if arch == "factored_transformer":
        return FactoredAttentionWavefunction(
            N=geo.N, d=config.get("d", 16), n_heads=config.get("n_heads", 4),
            n_layers=config.get("n_layers", 4), mlp_ratio=config.get("mlp_ratio", 2))
    raise ValueError(f"unknown arch {arch!r} (expected 'factored_transformer')")


def build_sampler(config: Dict[str, Any], hi, geo):
    """WeightedRule(LocalRule, vertex-cluster MultiRule) — the topological-phase fix.

    Same recipe as the 3D stack: each cluster is a vertex star's edges. OBC
    boundary stars are truncated (fewer edges); we strip the -1 padding in
    geo.vertex_all and pad each cluster back to a common width by repeating its
    last valid edge. MultiRule's `.at[cluster].set(-...)` is idempotent under
    duplicate indices, so a padded cluster flips exactly its distinct edges.
    """
    hetero = geo.get_vertex_all_hetero()                   # -1 stripped, ragged
    width = max(len(v) for v in hetero)
    vertex_clusters = np.array([v + [v[-1]] * (width - len(v)) for v in hetero])
    samp_ratio = geo.N / len(vertex_clusters)
    weighted = WeightedRule(
        (samp_ratio / (samp_ratio + 1), 1 - samp_ratio / (samp_ratio + 1)),
        [nk.sampler.rules.LocalRule(), MultiRule(vertex_clusters)],
    )
    n_sweeps = config.get("n_sweeps") or geo.N * 2
    common = dict(rule=weighted, n_chains=config.get("n_chains", 16), dtype=jnp.int8)
    # NetKet renamed the constructor kwarg n_sweeps -> sweep_size; support both.
    try:
        return nk.sampler.MetropolisSampler(hi, sweep_size=n_sweeps, **common)
    except TypeError:
        return nk.sampler.MetropolisSampler(hi, n_sweeps=n_sweeps, **common)


def build_state(config: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
    """Build everything: returns (geo, hi, Ham, vs)."""
    cfg = with_defaults(config)
    geo = build_geometry(cfg)
    hi = nk.hilbert.Spin(s=1/2, N=geo.N)
    Ham = build_hamiltonian(cfg, geo, hi)
    model = build_model(cfg, geo)
    sa = build_sampler(cfg, hi, geo)
    vs = nk.vqs.MCState(sa, model, n_samples=cfg["n_samples"],
                        n_discard_per_chain=cfg["n_discard"],
                        chunk_size=cfg["chunk_size"], seed=cfg["seed"])
    return geo, hi, Ham, vs


# =============================================================================
# Shared optimization loop
# =============================================================================

def run_loop(vs, Ham, n_iter: int, dt: float, diag_shift: float,
             on_step: Optional[Callable] = None, lr_min: Optional[float] = None,
             qgt: str = "minsr", start_step: int = 0,
             total_iter: Optional[int] = None):
    """VMC + Sgd + SR for n_iter steps.

    Learning rate: constant `dt`, or a cosine decay from `dt` to `lr_min` across
    `total_iter` steps if `lr_min` is given.

    `qgt` selects the SR representation:
      - "minsr": the kernel/"S-matrix" trick (netket.experimental VMC_SRt, the
        s42005-024-01732-4 method) — solves in sample space, the right choice when
        n_params > n_samples (the transformer regime). **Default here.**
      - "dense": QGTJacobianDense (form the matrix, direct solve); best when
        n_params <= n_samples.
      - "onthefly": NetKet's matrix-free CG.
      - "auto": dense when n_params <= n_samples, else minsr.

    `start_step`/`total_iter` support resuming: run `n_iter` more steps while the
    cosine-LR schedule and the `on_step` index are offset by `start_step`.
    """
    total_iter = total_iter or n_iter
    if lr_min is not None and lr_min != dt:
        import optax
        base = optax.cosine_decay_schedule(init_value=dt, decay_steps=total_iter,
                                           alpha=lr_min / dt)
        lr = (lambda s: base(s + start_step)) if start_step else base
    else:
        lr = dt
    opt = nk.optimizer.Sgd(learning_rate=lr)

    mode = qgt
    if mode == "auto":
        mode = "dense" if vs.n_parameters <= vs.n_samples else "minsr"

    if mode == "minsr":
        driver = nkx.driver.VMC_SRt(Ham, opt, diag_shift=diag_shift,
                                    variational_state=vs)
    else:
        if mode == "dense":
            sr = nk.optimizer.SR(qgt=nk.optimizer.qgt.QGTJacobianDense,
                                 diag_shift=diag_shift, holomorphic=False)
        else:  # onthefly
            sr = nk.optimizer.SR(diag_shift=diag_shift)
        driver = nk.driver.VMC(Ham, opt, variational_state=vs, preconditioner=sr)

    # Drive via the stable `driver.run(..., callback=...)` API (older NetKet's
    # `driver.advance` was removed). The driver already estimates the energy each
    # step, so `on_step` reads that Stats object rather than re-sampling.
    if on_step is not None:
        def _cb(step, log_data, dr):
            E = getattr(dr, "_loss_stats", None)
            if E is None:
                E = log_data.get(getattr(dr, "_loss_name", "Energy"))
            on_step(start_step + step, E, dr.state)
            return True
        driver.run(n_iter, callback=_cb, show_progress=False)
    else:
        driver.run(n_iter, show_progress=False)
    return vs
