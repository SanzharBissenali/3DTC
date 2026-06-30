# Progress log — 3D Toric Code NQS extension

Updated alongside `notes/3D_extension_plan.md` as work progresses. Numbered
checkpoints record discrete milestones; the most recent is at the top.

---

## Reference — exact unperturbed (h=0) ground-state energies

The unperturbed `H = −J·Σ A_v − J·Σ B_p` (J=1) is a commuting-stabiliser
Hamiltonian: the ground state satisfies **every** term = +1, so
**`E0 = −(#A_v + #B_p)`** exactly, at any L (BC only changes the counts). The
degeneracy (8 on the 3-torus, etc.) does *not* change E0. This is the **only exact
energy anchor at L>2** — train at h=0 and the net must hit these to machine precision.

- **PBC:** `#A_v = L³`, `#B_p = 3L³` → **`E0 = −4L³`**.
- **OBC** (complete-face plaquettes only, all L³ vertex stars kept):
  `#A_v = L³`, `#B_p = 3(L−1)²L` → **`E0 = −(L³ + 3(L−1)²L)`**.

| L | N (PBC) | E0 PBC | N (OBC) | E0 OBC |
|---|---|---|---|---|
| 2 | 24  | −32   | 12   | −14   |
| 4 | 192 | −256  | 144  | −172  |
| 6 | 648 | −864  | 540  | −666  |
| 8 | 1536| −2048 | 1344 | −1688 |

Verified: ED at L=2 gives exactly −32 (PBC) / −14 (OBC); counts from
`ThreeD_ToricCodeGeometry(L,L,L,bc)` (`len(vertex_all)`, `len(plaq_all)`).

---

## Checkpoint 6 — `ToricCNN_gridinv`: grid-conv invariant block (kernel → L)

The geometry-exact invariant conv reaches topological long-range order only by
spanning `Θ(L)`, which is costly in 3D (plaquette stencil has an `O=3` orientation
axis + 15-tap footprint per layer). Added **`ToricCNN_gridinv`**
(`Three_TC/model/networks.py:621`) — the **2D-paper architecture generalised to 3D**:
keep the Wilson sandwich, but make the *invariant* block a standard `nn.Conv3D` whose
kernel scales to `L`.

- After the per-channel Wilson product, `plaq_grid_layout` **folds** the flux field
  onto the cube-cell grid — each plaquette → cell `floor(centre)`, normal → one of 3
  channels, `(L,L,L,3·C)` + occupancy mask. This is the 2D "plaquettes on dual-lattice
  vertices" picture; a 3D cell holds 3 plaquettes folded into channels, collapsing the
  within-cell ½-offsets (**half-offset approximation** `GeoConv3D` avoids).
- `nn.Conv3D(kernel_size=L)` (override `--kernel_size`), CIRCULAR pad (PBC) / zero pad
  (OBC); final conv → `O` channels then a **masked mean** over occupied cells → real
  `log ψ`. Both BC supported; pre-Wilson noninv block stays geometry-exact.
- **Verified**: builds + forward + sampler at L=2 OBC (N=12, params 3543) and PBC;
  **Colab confirmed training to the same ~1e-3→1e-5 `delta`** as `ToricCNN_full` at L=2
  OBC. `nqs_sweep_colab_exp.ipynb` now defaults to it, with arch-named output folders
  `outputs/gridinv/…` and `NONINV`/`INV`/`KERNEL` knobs. Docs: `nqs_architecture.md`,
  `training_cli.md`, `log_and_plan.md`.
- **Use:** the cheap path to `kernel ∝ L` coverage at L≥3; A/B vs `ToricCNN_full`
  isolates half-offset exactness against plain-`nn.Conv3D` speed/scaling.

---

## Checkpoint 5 — Scaling the bosonic NQS past L=2 (no exact reference)

The L=2 architecture comparison is settled: the **symmetry-aware `ToricCNN_full`
reaches a relative energy error ~100× lower** than the symmetry-unaware pure CNN
(`GeoCNN`, no-Wilson control at matched params). Goal now: **scale to L≥3** and show
training is stable and the two architectures converge to **distinguishable energies**.

### What scales for free (verified by review + cheap proxy)

- `Three_TC/builders.py` + `train.py` carry **no L=2 hardcoding** in the
  VMC/sampler/model path. Network params are **L-independent** (site-shared weights):
  `ToricCNN_full`=2571, `GeoCNN`=1839 at **both** L=2 and L=3. Sampler `n_sweeps`
  auto-scales to `2N`; dense QGT stays cheap (~2.6k params).
- Smoke (no ED): both archs `build_state` + `expect(H)` at L=3 PBC (N=81). VMC at
  L=3/4 is laptop-fine (Checkpoint 1 ran L=4 h=0). **The 8 GB OOM rule is ED-only.**

### What does NOT scale — the exact reference (now guarded in the notebook)

ED is tractable only at **L=2** (PBC N=24 / OBC N=12). At L≥3 the Hilbert space
explodes (PBC 2⁸¹, OBC 2⁵⁴) ⇒ no `E_exact`. `nqs_sweep_colab_exp.ipynb` configure
cell now sets `HAS_GROUND_TRUTH = (L==2)`:

- **L=2** keeps the exact path: `--hz_preset` (PBC) or on-the-fly `eigsh` ED (OBC).
- **L≥3** passes `--hz` directly, **no `--exact_E0`** → `delta=None` (handled at
  `train.py:124`); runs are compared by **final variational energy** (lower wins).
- `N_SWEEPS=0` ⇒ code default `2N`. Intro markdown documents the L-scaling regime.
- `HZ_PRESETS` (`train.py:54`) are L=2-only — they must not be used at L>2.

### Next

On Colab, A/B `ToricCNN_full` vs `GeoCNN` at L=3 PBC (matched params via the
param-count helper, same hx,hz): confirm `R̂≈1` / shrinking `√Var` and a clear
energy gap. Expect to **raise `diag_shift`** for the larger SR solve.

---

## Checkpoint 4 — OBC enabled for the symmetry-aware CNN (bosonic, L=2)

OBC is now a first-class boundary condition via the existing `--bc OBC` flag (no new
flag). PBC paths are **byte-identical**; bosonic only (fermionic OBC deferred).

### What changed

- **Geometry** — OBC `plaq_all` keeps only **complete 4-edge faces** (incomplete
  boundary faces dropped: they'd corrupt the ED Z-strings *and* the Wilson product).
  New `plaq_centers` + `_plaq_center_to_idx` map (single source of truth for the
  plaquette conv).
- **`KernelManager3D`** — no longer rejects OBC. Edge stencils **mask** out-of-box
  neighbours instead of `% L` wrapping; plaquette stencils are now **coordinate-
  based** (via the centre map) for both BCs. `build_model` errors on Vanilla\*+OBC
  (CIRCULAR padding is PBC-only).
- **Sampler** (`build_sampler`) — clusters strip `-1` and pad to width 6; fixes the
  L=2 OBC divide-by-zero (no full bulk stars exist) and the `MultiRule` `-1` flip.
- **`bc` threaded** through `validation.py` / `optimize.py` (default PBC).
- **New Colab** `colab/3D_TC_OBC_ED_colab.ipynb` — self-contained OBC ED, 5×5
  `(hx,hz)` grid → per-point reference JSONs + E₀/gap heatmaps.

### Geometry facts (L=2 OBC)

`N = 3L³−3L² = 12`, 6 complete-face plaquettes, 8 truncated vertex stars (no full
bulk star exists at L=2). All `A_v`/`B_p` commute. Per-orientation edge/plaquette
counts are uniform (cube symmetry), so the `(3,P,S)` stencil tensors stay rectangular.

### Verification

- PBC plaquette stencils **byte-identical** to the old `pidx` arithmetic.
- `test_geometry.py` OBC assertions pass at L=2 and L=3.
- Inline notebook ED == repo `model/exact_diag.py`: `E₀(hx=hz=0.2)=−14.279396`.
- `ToricCNN_full` under OBC → **`eps_E=1.1e-4`** (150 iters, dense QGT, 4096 samples).

### Next

Sweep OBC vs PBC `eps_E` across the grid to confirm the OBC-does-better claim; then
fermionic OBC (needs `fermionic_decoration._idx` BC gating).

---

## Checkpoint 3 — MCMC acceptance pinned down as phase-dependent; lr×diag_shift sweep launched

### The acceptance puzzle, resolved

Training + MCMC sampling are **stable in the topological phase but collapse in the
trivial phase** — and this is **correct physics, not a bug**. The sampler is
`WeightedRule(LocalRule 75%, MultiRule star-flip 25%)`; star-cluster moves are
"free" (always accepted) only when ψ is A_v-symmetric. Only `h_z` (σ_z) breaks
A_v conservation; `h_x` (σ_x) commutes with A_v.

- `h_z=0`: acceptance pins at **0.25** (= MultiRule weight).
- Topological side (`h_z≈0.1–0.2`, transition ≈0.3): mild A_v breaking → **~0.2**,
  stable `R_hat≈1`. Healthy.
- Deep trivial (`h_z=0.553`, `easy` preset): true state strongly polarized → A_v
  genuinely broken *and* distribution sharply peaked → both cluster and local
  moves reject → **~0.01**. Expected, not a stall to fix.

### How it was diagnosed (de-confounded)

1. Reproduced the pre-GeoConv3D run (`VanillaWilsonCNN`, plain grid conv + Wilson
   sandwich, added to `networks.py`): held 0.2 → matched the old good run.
2. **De-confound run — GeoConv3D at the small `[1]/[8,1]` shape**: also held 0.2.
   → **the GeoConv3D kernel is exonerated**; it was never the cause.
3. Compared the one stalling run (`hz_preset easy`, `n_noninv=2`, 1024 chains)
   vs a stable one (`hz=0.2`, `n_noninv=1`, 32 chains) — same code (commit
   `3180500`), only config differed. Driver: **`h_z` (phase)**, secondary:
   noninv depth; the `1024 chains / 8 samples-per-chain` also made that run's
   `R_hat`/`tau_corr` unreliable.

### Rules adopted

- **Judge runs by `R_hat≈1`, stable `tau_corr`, converged energy — NOT by
  `mcmc_acceptance`** (it is phase-dependent and drops legitimately toward the
  trivial phase).
- Keep ≥ a few hundred samples/chain (16–64 chains at L=2).
- Let `h_z` drive symmetry breaking; keep `n_noninv` moderate (1–2).
- New CLI knobs added: `VanillaCNN`, `VanillaWilsonCNN` (`--noninv_random`),
  `--kernel_size`, `--vanilla_depth`. Full flag reference: `notes/training_cli.md`.

### In flight

`hz_preset=hard` (h_z=0.118, deep topological, small gap Δ≈0.062) **lr×diag_shift
sweep** (3×3, both inv & noninv = 2 layers, `n_iter=300`), split across two Colab
notebooks under `--wandb_group hard_lr_ds_sweep`:
`dt ∈ {5e-3, 1e-2, 2e-2}` × `diag_shift ∈ {1e-3, 3e-3, 1e-2}`, `lr_min = dt/10`.
Goal: the (lr, diag_shift) pair with **lowest `delta`** (% from exact E₀=−32.297),
gated on `R_hat≈1`. Expectation: best near mid-grid (`dt≈1e-2`, `diag_shift≈3e-3`).

---

## Checkpoint 2 — Validation harness + fermionic Hamiltonian (both models scorable)

### What's built

- **`Three_TC/validation.py`** — NQS goodness harness scoring ansätze against the
  Colab L=2 exact reference (expectation-value JSON). Metrics per
  (model, architecture, config, h_z regime): `eps_E`, `Vscore`, absolute
  deviations `dA, dB, dMz, dMx` each with MC error + pull, plus cost
  (`n_params, runtime_s`). Functions: `load_reference`/`find_reference`,
  `build_model`, `build_sampler`, `_mean_operators`, `nqs_metrics`,
  `train_one(fermionic=…)`, `run_validation(fermionic=…)`. See `notes/pipeline.md`.
- **`create_hamiltonian_fermionic`** in `Three_TC/model/hamiltonian.py` — the
  NetKet decorated-plaquette Hamiltonian (B̃_p = ZZZZ·XX from
  `fermionic_plaquettes`), for training the fermionic NQS. Bosonic version
  unchanged.
- **`colab_exact_diag.py` fermionic mode** — `PARAMS["fermionic"]=True` decorates
  the plaquettes (self-contained port), emits a JSON tagged `"model":"fermionic"`
  with `B_p_mean = ⟨B̃_p⟩`.
- **Notebook** (`2D_TC_phase_diag.ipynb`) — added h_z-derivative plots for
  ⟨A_v⟩/⟨B_p⟩/⟨σ_z⟩ (2D, rotated-surface, 3D bosonic, 3D fermionic), and a
  validation section (driver runs both models; table + Pareto + claim panel).

### Key result (verified, corrects the handoff)

`ToricCNN` is **exactly** global-flip symmetric: `log ψ(x)=log ψ(−x)` to 0.0, so
it is pinned to **⟨σ_z⟩=0** and ⟨A_v⟩=1 at all parameters — the handoff's
earlier "⟨σ_x⟩=0" was a slip (σ_x is free, ≈0.96). `ToricCNN_full`'s
non-invariant block breaks this (diff jumps 1e-7→~1 when perturbed). So the
architecture discriminators under the h_z sweep are **Δ⟨σ_z⟩ and Δ⟨A_v⟩**.

### Verification (cheap proxies; no local 2²⁴ ED)

- Fermionic NetKet Ham: 32 terms at h=0 (8 `XXXXXX` + 24 `ZZZZXX`, weight-6,
  coef −1); supports match `fermionic_plaquettes` exactly; VMC-compatible.
- Colab fermionic ED: geometry + decoration indexing identical to the repo;
  **matvec matches the verified `hamiltonian_linop` to 2e-14** on a random vector.
- Both architectures train (2-step smoke) for bosonic and fermionic.

### Same ansatz, both models

`ToricCNN`/`ToricCNN_full` serve both models: the decoration changes only the
plaquette; the vertex star A_v (what the Wilson product enforces) is unchanged.

### Next

Produce the 6 Colab reference JSONs (3 regimes × {bosonic, fermionic}, hx=0.2),
run `run_validation` for both, read the claim panel. Then scale (L=3: lose the
exact reference, lean on V-score / stabilizer saturation).

---

## Checkpoint 1 — Minimal symmetric-only network working at L=2,3,4 PBC, h=0

### What's built

Under `Three_TC/`:
```
Three_TC/
├── model/
│   ├── geometry.py        3D lattice, PBC + OBC, vertex_all (6-tuples),
│   │                      plaq_all (4-tuples × 3 orientations), bonds
│   └── hamiltonian.py     Reused from 2D verbatim — the loop iterates
│                          len(vertex_all[v]) so 6-tuples work unchanged
└── tests/
    ├── test_geometry.py
    ├── test_hamiltonian.py
    └── test_tiny_MLP.py   Minimal NQS training loop (this checkpoint's work)
```

**Network architecture (minimal, in `test_tiny_MLP.py`)**:
```
σ ∈ {±1}^N
  → Wilson 4-product over each plaquette          (no parameters, A_v invariant)
  → Dense(16) → tanh → Dense(1)                   (~400 parameters at L=2)
  → log ψ ∈ ℝ
```

**Training stack**: NetKet's `MCState` + `VMC` driver with `SR` preconditioner.
Single-spin Metropolis sampling. Same TDVP math as the 2D code.

### Validation results

| Run | System | Target E₀ | Achieved | Notes |
|---|---|---|---|---|
| L=2 PBC, h=0 | 24 qubits  | −32  | converged                        | clean, fast |
| L=3 PBC, h=0 | 81 qubits  | −108 | converged after raising diag_shift | was unstable until QGT regularisation bumped to ~1e-3 |
| L=4 PBC, h=0 | 192 qubits | −256 | converged                        | first non-trivial scale — 2¹⁹² Hilbert space, on a laptop |
| Vertex-flip symmetry | architecture check | log ψ identical | machine-precision | confirms Wilson 4-product enforces A_v invariance in 3D |
| ⟨A_v⟩, ⟨B_p⟩         | stabilizer check   | both → +1       | both at +1       | vertex and plaquette terms saturate independently |

### Key conceptual insights gained

1. **The Wilson 4-product generalises to 3D unchanged.** A_v flips 6 edges,
   but every plaquette intersects those 6 in 0 or 2 edges → the 4-product over
   any plaquette is A_v-invariant. The geometry took work, the symmetry trick
   was free.

2. **Vertex constraint hard-coded; plaquette constraint learned.** Network
   has vertex symmetry baked in via Wilson; MLP learns to suppress
   configurations with violated plaquettes.

3. **MLP is "free lunch" at h=0 only.** It works trivially when GS is the
   closed-flux superposition. Three failure modes (no translation equivariance,
   quadratic parameter scaling in N_plaq, no locality for quasi-adiabatic
   corrections) only bite when h ≠ 0.

4. **NetKet abstracts the VMC plumbing.** Designer's job is *just* the Flax
   `__call__`. Sampling, gradient estimation, QGT, Lanczos — already there.

5. **Single-flip MCMC fine for L=2,3,4 at h=0** with `diag_shift ≈ 10⁻³`.
   Custom vertex-update sampler skipped for now; will need it once
   perturbations + larger systems sharpen the wavefunction further.

### What's not yet built

- 3D KernelManager — all shift logic from 2D is non-portable.
- 3D CNN_noninvariant — three edge orientations (x/y/z), weight-tied recommended.
- 3D CNN_invariant — three plaquette orientations.
- Vertex-update sampler for 3D — `MultiRule(np.array(geo.vertex_all))` plugged
  into a `WeightedRule`. ~10 lines, trivially adapted from 2D.
- Observables module — 1D Wilson loops, closed-surface (2D) operators for
  m-loop BFFM order parameter.
- Config / main.py wrapper for clean 3D runs.

### Open research questions noted

- **Transformer alternative to CNN_invariant.** Hybrid (Wilson → transformer →
  log ψ) is the natural drop-in. Prior art: Luo et al. 2021 (autoregressive
  transformer for 3D TC), Viteritti/Rende 2023–24 (transformer NQS SOTA on
  several spin systems). Decision: build CNN baseline first, then ablate.

### Next concrete steps (per `notes/3D_extension_plan.md`)

1. **Step 5a**: Build minimal 3D CNN_invariant. Replace MLP in TinyToricMLP
   with one or two convolution layers over plaquette positions, weight-shared
   across orientations. Test at L=2, L=3 with h=0 — same energy, fewer params.

2. **Step 5b**: Build CNN_noninvariant, add before Wilson nonlinearity. Three
   kernel sets (x/y/z edges), identity-initialised. Test at L=2 with small
   hx=0.1, hz=0.1 — compare to exact diag.

3. **Step 5c**: Re-introduce vertex-update sampler.

4. **Step 6**: scale to L=3 PBC with perturbations.

5. **Step 7**: non-stoquastic perturbations (hy ≠ 0).
