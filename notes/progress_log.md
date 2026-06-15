# Progress log — 3D Toric Code NQS extension

Updated alongside `notes/3D_extension_plan.md` as work progresses. Numbered
checkpoints record discrete milestones; the most recent is at the top.

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
