# Symmetry restoration by group averaging (and what NetKet gives us for free)

Notes on *staged lattice-symmetry restoration* — the technique behind the
sentence in the ViT-NQS papers:

> "an initial phase of 10⁴ steps is performed without enforcing symmetries,
> during which the ViT state retains only translational symmetry among patches.
> This is followed by restoration of translational symmetry (4×10³ steps), then
> C₄ rotational (2×10³) and reflection symmetry (10³)."

This is **not** RL and **not** a penalty term. It is exact *quantum-number
projection* onto the trivial irrep, applied as a curriculum.

## 1. What it is

Given any log-wavefunction `log ψ_θ(σ)`, the symmetrized state over a discrete
permutation group `G` is

```
log ψ_sym(σ) = (1/|G|) · logsumexp_{g∈G} [ log χ_g + log ψ_θ(T_g σ) ]
```

- `T_g σ` is the group element `g` acting on the configuration = a **permutation
  of the sites** (translation / rotation / reflection reindexes the DoF).
- `χ_g` are the characters of the target irrep. For a ground state one almost
  always takes the **trivial irrep, χ_g = 1** (equal amplitude on the orbit).
  A momentum-`k` or odd-parity sector uses `χ_g = e^{ik·r}`, `±1`, … — this is
  how phases (not just equal amplitudes) are enforced.
- It is **parameter-free** (a fixed projection wrapped around the network) and
  **variational-monotone**: applied to a state already near a symmetric ground
  state, projecting onto the sector that contains the true GS can only *lower*
  the energy. So the schedule is a sequence of safe polishes.
- Cost is `×|G|` forward passes per configuration (fused by `logsumexp`), which
  flows straight into the sample / SR-Jacobian cost — hence the staging.

## 2. Why staged

The architecture supplies some symmetry for free; you only *restore* the rest,
and you do it late and cheaply:

| stage | group added | why |
|---|---|---|
| 1 (bulk, 10⁴) | none explicit — **patch-translations are architectural** | converge cheaply at `×1` cost; a patched ViT with circulant attention is already equivariant to whole-patch translations |
| 2 (4×10³) | full translation (the `b²` within-patch shifts) | completes the site-translation group → `k=0` sector |
| 3 (2×10³) | point-group rotations (`C₄` in 2D) | |
| 4 (10³) | reflections | full space group |

Decreasing step budgets track **diminishing returns** (each added factor buys a
smaller energy drop) and **rising per-step cost** (`|G|` grows). "Bake in what
the architecture gives cheaply, project the rest" — the same split our
Wilson-sandwich already embodies for the A_v gauge symmetry.

## 3. The NetKet primitive — `nk.nn.blocks.SymmExpSum`

NetKet 3.16 implements *exactly* the formula above:

```python
nk.nn.blocks.SymmExpSum(module, symm_group, character_id=None)
#  log ψ = (1/|G|) log Σ_g χ_g exp[ log ψ_module(T_g σ) ]
```

- `module` — **any** Flax `nn.Module` whose `__call__(x)` returns `log ψ`. Our
  `ToricCNN`, `ToricCNN_full`, `ToricCNN_gridinv`, `GeoCNN` all qualify verbatim.
- `symm_group` — a `netket.utils.group.PermutationGroup`.
- `character_id` — selects the irrep (default trivial). **Adds no parameters.**
- Its own docstring notes the staged split: *"If you have a Conv NN already
  invariant under translations, you might want to only symmetrize over the
  point-group."* — precisely the ViT logic.

Graph helpers that build the groups (for a NetKet lattice `g`):
`g.translation_group()`, `g.point_group()`, `g.space_group()`,
`g.automorphisms()`. E.g. `Hypercube(L, n_dim=3)` → `point_group()` = 48 = the
full cubic group `O_h`.

## 4. Status in THIS repo — what runs today, what's missing

| piece | status |
|---|---|
| **A_v gauge symmetry** | ✅ baked in via the Wilson 4-product (inductive bias, not averaging). This is a *gauge* symmetry — **orthogonal** to space-group restoration, not a substitute. |
| **Group-averaging mechanism** (`SymmExpSum`) | ✅ available in the installed NetKet; wraps our Flax modules directly, adds no params. |
| **Space-group symmetry of the 3D TC** (translations / cubic rotations / reflections) | ❌ **not applied** to the 3D ansatz today. |
| **The group `G` itself** | ⚠️ **the one real task.** `g.translation_group()` etc. permute the *graph's vertices*. Our DoF live on **edges** (`N = 3L³` qubits), and the Hilbert space is a bare `nk.hilbert.Spin(N=geo.N)` with no lattice attached. So the built-in groups do **not** apply — you must construct a `PermutationGroup` over the **edge indices** by hand. |
| **Staging across training** | ⚠️ feasible but needs plumbing: `MCState` holds a fixed model, so each stage = rebuild `MCState(sa, SymmExpSum(inner, G_stage), …)` and transfer the *unchanged* inner params (the checkpoint/resume path — see `[[nqs-checkpoint-resume]]` — makes this clean, since `SymmExpSum` adds no params). |

### The edge-permutation group (the only nontrivial bit)

`geometry.py` already stores edge-midpoint coordinates and a `_coord_to_idx`
map. For each lattice symmetry `R` (a signed axis permutation + translation),
the induced **edge** permutation is: for edge `i` at midpoint `c`, its image is
the edge whose midpoint is `R·c` (PBC-wrapped) — a coordinate lookup. Collect
those permutation arrays into `netket.utils.group.PermutationGroup([...],
degree=geo.N)`. Verified that a hand-built `PermutationGroup` over arbitrary `N`
(i.e. edges, not vertices) constructs fine.

## 5. Minimal recipe

```python
from netket.nn.blocks import SymmExpSum
from netket.utils.group import PermutationGroup, Permutation, Identity

# 1. build edge-permutation group from geo coordinates (translations first;
#    add cubic rotations/reflections stage by stage)
G_trans = PermutationGroup([Identity()] + [Permutation(p) for p in edge_perms],
                           degree=geo.N)

# 2. wrap the *existing* ansatz — no retraining of a new architecture
inner  = build_model(cfg, geo)          # ToricCNN_full, etc.
model  = SymmExpSum(module=inner, symm_group=G_trans)   # χ_g = 1 (trivial irrep)

# 3. drop into the usual state; cost is ×|G_trans| forward passes
vs = nk.vqs.MCState(sa, model, n_samples=cfg["n_samples"], ...)
```

Staging = repeat step 2 with a larger `G` (translations → +rotations →
+reflections) between checkpointed runs, carrying the inner params across.

## 6. Bottom line

The **mechanism is fully implemented** (`SymmExpSum`) and wraps our nets with
zero architectural change; the **A_v gauge symmetry is already handled** by the
Wilson product (complementary, not overlapping). The **only work** to reproduce
the staged ViT recipe on the 3D TC is (a) building the *edge*-permutation group
from `geo` coordinates — the NetKet graph groups act on vertices and don't apply
— and (b) light staging plumbing to swap the wrapped group between checkpointed
training stages. No new architecture, no RL, no penalty term.
