# NetKet VMC loop — internals reference

A reference for the moving parts inside a single NetKet VMC training step.
Use this to remind yourself what `driver.advance` actually does, and what
the sampler hyperparameters are trading off.

---

## What happens inside `driver.advance(1)`

One TDVP/SR step decomposes into six pieces:

```
1.  Sample.            MCMC chains draw {σ_1, ..., σ_M} from |ψ_θ|².
                       Uses the sampler, runs n_sweeps + n_discard, returns M samples.

2.  Local energies.    For each σ_k, compute
                           E_loc(σ_k) = Σ_σ' ⟨σ_k|H|σ'⟩ · ψ(σ')/ψ(σ_k).
                       Sum is over connected configurations σ' (sparse — usually
                       fewer than ~100 terms per σ_k). This is the dominant
                       compute cost per step.

3.  Energy + gradient. ⟨H⟩ ≈ (1/M) Σ E_loc(σ_k).
                       ∇⟨H⟩ ≈ 2·Re ⟨ (E_loc - ⟨E_loc⟩) · ∂_θ log ψ*(σ) ⟩
                       The ∂_θ log ψ is computed by JAX autograd; the formula
                       above is the standard VMC gradient identity.

4.  Quantum Geometric  S_ij = ⟨∂_i log ψ · ∂_j log ψ*⟩ - ⟨∂_i log ψ⟩⟨∂_j log ψ*⟩
    Tensor (QGT).      i.e. the Fisher information metric on parameter space.
                       Cost is O(M · n_params²); often the second-dominant cost.

5.  SR preconditioner. Solve (S + ε·I) Δθ = -∇⟨H⟩ via pinv_smooth.
                       The ε is `diag_shift`. This is the imaginary-time TDVP step.

6.  Parameter update.  θ ← θ + lr · Δθ.    (lr = dt in TDVP language.)
```

A few practical notes:

- `vs.expect(Ham)` after `advance` regenerates fresh samples, independent of
  the ones used in the update — so MC bias from the gradient step doesn't
  contaminate the reported energy.
- The sampler state **persists** between `advance` calls; chains don't
  restart unless you `vs.reset()`. Burn-in is only paid once.
- `driver.advance(k)` is `advance(1)` done k times, but without
  re-evaluating `vs.expect(Ham)` between steps — faster when per-step
  logging isn't needed.

---

## Swappable parts

Everything below can be swapped out without rewriting the loop. This is the
main reason NetKet is pleasant to do research with.

| Component | Effect of swapping |
|---|---|
| `Ham` | Change the physics being studied. |
| `sa` (sampler) | Single-flip ↔ vertex updates ↔ Hamiltonian Monte Carlo. |
| `model` | MLP → CNN → transformer; the architecture under study. |
| `opt` | Outer optimizer: SGD ↔ Adam ↔ Adagrad. |
| `sr` | Preconditioner: full SR ↔ no preconditioner ↔ `QGTJacobianDense` (cheaper). |

---

## Sampler hyperparameters

### `n_chains` — number of independent Markov chains

Each chain is its own random walk through configuration space. They share
the same target distribution |ψ(σ)|² but start independently and follow
independent random proposals. NetKet runs them vectorised in parallel.

Why multiple chains instead of one really long one:

1. **Catches non-mixing.** If many chains starting from different places
   all converge to the same distribution of observables → confidence that
   the chain has mixed. Exposed as the **R-hat statistic**
   (`vs.expect(...).R_hat`). Values close to 1.0 are good, > 1.1 is alarming.

2. **Parallelism is free on GPU.** Vectorising over chains doesn't cost
   wall-clock time on parallel hardware.

3. **Independent burn-ins.** Each chain pays its own burn-in cost
   (`n_discard_per_chain` steps thrown away).

**Trade-off**: more chains = better statistics, more burn-in waste.
16 is a reasonable CPU default; the 2D paper uses 1024 on GPU.

**Critical relationship**: `n_samples = n_chains × (samples per chain)`.
With `n_samples=1024, n_chains=16` each chain produces 64 samples. If you
crank `n_chains` without raising `n_samples`, each chain produces fewer
samples → less decorrelated each, more burn-in waste.

### `n_sweeps` — silent Metropolis steps between recorded samples

A single Metropolis step changes at most one spin. Consecutive samples are
therefore nearly identical (high autocorrelation). If every Metropolis step
were returned as a "sample," effective sample size could be 30 out of 1024.

So between successive **returned** samples, NetKet runs `n_sweeps` more
proposals silently (accepting or rejecting per Metropolis rule) and only
returns the final state. This **thins** the chain.

The heuristic `n_sweeps = N // 2`:

- A "sweep" in MCMC conventionally means N single-site updates — each of
  the N qubits gets *proposed* once on average.
- Half a sweep (`N // 2`) means each qubit has ~50% chance of being touched
  between samples. Cheap, works well in practice.
- Some codes use `N` or `2N` for harder problems where mixing is slow.

**Trade-off**: more sweeps = better decorrelation, but linear in compute.
Per-sample cost scales as `n_sweeps × (single-Metropolis cost)`.

**How to tell if `n_sweeps` is too low**: NetKet logs `tau_corr`,
the integrated autocorrelation time.

- `tau_corr >> 1`: under-sampling effectively; bump `n_sweeps`.
- `tau_corr ≈ 1`: fine.
- `tau_corr < 1`: wasting time on extra sweeps.

---

## How the knobs interact per iteration

Roughly:

```
for each chain in parallel:
    state = random_initial_config()
    repeat n_discard_per_chain times:                    # burn-in
        for n_sweeps + 1 metropolis steps: state = step(state)
    for j in range(n_samples / n_chains):                # productive samples
        for n_sweeps + 1 metropolis steps: state = step(state)
        emit state
```

Total Metropolis proposals per iteration:
`n_chains × (n_discard + n_samples/n_chains) × (n_sweeps + 1)`.

Example with N=24 qubits, `n_chains=16, n_discard=8, n_samples=1024,
n_sweeps=12`:  `16 × (8 + 64) × 13 ≈ 15,000 proposals`.

Each proposal is a single-spin flip plus a log-amplitude evaluation. The
log-amplitude evaluation is the dominant cost.

---

## Practical tuning rules

1. **Default starting point**: `n_chains = 16` (or = number of cores),
   `n_sweeps = N // 2`, `n_discard ≈ 8`.
2. **If R-hat > 1.1**: bump `n_discard` first (longer burn-in), then `n_chains`.
3. **If `tau_corr >> 1`**: bump `n_sweeps`.
4. **If gradient noise is killing convergence**: bump `n_samples`.
5. **If acceptance crashes**: you need a *better proposal* (vertex updates),
   not more sweeps. More sweeps of *rejected* proposals do nothing.

That last point is important: for a wavefunction concentrated on the
closed-flux sector, more `n_sweeps` of rejected single-flip proposals
doesn't help at all. Vertex-update samplers fix acceptance, which is the
only thing that actually decorrelates samples in the topological phase.

---

## Where this slots into your codebase

- 2D code: `simulation/optimizer.py:run_tdvp` inlines steps 1–6 explicitly
  with custom logging. Easier to read than the `driver` abstraction once
  you know what's going on.
- 3D code (`Three_TC/tests/test_tiny_MLP.py`): uses the `driver` abstraction.
  Equivalent math, fewer lines.
- 2D custom sampler: `simulation/custom_sampler.py` shows the
  `WeightedRule + MultiRule` pattern. Same code drops into 3D with
  `vertex_clusters = np.array(geo.vertex_all)` from the 3D geometry.
