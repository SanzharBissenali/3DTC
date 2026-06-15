"""
Let's test tiny MLP to see if it solves the toy example with
unperturbed Hamiltonian, with L = 2, so that number of qubits 
is N = 24 (PBC -> 3 * 2 * 2 * 2)
"""

import _path  # noqa: F401   <-- ADD THIS FIRST
import json
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import netket as nk
from tqdm import tqdm
from simulation.custom_sampler import WeightedRule, MultiRule

from Three_TC.model.geometry import ThreeD_ToricCodeGeometry
from Three_TC.model.hamiltonian import create_hamiltonian

geo = ThreeD_ToricCodeGeometry(Lx=2, Ly=2, Lz=2, bc='PBC')
hi = nk.hilbert.Spin(s=1/2, N=geo.N)
Ham = create_hamiltonian(
    hi=hi, vertex_all=geo.vertex_all, plaq_all=geo.plaq_all, 
    bonds=geo.bonds
)


# Firstly, let's define a model 

class TinyToricMLP(nn.Module):
    plaq_all: tuple        # tuple-of-tuples for hashability (Flax static field)
    hidden: int = 32

    @nn.compact
    def __call__(self, x):
        # x has shape (..., N) where ... is the batch dim(s) NetKet supplies.
        # 1. Wilson 4-product per plaquette  -> shape (..., N_plaq), values in {-1, +1}
        plaq_idx = jnp.asarray(self.plaq_all)                # (N_plaq, 4)
        wilson   = jnp.prod(x[..., plaq_idx], axis=-1)       # (..., N_plaq)

        # Below is more readable version
        # result = np.empty((B, N_plaq))
        # for b in range(B):
        #     for p in range(N_plaq):
        #         result[b, p] = x[b, plaq_idx[p, 0]] * x[b, plaq_idx[p, 1]] * \
        #                     x[b, plaq_idx[p, 2]] * x[b, plaq_idx[p, 3]]

        # 2. Tiny MLP across plaquettes
        h = nn.Dense(self.hidden)(wilson)
        h = nn.tanh(h)
        h = nn.Dense(1)(h)                                    # (..., 1)
        return h.squeeze(-1)                                  # scalar log ψ per sample

plaq_tuple = tuple(tuple(p) for p in geo.plaq_all)
model = TinyToricMLP(plaq_all=plaq_tuple, hidden=16)

# Secondly, we have to implement sampling

# MCMC sampler: single-spin flips, Metropolis
# sa = nk.sampler.MetropolisSampler(
#     hi,
#     rule=nk.sampler.rules.LocalRule(),
#     n_chains=16,
#     n_sweeps=geo.N // 2,      # standard heuristic
#     dtype=jnp.int8,
# )

vertex_clusters = np.array(geo.vertex_all)              # shape (N_v, 6)

samp_ratio = geo.N / len(vertex_clusters)
weighted = WeightedRule(
    (samp_ratio / (samp_ratio + 1), 1 - samp_ratio / (samp_ratio + 1)),
    [nk.sampler.rules.LocalRule(), MultiRule(vertex_clusters)],
)

sa = nk.sampler.MetropolisSampler(
    hi, rule=weighted,
    n_chains=16, n_sweeps=geo.N // 2, dtype=jnp.int8,
)

# Variational state: model + sampler + sample budget

# vs = nk.vqs.MCState(
#     sa, model,
#     n_samples=1024,
#     n_discard_per_chain=8,
# )

# Rebuild the variational state with the new sampler
vs = nk.vqs.MCState(sa, model, n_samples=4096, n_discard_per_chain=8)

print(f"N qubits: {geo.N}  |  n_params: {vs.n_parameters}")

# Thirdly, a training loop that lasts for 200 steps, where
# each step calls the sampling, evaluates the Energy, and gradients, 
# update the model weights.

dt        = 0.02      # learning rate (= TDVP timestep)
n_iter    = 200
diag_shift= 2*1e-3

opt = nk.optimizer.Sgd(learning_rate=dt)
sr  = nk.optimizer.SR(diag_shift=diag_shift)

driver = nk.driver.VMC(Ham, opt, variational_state=vs, preconditioner=sr)

energies = []
for step in tqdm(range(n_iter)):
    driver.advance(1)                        # one TDVP step
    E = vs.expect(Ham)
    energies.append(float(E.mean.real))

    acc_ratio = float(vs.sampler_state.n_accepted) / float(vs.sampler_state.n_steps)
    print(f"  acceptance = {acc_ratio:.3f}")
    print(f"step {step:3d}: E = {E.mean.real:+.4f} ± {E.error_of_mean:.4f}  Var={E.variance:.2e}")


# Fourtly, return the list of energies, and the model weights. 
with open("test_tiny_MLP_results.json", "w") as f:
    json.dump({
        "energies": energies,
        "n_params": int(vs.n_parameters),
        "N": int(geo.N),
        "exact_E0": -32,
    }, f)

