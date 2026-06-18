"""
Three_TC/model/networks.py
─────────────────────────────────────────────────────────────────────────────
Neural-network ansätze for the 3D toric code.

Two building blocks and two composed models:

    CNN_invariant_3D     — 3D conv applied DOWNSTREAM of the Wilson
                           nonlinearity. Default random init, ELU activation.
                           Its outputs are automatically A_v-invariant
                           because its inputs are.

    CNN_noninvariant_3D  — 3D conv applied UPSTREAM of the Wilson
                           nonlinearity. IDENTITY-initialised so that at
                           step 0 it is a pass-through, recovering the
                           symmetric-only network. Activation is a
                           normalised sigmoid that maps ±1 → ±1 so the
                           identity property survives the nonlinearity.

    ToricCNN             — Wilson → CNN_invariant ×2 → Final  (Step 5a)
    ToricCNN_full        — CNN_noninvariant → Wilson → CNN_invariant ×2 → Final
                           (Step 5b, the full paper-style architecture)

Geometry helper:

    compute_edges_3D(geo)  — produces a (3, L, L, L) array of qubit indices
                             arranged in (orientation, ix, iy, iz) order.
                             Needed by ToricCNN_full because the flat spin
                             array x is in qubit-index order (lexsort over
                             coordinates), but a 3D conv needs positions on
                             a regular 3D grid. Plaquette features (output
                             of Wilson) are already in (c, ix, iy, iz) order
                             because that is the loop order of geo.plaq_all,
                             so no permutation is needed downstream of Wilson.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import flax.linen as nn

# Geometry helper


def compute_edges_3D(geo) -> np.ndarray:
    """
    Returns a (3, Lx, Ly, Lz) array of qubit indices such that

        edges_3D[c, ix, iy, iz] = flat qubit index of the edge sitting
                                  on cubic vertex (ix, iy, iz) and pointing
                                  in direction c ∈ {0=x, 1=y, 2=z}.
    """
    Lx, Ly, Lz = geo.Lx, geo.Ly, geo.Lz
    e = np.eye(3)
    L_box = np.array([Lx, Ly, Lz])
    edges_3D = np.zeros((3, Lx, Ly, Lz), dtype=int)
    for c in range(3):
        offset = 0.5 * e[c]
        for ix in range(Lx):
            for iy in range(Ly):
                for iz in range(Lz):
                    coord = np.array([ix, iy, iz], dtype=float) + offset
                    if geo.bc == "PBC":
                        coord = coord % L_box
                    edges_3D[c, ix, iy, iz] = geo._mapping3Dto1D(coord)
    return edges_3D

# Initialiser and activation for the non-invariant block

def identity_initializer_3D(kernel_size: int):
    """
    Kernel initialiser that makes a 3D conv act as identity at step 0.

    Builds a kernel of shape (k, k, k, C_in, C_out) where every spatial
    position is zero except the centre, which holds an identity matrix in
    channel space. At step 0 the convolution is exactly

        output[..., c, x, y, z] = input[..., c, x, y, z]   (when C_in == C_out)

    i.e. a pure pass-through. Training adjusts these weights to learn the
    quasi-adiabatic deformation away from the symmetric fixed point.
    """
    def init(key, shape, dtype=jnp.float64):
        # shape = (k, k, k, C_in, C_out)
        w = jnp.zeros(shape, dtype=dtype)
        c = kernel_size // 2
        c_in, c_out = shape[-2], shape[-1]
        w = w.at[c, c, c].set(jnp.eye(c_in, c_out, dtype=dtype))
        return w
    return init


def _normalised_sigmoid(x):
    """
    The 2D paper's activation: matches sigmoid in shape but rescaled so
    that ±1 → ±1 exactly. This preserves the identity-init property of
    the non-invariant block at step 0 (a plain sigmoid would squash ±1
    to ≈ ±0.73, destroying the pass-through).
    """
    return (nn.sigmoid(x) - 0.5) * (2 + 2 * jnp.e) / (jnp.e - 1)


# Building blocks

class CNN_invariant_3D(nn.Module):
    """
    3D conv on A_v-invariant features (plaquette space).

        Input shape:  (..., features_in,  L, L, L)
        Output shape: (..., features_out, L, L, L)

    Default Flax kernel init (lecun_normal), zeros for bias, ELU activation.
    """
    features_out: int
    L: int
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x):
        x = jnp.moveaxis(x, -4, -1)                       # to channels-last
        x = nn.Conv(features=self.features_out,
                    kernel_size=(self.kernel_size,) * 3,
                    padding="CIRCULAR")(x)
        x = nn.elu(x)
        return jnp.moveaxis(x, -1, -4)                    # back to channels-first


class CNN_noninvariant_3D(nn.Module):
    """
    3D conv on raw edge features (edge space). Differs from
    CNN_invariant_3D in three specific places:

      1. `kernel_init` is identity-at-centre  → step 0 is pass-through.
      2. `bias_init` is zeros                  → step 0 is pass-through.
      3. Activation is a normalised sigmoid    → ±1 → ±1, identity preserved.

    Input/output shape: (..., 3, L, L, L). The "3" is edge orientation,
    treated as input/output channel. Output also has 3 channels: this
    minimal block does not grow channel count. Feature expansion is left
    to the invariant block downstream of Wilson.
    """
    L: int
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x):
        x = jnp.moveaxis(x, -4, -1)                       # (..., L, L, L, 3)
        x = nn.Conv(features=3,
                    kernel_size=(self.kernel_size,) * 3,
                    padding="CIRCULAR",
                    kernel_init=identity_initializer_3D(self.kernel_size),
                    bias_init=nn.initializers.zeros)(x)
        x = _normalised_sigmoid(x)
        return jnp.moveaxis(x, -1, -4)                    # back to channels-first


# Composed models

class ToricCNN(nn.Module):
    """
    Symmetric-only architecture (Step 5a): Wilson → CNN_invariant ×2 → Final.
    Sufficient for h = 0; will plateau under perturbations.
    """
    plaq_all: tuple                # (N_plaq, 4) flat qubit indices
    L: int
    hidden: int = 8

    @nn.compact
    def __call__(self, x):
        plaq_idx = jnp.asarray(self.plaq_all)
        wilson   = jnp.prod(x[..., plaq_idx], axis=-1)             # (..., N_plaq)
        wilson   = wilson.reshape(*wilson.shape[:-1],
                                  3, self.L, self.L, self.L)       # (..., 3, L, L, L)
        h = CNN_invariant_3D(features_out=self.hidden, L=self.L)(wilson)
        h = CNN_invariant_3D(features_out=1, L=self.L)(h)
        return jnp.mean(h, axis=tuple(range(-4, 0)))                # scalar log ψ


class ToricCNN_full(nn.Module):
    """
    Full architecture (Step 5b):

        flat spins x ∈ {±1}^N
        ─ gather  ─→ (..., 3, L, L, L)               [edges_3D permutation]
        ─ CNN_noninvariant ─→ (..., 3, L, L, L)      [identity-init]
        ─ scatter ─→ (..., N)                        [inverse permutation]
        ─ Wilson 4-product ─→ (..., N_plaq)
        ─ reshape ─→ (..., 3, L, L, L)               [natural plaq order]
        ─ CNN_invariant ×2 ─→ (..., 1, L, L, L)
        ─ mean ─→ scalar log ψ
    """
    plaq_all: tuple               # (N_plaq, 4) flat qubit indices
    edges_3D: tuple               # (3, L, L, L) qubit-index permutation
    L: int
    hidden_inv: int = 8

    @nn.compact
    def __call__(self, x):
        plaq_idx  = jnp.asarray(self.plaq_all)
        edges_idx = jnp.asarray(self.edges_3D)

        # 1. Gather edges into the (orientation, ix, iy, iz) grid.
        x = x[..., edges_idx]                                        # (..., 3, L, L, L)

        # 2. Non-invariant block. At step 0 this is identity, so the
        #    whole network reduces to ToricCNN (symmetric-only).
        x = CNN_noninvariant_3D(L=self.L)(x)                         # (..., 3, L, L, L)

        # 3. Scatter back to flat qubit-index order so Wilson can index.
        edges_flat = edges_idx.reshape(-1)                           # (N,)
        inv_perm   = jnp.argsort(edges_flat)                         # (N,)
        x = x.reshape(*x.shape[:-4], -1)                             # (..., N) natural order
        x = x[..., inv_perm]                                         # (..., N) qubit order

        # 4. Wilson 4-product per plaquette.
        wilson = jnp.prod(x[..., plaq_idx], axis=-1)                 # (..., N_plaq)
        wilson = wilson.reshape(*wilson.shape[:-1],
                                3, self.L, self.L, self.L)          # (..., 3, L, L, L)

        # 5. Invariant block on plaquette features.
        h = CNN_invariant_3D(features_out=self.hidden_inv, L=self.L)(wilson)
        h = CNN_invariant_3D(features_out=1, L=self.L)(h)

        # 6. Aggregate to scalar log ψ.
        return jnp.mean(h, axis=tuple(range(-4, 0)))
