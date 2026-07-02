"""
model/transformer.py
─────────────────────────────────────────────────────────────────────────────
Transformer wavefunction for the 2D toric code — Step 1 of the transformer-NQS
subproject: a 4-layer ViT-style ansatz with **factored attention** and NO
symmetry baked in yet.

"Factored attention" (following the NQS-ViT literature, 10.1088/2632-2153/ada1a0
and arXiv:2311.16889) means the token-mixing map is a *learnable parameter matrix*
`A^μ ∈ R^{N×N}` per head — there are no queries/keys and no softmax, so the mixing
is input-independent. This is the OBC-compatible form; it is NOT translation
invariant (that is Step 2: tie `A^μ_{ij} = A^μ(i-j)` on PBC).

I/O contract (identical to every other ansatz in this repo): `__call__(x)` takes
`x` of shape `(..., N)` with ±1 spins and returns a real `(...,)` `log ψ`. With
only h_x, h_z fields the Hamiltonian is stoquastic, so a real-valued log-amplitude
is correct — the whole network is real (float64).

Parameter count ≈ per layer  h·N² (A) + 2d² (W_V, W_O) + 4d² (MLP).  The h·N²
term is the only size-dependent piece, which is exactly the scaling Step 2
(circulant tying) is meant to remove.
"""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import flax.linen as nn


class FactoredAttention(nn.Module):
    """Multi-head factored attention: token mixing by a learnable A^μ (no Q/K)."""

    N: int
    d: int
    n_heads: int
    param_dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, x):                       # x: (..., N, d)
        h, dh = self.n_heads, self.d // self.n_heads
        assert self.d % self.n_heads == 0, "d must be divisible by n_heads"

        # value projection, split into heads: (..., N, d) -> (..., N, h, dh)
        v = nn.Dense(self.d, use_bias=False, param_dtype=self.param_dtype,
                     name="W_V")(x)
        v = v.reshape(x.shape[:-1] + (h, dh))

        # learnable per-head token-mixing matrix A[head, m, n]; small init so the
        # block starts near identity (residual dominates).
        A = self.param("A", nn.initializers.normal(stddev=1.0 / self.N),
                       (h, self.N, self.N), self.param_dtype)

        # o[..., m, head, f] = sum_n A[head, m, n] v[..., n, head, f]
        o = jnp.einsum("hmn,...nhf->...mhf", A, v)
        o = o.reshape(x.shape[:-1] + (self.d,))

        # output projection, zero-init -> each block is identity at step 0
        return nn.Dense(self.d, use_bias=False, param_dtype=self.param_dtype,
                        kernel_init=nn.initializers.zeros, name="W_O")(o)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm block: x + Attn(LN x); then y + MLP(LN y), MLP hidden = mlp_ratio·d."""

    N: int
    d: int
    n_heads: int
    mlp_ratio: int = 2
    param_dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, x):                       # x: (..., N, d)
        y = nn.LayerNorm(param_dtype=self.param_dtype)(x)
        x = x + FactoredAttention(self.N, self.d, self.n_heads, self.param_dtype)(y)

        z = nn.LayerNorm(param_dtype=self.param_dtype)(x)
        z = nn.Dense(self.mlp_ratio * self.d, param_dtype=self.param_dtype)(z)
        z = nn.relu(z)
        z = nn.Dense(self.d, param_dtype=self.param_dtype,
                     kernel_init=nn.initializers.zeros)(z)   # zero-init residual branch
        return x + z


class FactoredAttentionWavefunction(nn.Module):
    """4-layer factored-attention transformer: (..., N) ±1 spins -> real (...,) log ψ."""

    N: int
    d: int = 16
    n_heads: int = 4
    n_layers: int = 4
    mlp_ratio: int = 2
    param_dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, x):                       # x: (..., N) spins ±1
        x = x.astype(self.param_dtype)

        # per-spin embedding: (..., N, 1) -> (..., N, d), plus a learned position code
        h = nn.Dense(self.d, param_dtype=self.param_dtype, name="embed")(x[..., None])
        pos = self.param("pos_embed", nn.initializers.normal(stddev=0.02),
                         (self.N, self.d), self.param_dtype)
        h = h + pos

        for _ in range(self.n_layers):
            h = TransformerBlock(self.N, self.d, self.n_heads, self.mlp_ratio,
                                 self.param_dtype)(h)

        # shallow readout on the mean-pooled tokens -> scalar log ψ
        h = jnp.mean(h, axis=-2)                                   # (..., d)
        h = nn.LayerNorm(param_dtype=self.param_dtype)(h)
        h = nn.Dense(self.d, param_dtype=self.param_dtype)(h)
        h = nn.gelu(h)
        h = nn.Dense(1, param_dtype=self.param_dtype)(h)           # (..., 1)
        return h[..., 0]
