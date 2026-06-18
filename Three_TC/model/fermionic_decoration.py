"""
3D fermionic toric code: bosonic plaquettes decorated with two sigma^x.

Each plaquette stabilizer is
    B~_p = (prod_{e in dp} sigma^z_e) * sigma^x_{e+} * sigma^x_{e-}
where e+ and e- are the perpendicular ("transverse") corner edges at the
(+a,+b) corner on the +perp side and the (-a,-b) corner on the -perp side
(a body diagonal; a, b are the in-plane axes, c the plaquette normal).  The
same pattern is used for all three orientations; vertex stars A_v are unchanged.

This is the minimal decoration giving a valid commuting-stabilizer model
(verify_xz_commutation -> 0 violations at L=2,3 PBC; the pattern is translation
invariant, so it holds for all L).  It replaces the Wang-Levin 10-edge dressing
of PRL 113, 080403 (2014) with two sigma^x per plaquette.
"""

from __future__ import annotations

import numpy as np

_E = np.eye(3)


def _idx(geom, coord) -> int:
    """Qubit index at edge-midpoint `coord`, PBC-wrapped (2x-integer keys)."""
    L = (geom.Lx, geom.Ly, geom.Lz)
    key = tuple(int(round(2 * coord[d])) % (2 * L[d]) for d in range(3))
    return int(geom._coord_to_idx[key])


def fermionic_plaquettes(geom, J: float = 1.0):
    """Decorated plaquette stabilizers as (z_edges, x_edges, coef) triples.

    Drop-in for hamiltonian_linop(geom, ..., xz_stabs=fermionic_plaquettes(geom)).
    Per plaquette: the 4 sigma^z boundary edges plus two sigma^x on the corner
    edges at ctr +/- 0.5*(e_a + e_b + e_c).
    """
    out = []
    for c in range(3):                                   # plaquette normal axis
        a, b = (d for d in range(3) if d != c)           # in-plane axes
        for ix in range(geom.Lx):
            for iy in range(geom.Ly):
                for iz in range(geom.Lz):
                    ctr = np.array([ix, iy, iz], float) + 0.5 * _E[a] + 0.5 * _E[b]
                    z_edges = [_idx(geom, ctr + s * 0.5 * _E[ax])
                               for ax in (a, b) for s in (+1, -1)]
                    diag = 0.5 * (_E[a] + _E[b] + _E[c])
                    x_edges = [_idx(geom, ctr + diag), _idx(geom, ctr - diag)]
                    out.append((z_edges, x_edges, -float(J)))
    return out


def verify_xz_commutation(stabs, vertex_all) -> dict:
    """Pairwise commutation check on (decorated plaquettes) + (vertex stars).

    Returns {"ok": bool, "violations": list, "n_stabilizers": int}.
    """
    def mask(qs):
        m = 0
        for q in qs:
            if q != -1:
                m |= 1 << int(q)
        return m

    entries = [(mask(z), mask(x)) for z, x, _ in stabs]
    entries += [(0, mask(v)) for v in vertex_all]

    viol = []
    n = len(entries)
    for i in range(n):
        z1, x1 = entries[i]
        for j in range(i + 1, n):
            z2, x2 = entries[j]
            if (bin(z1 & x2).count("1") + bin(z2 & x1).count("1")) & 1:
                viol.append((i, j))
    return {"ok": not viol, "violations": viol, "n_stabilizers": n}
