"""
GF(2) index-identity tests for the Fredenhagen–Marcu builders in Three_TC/fm.py.
Geometry + index combinatorics + a logistic-fit check — NO ED, NO sampling.

Run directly:
    python test_fm.py
"""

import _path  # noqa: F401
import numpy as np
from Three_TC.model.geometry import ThreeD_ToricCodeGeometry
from Three_TC.fm import (electric_loop_edges, magnetic_membrane_edges,
                         fit_transition)


def _xor(sets):
    acc = set()
    for s in sets:
        acc ^= set(s)
    return acc


def _odd_overlap(string, stabilizers):
    """How many stabilizers share an odd number of edges with `string`
    (i.e. anticommute with it)."""
    return sum(len(set(string) & set(s)) % 2 == 1 for s in stabilizers)


def test_electric_closed_loop_is_product_of_enclosed_plaquettes(geo, R=2):
    """∏σ^z around the rectangle == ∏ enclosed B_p (the magnetic Wilson loop)."""
    plane_axis, plane_at, corner = 2, 0, (0, 0)
    closed, _ = electric_loop_edges(geo, plane_axis=plane_axis,
                                    plane_at=plane_at, corner=corner, R=R)
    enclosed = []
    for p, (cen, ori) in enumerate(zip(geo.plaq_centers, geo.plaq_orient)):
        if ori != plane_axis:
            continue
        x, y, z = cen
        if abs(z - plane_at) < 1e-9 and corner[0] < x < corner[0] + R \
                and corner[1] < y < corner[1] + R:
            enclosed.append(set(geo.plaq_all[p]))
    assert len(enclosed) == R * R
    assert _xor(enclosed) == set(closed)
    assert len(set(closed)) == 4 * R


def test_electric_open_string_is_half_square_with_two_charges(geo, R=2):
    """BFFM open string = HALF the square: |open| = 2R = ½·|closed| (the perimeter-
    law cancellation that gives a finite ℓ→∞ limit). It flips exactly 2 vertex
    stars (its e-charge ends); the closed loop commutes with all A_v."""
    closed, open_ = electric_loop_edges(geo, plane_axis=2, plane_at=0, R=R)
    assert len(set(open_)) == 2 * R == len(set(closed)) // 2
    verts = geo.get_vertex_all_hetero()
    assert _odd_overlap(open_, verts) == 2
    assert _odd_overlap(closed, verts) == 0


def test_magnetic_membrane_flux(geo):
    """Option A half-sheet: the full σ^x sheet is boundary-free (commutes with
    every B_p — it is ∏A_v over the slab, so =1 on the GS); the half-sheet opens a
    non-empty flux loop along the cut, and its area is L_a//2 of the full sheet."""
    closed, open_ = magnetic_membrane_edges(geo, normal=2, plane_at=0)
    plaqs = [set(p) for p in geo.plaq_all]
    assert _odd_overlap(closed, plaqs) == 0          # closed sheet: no flux
    assert _odd_overlap(open_, plaqs) > 0            # half sheet: a flux loop
    assert len(set(closed)) == geo.Lx * geo.Ly       # full xy sheet of x-edges
    assert len(set(open_)) == (geo.Lx // 2) * geo.Ly  # exactly half (a-cut)


def test_logistic_fit_recovers_inflection():
    """fit_transition's logistic inflection h_c recovers a known midpoint."""
    h = np.linspace(0.0, 1.0, 11)
    h0 = 0.42
    O = 1.0 / (1.0 + np.exp(-(h - h0) / 0.05))
    fit = fit_transition(h, O, Oe=0.01 * np.ones_like(h))
    assert abs(fit["h_c"] - h0) < 0.02


def main():
    for bc in ("OBC", "PBC"):
        geo = ThreeD_ToricCodeGeometry(3, 3, 3, bc=bc)
        test_electric_closed_loop_is_product_of_enclosed_plaquettes(geo)
        test_electric_open_string_is_half_square_with_two_charges(geo)
        test_magnetic_membrane_flux(geo)
        print(f"[PASS] index identities, bc={bc}")
    test_logistic_fit_recovers_inflection()
    print("[PASS] logistic fit recovers inflection")
    print("All FM tests passed.")


if __name__ == "__main__":
    main()
