"""
Rotated surface code geometry on a d x d square lattice of qubits.

Qubits sit on integer sites (i, j) with i, j in {0, ..., d-1} and linear
index q(i, j) = i*d + j, giving N = d**2 qubits total.  Stabilizers form a
checkerboard of weight-4 X- and Z-plaquettes on the (d-1)**2 bulk faces:
the face between qubits (i, j), (i+1, j), (i, j+1), (i+1, j+1) is X-type
when (i+j) is even and Z-type otherwise.  Weight-2 boundary stabilizers are
attached wherever the natural checkerboard extends past the lattice with
the matching colour: X on the top/bottom edges, Z on the left/right edges.

For odd d this is the standard rotated surface code with one logical qubit.
For even d the X/Z counts come out asymmetric (e.g. 7 X + 8 Z at d=4), but
the bulk physics and the field-driven topological transition are the same.

The class exposes the same attributes the rest of the codebase expects from
ToricCodeGeometry, so model.hamiltonian.create_hamiltonian can be reused
verbatim: X-stabilizers go in `vertex_all`, Z-stabilizers in `plaq_all`.
"""

from typing import List, Tuple


class RotatedSurfaceGeometry:
    """Rotated surface code on a d x d grid of vertex qubits."""

    def __init__(self, d: int):
        if d < 2:
            raise ValueError("d must be >= 2")
        self.d = d
        self.N = d * d
        self.vertex_all, self.plaq_all = self._build_stabilizers()
        self.bonds = self._build_bonds()

    def _q(self, i: int, j: int) -> int:
        return i * self.d + j

    def _build_stabilizers(self) -> Tuple[List[List[int]], List[List[int]]]:
        d = self.d
        x_stabs: List[List[int]] = []
        z_stabs: List[List[int]] = []

        # Bulk weight-4 plaquettes in a checkerboard.
        for i in range(d - 1):
            for j in range(d - 1):
                face = [self._q(i, j),     self._q(i + 1, j),
                        self._q(i, j + 1), self._q(i + 1, j + 1)]
                (x_stabs if (i + j) % 2 == 0 else z_stabs).append(face)

        # Boundary weight-2 stabilizers: keep only those virtual faces just
        # outside the lattice whose checkerboard colour matches the edge type.
        # Top/bottom edges carry X; left/right edges carry Z.
        for i in range(d - 1):
            if (i + (d - 1)) % 2 == 0:
                x_stabs.append([self._q(i, d - 1), self._q(i + 1, d - 1)])
            if (i - 1) % 2 == 0:
                x_stabs.append([self._q(i, 0), self._q(i + 1, 0)])

        for j in range(d - 1):
            if (j - 1) % 2 == 1:
                z_stabs.append([self._q(0, j), self._q(0, j + 1)])
            if ((d - 1) + j) % 2 == 1:
                z_stabs.append([self._q(d - 1, j), self._q(d - 1, j + 1)])

        return x_stabs, z_stabs

    def _build_bonds(self) -> List[List[int]]:
        d = self.d
        bonds: List[List[int]] = []
        for i in range(d):
            for j in range(d):
                if i + 1 < d:
                    bonds.append([self._q(i, j), self._q(i + 1, j)])
                if j + 1 < d:
                    bonds.append([self._q(i, j), self._q(i, j + 1)])
        return bonds

    def get_vertex_all_hetero(self) -> List[List[int]]:
        """Mirror of ToricCodeGeometry.get_vertex_all_hetero (no -1 entries here)."""
        return [list(s) for s in self.vertex_all]

    def wilson_paths(self, center=None
                     ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """Closed / open Wilson paths for the BFFM string order parameter.

        Our boundary convention (rough X on top/bottom = j in {0, d-1}, smooth
        Z on left/right = i in {0, d-1}) means:
          - sigma^x logical = vertical column (fixed i, varies j) — commutes
            with every Z-stabilizer (bulk and boundary).
          - sigma^z logical = horizontal row (fixed j, varies i) — commutes
            with every X-stabilizer.

        Construction:
          - closed X-loop = full sigma^x column connecting top/bottom X-boundary
          - open X-string = first half of that column
          - closed Z-loop = full sigma^z row connecting left/right Z-boundary
          - open Z-string = first half of that row

        Returns: (closed_x, open_x, closed_z, open_z) — qubit-index lists.
        The center kwarg picks (i_col, j_row); defaults to the lattice middle.
        """
        d = self.d
        cx, cy = (d // 2, d // 2) if center is None else center
        half = d // 2 + 1
        # X-loop: column at i = cx, varies j (top -> bottom).
        closed_x = [self._q(cx, j) for j in range(d)]
        open_x   = [self._q(cx, j) for j in range(half)]
        # Z-loop: row at j = cy, varies i (left -> right).
        closed_z = [self._q(i, cy) for i in range(d)]
        open_z   = [self._q(i, cy) for i in range(half)]
        return closed_x, open_x, closed_z, open_z
