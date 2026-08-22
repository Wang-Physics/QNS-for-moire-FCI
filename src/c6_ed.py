"""Translation-resolved ED on C6-symmetric triangular supercells.

The paper's 27-cell cluster is not a rectangular 3x9 mesh. Its real-space
translation matrix is ``[[3,-3],[3,6]]`` (columns in the primitive direct
lattice basis), giving a 3sqrt(3) x 3sqrt(3) torus. This module keeps the
finite momentum group exact using integer residues.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .continuum import ContinuumModel, reciprocal_shells
from .multiband_ed import COULOMB_MEV_NM, _apply_pair, shifted_overlap


@dataclass(frozen=True)
class TriangularSupercell:
    matrix: np.ndarray
    determinant: int
    numerators: np.ndarray
    fractions: np.ndarray
    centered_fractions: np.ndarray
    center_shifts: np.ndarray
    add_table: np.ndarray
    negation: np.ndarray
    residue_lookup: np.ndarray
    rotation60: np.ndarray

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "TriangularSupercell":
        matrix = np.asarray(matrix, dtype=np.int64)
        if matrix.shape != (2, 2):
            raise ValueError("supercell matrix must be 2x2")
        determinant = int(round(np.linalg.det(matrix)))
        if determinant <= 0:
            raise ValueError("supercell matrix must have positive determinant")

        st = matrix.T
        adjugate = np.array(
            [[st[1, 1], -st[0, 1]], [-st[1, 0], st[0, 0]]], dtype=np.int64
        )
        residues: set[tuple[int, int]] = set()
        for m0 in range(determinant):
            for m1 in range(determinant):
                num = tuple(
                    (adjugate @ np.array([m0, m1], dtype=np.int64) % determinant).tolist()
                )
                residues.add(num)
        if len(residues) != determinant:
            raise RuntimeError(f"expected {determinant} momenta, found {len(residues)}")

        def centered(num: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
            frac = np.asarray(num, dtype=float) / determinant
            candidates = []
            for dm in range(-2, 2):
                for dn in range(-2, 2):
                    value = frac + np.array([dm, dn], dtype=float)
                    norm2 = value[0] ** 2 + value[1] ** 2 - value[0] * value[1]
                    candidates.append((norm2, dm, dn, value))
            _, dm, dn, value = min(candidates, key=lambda item: (item[0], item[1], item[2]))
            return value, np.array([dm, dn], dtype=np.int64)

        def sort_key(num: tuple[int, int]) -> tuple[float, float, int, int]:
            value, _ = centered(num)
            cart = np.array(
                [value[0] - 0.5 * value[1], np.sqrt(3.0) * value[1] / 2.0]
            )
            return (
                float(cart @ cart),
                float(np.mod(np.arctan2(cart[1], cart[0]), 2.0 * np.pi)),
                num[0],
                num[1],
            )

        ordered = [(0, 0)] + sorted(residues - {(0, 0)}, key=sort_key)
        numerators = np.asarray(ordered, dtype=np.int64)
        fractions = numerators.astype(float) / determinant
        centered_values, shifts = zip(*(centered(tuple(num)) for num in ordered))
        centered_fractions = np.asarray(centered_values)
        center_shifts = np.asarray(shifts, dtype=np.int64)

        residue_lookup = -np.ones((determinant, determinant), dtype=np.int32)
        for index, (m, n) in enumerate(numerators):
            residue_lookup[m, n] = index
        add_table = np.empty((determinant, determinant), dtype=np.int32)
        negation = np.empty(determinant, dtype=np.int32)
        rotation60 = np.empty(determinant, dtype=np.int32)
        for left in range(determinant):
            for right in range(determinant):
                residue = tuple(
                    ((numerators[left] + numerators[right]) % determinant).tolist()
                )
                add_table[left, right] = residue_lookup[residue]
            minus = tuple((-numerators[left] % determinant).tolist())
            negation[left] = residue_lookup[minus]
            x, y = numerators[left]
            rotated = ((x - y) % determinant, x % determinant)
            rotation60[left] = residue_lookup[rotated]
            if rotation60[left] < 0:
                raise ValueError("supercell is not closed under C6 rotation")
        return cls(
            matrix=matrix,
            determinant=determinant,
            numerators=numerators,
            fractions=fractions,
            centered_fractions=centered_fractions,
            center_shifts=center_shifts,
            add_table=add_table,
            negation=negation,
            residue_lookup=residue_lookup,
            rotation60=rotation60,
        )

    @classmethod
    def c6_27(cls) -> "TriangularSupercell":
        return cls.from_matrix(np.array([[3, -3], [3, 6]], dtype=np.int64))

    @classmethod
    def c6_12(cls) -> "TriangularSupercell":
        return cls.from_matrix(np.array([[2, -2], [2, 4]], dtype=np.int64))

    def index_of_fraction(self, first: int, second: int, denominator: int) -> int:
        if self.determinant % denominator:
            raise ValueError("requested fraction is incompatible with this cluster")
        scale = self.determinant // denominator
        result = int(
            self.residue_lookup[
                (first * scale) % self.determinant,
                (second * scale) % self.determinant,
            ]
        )
        if result < 0:
            raise ValueError("requested momentum is absent from this cluster")
        return result

    def c6_orbits(self) -> list[list[int]]:
        unseen = set(range(self.determinant))
        orbits: list[list[int]] = []
        while unseen:
            start = min(unseen)
            orbit = []
            point = start
            while point not in orbit:
                orbit.append(point)
                point = int(self.rotation60[point])
            for item in orbit:
                unseen.discard(item)
            orbits.append(orbit)
        return orbits


def cluster_bloch_states(
    model: ContinuumModel, cluster: TriangularSupercell, n_bands: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nk = cluster.determinant
    energies = np.empty((nk, n_bands), dtype=float)
    vectors = np.empty((nk, model.dim, n_bands), dtype=np.complex128)
    k_points = np.empty((nk, 2), dtype=float)
    for index, fraction in enumerate(cluster.fractions):
        k = fraction[0] * model.b1 + fraction[1] * model.b2
        vals, vecs = model.solve(k)
        energies[index] = vals[:n_bands]
        vectors[index] = vecs[:, :n_bands]
        k_points[index] = model.wrap_to_hexagon(k)
    return energies, vectors, k_points


def density_matrix_for_cluster_transfer(
    model: ContinuumModel,
    cluster: TriangularSupercell,
    vectors: np.ndarray,
    n_bands: int,
    transfer: int,
    reciprocal_shift: tuple[int, int],
) -> np.ndarray:
    nk = cluster.determinant
    rho = np.zeros((nk * n_bands, nk * n_bands), dtype=np.complex128)
    qm, qn = reciprocal_shift
    qfrac = cluster.centered_fractions[transfer]
    for initial in range(nk):
        final = int(cluster.add_table[initial, transfer])
        wrap_float = cluster.fractions[initial] + qfrac - cluster.fractions[final]
        wrap = np.rint(wrap_float).astype(np.int64)
        if not np.allclose(wrap_float, wrap, atol=1.0e-12):
            raise RuntimeError("noninteger reciprocal wrap in cluster density operator")
        form = shifted_overlap(
            model,
            vectors[final, :, :n_bands],
            vectors[initial, :, :n_bands],
            (int(qm + wrap[0]), int(qn + wrap[1])),
        )
        initial0 = initial * n_bands
        final0 = final * n_bands
        rho[final0 : final0 + n_bands, initial0 : initial0 + n_bands] = form
    return rho


def build_cluster_pair_interaction(
    model: ContinuumModel,
    cluster: TriangularSupercell,
    vectors: np.ndarray,
    n_bands: int,
    dielectric: float,
    reciprocal_shell_count: int,
) -> tuple[np.ndarray, list[tuple[int, int]], float]:
    n_orbitals = cluster.determinant * n_bands
    pairs = list(combinations(range(n_orbitals), 2))
    first = np.asarray([x for x, _ in pairs], dtype=int)
    second = np.asarray([y for _, y in pairs], dtype=int)
    interaction = np.zeros((len(pairs), len(pairs)), dtype=np.complex128)
    area = cluster.determinant * model.cell_area
    reciprocal_vectors = reciprocal_shells(reciprocal_shell_count)
    for transfer, qfrac in enumerate(cluster.centered_fractions):
        qmesh = qfrac[0] * model.b1 + qfrac[1] * model.b2
        for qm, qn in reciprocal_vectors:
            q = qmesh + qm * model.b1 + qn * model.b2
            qnorm = float(np.linalg.norm(q))
            if qnorm < 1.0e-13:
                continue
            rho = density_matrix_for_cluster_transfer(
                model, cluster, vectors, n_bands, transfer, (qm, qn)
            )
            rho_minus = rho.conj().T
            direct = rho[first[:, None], first[None, :]] * rho_minus[
                second[:, None], second[None, :]
            ]
            exchange = rho[first[:, None], second[None, :]] * rho_minus[
                second[:, None], first[None, :]
            ]
            interaction += (
                2.0 * np.pi * COULOMB_MEV_NM / (dielectric * qnorm * area)
            ) * (direct - exchange)
    herm_error = np.linalg.norm(interaction - interaction.conj().T) / max(
        np.linalg.norm(interaction), 1.0e-30
    )
    if herm_error < 1.0e-7:
        interaction = 0.5 * (interaction + interaction.conj().T)
    return interaction, pairs, float(herm_error)


def particle_hole_transform_one_band(
    orbital_energies: np.ndarray,
    pair_interaction: np.ndarray,
    pairs: list[tuple[int, int]],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return exact hole constant, one-body energies and two-body matrix."""
    energies = np.asarray(orbital_energies, dtype=float).reshape(-1)
    n_orbitals = len(energies)
    if n_orbitals > 63:
        raise ValueError("bit representation supports at most 63 orbitals")
    full = (1 << n_orbitals) - 1
    pair_lookup = {pair: index for index, pair in enumerate(pairs)}
    full_energy = float(np.sum(energies) + np.trace(pair_interaction).real)
    hole_one_body = np.empty(n_orbitals, dtype=float)
    for hole in range(n_orbitals):
        occupied = [orb for orb in range(n_orbitals) if orb != hole]
        diagonal = sum(energies[occupied])
        diagonal += sum(
            pair_interaction[pair_lookup[pair], pair_lookup[pair]].real
            for pair in combinations(occupied, 2)
        )
        hole_one_body[hole] = diagonal - full_energy

    two_hole = np.zeros_like(pair_interaction)
    for column, (a, b) in enumerate(pairs):
        electron_state = full ^ (1 << a) ^ (1 << b)
        occupied = [orb for orb in range(n_orbitals) if (electron_state >> orb) & 1]
        diagonal = sum(energies[occupied])
        diagonal += sum(
            pair_interaction[pair_lookup[pair], pair_lookup[pair]].real
            for pair in combinations(occupied, 2)
        )
        two_hole[column, column] = diagonal
        column_phase = -1.0 if (a + b) % 2 else 1.0
        for c, d in combinations(occupied, 2):
            initial_pair = pair_lookup[(c, d)]
            destination_pair = pair_lookup[(a, b)]
            amplitude = pair_interaction[destination_pair, initial_pair]
            if abs(amplitude) < 1.0e-13:
                continue
            row = pair_lookup[(c, d)]
            applied = _apply_pair(electron_state, c, d, a, b)
            expected = full ^ (1 << c) ^ (1 << d)
            if applied is None or applied[0] != expected:
                raise RuntimeError("particle-hole pair map failed")
            row_phase = -1.0 if (c + d) % 2 else 1.0
            two_hole[row, column] += (
                row_phase * column_phase * applied[1] * amplitude
            )

    hole_pair = two_hole.copy()
    for index, (a, b) in enumerate(pairs):
        hole_pair[index, index] -= full_energy + hole_one_body[a] + hole_one_body[b]
    herm_error = np.linalg.norm(hole_pair - hole_pair.conj().T) / max(
        np.linalg.norm(hole_pair), 1.0e-30
    )
    if herm_error > 1.0e-7:
        raise RuntimeError(f"particle-hole interaction is not Hermitian: {herm_error:.3e}")
    return full_energy, hole_one_body, 0.5 * (hole_pair + hole_pair.conj().T)
