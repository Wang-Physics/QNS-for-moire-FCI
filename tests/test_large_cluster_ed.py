from __future__ import annotations

import numpy as np

from src.c6_ed import (
    TriangularSupercell,
    build_cluster_pair_interaction,
    cluster_bloch_states,
    particle_hole_transform_one_band,
)
from src.continuum import ContinuumModel, ContinuumParameters
from src.group_sparse_ed import all_group_sector_bases, diagonalize_sparse_basis
from src.multiband_ed import build_pair_interaction, mesh_bloch_states


def test_c6_cluster_contains_gamma_k_kprime() -> None:
    cluster = TriangularSupercell.c6_27()
    assert cluster.determinant == 27
    assert cluster.index_of_fraction(2, 1, 3) != 0
    assert cluster.index_of_fraction(1, 2, 3) != 0
    assert sorted(item for orbit in cluster.c6_orbits() for item in orbit) == list(range(27))


def test_general_cluster_operator_matches_rectangular_implementation() -> None:
    model = ContinuumModel(
        ContinuumParameters(theta_deg=2.6, plane_wave_shells=1)
    )
    cluster = TriangularSupercell.from_matrix(np.array([[3, 0], [0, 3]]))
    energies, vectors, _ = cluster_bloch_states(model, cluster, 1)
    old_energies, old_vectors, _ = mesh_bloch_states(model, 3, 3, 1)
    mapping = np.asarray(
        [
            (int(round(fraction[0] * 3)) % 3) * 3
            + int(round(fraction[1] * 3)) % 3
            for fraction in cluster.fractions
        ]
    )
    assert np.max(np.abs(energies[:, 0] - old_energies.reshape(-1)[mapping])) < 1.0e-12
    pair, pairs, _ = build_cluster_pair_interaction(
        model, cluster, vectors, 1, dielectric=5.0, reciprocal_shell_count=1
    )
    old_pair, old_pairs, _ = build_pair_interaction(
        model, old_vectors, 3, 3, 1, dielectric=5.0, reciprocal_shell_count=1
    )
    old_lookup = {value: index for index, value in enumerate(old_pairs)}
    pair_mapping = []
    signs = []
    for first, second in pairs:
        a, b = int(mapping[first]), int(mapping[second])
        pair_mapping.append(old_lookup[tuple(sorted((a, b)))])
        signs.append(1 if a < b else -1)
    signs = np.asarray(signs)
    reference = old_pair[np.ix_(pair_mapping, pair_mapping)] * signs[:, None] * signs[None, :]
    assert np.linalg.norm(pair - reference) / np.linalg.norm(reference) < 1.0e-12


def test_particle_hole_transform_is_sector_isospectral() -> None:
    model = ContinuumModel(
        ContinuumParameters(theta_deg=2.6, plane_wave_shells=1)
    )
    cluster = TriangularSupercell.from_matrix(np.array([[3, 0], [0, 3]]))
    energies, vectors, _ = cluster_bloch_states(model, cluster, 1)
    pair, pairs, _ = build_cluster_pair_interaction(
        model, cluster, vectors, 1, dielectric=5.0, reciprocal_shell_count=1
    )
    electron_bases = all_group_sector_bases(
        cluster.numerators, 9, cluster.residue_lookup, 1, 6
    )
    hole_bases = all_group_sector_bases(
        cluster.numerators, 9, cluster.residue_lookup, 1, 3
    )
    constant, hole_energies, hole_pair = particle_hole_transform_one_band(
        energies[:, 0], pair, pairs
    )
    for electron_sector in (0, cluster.index_of_fraction(2, 1, 3)):
        hole_sector = int(cluster.negation[electron_sector])
        electron = diagonalize_sparse_basis(
            electron_bases[electron_sector], energies[:, 0], pair, pairs,
            (electron_sector, 0), n_eigenvalues=3,
        )
        hole = diagonalize_sparse_basis(
            hole_bases[hole_sector], hole_energies, hole_pair, pairs,
            (hole_sector, 0), n_eigenvalues=3, energy_constant=constant,
        )
        assert np.max(np.abs(electron.energies - hole.energies)) < 1.0e-9
