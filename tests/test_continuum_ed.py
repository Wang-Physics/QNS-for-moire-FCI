from __future__ import annotations

import unittest

import numpy as np

from src.continuum import ContinuumModel, ContinuumParameters, reciprocal_shells
from src.multiband_ed import (
    build_pair_interaction,
    collect_low_states,
    diagonalize_projection,
    mesh_bloch_states,
    momentum_occupation,
)


class ContinuumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ContinuumModel(ContinuumParameters(plane_wave_shells=3))

    def test_reciprocal_shells_are_inversion_symmetric(self):
        vectors = set(reciprocal_shells(5))
        self.assertIn((0, 0), vectors)
        for m, n in vectors:
            self.assertIn((-m, -n), vectors)

    def test_hamiltonian_is_hermitian(self):
        k = 0.173 * self.model.b1 - 0.219 * self.model.b2
        h = self.model.hamiltonian(k)
        relative = np.linalg.norm(h - h.conj().T) / np.linalg.norm(h)
        self.assertLess(relative, 1.0e-13)

    def test_high_symmetry_band_cutoff_convergence(self):
        model4 = ContinuumModel(ContinuumParameters(plane_wave_shells=4))
        model6 = ContinuumModel(ContinuumParameters(plane_wave_shells=6))
        for name in ("Gamma", "K", "M"):
            e4, _ = model4.solve(model4.high_symmetry_points()[name])
            e6, _ = model6.solve(model6.high_symmetry_points()[name])
            self.assertLess(np.max(np.abs(e4[:3] - e6[:3])), 2.0e-3)

    def test_lowest_band_chern_is_near_integer(self):
        # Midpoint quadrature avoids sampling the sharp high-symmetry peak.
        values = []
        size = 17
        for i in range(size):
            for j in range(size):
                k = ((i + 0.5) / size) * self.model.b1 + ((j + 0.5) / size) * self.model.b2
                values.append(self.model.quantum_geometry(k, 1)["berry_curvature"][0])
        chern = np.mean(values) * self.model.bz_area / (2.0 * np.pi)
        self.assertLess(abs(abs(chern) - 1.0), 0.35)


class EDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ContinuumModel(ContinuumParameters(plane_wave_shells=3))
        cls.energies, cls.vectors, _ = mesh_bloch_states(cls.model, 3, 3, 2)

    def test_one_band_pair_and_ed_checks(self):
        pair, pairs, pair_herm = build_pair_interaction(
            self.model, self.vectors, 3, 3, 1, dielectric=5.0, reciprocal_shell_count=1
        )
        self.assertLess(pair_herm, 1.0e-12)
        sectors = diagonalize_projection(self.energies, pair, pairs, 3, 3, 1, 3)
        self.assertEqual(sum(len(s.basis) for s in sectors), 84)
        self.assertLess(max(s.hermiticity_error for s in sectors), 1.0e-12)
        self.assertLess(max(s.residual for s in sectors), 1.0e-9)
        low = collect_low_states(sectors, 3)
        occupation = momentum_occupation(low, 3, 3, 1)
        self.assertAlmostEqual(float(occupation.sum()), 3.0, places=11)

    def test_two_band_space_is_variationally_larger(self):
        ground = []
        for n_bands in (1, 2):
            pair, pairs, _ = build_pair_interaction(
                self.model, self.vectors, 3, 3, n_bands, dielectric=5.0, reciprocal_shell_count=1
            )
            sectors = diagonalize_projection(self.energies, pair, pairs, 3, 3, n_bands, 3)
            ground.append(collect_low_states(sectors, 1)[0][0])
        self.assertLessEqual(ground[1], ground[0] + 1.0e-8)


if __name__ == "__main__":
    unittest.main()
