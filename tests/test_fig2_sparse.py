from __future__ import annotations

import math
import unittest
import json
from pathlib import Path

import numpy as np

from src.continuum import ContinuumModel, ContinuumParameters
from src.multiband_ed import (
    build_pair_interaction,
    diagonalize_projection,
    mesh_bloch_states,
)
from src.sparse_ed import diagonalize_sparse_sector
from src.make_figures import band_population_from_state


class SparseKernelRegression(unittest.TestCase):
    def test_native_sparse_matches_dense_block(self):
        model = ContinuumModel(ContinuumParameters(plane_wave_shells=3))
        energies, vectors, _ = mesh_bloch_states(model, 3, 3, 1)
        interaction, pairs, _ = build_pair_interaction(model, vectors, 3, 3, 1, 5.0, 1)
        dense = diagonalize_projection(energies, interaction, pairs, 3, 3, 1, 3)
        dense_gamma = next(result for result in dense if result.momentum == (0, 0))
        sparse_gamma = diagonalize_sparse_sector(
            energies, interaction, pairs, 3, 3, 1, 3, (0, 0), n_eigenvalues=3
        )
        np.testing.assert_allclose(sparse_gamma.energies, dense_gamma.energies[:3], atol=1.0e-10)
        self.assertLess(sparse_gamma.residual, 1.0e-9)

    def test_fig2_hilbert_dimensions(self):
        self.assertEqual(math.comb(9, 6), 84)
        self.assertEqual(math.comb(27, 6), 296010)
        self.assertEqual(math.comb(45, 6), 8145060)

    def test_saved_fig2_first_three_points_are_within_digitization_error(self):
        path = Path(__file__).resolve().parents[1] / "result" / "data" / "fig2_benchmark_summary.json"
        if not path.exists():
            self.skipTest("benchmark has not been run")
        summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(summary["checks"]["paper_agreement"])
        for result in summary["projections"].values():
            self.assertLess(abs(result["difference_from_paper_meV_per_hole"]), 0.10)

    def test_band_resolved_weights_are_additive(self):
        for n_bands in (1, 3, 5):
            weights = band_population_from_state(n_bands)
            self.assertEqual(len(weights), n_bands)
            self.assertTrue(np.all(weights >= 0.0))
            self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)

    def test_four_band_energy_is_variationally_between_three_and_five(self):
        data = Path(__file__).resolve().parents[1] / "result" / "data"
        summary = json.loads((data / "fig2_benchmark_summary.json").read_text())
        four = json.loads((data / "fig2_4band_gamma_summary.json").read_text())
        five = json.loads((data / "fig2_5band_gamma_summary.json").read_text())
        e3 = summary["projections"]["3"]["energy_per_hole_meV"]
        e4 = four["ground_energy_per_hole_meV"]
        e5 = five["energies_per_hole_meV"][0]
        self.assertGreaterEqual(e3, e4)
        self.assertGreaterEqual(e4, e5)
        self.assertLess(four["residual_meV"], 1.0e-4)


if __name__ == "__main__":
    unittest.main()
