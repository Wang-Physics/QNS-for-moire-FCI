from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from src.c6_ed import TriangularSupercell
from src.run_fig5_observables import paper_momentum_coordinates


ROOT = Path(__file__).resolve().parents[1]


class FinalBenchmarkRegression(unittest.TestCase):
    def test_paper_momentum_index_is_bijective(self) -> None:
        cluster = TriangularSupercell.c6_27()
        coordinates = paper_momentum_coordinates(cluster)
        indices = coordinates[:, 0] + 3 * coordinates[:, 1]
        self.assertEqual(sorted(indices.tolist()), list(range(27)))
        self.assertEqual(int(indices[0]), 0)
        self.assertEqual(int(indices[cluster.index_of_fraction(1, 2, 3)]), 9)
        self.assertEqual(int(indices[cluster.index_of_fraction(2, 1, 3)]), 18)

    def test_saved_fig5_observables_have_expected_cdw_contrast(self) -> None:
        path = ROOT / "result" / "data" / "fig5_27cell_observables.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        one = data["fillings"]["1/3"]
        two = data["fillings"]["2/3"]
        self.assertAlmostEqual(sum(one["n_k"]), 9.0, places=10)
        self.assertAlmostEqual(sum(two["n_k"]), 18.0, places=10)
        indices = np.asarray(data["momentum_indices"])
        k_sector = int(np.flatnonzero(indices == 18)[0])
        self.assertGreater(one["S_q"][k_sector], 4.0 * two["S_q"][k_sector])
        self.assertEqual(one["S_q_max_sector"], int(np.argmax(one["S_q"])))

    def test_manybody_bundle_chern_distinguishes_cdw_and_fci(self) -> None:
        data_dir = ROOT / "result" / "data"
        one = json.loads((data_dir / "manybody_berry_27cell_nu1of3.json").read_text())
        two = json.loads((data_dir / "manybody_berry_27cell_nu2of3.json").read_text())
        self.assertLess(abs(one["bundle_chern_number"]), 1.0e-4)
        self.assertAlmostEqual(two["bundle_chern_number"], 2.0, places=5)
        self.assertGreater(one["minimum_sampled_manifold_gap_meV"], 1.0)
        self.assertGreater(two["minimum_manifold_gap_meV"], 5.0)
        self.assertGreater(one["minimum_link_overlap"], 0.8)
        self.assertGreater(two["minimum_link_singular_value"], 0.8)
        self.assertEqual(np.asarray(one["berry_curvature"]).shape, (6, 6))
        self.assertEqual(np.asarray(two["berry_curvature"]).shape, (6, 6))

    def test_final_report_and_four_vector_figures_exist(self) -> None:
        targets = [
            ROOT / "result" / "AI_for_Physics.pdf",
            ROOT / "result" / "figures" / "fig1_bands_quantum_geometry.pdf",
            ROOT / "result" / "figures" / "fig2_multiband_ed.pdf",
            ROOT / "result" / "figures" / "fig3_fig5_spectra_observables.pdf",
            ROOT / "result" / "figures" / "fig4_manybody_berry_curvature.pdf",
        ]
        for target in targets:
            self.assertTrue(target.exists(), str(target))
            self.assertGreater(target.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
