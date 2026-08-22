from __future__ import annotations

import json
import unittest
from itertools import combinations
from pathlib import Path

import numpy as np

from src.observables import DensityContext


class DensityObservableTests(unittest.TestCase):
    def test_native_one_body_operator_matches_enumeration(self):
        rng = np.random.default_rng(7)
        basis = np.asarray([
            sum(1 << orbital for orbital in occupied)
            for occupied in combinations(range(4), 2)
        ], dtype=np.uint64)
        operator = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        vector = rng.normal(size=len(basis)) + 1j * rng.normal(size=len(basis))
        context = DensityContext(basis)
        try:
            calculated = context.apply(basis, operator, vector)
        finally:
            context.close()

        lookup = {int(state): index for index, state in enumerate(basis)}
        expected = np.zeros(len(basis), dtype=np.complex128)
        for column, source in enumerate(map(int, basis)):
            for final in range(4):
                for initial in range(4):
                    if not (source >> initial) & 1:
                        continue
                    sign = -1 if (source & ((1 << initial) - 1)).bit_count() % 2 else 1
                    intermediate = source ^ (1 << initial)
                    if (intermediate >> final) & 1:
                        continue
                    sign *= -1 if (intermediate & ((1 << final) - 1)).bit_count() % 2 else 1
                    target = intermediate | (1 << final)
                    expected[lookup[target]] += sign * operator[final, initial] * vector[column]
        np.testing.assert_allclose(calculated, expected, atol=1.0e-12)

    def test_saved_dual_filling_observables_are_normalized(self):
        data = Path(__file__).resolve().parents[1] / "result" / "data"
        path = data / "dual_filling_5band_observables.json"
        if not path.exists():
            self.skipTest("dual-filling observables have not been run")
        result = json.loads(path.read_text())
        for filling, n_holes in (("1/3", 3), ("2/3", 6)):
            record = result["fillings"][filling]
            self.assertAlmostEqual(record["n_k_sum"], n_holes, places=11)
            charge = np.asarray(record["S_charge_q"])
            spin = np.asarray(record["S_zz_q"])
            np.testing.assert_allclose(spin, charge / 4.0, atol=1.0e-13)
            self.assertAlmostEqual(charge[0, 0], 0.0, places=14)


if __name__ == "__main__":
    unittest.main()
