from __future__ import annotations

import unittest

from test_large_cluster_ed import (
    test_c6_cluster_contains_gamma_k_kprime,
    test_general_cluster_operator_matches_rectangular_implementation,
    test_particle_hole_transform_is_sector_isospectral,
)


class LargeClusterRegression(unittest.TestCase):
    def test_cluster_geometry(self) -> None:
        test_c6_cluster_contains_gamma_k_kprime()

    def test_projected_operator(self) -> None:
        test_general_cluster_operator_matches_rectangular_implementation()

    def test_particle_hole_map(self) -> None:
        test_particle_hole_transform_is_sector_isospectral()


if __name__ == "__main__":
    unittest.main()
