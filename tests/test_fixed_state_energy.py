import unittest

from src.fixed_state_energy_regression import monte_carlo_components


class FixedStateHamiltonianRegression(unittest.TestCase):
    """Independent delta-r=0, J=1, fixed-M Bloch-Slater audit."""

    @classmethod
    def setUpClass(cls):
        cls.result = monte_carlo_components(
            walkers=2048, burn_sweeps=80, measurement_sweeps=10, seed=2701
        )

    def test_validation_samples_never_update_parameters(self):
        self.assertFalse(self.result["used_for_parameter_updates"])

    def test_all_local_energy_components_match_finite_basis(self):
        # A stochastic 5-sigma guard catches normalization, q=0, duplicate-q,
        # layer, tunneling-phase, and Coulomb-prefactor convention errors.
        for name in ("kinetic", "moire", "tunneling", "coulomb", "total"):
            difference = abs(self.result["difference_meV"][name])
            sem = self.result["sem_meV"][name]
            self.assertLess(
                difference, 5.0 * sem + 0.08,
                msg=f"{name}: difference={difference}, sem={sem}",
            )


if __name__ == "__main__":
    unittest.main()
