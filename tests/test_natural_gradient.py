import unittest

import torch

from src.run_neural_bloch import parser
from src.natural_gradient import (
    energy_force,
    fisher_vector_product,
    natural_gradient_step,
)


class ToyComplexWavefunction(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.coefficients = torch.nn.Parameter(
            torch.tensor([0.2, -0.3, 0.4], dtype=torch.float64)
        )

    def scores(self, positions, layers):
        x = positions[:, 0, 0]
        y = positions[:, 0, 1]
        layer = layers[:, 0].to(torch.float64)
        real = torch.stack([x, y, layer], dim=1)
        imag = torch.stack([y, -x, 0.5 * layer], dim=1)
        return real, imag

    def forward(self, positions, layers):
        real, imag = self.scores(positions, layers)
        return real @ self.coefficients + 1j * (imag @ self.coefficients)


class NaturalGradientTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        generator = torch.Generator().manual_seed(7)
        self.positions = torch.rand(9, 1, 2, generator=generator)
        self.layers = torch.randint(0, 2, (9, 1), generator=generator)
        self.energy = torch.complex(
            torch.linspace(-1.0, 1.0, 9),
            torch.linspace(0.4, -0.2, 9),
        )
        self.wavefunction = ToyComplexWavefunction()

    def test_force_and_fisher_match_explicit_scores(self):
        real, imag = self.wavefunction.scores(self.positions, self.layers)
        centered_real = real - real.mean(0)
        centered_imag = imag - imag.mean(0)
        centered_energy = self.energy - self.energy.mean()
        expected_force = 2.0 * torch.mean(
            centered_energy.real[:, None] * real
            + centered_energy.imag[:, None] * imag, dim=0
        )
        force = energy_force(
            self.wavefunction, self.positions, self.layers, self.energy, chunk_size=4
        )
        torch.testing.assert_close(force, expected_force)

        vector = torch.tensor([0.7, -0.2, 0.5])
        damping = 0.03
        fisher = (
            centered_real.T @ centered_real / len(real)
            + centered_imag.T @ centered_imag / len(real)
        )
        expected_product = fisher @ vector + damping * vector
        product = fisher_vector_product(
            self.wavefunction, self.positions, self.layers, vector,
            damping=damping, chunk_size=4,
        )
        torch.testing.assert_close(product, expected_product)

    def test_cli_defaults_match_paper_optimizer_protocol(self):
        args = parser().parse_args([])
        self.assertEqual(args.optimizer, "natural_gradient")
        self.assertEqual(args.learning_rate, 2.0e-3)
        self.assertEqual(args.walkers, 4128)
        self.assertEqual(args.energy_batch, 4128)
        self.assertEqual(args.steps, 1000)
        self.assertIsNone(args.correction_init_scale)
        self.assertFalse(args.ed_warmstart)

    def test_natural_gradient_step_is_finite_and_updates_parameters(self):
        before = self.wavefunction.coefficients.detach().clone()
        diagnostics = natural_gradient_step(
            self.wavefunction, self.positions, self.layers, self.energy,
            learning_rate=2.0e-3, damping=1.0e-2, chunk_size=3,
            cg_iterations=8, cg_tolerance=1.0e-10,
            maximum_direction_norm=100.0,
        )
        self.assertTrue(torch.isfinite(self.wavefunction.coefficients).all())
        self.assertGreater(float(torch.linalg.norm(
            self.wavefunction.coefficients.detach() - before
        )), 0.0)
        self.assertGreater(diagnostics.force_norm, 0.0)
        self.assertGreater(diagnostics.direction_norm, 0.0)
        self.assertLessEqual(diagnostics.cg_iterations, 8)


if __name__ == "__main__":
    unittest.main()
