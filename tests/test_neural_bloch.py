import unittest

import numpy as np
import torch

from src.neural_bloch import NeuralBlochConfig
from src.neural_bloch_diagnostics import (
    five_band_ed_population,
    projected_momentum_occupation,
)
from src.neural_bloch_overlap import (
    bloch_orbital_values,
    ed_coordinate_wavefunction,
)
from src.run_neural_bloch import DATA, build_wavefunction


class NeuralBlochTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_default_dtype(torch.float64)
        cls.continuum, cls.wavefunction = build_wavefunction(
            6,
            NeuralBlochConfig(
                width=24,
                message_passing_steps=2,
                determinants=12,
                orbital_hidden=24,
            ),
            seed=19,
        )
        state = np.load(DATA / "fig2_1band_ground_state.npz")
        cls.basis = np.asarray(state["basis"])
        cls.amplitudes = np.asarray(state["amplitudes"])
        cls.wavefunction.initialize_from_fock_expansion(
            cls.basis, cls.amplitudes
        )

    def test_full_rdm_diagonal_gives_bloch_momentum_occupations(self):
        rho = np.zeros((45, 45), dtype=np.complex128)
        diagonal = np.arange(45, dtype=float).reshape(9, 5) / 100.0
        np.fill_diagonal(rho, diagonal.ravel())
        rho[0, 17] = 4.0 + 2.0j
        rho[17, 0] = 4.0 - 2.0j
        band1, first_five = projected_momentum_occupation(rho)
        np.testing.assert_allclose(band1, diagonal[:, 0])
        np.testing.assert_allclose(first_five, diagonal.sum(axis=1))

    def test_five_band_ed_population_supports_both_fillings(self):
        for particles in (3, 6):
            population = five_band_ed_population(particles)
            self.assertEqual(population.shape, (5,))
            self.assertAlmostEqual(float(population.sum()), 1.0, places=10)

    def test_all_parameters_are_trainable(self):
        self.assertTrue(all(parameter.requires_grad for parameter in self.wavefunction.parameters()))

    def test_default_lecun_trial_reaches_all_active_parameter_blocks(self):
        _, generic = build_wavefunction(
            6,
            NeuralBlochConfig(
                width=24, message_passing_steps=1, determinants=12,
                orbital_hidden=24, c3_irrep=2, outer_c3_projector=True,
            ),
            seed=37,
        )
        generator = torch.Generator().manual_seed(41)
        positions = torch.rand(6, 6, 2, generator=generator)
        layers = torch.randint(0, 2, (6, 6), generator=generator)
        logpsi = generic(positions, layers)
        (logpsi.real.square().mean() + 0.1 * logpsi.imag.square().mean()).backward()
        prefixes = (
            "node_hidden_initial", "edge_hidden_initial", "steps", "backflow_heads",
            "orbital_mlps", "momentum_", "determinant_logits",
        )
        for prefix in prefixes:
            gradients = [
                parameter.grad
                for name, parameter in generic.named_parameters()
                if name.startswith(prefix)
            ]
            self.assertTrue(gradients)
            self.assertTrue(any(
                gradient is not None and torch.linalg.norm(gradient) > 0
                for gradient in gradients
            ))

    def test_exact_c3_projection_matches_ed_irrep_and_is_covariant(self):
        _, projected = build_wavefunction(
            6,
            NeuralBlochConfig(
                width=24, message_passing_steps=2, determinants=12,
                orbital_hidden=24, c3_irrep=2, outer_c3_projector=True,
            ),
            seed=23,
        )
        projected.initialize_from_fock_expansion(self.basis, self.amplitudes)
        generator = torch.Generator().manual_seed(29)
        positions = torch.rand(7, 6, 2, generator=generator)
        layers = torch.randint(0, 2, (7, 6), generator=generator)
        raw = projected._unprojected_determinant_values(positions, layers)
        symmetric = projected.determinant_values(positions, layers)
        relative = torch.abs(symmetric - raw) / torch.abs(raw).clamp_min(1.0e-12)
        self.assertLess(float(torch.max(relative).detach()), 4.0e-4)

        projected.reference_fast_path = False
        with torch.no_grad():
            projected.backflow_heads[0].weight.normal_(std=2.0e-2)
        rotated = projected.c3_rotate_positions(positions)
        acted = projected.c3_gauge_phase(positions, layers) * (
            projected.determinant_values(rotated, layers)
        )
        target = np.exp(4j * np.pi / 3.0) * projected.determinant_values(
            positions, layers
        )
        covariance = torch.abs(acted - target) / torch.abs(target).clamp_min(1.0e-12)
        self.assertLess(float(torch.max(covariance).detach()), 2.0e-9)

    def test_one_band_ed_warm_start_is_exact_in_coordinates(self):
        generator = torch.Generator().manual_seed(31)
        positions = torch.rand(5, 6, 2, generator=generator)
        layers = torch.randint(0, 2, (5, 6), generator=generator)
        neural = self.wavefunction.determinant_values(positions, layers)
        orbital = bloch_orbital_values(
            positions, layers, 1, self.continuum
        )
        exact = ed_coordinate_wavefunction(
            orbital, self.basis, self.amplitudes, chunk_size=12
        )
        ratio = neural / exact
        normalized = ratio / ratio[0]
        self.assertLess(float(torch.max(torch.abs(normalized - 1.0)).detach()), 2.0e-11)

    def test_particle_exchange_changes_only_determinant_sign(self):
        generator = torch.Generator().manual_seed(53)
        positions = torch.rand(4, 6, 2, generator=generator)
        layers = torch.randint(0, 2, (4, 6), generator=generator)
        original = self.wavefunction.determinant_values(positions, layers)
        exchanged_positions = positions.clone()
        exchanged_layers = layers.clone()
        exchanged_positions[:, [0, 1]] = exchanged_positions[:, [1, 0]]
        exchanged_layers[:, [0, 1]] = exchanged_layers[:, [1, 0]]
        exchanged = self.wavefunction.determinant_values(
            exchanged_positions, exchanged_layers
        )
        relative = torch.abs(exchanged + original) / torch.abs(original).clamp_min(1.0e-12)
        self.assertLess(float(torch.max(relative).detach()), 2.0e-10)

    def test_orbital_factor_J_is_applied_exactly_once(self):
        _, state = build_wavefunction(
            2,
            NeuralBlochConfig(
                width=24, message_passing_steps=1, determinants=1,
                orbital_hidden=24,
                fixed_gamma_no_m=True,
            ),
            seed=61,
        )
        state.reference_fast_path = False
        with torch.no_grad():
            for parameter in state.parameters():
                parameter.zero_()
            state.orbital_mlps[0][-1].bias[0] = 0.5
            state.sector_mixing_real[0].copy_(
                torch.linspace(0.25, 1.25, len(state.gamma_sector_indices))
            )
            state.sector_mixing_imag[0].copy_(
                torch.linspace(-0.4, 0.3, len(state.gamma_sector_indices))
            )
        generator = torch.Generator().manual_seed(67)
        positions = torch.rand(3, 2, 2, generator=generator)
        layers = torch.randint(0, 2, (3, 2), generator=generator)
        cartesian = state.cartesian(positions).to(torch.complex128)
        all_bare = state._bloch_values(cartesian, layers)
        sector_determinants = torch.linalg.det(
            all_bare[:, state.gamma_sector_indices, :]
        )
        mixing = torch.complex(
            state.sector_mixing_real[0], state.sector_mixing_imag[0]
        )
        bare = torch.einsum("m,bm->b", mixing, sector_determinants)
        actual = state._unprojected_determinant_values(positions, layers)
        expected = bare * 0.5 ** state.n_particles
        torch.testing.assert_close(actual, expected, rtol=2.0e-12, atol=2.0e-12)


if __name__ == "__main__":
    unittest.main()
