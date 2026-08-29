import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import torch
from unittest.mock import patch

import src.jax_neural_bloch as jnb

from src.continuum_vmc import ContinuumTorusHamiltonian
from src.jax_neural_bloch import (
    JaxNeuralBlochSpec,
    _bloch_values,
    copy_to_torch,
    from_torch,
    initialize,
    local_energy,
    logpsi,
)
from src.neural_bloch import NeuralBlochConfig
from src.run_neural_bloch import build_wavefunction
from src.run_jax_neural_bloch import (
    MatrixFreeNaturalGradient,
    _adaptive_next_state,
    _adaptive_update_is_accepted,
    parser,
)


class JaxNeuralBlochParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_default_dtype(torch.float64)
        cls.spec = JaxNeuralBlochSpec(
            n_particles=2,
            width=24,
            message_passing_steps=1,
            determinants=1,
            orbital_hidden=24,
        )
        cls.continuum, cls.torch_wavefunction = build_wavefunction(
            2,
            NeuralBlochConfig(
                width=24,
                message_passing_steps=1,
                determinants=1,
                orbital_hidden=24,
            ),
            seed=19,
        )
        _, cls.constants = initialize(cls.spec, seed=19)
        cls.parameters = from_torch(cls.torch_wavefunction)
        generator = torch.Generator().manual_seed(29)
        cls.positions = torch.rand(2, 2, 2, generator=generator)
        cls.layers = torch.randint(0, 2, (2, 2), generator=generator)

    def test_paper_protocol_defaults_are_fixed(self):
        arguments = parser().parse_args([])
        self.assertEqual(arguments.samples, 4128)
        self.assertEqual(arguments.steps, 1000)
        self.assertEqual(arguments.learning_rate, 2.0e-3)
        self.assertEqual(arguments.width, 32)
        self.assertEqual(arguments.message_passing_steps, 2)
        self.assertEqual(arguments.determinants, 1)
        self.assertIsNone(arguments.c3_irrep)
        self.assertFalse(arguments.fixed_gamma_no_m)

    def test_adaptive_cg_acceptance_and_recovery_state_machine(self):
        args = parser().parse_args([
            "--adaptive-cg-120-step", "--steps", "120",
            "--checkpoint-interval", "10",
        ])
        diagnostic = {
            "force_dot_direction": 2.0,
            "cg_termination_reason": "true_residual_tolerance",
            "relative_residual_norm": 0.019,
            "cg_iterations": 20,
        }
        self.assertTrue(_adaptive_update_is_accepted(diagnostic, args))
        diagnostic.update({
            "cg_termination_reason": "maximum_iterations",
            "relative_residual_norm": 0.049,
            "cg_iterations": 60,
        })
        self.assertTrue(_adaptive_update_is_accepted(diagnostic, args))
        diagnostic["relative_residual_norm"] = 0.051
        self.assertFalse(_adaptive_update_is_accepted(diagnostic, args))
        diagnostic["relative_residual_norm"] = 0.019
        diagnostic["force_dot_direction"] = -1.0
        self.assertFalse(_adaptive_update_is_accepted(diagnostic, args))

        state = _adaptive_next_state(False, False, 0, args)
        self.assertEqual(state, (True, 0, 1.0e-3, 0.03))
        state = _adaptive_next_state(True, state[0], state[1], args)
        self.assertEqual(state, (True, 1, 1.0e-3, 0.03))
        state = _adaptive_next_state(True, state[0], state[1], args)
        self.assertEqual(state, (False, 0, 2.0e-3, 0.01))

    def test_true_residual_cg_checks_start_at_minimum_iteration(self):
        size = 48
        matrix = jnp.diag(jnp.geomspace(1.0, 200.0, size))
        force = jnp.linspace(0.5, 1.5, size)
        optimizer = MatrixFreeNaturalGradient.__new__(MatrixFreeNaturalGradient)
        optimizer.force = lambda *unused: force
        optimizer.product = lambda unused_flat, vector, *unused: matrix @ vector
        solution, diagnostics = optimizer.direction(
            jnp.zeros(size), None, None, None, 60, 0.0,
            min_iterations=20, true_tolerance=0.02,
        )
        explicit = np.asarray(force - matrix @ solution)
        expected = np.linalg.norm(explicit) / np.linalg.norm(np.asarray(force))
        self.assertGreaterEqual(diagnostics["cg_iterations"], 20)
        self.assertLessEqual(diagnostics["cg_iterations"], 60)
        self.assertGreaterEqual(diagnostics["true_residual_checks"], 1)
        self.assertAlmostEqual(diagnostics["relative_residual_norm"], expected, 12)
        self.assertEqual(diagnostics["preconditioner"], "none")

    def test_logpsi_matches_pytorch(self):
        expected = self.torch_wavefunction(
            self.positions, self.layers
        ).detach().numpy()
        actual = np.asarray(logpsi(
            self.parameters,
            jnp.asarray(self.positions.numpy()),
            jnp.asarray(self.layers.numpy()),
            self.constants,
            self.spec,
        ))
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)

    def test_bloch_coefficients_are_fixed_and_only_coordinates_shift(self):
        self.assertNotIn("bloch_coefficients", self.parameters)
        self.assertIn(
            "bloch_coefficients", dict(self.torch_wavefunction.named_buffers())
        )
        self.assertNotIn(
            "bloch_coefficients", dict(self.torch_wavefunction.named_parameters())
        )

        positions = jnp.asarray(self.positions.numpy())
        layers = jnp.asarray(self.layers.numpy())
        cartesian = jnp.einsum(
            "ac,bnc->bna", self.constants["supercell_lattice"], positions
        ).astype(jnp.complex128)
        displacement = (
            jnp.asarray([0.013 + 0.004j, -0.009 + 0.002j])[None, None, :]
            * jnp.ones_like(cartesian)
        )
        shifted = _bloch_values(cartesian + displacement, layers, self.constants)

        plane_momenta = (
            self.constants["momenta"][:, None, :]
            + self.constants["g_vectors"][None, :, :]
        )
        bare_phase = jnp.einsum("bna,kga->bnkg", cartesian, plane_momenta)
        shift_phase = jnp.einsum("bna,kga->bnkg", displacement, plane_momenta)
        fixed_coefficients = jnp.take(
            self.constants["bloch_coefficients"], layers, axis=1
        )
        fixed_coefficients = jnp.transpose(fixed_coefficients, (1, 2, 0, 3))
        manual = jnp.transpose(
            jnp.sum(
                fixed_coefficients
                * jnp.exp(1j * bare_phase)
                * jnp.exp(1j * shift_phase),
                axis=-1,
            ),
            (0, 2, 1),
        )
        np.testing.assert_allclose(
            np.asarray(shifted), np.asarray(manual), rtol=2e-13, atol=2e-13
        )

    def test_c3_projected_logpsi_matches_pytorch(self):
        spec = JaxNeuralBlochSpec(
            n_particles=2, width=24, message_passing_steps=1,
            determinants=1, orbital_hidden=24, c3_irrep=2, outer_c3_projector=True,
        )
        _, torch_state = build_wavefunction(
            2,
            NeuralBlochConfig(
                width=24, message_passing_steps=1, determinants=1,
                orbital_hidden=24, c3_irrep=2, outer_c3_projector=True,
            ),
            seed=43,
        )
        parameters = from_torch(torch_state)
        _, constants = initialize(spec, seed=43)
        expected = torch_state(self.positions, self.layers).detach().numpy()
        actual = np.asarray(logpsi(
            parameters, jnp.asarray(self.positions.numpy()),
            jnp.asarray(self.layers.numpy()), constants, spec,
        ))
        np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)

    def test_fixed_gamma_no_m_matches_torch_and_is_translation_invariant(self):
        spec = JaxNeuralBlochSpec(
            n_particles=2, width=24, message_passing_steps=1,
            determinants=1, orbital_hidden=24, fixed_gamma_no_m=True,
        )
        _, torch_state = build_wavefunction(
            2,
            NeuralBlochConfig(
                width=24, message_passing_steps=1, determinants=1,
                orbital_hidden=24, fixed_gamma_no_m=True,
            ),
            seed=47,
        )
        parameters, constants = initialize(spec, seed=47)
        copy_to_torch(parameters, torch_state)
        base = np.asarray(logpsi(
            parameters, jnp.asarray(self.positions.numpy()),
            jnp.asarray(self.layers.numpy()), constants, spec,
        ))
        expected = torch_state(self.positions, self.layers).detach().numpy()
        np.testing.assert_allclose(base, expected, rtol=3e-12, atol=3e-12)
        for shift in ((1.0 / 3.0, 0.0), (0.0, 1.0 / 3.0)):
            translated = np.remainder(
                self.positions.numpy() + np.asarray(shift), 1.0
            )
            moved = np.asarray(logpsi(
                parameters, jnp.asarray(translated),
                jnp.asarray(self.layers.numpy()), constants, spec,
            ))
            np.testing.assert_allclose(
                np.exp(moved - base), np.ones(len(base)),
                rtol=3e-11, atol=3e-11,
            )
        self.assertNotIn("momentum_real", parameters)
        self.assertNotIn("momentum_imag", parameters)

    def test_jax_parameters_export_losslessly_to_torch(self):
        _, exported = build_wavefunction(
            2,
            NeuralBlochConfig(
                width=24, message_passing_steps=1, determinants=1,
                orbital_hidden=24,
            ),
            seed=97,
        )
        copy_to_torch(self.parameters, exported)
        expected = np.asarray(logpsi(
            self.parameters,
            jnp.asarray(self.positions.numpy()),
            jnp.asarray(self.layers.numpy()),
            self.constants,
            self.spec,
        ))
        actual = exported(self.positions, self.layers).detach().numpy()
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)

    def test_local_energy_recomputes_full_graph_for_every_configuration(self):
        positions = jnp.asarray(self.positions[:1].numpy())
        layers = jnp.asarray(self.layers[:1].numpy())
        with patch.object(jnb, "_encoded_graph", wraps=jnb._encoded_graph) as graph:
            jnb.local_energy(
                self.parameters, positions, layers, self.constants, self.spec
            )

        # base X, the full batches of all X+h and X-h configurations, and the
        # full batch of all one-particle layer-flipped configurations.
        self.assertEqual(graph.call_count, 4)
        graph_positions = [np.asarray(call.args[1]) for call in graph.call_args_list]
        graph_layers = [np.asarray(call.args[2]) for call in graph.call_args_list]
        derivatives = 2 * self.spec.n_particles
        expected_shifted_shape = (
            derivatives, self.spec.n_particles, 2
        )
        self.assertEqual(graph_positions[0].shape, (1, self.spec.n_particles, 2))
        self.assertEqual(graph_positions[1].shape, expected_shifted_shape)
        self.assertEqual(graph_positions[2].shape, expected_shifted_shape)
        self.assertEqual(
            graph_positions[3].shape,
            (self.spec.n_particles, self.spec.n_particles, 2),
        )
        np.testing.assert_allclose(graph_positions[0], np.asarray(positions))
        expected_plus = np.remainder(
            np.asarray(positions)[:, None]
            + np.asarray(self.constants["coordinate_deltas"])[None],
            1.0,
        ).reshape(expected_shifted_shape)
        expected_minus = np.remainder(
            np.asarray(positions)[:, None]
            - np.asarray(self.constants["coordinate_deltas"])[None],
            1.0,
        ).reshape(expected_shifted_shape)
        np.testing.assert_allclose(graph_positions[1], expected_plus)
        np.testing.assert_allclose(graph_positions[2], expected_minus)
        np.testing.assert_allclose(
            graph_positions[3],
            np.broadcast_to(
                np.asarray(positions)[:, None],
                (1, self.spec.n_particles, self.spec.n_particles, 2),
            ).reshape(self.spec.n_particles, self.spec.n_particles, 2),
        )
        np.testing.assert_array_equal(
            graph_layers[1],
            np.broadcast_to(
                np.asarray(layers)[:, None],
                (1, derivatives, self.spec.n_particles),
            ).reshape(derivatives, self.spec.n_particles),
        )
        np.testing.assert_array_equal(graph_layers[1], graph_layers[2])
        expected_flipped = np.broadcast_to(
            np.asarray(layers)[:, None],
            (1, self.spec.n_particles, self.spec.n_particles),
        ).copy()
        index = np.arange(self.spec.n_particles)
        expected_flipped[:, index, index] = 1 - expected_flipped[:, index, index]
        np.testing.assert_array_equal(
            graph_layers[3],
            expected_flipped.reshape(self.spec.n_particles, self.spec.n_particles),
        )

    def test_full_local_energy_matches_pytorch(self):
        hamiltonian = ContinuumTorusHamiltonian(
            self.continuum,
            3,
            3,
            dielectric=5.0,
            reciprocal_shell_count=8,
            finite_difference_fraction=1.5e-3,
        )
        expected = hamiltonian.local_energy(
            self.torch_wavefunction,
            self.positions[:1],
            self.layers[:1],
        ).numpy()
        actual = np.asarray(local_energy(
            self.parameters,
            jnp.asarray(self.positions[:1].numpy()),
            jnp.asarray(self.layers[:1].numpy()),
            self.constants,
            self.spec,
        ))
        np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-7)


    def test_natural_gradient_matches_explicit_score_jacobian(self):
        positions = jnp.asarray(self.positions.numpy())
        layers = jnp.asarray(self.layers.numpy())
        optimizer = MatrixFreeNaturalGradient(
            self.constants, self.spec, chunk_size=1, damping=0.03
        )
        flat = optimizer.bind(self.parameters)
        energies = jnp.asarray([1.2 + 0.3j, -0.4 - 0.2j])
        vector = jnp.linspace(-0.2, 0.3, len(flat))

        def scores(values):
            result = logpsi(
                optimizer.unravel(values), positions, layers,
                self.constants, self.spec,
            )
            return jnp.concatenate([result.real, result.imag])

        jacobian = jax.jacrev(scores)(flat)
        real, imag = jacobian[:2], jacobian[2:]
        centered_energy = energies - jnp.mean(energies)
        expected_force = 2.0 * jnp.mean(
            centered_energy.real[:, None] * real
            + centered_energy.imag[:, None] * imag,
            axis=0,
        )
        centered_real = real - jnp.mean(real, axis=0)
        centered_imag = imag - jnp.mean(imag, axis=0)
        expected_product = (
            centered_real.T @ (centered_real @ vector)
            + centered_imag.T @ (centered_imag @ vector)
        ) / 2.0 + 0.03 * vector
        actual_force = optimizer.force(flat, positions, layers, energies)
        actual_product = optimizer.product(flat, vector, positions, layers)
        np.testing.assert_allclose(
            np.asarray(actual_force), np.asarray(expected_force),
            rtol=2e-12, atol=2e-12,
        )
        np.testing.assert_allclose(
            np.asarray(actual_product), np.asarray(expected_product),
            rtol=2e-12, atol=2e-12,
        )


if __name__ == "__main__":
    unittest.main()
