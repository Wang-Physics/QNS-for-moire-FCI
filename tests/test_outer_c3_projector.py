"""Regression tests for complete-wavefunction outer C3 projection."""

from dataclasses import replace
import unittest

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import torch

from src.continuum_vmc import ContinuumTorusHamiltonian
from src.jax_neural_bloch import (
    JaxNeuralBlochSpec,
    _unprojected_determinant_values,
    c3_orbit_positions_and_gauge,
    c3_project_from_orbit,
    copy_to_torch,
    determinant_values,
    initialize,
    local_energy,
    logpsi,
)
from src.neural_bloch import NeuralBlochConfig
from src.run_jax_neural_bloch import JaxMetropolis, MatrixFreeNaturalGradient
from src.run_neural_bloch import build_wavefunction


class OuterC3ProjectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_default_dtype(torch.float64)
        cls.raw_spec = JaxNeuralBlochSpec(
            n_particles=2,
            width=24,
            message_passing_steps=1,
            determinants=1,
            orbital_hidden=24,
            fixed_gamma_no_m=True,
        )
        cls.parameters, cls.constants = initialize(cls.raw_spec, seed=149)
        rng = np.random.default_rng(151)
        cls.positions = jnp.asarray(rng.random((3, 2, 2)))
        cls.layers = jnp.asarray(rng.integers(0, 2, (3, 2)), dtype=jnp.int32)

    @classmethod
    def projected_spec(cls, irrep):
        return replace(
            cls.raw_spec, c3_irrep=irrep, outer_c3_projector=True
        )

    def apply_projector_to_values(self, evaluator, irrep):
        rotated, gauge = c3_orbit_positions_and_gauge(
            self.positions, self.layers, self.constants
        )
        values = evaluator(
            rotated.reshape(3 * len(self.positions), 2, 2),
            jnp.tile(self.layers, (3, 1)),
        ).reshape(3, len(self.positions))
        return c3_project_from_orbit(values, gauge, irrep)

    def test_projected_character_for_all_three_sectors(self):
        rotated, gauge = c3_orbit_positions_and_gauge(
            self.positions, self.layers, self.constants
        )
        for irrep in range(3):
            spec = self.projected_spec(irrep)
            base = determinant_values(
                self.parameters, self.positions, self.layers,
                self.constants, spec,
            )
            moved = determinant_values(
                self.parameters, rotated[1], self.layers,
                self.constants, spec,
            )
            ratio = gauge[1] * moved / base
            target = np.exp(2j * np.pi * irrep / 3.0)
            np.testing.assert_allclose(
                np.asarray(ratio), target * np.ones(len(base)),
                rtol=2e-10, atol=2e-10,
            )

    def test_projector_idempotence_and_orthogonality(self):
        raw = _unprojected_determinant_values(
            self.parameters, self.positions, self.layers,
            self.constants, self.raw_spec,
        )
        scale = max(float(jnp.linalg.norm(raw)), 1.0e-14)
        for sector_n in range(3):
            spec_n = self.projected_spec(sector_n)
            first = determinant_values(
                self.parameters, self.positions, self.layers,
                self.constants, spec_n,
            )
            evaluator = lambda x, l, spec=spec_n: determinant_values(
                self.parameters, x, l, self.constants, spec
            )
            for sector_m in range(3):
                second = self.apply_projector_to_values(evaluator, sector_m)
                if sector_m == sector_n:
                    error = jnp.linalg.norm(second - first) / max(
                        float(jnp.linalg.norm(first)), 1.0e-14
                    )
                else:
                    error = jnp.linalg.norm(second) / scale
                self.assertLess(float(error), 3.0e-10)

    def test_all_sector_projection_norms_are_nonzero(self):
        raw = _unprojected_determinant_values(
            self.parameters, self.positions, self.layers,
            self.constants, self.raw_spec,
        )
        raw_norm = float(jnp.mean(jnp.abs(raw) ** 2))
        self.assertGreater(raw_norm, 0.0)
        for irrep in range(3):
            projected = determinant_values(
                self.parameters, self.positions, self.layers,
                self.constants, self.projected_spec(irrep),
            )
            relative_norm = float(jnp.mean(jnp.abs(projected) ** 2) / raw_norm)
            self.assertGreater(relative_norm, 1.0e-10)

    def test_local_energy_metropolis_and_sr_use_projected_logpsi(self):
        spec = self.projected_spec(1)
        continuum, torch_state = build_wavefunction(
            2,
            NeuralBlochConfig(
                width=24, message_passing_steps=1, determinants=1,
                orbital_hidden=24, fixed_gamma_no_m=True,
                c3_irrep=1, outer_c3_projector=True,
            ),
            seed=149,
        )
        copy_to_torch(self.parameters, torch_state)
        hamiltonian = ContinuumTorusHamiltonian(
            continuum, 3, 3, dielectric=5.0,
            reciprocal_shell_count=8, finite_difference_fraction=1.5e-3,
        )
        expected_energy = hamiltonian.local_energy(
            torch_state,
            torch.as_tensor(np.asarray(self.positions[:1]).copy()),
            torch.as_tensor(np.asarray(self.layers[:1]).copy()),
        ).detach().numpy()
        actual_energy = np.asarray(local_energy(
            self.parameters, self.positions[:1], self.layers[:1],
            self.constants, spec,
        ))
        np.testing.assert_allclose(
            actual_energy, expected_energy, rtol=3e-9, atol=3e-7
        )

        sampler = JaxMetropolis(
            self.parameters, self.constants, spec, walkers=2,
            proposal_scale=0.06, seed=157, batch_size=2,
        )
        candidate = sampler.positions.at[:, 0, 0].set(
            jnp.remainder(sampler.positions[:, 0, 0] + 0.017, 1.0)
        )
        actual_ratio = jnp.exp(
            sampler.evaluate(self.parameters, candidate, sampler.layers)
            - sampler.values
        )
        expected_ratio = jnp.exp(
            logpsi(self.parameters, candidate, sampler.layers, self.constants, spec)
            - logpsi(
                self.parameters, sampler.positions, sampler.layers,
                self.constants, spec,
            )
        )
        np.testing.assert_allclose(
            np.asarray(actual_ratio), np.asarray(expected_ratio),
            rtol=2e-12, atol=2e-12,
        )

        optimizer = MatrixFreeNaturalGradient(
            self.constants, spec, chunk_size=1, damping=0.03
        )
        flat = optimizer.bind(self.parameters)
        positions = self.positions[:2]
        layers = self.layers[:2]
        energies = jnp.asarray([0.7 + 0.2j, -0.3 - 0.1j])
        vector = jnp.linspace(-0.1, 0.15, len(flat))

        def scores(values):
            result = logpsi(
                optimizer.unravel(values), positions, layers,
                self.constants, spec,
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
        np.testing.assert_allclose(
            np.asarray(optimizer.force(flat, positions, layers, energies)),
            np.asarray(expected_force), rtol=3e-11, atol=3e-11,
        )
        np.testing.assert_allclose(
            np.asarray(optimizer.product(flat, vector, positions, layers)),
            np.asarray(expected_product), rtol=3e-11, atol=3e-11,
        )

    def test_disabling_projector_exactly_restores_no_m(self):
        raw = _unprojected_determinant_values(
            self.parameters, self.positions, self.layers,
            self.constants, self.raw_spec,
        )
        disabled = determinant_values(
            self.parameters, self.positions, self.layers,
            self.constants, replace(
                self.raw_spec, outer_c3_projector=False, c3_irrep=None
            ),
        )
        np.testing.assert_array_equal(np.asarray(disabled), np.asarray(raw))


if __name__ == '__main__':
    unittest.main()
