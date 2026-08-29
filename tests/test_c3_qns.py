"""Regression tests for the internally C3-equivariant neural-Bloch ansatz."""

import unittest

import jax.numpy as jnp
import numpy as np
import torch

from src.jax_neural_bloch import (
    JaxNeuralBlochSpec,
    copy_to_torch,
    determinant_values,
    initialize,
    logpsi,
)
from src.neural_bloch import NeuralBlochConfig
from src.run_neural_bloch import build_wavefunction


class C3QNSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_default_dtype(torch.float64)

    @staticmethod
    def spec(particles: int, irrep: int) -> JaxNeuralBlochSpec:
        return JaxNeuralBlochSpec(
            n_particles=particles,
            width=32,
            message_passing_steps=2,
            determinants=1,
            orbital_hidden=32,
            c3_irrep=irrep,
            fixed_gamma_no_m=True,
            c3_qns=True,
        )

    def test_invalid_architecture_combinations_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "fixed_gamma_no_m"):
            initialize(
                JaxNeuralBlochSpec(
                    n_particles=3, c3_irrep=0, c3_qns=True
                )
            )
        with self.assertRaisesRegex(ValueError, "explicit c3_irrep"):
            initialize(
                JaxNeuralBlochSpec(
                    n_particles=3, fixed_gamma_no_m=True, c3_qns=True
                )
            )

    def test_jax_and_torch_match_and_have_the_requested_character(self):
        spec = self.spec(3, 0)
        parameters, constants = initialize(spec, seed=83)
        config = NeuralBlochConfig(
            width=spec.width,
            message_passing_steps=spec.message_passing_steps,
            determinants=1,
            orbital_hidden=spec.orbital_hidden,
            c3_irrep=spec.c3_irrep,
            fixed_gamma_no_m=True,
            c3_qns=True,
        )
        _, torch_state = build_wavefunction(3, config, seed=83)
        copy_to_torch(parameters, torch_state)

        rng = np.random.default_rng(130)
        positions = rng.random((8, 3, 2))
        layers = rng.integers(0, 2, (8, 3))
        jax_log = np.asarray(logpsi(
            parameters, jnp.asarray(positions), jnp.asarray(layers),
            constants, spec,
        ))
        torch_log = torch_state(
            torch.as_tensor(positions), torch.as_tensor(layers)
        ).detach().numpy()
        np.testing.assert_allclose(jax_log, torch_log, rtol=5e-13, atol=5e-13)

        base = np.asarray(determinant_values(
            parameters, jnp.asarray(positions), jnp.asarray(layers),
            constants, spec,
        ))
        rotated_positions = jnp.mod(
            jnp.einsum(
                "ac,bnc->bna",
                constants["c3_fractional_rotation"],
                jnp.asarray(positions),
            ),
            1.0,
        )
        rotated = np.asarray(determinant_values(
            parameters, rotated_positions, jnp.asarray(layers),
            constants, spec,
        ))
        cartesian = np.einsum(
            "ac,bnc->bna", np.asarray(constants["supercell_lattice"]), positions
        )
        shifts = np.asarray(constants["c3_layer_shifts"])[layers]
        gauge = np.exp(1j * np.sum(shifts * cartesian, axis=(1, 2)))
        np.testing.assert_allclose(
            gauge * rotated / base,
            np.ones(len(base), dtype=np.complex128),
            rtol=5e-4,
            atol=5e-4,
        )

    def test_two_thirds_character(self):
        spec = self.spec(6, 2)
        parameters, constants = initialize(spec, seed=89)
        rng = np.random.default_rng(131)
        positions = rng.random((6, 6, 2))
        layers = rng.integers(0, 2, (6, 6))
        base = np.asarray(determinant_values(
            parameters, jnp.asarray(positions), jnp.asarray(layers),
            constants, spec,
        ))
        rotated_positions = jnp.mod(
            jnp.einsum(
                "ac,bnc->bna",
                constants["c3_fractional_rotation"],
                jnp.asarray(positions),
            ),
            1.0,
        )
        rotated = np.asarray(determinant_values(
            parameters, rotated_positions, jnp.asarray(layers),
            constants, spec,
        ))
        cartesian = np.einsum(
            "ac,bnc->bna", np.asarray(constants["supercell_lattice"]), positions
        )
        shifts = np.asarray(constants["c3_layer_shifts"])[layers]
        gauge = np.exp(1j * np.sum(shifts * cartesian, axis=(1, 2)))
        target = np.exp(4j * np.pi / 3.0)
        np.testing.assert_allclose(
            gauge * rotated / base,
            target * np.ones(len(base), dtype=np.complex128),
            rtol=7e-4,
            atol=7e-4,
        )


if __name__ == "__main__":
    unittest.main()
