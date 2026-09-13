import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from src.mixture_sr import mixture_weights, WeightedNaturalGradient
from src.run_jax_neural_bloch import MatrixFreeNaturalGradient


class MixtureWeightTests(unittest.TestCase):
    def test_exact_discrete_mixture_and_amplitude_invariance(self):
        probabilities = np.array([[.6,.3,.1], [.1,.6,.3], [.3,.1,.6]])
        # Equal mixture is uniform. These multiplicities exactly integrate it.
        states = np.repeat(np.arange(3), 100)
        density = probabilities[:, states]
        weights, diagnostics = mixture_weights(np.log(density), [100]*3)
        np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-12)
        for m in range(3):
            measured = np.bincount(states, weights=weights[m], minlength=3)
            np.testing.assert_allclose(measured, probabilities[m], atol=1e-10)
        offsets = np.array([73., -54., 13.])
        shifted, info = mixture_weights(np.log(density)+offsets[:,None], [100]*3)
        np.testing.assert_allclose(weights, shifted, atol=1e-12)
        np.testing.assert_allclose(info['log_relative_norms'], offsets-offsets[0], atol=1e-9)
        self.assertGreater(diagnostics['overlap_spectral_gap'], .1)

    def test_identical_states_uniform_weights(self):
        logs = np.tile(np.linspace(-100, 100, 60), (3,1))
        weights, _ = mixture_weights(logs, [20]*3)
        np.testing.assert_allclose(weights, np.full((3,60), 1/60), atol=1e-12)

    def test_nonuniform_pool_recovers_unequal_normalizers(self):
        probabilities = np.array([[.6,.3,.1], [.2,.5,.3], [.1,.2,.7]])
        states = np.repeat(np.arange(3), [90,100,110])
        norms = np.array([2.,3.,7.])
        logs = np.log(norms[:,None]*probabilities[:,states])
        weights, info = mixture_weights(logs, [100]*3)
        np.testing.assert_allclose(info['log_relative_norms'], np.log(norms/norms[0]), atol=1e-9)
        for m in range(3):
            np.testing.assert_allclose(np.bincount(states,weights=weights[m]), probabilities[m], atol=1e-10)

    def test_disconnected_or_nonfinite_pool_fails(self):
        logs = np.full((3,30), -1000.)
        for m in range(3):
            logs[m,m*10:(m+1)*10] = 0
        with self.assertRaisesRegex(RuntimeError, 'disconnected'):
            mixture_weights(logs, [10]*3)
        logs[0,0] = np.nan
        with self.assertRaises(ValueError):
            mixture_weights(logs, [10]*3)


class WeightedSRTests(unittest.TestCase):
    def test_weighted_force_metric_and_uniform_limit(self):
        x = jnp.asarray([[1.,2.,3.],[2.,-.4,.5],[.1,1.,-.7],[.4,-.3,.8]])
        l = jnp.asarray([[.1,.5,-1.],[-.2,.7,1.],[1.,.2,.3],[-.4,1.,.2]])
        energies = jnp.asarray([1.+.3j, -.4-.2j, .7+.1j, 2.-.6j])
        parameters = {'v': jnp.asarray([.1,.2,.3])}
        vector = jnp.asarray([.5,-.2,.8])
        w = jnp.asarray([.1,.2,.3,.4])
        def toy_logpsi(p, positions, layers, constants, spec):
            return positions @ p['v'] + 1j*(layers @ p['v'])
        with patch('src.run_jax_neural_bloch.logpsi', toy_logpsi):
            optimizer = WeightedNaturalGradient(None, None, 2, .03)
            flat = optimizer.bind(parameters)
            optimizer.set_weights(w)
            mean_e = jnp.sum(w*energies)
            expected_force = 2*jnp.sum(w[:,None]*(
                (energies-mean_e).real[:,None]*x + (energies-mean_e).imag[:,None]*l), axis=0)
            dx, dl = x-jnp.sum(w[:,None]*x,axis=0), l-jnp.sum(w[:,None]*l,axis=0)
            expected_product = dx.T@(w*(dx@vector)) + dl.T@(w*(dl@vector)) + .03*vector
            np.testing.assert_allclose(optimizer.force(flat,x,l,energies),expected_force,atol=1e-12)
            np.testing.assert_allclose(optimizer.product(flat,vector,x,l),expected_product,atol=1e-12)
            optimizer.set_weights(np.ones(4))
            legacy = MatrixFreeNaturalGradient(None,None,2,.03)
            legacy.bind(parameters)
            np.testing.assert_allclose(optimizer.force(flat,x,l,energies),legacy.force(flat,x,l,energies),atol=1e-12)
            np.testing.assert_allclose(optimizer.product(flat,vector,x,l),legacy.product(flat,vector,x,l),atol=1e-12)
            optimizer.damping = .08
            np.testing.assert_allclose(optimizer.product(flat,vector,x,l),
                legacy.product(flat,vector,x,l)+.05*vector,atol=1e-12)


if __name__ == '__main__':
    unittest.main()
