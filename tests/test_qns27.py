import unittest
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from src.qns27 import cluster, inputs, gamma_indices
from src.jax_neural_bloch import (
    JaxNeuralBlochSpec,
    complete_minor_sum,
    c3_orbit_positions_and_gauge,
    finite_translation_representatives,
    initialize,
    logpsi,
)
from src.qns27_diagnostics import (
    _coordinate_observables,
    _first_bz_momenta,
    _plane_wave_momentum_occupation,
)


class QNS27Checks(unittest.TestCase):
    def test_log_domain_full_m(self):
        from src.jax_neural_bloch import JaxNeuralBlochSpec, initialize, determinant_values, logpsi
        spec=JaxNeuralBlochSpec(cells=27,n_particles=9,width=64,orbital_hidden=64)
        p,c=initialize(spec,3184); rng=np.random.default_rng(11)
        x=jnp.asarray(rng.random((2,9,2))); l=jnp.asarray(rng.integers(0,2,(2,9)))
        raw=lambda x:determinant_values(p,x,l,c,spec)
        logs=lambda x:logpsi(p,x,l,c,spec)
        np.testing.assert_allclose(jnp.exp(logs(x)),raw(x),rtol=1e-10)
        np.testing.assert_allclose(jax.grad(lambda x:jnp.log(jnp.abs(raw(x))).sum())(x),
                                   jax.grad(lambda x:logs(x).real.sum())(x),rtol=1e-8,atol=1e-8)

    def test_geometry(self):
        model,data=inputs(); c=cluster()
        np.testing.assert_allclose(data['momenta']@data['supercell_lattice']/(2*np.pi),
                                   np.rint(data['momenta']@data['supercell_lattice']/(2*np.pi)),atol=1e-12)
        for n in (9,18):
            ids=gamma_indices(n)
            self.assertEqual(ids.shape,(173613,n))
            self.assertTrue(np.all(c.numerators[ids].sum(1)%27==0))
        angle=-2*np.pi/3; r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        fractional=np.linalg.solve(data['supercell_lattice'],r@data['supercell_lattice'])
        np.testing.assert_allclose(fractional,np.rint(fractional),atol=1e-12)
        # Same finite Coulomb mode set and normalization as c6_ed.
        self.assertEqual(len(data['q_vectors']),27*len(model.g_vectors)-1)

    def test_complete_translation_character_table(self):
        _, data = inputs()
        c = cluster()
        translations = finite_translation_representatives(c.matrix)
        characters = np.exp(
            2j * np.pi * translations @ data['momentum_fractions'].T
        )
        self.assertEqual(translations.shape, (27, 2))
        np.testing.assert_allclose(
            characters.conj().T @ characters,
            27.0 * np.eye(27),
            rtol=0.0,
            atol=2.0e-10,
        )

    def test_direct_plane_wave_fourier_estimator(self):
        _, data = inputs()
        c = cluster()
        translations = finite_translation_representatives(cluster().matrix)
        fractional = (np.linalg.inv(c.matrix) @ translations.T).T
        cartesian = np.einsum('ac,tc->ta', data['supercell_lattice'], fractional)
        momenta = _first_bz_momenta()
        characters = np.exp(1j * cartesian @ momenta.T)
        target = 17
        # Exact quadrature over the 27 primitive translations for a single
        # G=0 plane wave. The second draw has the wrong layer, so the factor
        # two in the estimator restores a unit occupation at the target only.
        selected_position = np.zeros((2, 27, 2))
        proposal_position = np.broadcast_to(fractional, (2, 27, 2)).copy()
        selected_layer = np.zeros((2, 27), dtype=int)
        proposal_layer = np.stack([np.zeros(27, dtype=int), np.ones(27, dtype=int)])
        ratios = np.broadcast_to(characters[:, target].conj(), (2, 27)).copy()
        occupation, _, imaginary = _plane_wave_momentum_occupation(
            selected_position, selected_layer, proposal_position, proposal_layer,
            ratios, 1, 1,
        )
        expected = np.zeros(27)
        expected[target] = 1.0
        np.testing.assert_allclose(occupation, expected, rtol=0.0, atol=2.0e-12)
        self.assertLess(imaginary, 2.0e-12)

    def test_full_structure_factor_uses_plotted_first_bz_q(self):
        _, data = inputs()
        rng = np.random.default_rng(8)
        positions = rng.random((16, 9, 2))
        result = _coordinate_observables(positions, 4)
        expected = _first_bz_momenta()
        np.testing.assert_allclose(result['structure_q_vectors'], expected, atol=1.0e-14)
        published = np.asarray(json.loads(
            (Path(__file__).resolve().parents[1] / 'result/data/fig5_27cell_observables.json')
            .read_text()
        )['k_points_hex'])
        np.testing.assert_allclose(expected, published, rtol=0.0, atol=1.0e-14)
        old = cluster().centered_fractions @ data['primitive_reciprocal']
        self.assertEqual(np.count_nonzero(np.linalg.norm(old - expected, axis=1) > 1e-10), 5)

    def test_v4_gamma_projection_is_translation_invariant(self):
        spec = JaxNeuralBlochSpec(
            cells=27, n_particles=9, width=64, orbital_hidden=64,
            translation_projected_rank=1,
        )
        parameters, constants = initialize(spec, 2718)
        rng = np.random.default_rng(314)
        positions = jnp.asarray(rng.random((1, 9, 2)))
        layers = jnp.asarray(rng.integers(0, 2, (1, 9)))
        evaluate = jax.jit(lambda x: logpsi(
            parameters, x, layers, constants, spec
        ))
        base = np.asarray(evaluate(positions))
        matrix = cluster().matrix
        for primitive in np.eye(2, dtype=np.int64):
            shift = np.linalg.solve(matrix, primitive)
            moved = np.asarray(evaluate(jnp.remainder(positions + shift, 1.0)))
            np.testing.assert_allclose(
                np.exp(moved - base), np.ones_like(base),
                rtol=2.0e-9, atol=2.0e-9,
            )

    def test_v4_outer_projection_retains_gamma_and_c3_character(self):
        irrep = 2
        spec = JaxNeuralBlochSpec(
            cells=27, n_particles=9, width=64, orbital_hidden=64,
            translation_projected_rank=1, outer_c3_projector=True,
            c3_irrep=irrep,
        )
        parameters, constants = initialize(spec, 1618)
        rng = np.random.default_rng(271)
        positions = jnp.asarray(rng.random((1, 9, 2)))
        layers = jnp.asarray(rng.integers(0, 2, (1, 9)))
        evaluate = jax.jit(lambda x: logpsi(
            parameters, x, layers, constants, spec
        ))
        base = np.asarray(evaluate(positions))
        matrix = cluster().matrix
        for primitive in np.eye(2, dtype=np.int64):
            shift = np.linalg.solve(matrix, primitive)
            moved = np.asarray(evaluate(jnp.remainder(positions + shift, 1.0)))
            np.testing.assert_allclose(
                np.exp(moved - base), np.ones_like(base),
                rtol=3.0e-9, atol=3.0e-9,
            )
        rotated, gauge = c3_orbit_positions_and_gauge(
            positions, layers, constants
        )
        moved = np.asarray(evaluate(rotated[1]))
        physical_ratio = np.asarray(gauge[1]) * np.exp(moved - base)
        target = np.exp(2j * np.pi * irrep / 3.0)
        np.testing.assert_allclose(
            physical_ratio, target * np.ones_like(base),
            rtol=3.0e-9, atol=3.0e-9,
        )

    def test_complete_sum_and_derivatives(self):
        from itertools import combinations
        rng=np.random.default_rng(43)
        q=jnp.asarray(rng.normal(size=(2,7,3))+1j*rng.normal(size=(2,7,3)))
        ix=jnp.asarray(list(combinations(range(7),3)))
        w=jnp.asarray(rng.normal(size=len(ix))+1j*rng.normal(size=len(ix)))
        direct=lambda q,w:jnp.einsum('bm,m->b',jnp.linalg.det(q[:,ix,:]),w)
        streamed=lambda q,w:complete_minor_sum(q,ix,w,8)
        np.testing.assert_allclose(streamed(q,w),direct(q,w),rtol=1e-12)
        a=jax.grad(lambda q,w:jnp.abs(direct(q,w)).sum(),argnums=(0,1))(q,w)
        b=jax.grad(lambda q,w:jnp.abs(streamed(q,w)).sum(),argnums=(0,1))(q,w)
        for aa,bb in zip(a,b): np.testing.assert_allclose(aa,bb,rtol=1e-11,atol=1e-11)
        v=jnp.ones_like(q)
        np.testing.assert_allclose(jax.jvp(lambda q:direct(q,w),(q,),(v,))[1],
                                   jax.jvp(lambda q:streamed(q,w),(q,),(v,))[1],rtol=1e-11)
