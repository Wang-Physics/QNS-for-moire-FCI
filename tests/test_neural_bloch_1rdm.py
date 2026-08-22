import inspect
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.neural_bloch import NeuralBlochConfig
from src.neural_bloch_diagnostics import one_body_band_density_matrix
from src.run_neural_bloch import build_wavefunction


def synthetic_bare_bloch_values(positions, layers, n_bands, continuum):
    """Nine orthonormal bare torus orbitals; no neural/backflow phase."""
    if n_bands != 1:
        raise ValueError("synthetic audit uses nine momenta and one band")
    modes = torch.arange(9, dtype=positions.dtype, device=positions.device)
    phase = 2.0 * torch.pi * positions[..., :1] * modes
    return torch.exp(1j * phase) / np.sqrt(2.0)


class BareFixedMSlater(torch.nn.Module):
    """delta-r=0, J=1, fixed-M, one-determinant reference state."""

    def __init__(self, mixing):
        super().__init__()
        self.register_buffer("mixing", torch.as_tensor(mixing, dtype=torch.complex128))
        self.n_particles = int(self.mixing.shape[0])

    def determinant_values(self, positions, layers):
        bare = synthetic_bare_bloch_values(positions, layers, 1, None)
        generalized = torch.einsum("ra,bia->bri", self.mixing, bare)
        return torch.linalg.det(generalized)

    def forward(self, positions, layers):
        value = self.determinant_values(positions, layers)
        tiny = torch.finfo(value.real.dtype).tiny
        return torch.log(torch.abs(value).clamp_min(tiny)) + 1j * torch.angle(value)


def sample_fixed_m_slater(wavefunction, walkers=4096, burn_sweeps=100, seed=91):
    generator = torch.Generator().manual_seed(seed)
    positions = torch.rand(walkers, wavefunction.n_particles, 2, generator=generator)
    layers = torch.randint(0, 2, (walkers, wavefunction.n_particles), generator=generator)
    with torch.no_grad():
        logpsi = wavefunction(positions, layers)
        rows = torch.arange(walkers)
        for _ in range(burn_sweeps):
            particle = torch.randint(wavefunction.n_particles, (walkers,), generator=generator)
            proposed_positions = positions.clone()
            proposed_layers = layers.clone()
            proposed_positions[rows, particle] = torch.rand(walkers, 2, generator=generator)
            proposed_layers[rows, particle] = torch.randint(0, 2, (walkers,), generator=generator)
            proposed_logpsi = wavefunction(proposed_positions, proposed_layers)
            probability = torch.exp(2.0 * (proposed_logpsi.real - logpsi.real)).clamp(max=1.0)
            accept = torch.rand(walkers, generator=generator) < probability
            positions[accept] = proposed_positions[accept]
            layers[accept] = proposed_layers[accept]
            logpsi[accept] = proposed_logpsi[accept]
    return positions, layers


class NeuralBlochOneRdmAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_default_dtype(torch.float64)
        mixing = np.zeros((2, 9), dtype=np.complex128)
        mixing[0, 0] = 1.0 / np.sqrt(2.0)
        mixing[0, 1] = 1j / np.sqrt(2.0)
        mixing[1, 2] = 1.0 / np.sqrt(2.0)
        mixing[1, 3] = np.exp(0.37j) / np.sqrt(2.0)
        cls.mixing = mixing
        cls.wavefunction = BareFixedMSlater(mixing)
        cls.positions, cls.layers = sample_fixed_m_slater(cls.wavefunction)
        with patch(
            "src.neural_bloch_diagnostics.bloch_orbital_values",
            side_effect=synthetic_bare_bloch_values,
        ):
            cls.rho, cls.weight_sem, cls.occupations, _, _, cls.trace_sem = (
                one_body_band_density_matrix(
                    cls.wavefunction, None, cls.positions, cls.layers,
                    n_bands=1, auxiliary_draws=32, batch_size=256, seed=1701,
                )
            )
        cls.exact = mixing.T @ mixing.conj()

    def test_fixed_m_bare_slater_occupation_and_full_trace(self):
        error = np.linalg.norm(self.rho - self.exact)
        self.assertLess(error, 0.16)
        self.assertAlmostEqual(np.trace(self.rho).real, 2.0, delta=0.08)
        self.assertAlmostEqual(self.occupations[:2].sum(), 2.0, delta=0.08)
        self.assertLess(np.abs(self.occupations[2:]).max(), 0.08)

    def test_hermiticity_and_psd_within_sampling_error(self):
        self.assertLess(np.linalg.norm(self.rho - self.rho.conj().T), 1.0e-12)
        self.assertGreater(self.occupations.min(), -0.08)

    def test_projector_is_fixed_bare_bloch_without_backflow_phase(self):
        from src.neural_bloch_overlap import bloch_orbital_values
        source = inspect.getsource(bloch_orbital_values)
        self.assertIn("cache[\"vectors\"]", source)
        self.assertNotIn("backflow", source.lower())
        self.assertNotIn("displacement", source.lower())

    def test_particle_replacement_recomputes_complete_ansatz(self):
        _, wavefunction = build_wavefunction(
            2, NeuralBlochConfig(width=32, message_passing_steps=2, determinants=1),
            seed=13,
        )
        with torch.no_grad():
            wavefunction.backflow_heads[0].weight.zero_()
            last = wavefunction.orbital_mlps[0][-1]
            last.weight.zero_()
            last.bias[:] = torch.tensor([1.0, 0.0])
            wavefunction.momentum_real.zero_()
            wavefunction.momentum_imag.zero_()
            wavefunction.momentum_real[0, 0, 0] = 1.0
            wavefunction.momentum_real[0, 1, 1] = 1.0
        counts = {"graph": 0, "bloch": 0, "determinant": 0, "backflow": 0, "J": 0}
        original_graph = wavefunction.encoded_graph
        original_bloch = wavefunction._bloch_values
        original_det = wavefunction._unprojected_determinant_values

        def graph(this, *args, **kwargs):
            counts["graph"] += 1
            return original_graph(*args, **kwargs)

        def bloch(this, *args, **kwargs):
            counts["bloch"] += 1
            return original_bloch(*args, **kwargs)

        def determinant(this, *args, **kwargs):
            counts["determinant"] += 1
            return original_det(*args, **kwargs)

        wavefunction.encoded_graph = types.MethodType(graph, wavefunction)
        wavefunction._bloch_values = types.MethodType(bloch, wavefunction)
        wavefunction._unprojected_determinant_values = types.MethodType(determinant, wavefunction)
        hooks = [
            wavefunction.backflow_heads[0].register_forward_hook(
                lambda *args: counts.__setitem__("backflow", counts["backflow"] + 1)
            ),
            wavefunction.orbital_mlps[0].register_forward_hook(
                lambda *args: counts.__setitem__("J", counts["J"] + 1)
            ),
        ]
        try:
            positions = torch.rand(4, 2, 2)
            layers = torch.randint(0, 2, (4, 2))
            with patch(
                "src.neural_bloch_diagnostics.bloch_orbital_values",
                side_effect=synthetic_bare_bloch_values,
            ):
                one_body_band_density_matrix(
                    wavefunction, None, positions, layers,
                    n_bands=1, auxiliary_draws=1, batch_size=4, seed=7,
                )
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(counts, {
            "graph": 2, "bloch": 2, "determinant": 2, "backflow": 2, "J": 2
        })


if __name__ == "__main__":
    unittest.main()
