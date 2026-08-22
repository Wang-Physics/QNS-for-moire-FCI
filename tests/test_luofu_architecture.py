import unittest

import torch

from src.neural_bloch import NeuralBlochConfig, _MessagePassingStep
from src.run_neural_bloch import build_wavefunction


class LuoFuArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_default_dtype(torch.float64)

    def test_full_m_uses_reproducible_complex_random_initialization(self):
        config = NeuralBlochConfig(
            width=24,
            message_passing_steps=1,
            determinants=4,
            orbital_hidden=24,
        )
        _, first = build_wavefunction(2, config, seed=101)
        _, repeated = build_wavefunction(2, config, seed=101)
        _, changed = build_wavefunction(2, config, seed=102)
        torch.testing.assert_close(first.momentum_real, repeated.momentum_real)
        torch.testing.assert_close(first.momentum_imag, repeated.momentum_imag)
        self.assertGreater(float(torch.linalg.norm(first.momentum_real).detach()), 0.0)
        self.assertGreater(float(torch.linalg.norm(first.momentum_imag).detach()), 0.0)
        self.assertFalse(torch.allclose(first.momentum_real, changed.momentum_real))
        selector_values = (first.momentum_real == 0) | (first.momentum_real == 1)
        self.assertFalse(bool(torch.all(selector_values)))

    def test_random_auxiliary_h0_is_shared_and_trainable(self):
        _, state = build_wavefunction(
            2,
            NeuralBlochConfig(
                width=24,
                message_passing_steps=1,
                determinants=1,
                orbital_hidden=24,
            ),
            seed=103,
        )
        named = dict(state.named_parameters())
        for name in ("node_hidden_initial", "edge_hidden_initial"):
            self.assertIn(name, named)
            self.assertTrue(named[name].requires_grad)
            self.assertGreater(float(torch.linalg.norm(named[name]).detach()), 0.0)
        self.assertEqual(state.node_hidden_initial.ndim, 1)
        self.assertEqual(state.edge_hidden_initial.ndim, 1)

    def test_luofu_eq19_includes_the_self_message(self):
        step = _MessagePassingStep(width=2, node_feature_dim=1, edge_feature_dim=1)
        with torch.no_grad():
            for parameter in step.parameters():
                parameter.zero_()
            step.query.weight.fill_(1.0)
            step.key.weight.fill_(1.0)
            step.tensor_filter.linear.weight.fill_(1.0)
            step.edge_gate.linear.weight.fill_(1.0)
            step.node_update.linear.weight[0, 0] = 1.0
        node = torch.zeros(1, 1, 2)
        edge = torch.ones(1, 1, 1, 2)
        updated, _ = step(node, edge)
        self.assertGreater(float(updated[0, 0, 1].detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
