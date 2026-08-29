"""Continuous-coordinate many-body neural Bloch wavefunction.

This module follows Eqs. (2)--(7) and Supplementary Eqs. (16)--(23) of
Luo, Zaklama and Fu, arXiv:2503.13585v3.  Particle coordinates and layer
pseudospins are the variational configurations; no band-projected Fock
configuration is used as neural input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .c3_symmetry import rotate_fock_state, single_particle_c3


@dataclass(frozen=True)
class NeuralBlochConfig:
    width: int = 32
    message_passing_steps: int = 2
    determinants: int = 1
    orbital_hidden: int = 32
    c3_irrep: int | None = None
    fixed_gamma_no_m: bool = False
    c3_qns: bool = False
    outer_c3_projector: bool = False
    correction_init_scale: float | None = None


def _lecun_normal(module: nn.Module) -> None:
    """JAX-compatible LeCunNormal initialization for real linear layers."""
    if isinstance(module, nn.Linear):
        fan_in = max(module.weight.shape[1], 1)
        target_std = fan_in ** -0.5
        # JAX variance_scaling corrects the variance of a [-2, 2] truncation.
        truncated_std = target_std / 0.87962566103423978
        nn.init.trunc_normal_(
            module.weight, mean=0.0, std=truncated_std,
            a=-2.0 * truncated_std, b=2.0 * truncated_std,
        )
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class _DenseGELU(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.linear(values))


class _MessagePassingStep(nn.Module):
    """One implementation of Supplementary Eqs. (17)--(21)."""

    def __init__(
        self, width: int, node_feature_dim: int, edge_feature_dim: int
    ) -> None:
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.tensor_filter = _DenseGELU(width * width, width)
        self.edge_gate = _DenseGELU(width, width)
        self.node_update = _DenseGELU(2 * width, width - node_feature_dim)
        self.edge_update = _DenseGELU(2 * width, width - edge_feature_dim)

    def forward(
        self, node: torch.Tensor, edge: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query(edge)
        key = self.key(edge)
        # Supplementary Eq. (18): sum over the intermediate particle n.
        tensor_path = torch.einsum("binp,bnjq->bijpq", query, key)
        message = self.tensor_filter(tensor_path.flatten(-2))
        message = message * self.edge_gate(edge)
        # Luo--Fu Appendix-B Eq. (19): the aggregation is the literal
        # sum_j m_ij, including the diagonal edge j=i.
        pooled = message.sum(dim=2)
        node_hidden = (
            node[..., self.node_feature_dim :]
            + self.node_update(torch.cat([pooled, node], -1))
        )
        edge_hidden = (
            edge[..., self.edge_feature_dim :]
            + self.edge_update(torch.cat([message, edge], -1))
        )
        # Eqs. (20)--(21): immutable physical features are reinserted.
        return (
            torch.cat([node[..., : self.node_feature_dim], node_hidden], -1),
            torch.cat([edge[..., : self.edge_feature_dim], edge_hidden], -1),
        )


class ManyBodyNeuralBloch(nn.Module):
    """Generalized Slater determinant with complex Bloch backflow.

    ``positions`` are fractional coordinates in the supercell, with shape
    ``(batch, particles, 2)``.  ``layers`` has values 0 or 1.  The output is
    the complex logarithm of the sum of generalized determinants.
    """

    def __init__(
        self,
        primitive_reciprocal: np.ndarray,
        supercell_lattice: np.ndarray,
        momentum_fractions: np.ndarray,
        momenta: np.ndarray,
        g_vectors: np.ndarray,
        bloch_coefficients: np.ndarray,
        one_body_energies: np.ndarray,
        n_particles: int,
        config: NeuralBlochConfig,
        seed: int,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.config = config
        self.reference_fast_path = False
        self.n_particles = int(n_particles)
        self.n_momenta = int(len(momenta))
        self.n_g = int(len(g_vectors))
        dtype = torch.get_default_dtype()
        self.register_buffer(
            "primitive_reciprocal",
            torch.as_tensor(primitive_reciprocal, dtype=dtype),
        )
        self.register_buffer(
            "supercell_lattice",
            torch.as_tensor(supercell_lattice, dtype=dtype),
        )
        self.register_buffer(
            "momentum_fractions",
            torch.as_tensor(momentum_fractions, dtype=dtype),
        )
        self.register_buffer("momenta", torch.as_tensor(momenta, dtype=dtype))
        self.register_buffer("g_vectors", torch.as_tensor(g_vectors, dtype=dtype))
        if config.c3_irrep is not None and config.c3_irrep not in (0, 1, 2):
            raise ValueError("c3_irrep must be None, 0, 1, or 2")
        if config.c3_qns and not config.fixed_gamma_no_m:
            raise ValueError("C3-QNS requires fixed_gamma_no_m=True")
        if config.c3_qns and config.c3_irrep is None:
            raise ValueError("C3-QNS requires an explicit c3_irrep")
        if config.c3_qns and config.determinants != 1:
            raise ValueError("C3-QNS uses one shared network and one three-minor orbit")
        if config.outer_c3_projector and config.c3_qns:
            raise ValueError("outer C3 projection and internal C3-QNS are mutually exclusive")
        if config.outer_c3_projector and config.c3_irrep is None:
            raise ValueError("outer C3 projection requires an explicit c3_irrep")
        if config.c3_irrep is not None and not (
            config.c3_qns or config.outer_c3_projector
        ):
            raise ValueError("c3_irrep requires c3_qns or outer_c3_projector")
        angle = -2.0 * np.pi / 3.0
        cartesian_rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        fractional_rotation = np.linalg.solve(
            np.asarray(supercell_lattice),
            cartesian_rotation @ np.asarray(supercell_lattice),
        )
        self.register_buffer(
            "c3_fractional_rotation",
            torch.as_tensor(fractional_rotation, dtype=dtype),
        )
        self.register_buffer(
            "c3_layer_shifts",
            torch.as_tensor(
                np.stack([primitive_reciprocal[0], -primitive_reciprocal[1]]),
                dtype=dtype,
            ),
        )
        inverse_cartesian_powers = np.stack([
            np.linalg.matrix_power(cartesian_rotation.T, power)
            for power in range(3)
        ])
        self.register_buffer(
            "c3_inverse_cartesian_powers",
            torch.as_tensor(inverse_cartesian_powers, dtype=dtype),
        )
        layer_shifts = np.stack([
            primitive_reciprocal[0], -primitive_reciprocal[1]
        ])
        backflow_gauge_vectors = np.stack([
            np.linalg.solve(cartesian_rotation.T - np.eye(2), shift)
            for shift in layer_shifts
        ])
        self.register_buffer(
            "c3_backflow_gauge_vectors",
            torch.as_tensor(backflow_gauge_vectors, dtype=dtype),
        )
        coefficients = np.asarray(bloch_coefficients, dtype=np.complex128)
        expected = (self.n_momenta, 2, self.n_g)
        if coefficients.shape != expected:
            raise ValueError(f"Bloch coefficients must have shape {expected}")
        self.register_buffer(
            "bloch_coefficients", torch.as_tensor(coefficients)
        )

        width = config.width
        self.node_feature_dim = 7
        self.edge_feature_dim = 2 * self.n_momenta + 1
        if width <= max(self.node_feature_dim, self.edge_feature_dim):
            raise ValueError(
                "width must exceed the fixed node and edge feature dimensions"
            )
        # Appendix B initializes only auxiliary h^0 randomly. Physical v^0/e^0
        # remain literal features and are reinserted at every propagation step.
        self.node_hidden_initial = nn.Parameter(
            torch.randn(width - self.node_feature_dim, dtype=dtype) / np.sqrt(width)
        )
        self.edge_hidden_initial = nn.Parameter(
            torch.randn(width - self.edge_feature_dim, dtype=dtype) / np.sqrt(width)
        )
        self.steps = nn.ModuleList(
            [
                _MessagePassingStep(
                    width, self.node_feature_dim, self.edge_feature_dim
                )
                for _ in range(config.message_passing_steps)
            ]
        )

        determinants = config.determinants
        self.backflow_heads = nn.ModuleList(
            [nn.Linear(width, 4, bias=False) for _ in range(determinants)]
        )
        self.orbital_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(width + 2, config.orbital_hidden),
                    nn.GELU(),
                    nn.Linear(config.orbital_hidden, 2),
                )
                for _ in range(determinants)
            ]
        )
        selected = self._reference_momenta(np.asarray(one_body_energies))
        self.register_buffer(
            "fixed_momentum_indices", torch.as_tensor(selected, dtype=torch.long)
        )
        orbit_indices = np.tile(np.asarray(selected, dtype=np.int64), (3, 1))
        orbit_weights = np.ones(3, dtype=np.complex128)
        rotated_momenta = np.stack([np.asarray(momenta)] * 3)
        if config.c3_qns:
            representation = single_particle_c3(1)
            inverse_map = np.argsort(representation.orbital_map)
            inverse_sewing = representation.sewing_phase[inverse_map]
            momentum_powers = [np.arange(self.n_momenta, dtype=np.int64)]
            for _ in range(2):
                momentum_powers.append(inverse_map[momentum_powers[-1]])
            rotated_momenta = np.asarray(momenta)[np.stack(momentum_powers)]
            state0 = sum(1 << int(index) for index in selected)
            state = state0
            chi = np.exp(2j * np.pi * int(config.c3_irrep) / 3.0)
            weight = 1.0 + 0.0j
            for power in range(3):
                occupied = [
                    index for index in range(self.n_momenta)
                    if (state >> index) & 1
                ]
                orbit_indices[power] = occupied
                orbit_weights[power] = weight
                state, sewing = rotate_fock_state(
                    state, inverse_map, inverse_sewing
                )
                weight = weight * chi / sewing
            if state != state0 or abs(weight - 1.0) > 1.0e-8:
                raise RuntimeError("inconsistent C3 sewing orbit closure")
        self.register_buffer(
            "c3_rotated_momenta", torch.as_tensor(rotated_momenta, dtype=dtype)
        )
        self.register_buffer(
            "c3_orbit_indices", torch.as_tensor(orbit_indices, dtype=torch.long)
        )
        self.register_buffer(
            "c3_orbit_weights", torch.as_tensor(orbit_weights)
        )
        if config.fixed_gamma_no_m:
            self.register_parameter("momentum_real", None)
            self.register_parameter("momentum_imag", None)
        else:
            self.momentum_real = nn.Parameter(
                torch.empty(determinants, self.n_particles, self.n_momenta)
            )
            self.momentum_imag = nn.Parameter(
                torch.empty(determinants, self.n_particles, self.n_momenta)
            )
        self.determinant_logits = nn.Parameter(torch.zeros(determinants, 2))

        self.apply(_lecun_normal)
        with torch.no_grad():
            if not config.fixed_gamma_no_m:
                # Complex LeCunNormal: split the target variance equally
                # between Re M and Im M, so E|M_{gamma k}|^2 = 1 / N_k.
                component_std = (2.0 * self.n_momenta) ** -0.5
                truncated_std = component_std / 0.87962566103423978
                for component in (self.momentum_real, self.momentum_imag):
                    nn.init.trunc_normal_(
                        component, mean=0.0, std=truncated_std,
                        a=-2.0 * truncated_std, b=2.0 * truncated_std,
                    )
            self.determinant_logits[:, 0] = -np.log(max(determinants, 1))
        # Optional small-correction ablation; production uses LeCunNormal.
        # Exact ED warm starts explicitly reset these heads to zero below.
        if config.correction_init_scale is not None:
            for layer in list(self.backflow_heads) + [
                mlp[-1] for mlp in self.orbital_mlps
            ]:
                nn.init.normal_(layer.weight, std=config.correction_init_scale)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _reference_momenta(self, energies: np.ndarray) -> list[int]:
        """Lowest one-body Slater determinant in the total-Gamma sector."""
        from itertools import combinations

        fractions = np.rint(3.0 * np.asarray(self.momentum_fractions.cpu())).astype(int)
        best: tuple[float, tuple[int, ...]] | None = None
        for occupied in combinations(range(self.n_momenta), self.n_particles):
            total = fractions[list(occupied)].sum(axis=0) % 3
            if np.any(total):
                continue
            candidate = (float(np.sum(energies[list(occupied)])), occupied)
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best is None:
            return list(np.argsort(energies)[: self.n_particles])
        return list(best[1])

    def initialize_from_fock_expansion(
        self, basis: np.ndarray, amplitudes: np.ndarray
    ) -> None:
        """Embed a one-band Fock expansion into the determinant sum exactly.

        This is a variational initialization, not a basis restriction: after
        initialization every determinant retains its independent complex
        backflow, orbital transformation and momentum-mixing parameters.
        """
        basis = np.asarray(basis, dtype=np.uint64)
        amplitudes = np.asarray(amplitudes, dtype=np.complex128)
        if len(basis) > self.config.determinants:
            raise ValueError("one determinant is required per warm-start configuration")
        occupied = []
        for state in basis:
            orbitals = [
                orbital for orbital in range(self.n_momenta)
                if (int(state) >> orbital) & 1
            ]
            if len(orbitals) != self.n_particles:
                raise ValueError("warm-start state has the wrong particle number")
            occupied.append(orbitals)
        with torch.no_grad():
            for head in self.backflow_heads:
                head.weight.zero_()
            for mlp in self.orbital_mlps:
                mlp[-1].weight.zero_()
                mlp[-1].bias.zero_()
                mlp[-1].bias[0] = 1.0
            self.momentum_real.zero_()
            self.momentum_imag.zero_()
            self.determinant_logits[:, 0].fill_(-30.0)
            self.determinant_logits[:, 1].zero_()
            for determinant, (orbitals, amplitude) in enumerate(
                zip(occupied, amplitudes)
            ):
                for row, orbital in enumerate(orbitals):
                    self.momentum_real[determinant, row, orbital] = 1.0
                self.determinant_logits[determinant, 0] = np.log(
                    max(abs(amplitude), 1.0e-30)
                )
                self.determinant_logits[determinant, 1] = np.angle(amplitude)
        self.reference_fast_path = True

    def cartesian(self, positions: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ac,bnc->bna", self.supercell_lattice, positions)

    def _raw_features(
        self, positions: torch.Tensor, layers: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cartesian = torch.einsum("ac,bnc->bna", self.supercell_lattice, positions)
        directions = torch.stack(
            [
                self.primitive_reciprocal[0],
                self.primitive_reciprocal[1],
                self.primitive_reciprocal[0] + self.primitive_reciprocal[1],
            ]
        )
        node_phase = torch.einsum("bna,ga->bng", cartesian, directions)
        layer_value = 2.0 * layers.to(cartesian.dtype) - 1.0
        node_raw = torch.cat(
            [torch.sin(node_phase), torch.cos(node_phase), layer_value[..., None]], -1
        )
        displacement = cartesian[:, :, None, :] - cartesian[:, None, :, :]
        edge_phase = torch.einsum("bija,ka->bijk", displacement, self.momenta)
        li = layer_value[:, :, None].expand(-1, -1, self.n_particles)
        edge_raw = torch.cat(
            [torch.sin(edge_phase), torch.cos(edge_phase), li[..., None]], -1
        )
        return cartesian, node_raw, edge_raw

    def encoded_graph(
        self, positions: torch.Tensor, layers: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cartesian, node_raw, edge_raw = self._raw_features(positions, layers)
        node_hidden = self.node_hidden_initial.reshape(1, 1, -1).expand(
            len(positions), self.n_particles, -1
        )
        edge_hidden = self.edge_hidden_initial.reshape(1, 1, 1, -1).expand(
            len(positions), self.n_particles, self.n_particles, -1
        )
        node = torch.cat([node_raw, node_hidden], -1)
        edge = torch.cat([edge_raw, edge_hidden], -1)
        for step in self.steps:
            node, edge = step(node, edge)
        return cartesian, node, edge

    def _bloch_values(
        self, positions: torch.Tensor, layers: torch.Tensor
    ) -> torch.Tensor:
        """Return phi[k,i] at possibly complex Cartesian coordinates."""
        plane_momenta = self.momenta[:, None, :] + self.g_vectors[None, :, :]
        phase = torch.einsum("bna,kga->bnkg", positions, plane_momenta.to(positions.dtype))
        exponent = torch.exp(1j * phase)
        coefficients = self.bloch_coefficients.permute(1, 0, 2)
        selected = coefficients[layers]
        # selected: batch, particle, momentum, g
        return torch.sum(selected * exponent, dim=-1).permute(0, 2, 1)

    def _orbital_factor(
        self, orbital_mlp: nn.Module, node: torch.Tensor, momenta: torch.Tensor
    ) -> torch.Tensor:
        node_k = node[:, :, None, :].expand(-1, -1, self.n_momenta, -1)
        k_feature = momenta[None, None].expand(
            len(node), self.n_particles, -1, -1
        )
        raw = orbital_mlp(torch.cat([node_k, k_feature], -1))
        return torch.complex(raw[..., 0], raw[..., 1])

    def _unprojected_determinant_values(
        self, positions: torch.Tensor, layers: torch.Tensor
    ) -> torch.Tensor:
        if self.reference_fast_path:
            cartesian = self.cartesian(positions).to(torch.complex128)
            bloch = self._bloch_values(cartesian, layers)
            determinant_values = []
            for index in range(self.config.determinants):
                mixing = torch.complex(
                    self.momentum_real[index], self.momentum_imag[index]
                )
                generalized = torch.einsum("rk,bki->bri", mixing, bloch)
                determinant_values.append(torch.linalg.det(generalized))
            determinants = torch.stack(determinant_values, -1)
            weights = torch.exp(
                torch.complex(
                    self.determinant_logits[:, 0], self.determinant_logits[:, 1]
                )
            )
            return torch.sum(determinants * weights, dim=-1)

        if self.config.c3_qns:
            graphs = []
            current = positions
            for _ in range(3):
                graphs.append(self.encoded_graph(current, layers))
                current = self.c3_rotate_positions(current)
            cartesian = graphs[0][0]
        else:
            cartesian, node, _ = self.encoded_graph(positions, layers)
        determinant_values = []
        for index, (backflow, orbital_mlp) in enumerate(
            zip(self.backflow_heads, self.orbital_mlps)
        ):
            if self.config.c3_qns:
                displacements = []
                orbitals = []
                for power, (_, rotated_node, _) in enumerate(graphs):
                    raw = backflow(rotated_node)
                    inverse = self.c3_inverse_cartesian_powers[power]
                    real = torch.einsum("ac,bnc->bna", inverse, raw[..., :2])
                    imag = torch.einsum("ac,bnc->bna", inverse, raw[..., 2:])
                    displacements.append(torch.complex(real, imag))
                    orbitals.append(self._orbital_factor(
                        orbital_mlp, rotated_node,
                        self.c3_rotated_momenta[power],
                    ))
                displacement = torch.stack(displacements).mean(0)
                orbital = torch.stack(orbitals).mean(0)
            else:
                displacement_raw = backflow(node)
                displacement = torch.complex(
                    displacement_raw[..., :2], displacement_raw[..., 2:]
                )
                orbital = self._orbital_factor(orbital_mlp, node, self.momenta)
            transformed = cartesian.to(torch.complex128) + displacement
            bloch = self._bloch_values(transformed, layers)
            q_matrix = bloch * orbital.permute(0, 2, 1)
            if self.config.c3_qns:
                gauge_vectors = self.c3_backflow_gauge_vectors[layers].to(
                    displacement.dtype
                )
                compensation = torch.exp(
                    1j * torch.einsum(
                        "bna,bna->bn", displacement, gauge_vectors
                    )
                )
                q_matrix = q_matrix * compensation[:, None, :]
                orbit_values = [
                    weight * torch.linalg.det(q_matrix[:, orbit, :])
                    for orbit, weight in zip(
                        self.c3_orbit_indices, self.c3_orbit_weights
                    )
                ]
                determinant_values.append(torch.stack(orbit_values).sum(0) / 3.0)
                continue
            if self.config.fixed_gamma_no_m:
                generalized = q_matrix[:, self.fixed_momentum_indices, :]
            else:
                mixing = torch.complex(
                    self.momentum_real[index], self.momentum_imag[index]
                )
                generalized = torch.einsum("rk,bki->bri", mixing, q_matrix)
            determinant_values.append(torch.linalg.det(generalized))
        determinants = torch.stack(determinant_values, -1)
        weights = torch.exp(
            torch.complex(self.determinant_logits[:, 0], self.determinant_logits[:, 1])
        )
        return torch.sum(determinants * weights, dim=-1)

    def c3_rotate_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Rotate a configuration by -120 degrees in fractional torus coordinates."""
        rotated = torch.einsum(
            "ac,bnc->bna", self.c3_fractional_rotation, positions
        )
        return torch.remainder(rotated, 1.0)

    def c3_gauge_phase(
        self, positions: torch.Tensor, layers: torch.Tensor
    ) -> torch.Tensor:
        """Layer-gauge factor in the continuum plane-wave C3 convention."""
        cartesian = self.cartesian(positions)
        shifts = self.c3_layer_shifts[layers]
        phase = torch.sum(shifts * cartesian, dim=(1, 2))
        return torch.exp(1j * phase)

    def determinant_values(
        self, positions: torch.Tensor, layers: torch.Tensor
    ) -> torch.Tensor:
        """Return the raw ansatz or its exact C3-character projection."""
        if self.config.c3_qns or not self.config.outer_c3_projector:
            return self._unprojected_determinant_values(positions, layers)
        root = torch.exp(
            torch.as_tensor(
                -2j * np.pi * self.config.c3_irrep / 3.0,
                dtype=torch.complex128,
                device=positions.device,
            )
        )
        current = positions
        gauge = torch.ones(
            len(positions), dtype=torch.complex128, device=positions.device
        )
        rotated_positions = []
        coefficients = []
        for power in range(3):
            rotated_positions.append(current)
            coefficients.append(root**power * gauge)
            gauge = gauge * self.c3_gauge_phase(current, layers)
            current = self.c3_rotate_positions(current)
        raw = self._unprojected_determinant_values(
            torch.cat(rotated_positions, dim=0), layers.repeat(3, 1)
        ).reshape(3, len(positions))
        return torch.sum(torch.stack(coefficients) * raw, dim=0) / 3.0

    def forward(self, positions: torch.Tensor, layers: torch.Tensor) -> torch.Tensor:
        if positions.ndim != 3 or positions.shape[1:] != (self.n_particles, 2):
            raise ValueError("positions must have shape (batch, particles, 2)")
        if layers.shape != positions.shape[:2]:
            raise ValueError("layers must have shape (batch, particles)")
        values = self.determinant_values(positions, layers)
        tiny = torch.finfo(values.real.dtype).tiny
        return torch.log(torch.abs(values).clamp_min(tiny)) + 1j * torch.angle(values)
