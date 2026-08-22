"""JAX implementation of the continuum many-body neural-Bloch ansatz."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import torch

from .continuum_vmc import ContinuumTorusHamiltonian
from .neural_bloch import NeuralBlochConfig
from .run_neural_bloch import neural_bloch_inputs


@dataclass(frozen=True)
class JaxNeuralBlochSpec:
    n_particles: int = 6
    width: int = 32
    message_passing_steps: int = 2
    determinants: int = 1
    orbital_hidden: int = 32
    c3_irrep: int | None = None
    fixed_gamma_no_m: bool = False


def _linear(parameters, values):
    result = jnp.einsum("...i,io->...o", values, parameters["weight"])
    if "bias" in parameters:
        result = result + parameters["bias"]
    return result


def _gelu_linear(parameters, values):
    return jax.nn.gelu(_linear(parameters, values), approximate=False)


def _message_step(parameters, node, edge, node_feature_dim, edge_feature_dim):
    query = _linear(parameters["query"], edge)
    key = _linear(parameters["key"], edge)
    tensor_path = jnp.einsum("binp,bnjq->bijpq", query, key)
    message = _gelu_linear(
        parameters["tensor_filter"], tensor_path.reshape(*tensor_path.shape[:-2], -1)
    ) * _gelu_linear(parameters["edge_gate"], edge)
    # Luo--Fu Appendix-B Eq. (19) is the literal sum_j m_ij.  In particular,
    # the diagonal edge E_ii participates in the node aggregation.
    pooled = jnp.sum(message, axis=2)
    node_hidden = node[..., node_feature_dim:] + _gelu_linear(
        parameters["node_update"], jnp.concatenate([pooled, node], -1)
    )
    edge_hidden = edge[..., edge_feature_dim:] + _gelu_linear(
        parameters["edge_update"], jnp.concatenate([message, edge], -1)
    )
    return (
        jnp.concatenate([node[..., :node_feature_dim], node_hidden], -1),
        jnp.concatenate([edge[..., :edge_feature_dim], edge_hidden], -1),
    )


def _encoded_graph(parameters, positions, layers, constants):
    cartesian = jnp.einsum("ac,bnc->bna", constants["supercell_lattice"], positions)
    node_phase = jnp.einsum("bna,ga->bng", cartesian, constants["node_directions"])
    layer_value = 2.0 * layers.astype(cartesian.dtype) - 1.0
    node_raw = jnp.concatenate(
        [jnp.sin(node_phase), jnp.cos(node_phase), layer_value[..., None]], -1
    )
    displacement = cartesian[:, :, None, :] - cartesian[:, None, :, :]
    edge_phase = jnp.einsum("bija,ka->bijk", displacement, constants["momenta"])
    count = positions.shape[1]
    li = jnp.broadcast_to(layer_value[:, :, None], (positions.shape[0], count, count))
    edge_raw = jnp.concatenate(
        [jnp.sin(edge_phase), jnp.cos(edge_phase), li[..., None]], -1
    )
    node_hidden = jnp.broadcast_to(
        parameters["node_hidden_initial"][None, None],
        (positions.shape[0], count, len(parameters["node_hidden_initial"])),
    )
    edge_hidden = jnp.broadcast_to(
        parameters["edge_hidden_initial"][None, None, None],
        (positions.shape[0], count, count, len(parameters["edge_hidden_initial"])),
    )
    node = jnp.concatenate([node_raw, node_hidden], -1)
    edge = jnp.concatenate([edge_raw, edge_hidden], -1)
    node_feature_dim = node_raw.shape[-1]
    edge_feature_dim = edge_raw.shape[-1]
    for step in parameters["steps"]:
        node, edge = _message_step(
            step, node, edge, node_feature_dim, edge_feature_dim
        )
    return cartesian, node


def _bloch_values(cartesian, layers, constants):
    plane_momenta = constants["momenta"][:, None, :] + constants["g_vectors"][None]
    phase = jnp.einsum("bna,kga->bnkg", cartesian, plane_momenta)
    exponent = jnp.exp(1j * phase)
    selected = jnp.take(constants["bloch_coefficients"], layers, axis=1)
    selected = jnp.transpose(selected, (1, 2, 0, 3))
    return jnp.transpose(jnp.sum(selected * exponent, axis=-1), (0, 2, 1))


def _unprojected_determinant_values(parameters, positions, layers, constants, spec):
    cartesian, node = _encoded_graph(parameters, positions, layers, constants)
    values = []
    for index in range(spec.determinants):
        raw_shift = _linear(parameters["backflow_heads"][index], node)
        displacement = raw_shift[..., :2] + 1j * raw_shift[..., 2:]
        transformed = cartesian.astype(jnp.complex128) + displacement
        bloch = _bloch_values(transformed, layers, constants)
        node_k = jnp.broadcast_to(
            node[:, :, None, :],
            (len(node), spec.n_particles, len(constants["momenta"]), spec.width),
        )
        k_feature = jnp.broadcast_to(
            constants["momenta"][None, None],
            (len(node), spec.n_particles, len(constants["momenta"]), 2),
        )
        orbital_parameters = parameters["orbital_mlps"][index]
        raw = _linear(
            orbital_parameters[1],
            jax.nn.gelu(
                _linear(orbital_parameters[0], jnp.concatenate([node_k, k_feature], -1)),
                approximate=False,
            ),
        )
        orbital = raw[..., 0] + 1j * raw[..., 1]
        q_matrix = bloch * jnp.transpose(orbital, (0, 2, 1))
        if spec.fixed_gamma_no_m:
            generalized = q_matrix[:, constants["fixed_momentum_indices"], :]
        else:
            mixing = (
                parameters["momentum_real"][index]
                + 1j * parameters["momentum_imag"][index]
            )
            generalized = jnp.einsum("rk,bki->bri", mixing, q_matrix)
        values.append(jnp.linalg.det(generalized))
    return jnp.sum(jnp.stack(values, -1), axis=-1)


def _c3_rotate_positions(positions, constants):
    rotated = jnp.einsum("ac,bnc->bna", constants["c3_fractional_rotation"], positions)
    return jnp.mod(rotated, 1.0)


def _c3_gauge_phase(positions, layers, constants):
    cartesian = jnp.einsum("ac,bnc->bna", constants["supercell_lattice"], positions)
    shifts = constants["c3_layer_shifts"][layers]
    return jnp.exp(1j * jnp.sum(shifts * cartesian, axis=(1, 2)))


def determinant_values(parameters, positions, layers, constants, spec):
    """Raw ansatz or its exact C3-character projection."""
    if spec.c3_irrep is None:
        return _unprojected_determinant_values(
            parameters, positions, layers, constants, spec
        )
    root = jnp.exp(-2j * jnp.pi * spec.c3_irrep / 3.0)
    current = positions
    gauge = jnp.ones(len(positions), dtype=jnp.complex128)
    rotated_positions = []
    coefficients = []
    for power in range(3):
        rotated_positions.append(current)
        coefficients.append(root**power * gauge)
        gauge = gauge * _c3_gauge_phase(current, layers, constants)
        current = _c3_rotate_positions(current, constants)
    raw = _unprojected_determinant_values(
        parameters,
        jnp.concatenate(rotated_positions, axis=0),
        jnp.tile(layers, (3, 1)),
        constants,
        spec,
    ).reshape(3, len(positions))
    return jnp.sum(jnp.stack(coefficients) * raw, axis=0) / 3.0

def logpsi(parameters, positions, layers, constants, spec):
    values = determinant_values(parameters, positions, layers, constants, spec)
    return jnp.log(jnp.maximum(jnp.abs(values), jnp.finfo(jnp.float64).tiny)) + 1j * jnp.angle(values)


def local_energy_components(parameters, positions, layers, constants, spec):
    base = logpsi(parameters, positions, layers, constants, spec)
    cartesian = jnp.einsum("ac,bnc->bna", constants["supercell_lattice"], positions)
    layer_phase = jnp.where(layers == 0, -constants["moire_phase"], constants["moire_phase"])
    arguments = jnp.einsum("bna,ga->bng", cartesian, constants["potential_directions"])
    moire = -2.0 * constants["potential_meV"] * jnp.sum(
        jnp.cos(arguments + layer_phase[..., None]), axis=(1, 2)
    )
    displacement = cartesian[:, :, None, :] - cartesian[:, None, :, :]
    qphase = jnp.einsum("bija,qa->bijq", displacement, constants["q_vectors"])
    pair_potential = jnp.einsum(
        "q,bijq->bij", constants["q_coefficients"], jnp.cos(qphase)
    )
    coulomb = jnp.sum(pair_potential * constants["pair_upper"], axis=(1, 2))

    batch = positions.shape[0]
    derivatives = spec.n_particles * 2
    plus = jnp.remainder(positions[:, None] + constants["coordinate_deltas"][None], 1.0)
    minus = jnp.remainder(positions[:, None] - constants["coordinate_deltas"][None], 1.0)
    repeated_layers = jnp.broadcast_to(
        layers[:, None], (batch, derivatives, spec.n_particles)
    ).reshape(batch * derivatives, spec.n_particles)
    plus_log = logpsi(
        parameters, plus.reshape(batch * derivatives, spec.n_particles, 2),
        repeated_layers, constants, spec,
    ).reshape(batch, derivatives)
    minus_log = logpsi(
        parameters, minus.reshape(batch * derivatives, spec.n_particles, 2),
        repeated_layers, constants, spec,
    ).reshape(batch, derivatives)
    plus_ratio = jnp.exp(plus_log - base[:, None])
    minus_ratio = jnp.exp(minus_log - base[:, None])
    h = constants["finite_difference_nm"]
    first = (plus_ratio - minus_ratio) / (2.0 * h)
    second = (plus_ratio + minus_ratio - 2.0) / (h * h)
    kappa = jnp.take(constants["kappas"], layers, axis=0).reshape(batch, derivatives)
    kinetic = jnp.sum(
        -constants["alpha"] * second + 2j * constants["alpha"] * kappa * first,
        axis=1,
    ) + constants["alpha"] * jnp.sum(kappa * kappa, axis=1)

    index = jnp.arange(spec.n_particles)
    flipped = jnp.broadcast_to(
        layers[:, None, :], (batch, spec.n_particles, spec.n_particles)
    )
    flipped = flipped.at[:, index, index].set(1 - flipped[:, index, index])
    repeated_positions = jnp.broadcast_to(
        positions[:, None], (batch, spec.n_particles, spec.n_particles, 2)
    )
    flipped_log = logpsi(
        parameters,
        repeated_positions.reshape(batch * spec.n_particles, spec.n_particles, 2),
        flipped.reshape(batch * spec.n_particles, spec.n_particles),
        constants,
        spec,
    ).reshape(batch, spec.n_particles)
    ratio = jnp.exp(flipped_log - base[:, None])
    t01 = constants["tunneling_meV"] * (
        1.0
        + jnp.exp(1j * jnp.einsum("bna,a->bn", cartesian, constants["b2"]))
        + jnp.exp(1j * jnp.einsum("bna,a->bn", cartesian, constants["b12"]))
    )
    matrix_element = jnp.where(layers == 0, t01, jnp.conj(t01))
    tunneling = jnp.sum(matrix_element * ratio, axis=1)
    return kinetic, moire, tunneling, coulomb


def local_energy(parameters, positions, layers, constants, spec):
    """Complete local energy; components are exposed for fixed-state audits."""
    kinetic, moire, tunneling, coulomb = local_energy_components(
        parameters, positions, layers, constants, spec
    )
    return kinetic + moire + tunneling + coulomb


def _reference_momenta(energies, fractions, particles):
    integer_fraction = np.rint(3.0 * fractions).astype(int)
    candidates = []
    for occupied in combinations(range(len(energies)), particles):
        if np.any(integer_fraction[list(occupied)].sum(0) % 3):
            continue
        candidates.append((float(np.sum(energies[list(occupied)])), occupied))
    return list(min(candidates)[1]) if candidates else list(np.argsort(energies)[:particles])


def initialize(spec: JaxNeuralBlochSpec, seed: int = 83):
    if spec.c3_irrep is not None and spec.c3_irrep not in (0, 1, 2):
        raise ValueError("c3_irrep must be None, 0, 1, or 2")
    continuum, inputs = neural_bloch_inputs()
    torch_hamiltonian = ContinuumTorusHamiltonian(
        continuum, 3, 3, dielectric=5.0, reciprocal_shell_count=8,
        finite_difference_fraction=1.5e-3,
    )
    key = jax.random.PRNGKey(seed)
    initializer = jax.nn.initializers.lecun_normal()

    def dense(input_dim, output_dim, bias=True):
        nonlocal key
        key, subkey = jax.random.split(key)
        result = {"weight": initializer(subkey, (input_dim, output_dim), jnp.float64)}
        if bias:
            result["bias"] = jnp.zeros(output_dim, dtype=jnp.float64)
        return result

    width = spec.width
    node_feature_dim = 7
    edge_feature_dim = 2 * len(inputs["momenta"]) + 1
    if width <= max(node_feature_dim, edge_feature_dim):
        raise ValueError("width must exceed fixed node and edge feature dimensions")
    key, node_key, edge_key = jax.random.split(key, 3)
    parameters = {
        "node_hidden_initial": jax.random.normal(
            node_key, (width - node_feature_dim,), dtype=jnp.float64
        ) / jnp.sqrt(width),
        "edge_hidden_initial": jax.random.normal(
            edge_key, (width - edge_feature_dim,), dtype=jnp.float64
        ) / jnp.sqrt(width),
        "steps": [],
        "backflow_heads": [],
        "orbital_mlps": [],
    }
    for _ in range(spec.message_passing_steps):
        parameters["steps"].append({
            "query": dense(width, width, False),
            "key": dense(width, width, False),
            "tensor_filter": dense(width * width, width),
            "edge_gate": dense(width, width),
            "node_update": dense(2 * width, width - node_feature_dim),
            "edge_update": dense(2 * width, width - edge_feature_dim),
        })
    for _ in range(spec.determinants):
        parameters["backflow_heads"].append(dense(width, 4, False))
        parameters["orbital_mlps"].append([
            dense(width + 2, spec.orbital_hidden), dense(spec.orbital_hidden, 2)
        ])
    selected = _reference_momenta(
        np.asarray(inputs["one_body_energies"]),
        np.asarray(inputs["momentum_fractions"]), spec.n_particles,
    )
    if not spec.fixed_gamma_no_m:
        # M is complex LeCunNormal.  Each real component has half of the
        # LeCun variance so E|M_{gamma k}|^2 = 1 / N_k.  Luo--Fu specify
        # LeCunNormal globally but do not separately document M's initializer;
        # this random complex extension avoids selecting one minor at step 0.
        key, momentum_real_key, momentum_imag_key = jax.random.split(key, 3)
        flat_shape = (9, spec.determinants * spec.n_particles)
        momentum_real = initializer(
            momentum_real_key, flat_shape, jnp.float64
        ).T.reshape(spec.determinants, spec.n_particles, 9) / jnp.sqrt(2.0)
        momentum_imag = initializer(
            momentum_imag_key, flat_shape, jnp.float64
        ).T.reshape(spec.determinants, spec.n_particles, 9) / jnp.sqrt(2.0)
        parameters["momentum_real"] = momentum_real
        parameters["momentum_imag"] = momentum_imag
    b1, b2 = continuum.b1, continuum.b2
    angle = -2.0 * np.pi / 3.0
    cartesian_rotation = np.asarray([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    fractional_rotation = np.linalg.solve(
        np.asarray(inputs["supercell_lattice"]),
        cartesian_rotation @ np.asarray(inputs["supercell_lattice"]),
    )
    c3_layer_shifts = np.stack([b1, -b2])
    deltas = np.zeros((spec.n_particles * 2, spec.n_particles, 2))
    for particle in range(spec.n_particles):
        for axis in range(2):
            cartesian_delta = np.zeros(2)
            cartesian_delta[axis] = torch_hamiltonian.finite_difference_nm
            fractional = torch_hamiltonian.inverse_supercell @ cartesian_delta
            deltas[2 * particle + axis, particle] = fractional
    constants = {
        "supercell_lattice": jnp.asarray(inputs["supercell_lattice"]),
        "momentum_fractions": jnp.asarray(inputs["momentum_fractions"]),
        "momenta": jnp.asarray(inputs["momenta"]),
        "g_vectors": jnp.asarray(inputs["g_vectors"]),
        "bloch_coefficients": jnp.asarray(inputs["bloch_coefficients"]),
        "c3_fractional_rotation": jnp.asarray(fractional_rotation),
        "c3_layer_shifts": jnp.asarray(c3_layer_shifts),
        "fixed_momentum_indices": jnp.asarray(selected, dtype=jnp.int32),
        "node_directions": jnp.asarray(np.stack([b1, b2, b1 + b2])),
        "potential_directions": jnp.asarray(np.stack([b1, b2, -(b1 + b2)])),
        "moire_phase": jnp.asarray(np.deg2rad(continuum.params.phase_deg)),
        "potential_meV": jnp.asarray(continuum.params.potential_meV),
        "tunneling_meV": jnp.asarray(continuum.params.tunneling_meV),
        "b2": jnp.asarray(b2),
        "b12": jnp.asarray(b1 + b2),
        "alpha": jnp.asarray(continuum.alpha),
        "kappas": jnp.asarray(continuum.kappas),
        "q_vectors": jnp.asarray(torch_hamiltonian.q_vectors.numpy()),
        "q_coefficients": jnp.asarray(torch_hamiltonian.q_coefficients.numpy()),
        "pair_upper": jnp.asarray(np.triu(np.ones((spec.n_particles, spec.n_particles)), 1)),
        "coordinate_deltas": jnp.asarray(deltas),
        "finite_difference_nm": jnp.asarray(torch_hamiltonian.finite_difference_nm),
    }
    return parameters, constants


def from_torch(wavefunction):
    """Convert a PyTorch checkpoint exactly for JAX/PyTorch parity tests."""
    def dense(module):
        result = {"weight": jnp.asarray(module.weight.detach().numpy().T)}
        if module.bias is not None:
            result["bias"] = jnp.asarray(module.bias.detach().numpy())
        return result

    parameters = {
        "node_hidden_initial": jnp.asarray(
            wavefunction.node_hidden_initial.detach().numpy()
        ),
        "edge_hidden_initial": jnp.asarray(
            wavefunction.edge_hidden_initial.detach().numpy()
        ),
        "steps": [], "backflow_heads": [], "orbital_mlps": [],
    }
    for step in wavefunction.steps:
        parameters["steps"].append({
            "query": dense(step.query), "key": dense(step.key),
            "tensor_filter": dense(step.tensor_filter.linear),
            "edge_gate": dense(step.edge_gate.linear),
            "node_update": dense(step.node_update.linear),
            "edge_update": dense(step.edge_update.linear),
        })
    for head, mlp in zip(wavefunction.backflow_heads, wavefunction.orbital_mlps):
        parameters["backflow_heads"].append(dense(head))
        parameters["orbital_mlps"].append([dense(mlp[0]), dense(mlp[2])])
    if wavefunction.momentum_real is not None:
        parameters["momentum_real"] = jnp.asarray(
            wavefunction.momentum_real.detach().numpy()
        )
        parameters["momentum_imag"] = jnp.asarray(
            wavefunction.momentum_imag.detach().numpy()
        )
    return parameters


def copy_to_torch(parameters, wavefunction):
    """Copy a JAX parameter pytree into the parity-matched PyTorch module."""
    def dense(source, module):
        module.weight.copy_(torch.as_tensor(np.array(source["weight"], copy=True).T))
        if module.bias is not None:
            module.bias.copy_(torch.as_tensor(np.array(source["bias"], copy=True)))

    with torch.no_grad():
        wavefunction.node_hidden_initial.copy_(
            torch.as_tensor(np.array(parameters["node_hidden_initial"], copy=True))
        )
        wavefunction.edge_hidden_initial.copy_(
            torch.as_tensor(np.array(parameters["edge_hidden_initial"], copy=True))
        )
        for source, target in zip(parameters["steps"], wavefunction.steps):
            dense(source["query"], target.query)
            dense(source["key"], target.key)
            dense(source["tensor_filter"], target.tensor_filter.linear)
            dense(source["edge_gate"], target.edge_gate.linear)
            dense(source["node_update"], target.node_update.linear)
            dense(source["edge_update"], target.edge_update.linear)
        for source, target in zip(parameters["backflow_heads"], wavefunction.backflow_heads):
            dense(source, target)
        for source, target in zip(parameters["orbital_mlps"], wavefunction.orbital_mlps):
            dense(source[0], target[0])
            dense(source[1], target[2])
        if wavefunction.momentum_real is not None:
            wavefunction.momentum_real.copy_(
                torch.as_tensor(np.array(parameters["momentum_real"], copy=True))
            )
            wavefunction.momentum_imag.copy_(
                torch.as_tensor(np.array(parameters["momentum_imag"], copy=True))
            )
        wavefunction.determinant_logits.zero_()
    wavefunction.reference_fast_path = False
    return wavefunction
