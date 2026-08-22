"""Independent fixed-Bloch-Slater Hamiltonian component regression.

The state has zero neural coordinate displacement, J=1, and a fixed one-hot M.
Monte Carlo local-energy components are compared with finite-plane-wave-basis
matrix elements.  Nothing in this module updates variational parameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from .jax_neural_bloch import (
    JaxNeuralBlochSpec,
    initialize,
    local_energy_components,
)
from .run_jax_neural_bloch import JaxMetropolis, _batched
from .run_neural_bloch import neural_bloch_inputs


def fixed_slater(occupied: tuple[int, ...], seed: int = 83):
    """Return a literal bare Bloch Slater determinant in the production tree."""
    spec = JaxNeuralBlochSpec(
        n_particles=len(occupied), width=32, message_passing_steps=2,
        determinants=1, orbital_hidden=32, fixed_gamma_no_m=False,
    )
    parameters, constants = initialize(spec, seed=seed)
    for head in parameters["backflow_heads"]:
        head["weight"] = jnp.zeros_like(head["weight"])
    for mlp in parameters["orbital_mlps"]:
        mlp[-1]["weight"] = jnp.zeros_like(mlp[-1]["weight"])
        mlp[-1]["bias"] = jnp.asarray([1.0, 0.0], dtype=jnp.float64)
    momentum_real = jnp.zeros_like(parameters["momentum_real"])
    momentum_imag = jnp.zeros_like(parameters["momentum_imag"])
    for row, orbital in enumerate(occupied):
        momentum_real = momentum_real.at[0, row, orbital].set(1.0)
    parameters["momentum_real"] = momentum_real
    parameters["momentum_imag"] = momentum_imag
    return spec, parameters, constants


def _one_body_components(occupied: tuple[int, ...]) -> dict[str, float]:
    model, inputs = neural_bloch_inputs()
    coefficients = np.asarray(inputs["bloch_coefficients"])
    momenta = np.asarray(inputs["momenta"])
    ng = model.n_g
    static = np.asarray(model._static)
    moire_matrix = static.copy()
    moire_matrix[:ng, ng:] = 0.0
    moire_matrix[ng:, :ng] = 0.0
    tunneling_matrix = static - moire_matrix
    values = {"kinetic": 0.0, "moire": 0.0, "tunneling": 0.0}
    for orbital in occupied:
        vector = coefficients[orbital].reshape(2 * ng)
        diagonal = np.concatenate([
            model.alpha * np.sum(
                (momenta[orbital][None] + model.g_vectors - model.kappas[layer]) ** 2,
                axis=1,
            )
            for layer in range(2)
        ])
        values["kinetic"] += float(np.vdot(vector, diagonal * vector).real)
        values["moire"] += float(np.vdot(vector, moire_matrix @ vector).real)
        values["tunneling"] += float(
            np.vdot(vector, tunneling_matrix @ vector).real
        )
    return values


def _density_matrix_element(
    left: int,
    right: int,
    q: np.ndarray,
    momenta: np.ndarray,
    g_vectors: np.ndarray,
    coefficients: np.ndarray,
    tolerance: float = 1.0e-7,
) -> complex:
    result = 0.0j
    for layer in range(2):
        left_momenta = momenta[left][None] + g_vectors
        right_momenta = momenta[right][None] + g_vectors
        difference = left_momenta[:, None, :] - right_momenta[None, :, :] - q
        matches = np.linalg.norm(difference, axis=-1) < tolerance
        result += np.sum(
            coefficients[left, layer, :, None].conj()
            * coefficients[right, layer, None, :]
            * matches
        )
    return complex(result)


def analytic_components(
    occupied: tuple[int, ...], constants: dict,
) -> dict[str, float]:
    """Finite-plane-wave matrix elements for exactly the same fixed state."""
    result = _one_body_components(occupied)
    momenta = np.asarray(constants["momenta"])
    g_vectors = np.asarray(constants["g_vectors"])
    coefficients = np.asarray(constants["bloch_coefficients"])
    coulomb = 0.0
    for q, coefficient in zip(
        np.asarray(constants["q_vectors"]),
        np.asarray(constants["q_coefficients"]),
    ):
        rho = np.asarray([
            [
                _density_matrix_element(
                    left, right, q, momenta, g_vectors, coefficients
                )
                for right in occupied
            ]
            for left in occupied
        ])
        diagonal = np.diag(rho)
        direct = np.sum(diagonal[:, None] * diagonal[None, :].conj()).real
        exchange = np.sum(np.abs(rho) ** 2).real
        coulomb += 0.5 * float(coefficient) * (direct - exchange)
    result["coulomb"] = float(coulomb)
    result["total"] = float(sum(result.values()))
    return result


def monte_carlo_components(
    occupied: tuple[int, ...] = (0, 4),
    walkers: int = 4096,
    burn_sweeps: int = 120,
    measurement_sweeps: int = 10,
    seed: int = 2701,
    wavefunction_batch: int = 516,
    local_energy_batch: int = 32,
) -> dict:
    spec, parameters, constants = fixed_slater(occupied)
    sampler = JaxMetropolis(
        parameters, constants, spec, walkers, 0.06, seed, wavefunction_batch
    )
    acceptance = []
    for _ in range(burn_sweeps):
        acceptance.append(sampler.sweep(parameters))
    for _ in range(measurement_sweeps):
        acceptance.append(sampler.sweep(parameters))
    evaluator = jax.jit(
        lambda p, x, l: jnp.stack(
            local_energy_components(p, x, l, constants, spec), axis=1
        )
    )
    samples = np.asarray(_batched(
        evaluator, parameters, sampler.positions, sampler.layers,
        local_energy_batch,
    )).real
    names = ("kinetic", "moire", "tunneling", "coulomb")
    exact = analytic_components(occupied, constants)
    mc = {name: float(samples[:, index].mean()) for index, name in enumerate(names)}
    sem = {
        name: float(samples[:, index].std(ddof=1) / np.sqrt(walkers))
        for index, name in enumerate(names)
    }
    mc["total"] = float(samples.sum(axis=1).mean())
    sem["total"] = float(samples.sum(axis=1).std(ddof=1) / np.sqrt(walkers))
    return {
        "definition": "fixed-state delta-r=0, J=1, fixed-M Bloch-Slater energy regression",
        "occupied_momentum_indices": list(occupied),
        "walkers": walkers,
        "burn_sweeps": burn_sweeps,
        "measurement_sweeps": measurement_sweeps,
        "analytic_meV": exact,
        "monte_carlo_meV": mc,
        "sem_meV": sem,
        "difference_meV": {name: mc[name] - exact[name] for name in (*names, "total")},
        "mean_position_acceptance": float(np.mean([x["position_acceptance"] for x in acceptance])),
        "mean_layer_acceptance": float(np.mean([x["layer_acceptance"] for x in acceptance])),
        "used_for_parameter_updates": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--walkers", type=int, default=4096)
    parser.add_argument("--burn-sweeps", type=int, default=120)
    parser.add_argument("--measurement-sweeps", type=int, default=10)
    args = parser.parse_args()
    result = monte_carlo_components(
        walkers=args.walkers,
        burn_sweeps=args.burn_sweeps,
        measurement_sweeps=args.measurement_sweeps,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
