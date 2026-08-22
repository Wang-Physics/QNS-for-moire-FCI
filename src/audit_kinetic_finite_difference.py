"""Finite-difference step-size audit on identical neural-Bloch configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import numpy as np

from .jax_neural_bloch import JaxNeuralBlochSpec, initialize, local_energy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    metadata = json.loads(args.checkpoint.with_suffix(".json").read_text())
    spec_data = metadata["spec"] if "spec" in metadata else metadata["protocol"]["spec"]
    spec = JaxNeuralBlochSpec(**spec_data)
    initial, constants = initialize(spec, seed=83)
    expected, unravel = ravel_pytree(initial)
    with np.load(args.checkpoint) as payload:
        flat = np.asarray(payload["flat_parameters"])
    if flat.shape != np.asarray(expected).shape:
        raise ValueError("checkpoint shape mismatch")
    parameters = unravel(jnp.asarray(flat))
    with np.load(args.samples) as payload:
        positions = np.asarray(payload["positions"][: args.sample_count])
        layers = np.asarray(payload["layers"][: args.sample_count])

    results: dict[str, dict] = {}
    energies_by_scale: dict[float, np.ndarray] = {}
    base_delta = constants["coordinate_deltas"]
    base_h = constants["finite_difference_nm"]
    for scale in (1.0, 0.5, 0.25):
        scaled = dict(constants)
        scaled["coordinate_deltas"] = base_delta * scale
        scaled["finite_difference_nm"] = base_h * scale
        evaluator = jax.jit(
            lambda x, l: local_energy(parameters, x, l, scaled, spec)
        )
        chunks = []
        for start in range(0, len(positions), args.batch_size):
            chunks.append(np.asarray(evaluator(
                jnp.asarray(positions[start:start + args.batch_size]),
                jnp.asarray(layers[start:start + args.batch_size]),
            )))
        energies = np.concatenate(chunks)
        energies_by_scale[scale] = energies
        real = energies.real
        results[str(scale)] = {
            "h_nm": float(base_h * scale),
            "mean_energy_meV": float(real.mean()),
            "mean_energy_per_particle_meV": float(real.mean() / spec.n_particles),
            "variance_meV2": float(real.var(ddof=1)),
            "imaginary_rms_meV": float(np.sqrt(np.mean(energies.imag ** 2))),
        }

    reference = energies_by_scale[0.25].real
    for scale in (1.0, 0.5):
        difference = energies_by_scale[scale].real - reference
        results[str(scale)]["difference_from_h_over_4_mean_meV"] = float(
            difference.mean()
        )
        results[str(scale)]["difference_from_h_over_4_rms_meV"] = float(
            np.sqrt(np.mean(difference ** 2))
        )
    payload = {
        "definition": "same-configuration central three-point finite-difference audit",
        "sample_count": len(positions),
        "particles": spec.n_particles,
        "scales": results,
        "automatic_differentiation_laplacian_checked": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
