"""Independent re-equilibrated validation energy for a JAX neural-Bloch checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import numpy as np

from .jax_neural_bloch import JaxNeuralBlochSpec, initialize, local_energy
from .run_jax_neural_bloch import JaxMetropolis, _batched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--walkers", type=int, default=4128)
    parser.add_argument("--burn-sweeps", type=int, default=300)
    parser.add_argument("--measurement-sweeps", type=int, default=20)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=9187)
    parser.add_argument("--wavefunction-batch", type=int, default=258)
    parser.add_argument("--local-energy-batch", type=int, default=32)
    args = parser.parse_args()

    metadata = json.loads(args.checkpoint.with_suffix(".json").read_text())
    spec_data = metadata["spec"] if "spec" in metadata else metadata["protocol"]["spec"]
    spec = JaxNeuralBlochSpec(**spec_data)
    initial, constants = initialize(spec, seed=83)
    expected, unravel = ravel_pytree(initial)
    with np.load(args.checkpoint) as payload:
        flat = jnp.asarray(payload["flat_parameters"])
    if flat.shape != expected.shape:
        raise ValueError("checkpoint shape mismatch")
    parameters = unravel(flat)

    sampler = JaxMetropolis(
        parameters, constants, spec, args.walkers, 0.06, args.seed,
        args.wavefunction_batch,
    )
    burn_acceptance = []
    for _ in range(args.burn_sweeps):
        burn_acceptance.append(sampler.sweep(parameters))
    evaluator = jax.jit(
        lambda p, x, l: local_energy(p, x, l, constants, spec)
    )
    measurement_acceptance = []
    measurement_energies = []
    for _ in range(args.measurement_sweeps):
        measurement_acceptance.append(sampler.sweep(parameters))
        measurement_energies.append(np.asarray(_batched(
            evaluator, parameters, sampler.positions, sampler.layers,
            args.local_energy_batch,
        )).real)
    real_by_sweep = np.asarray(measurement_energies)
    real = real_by_sweep.reshape(-1)
    # Keep complete sweep-major slices together in the blocking analysis.
    block_values = np.asarray([
        values.mean()
        for values in np.array_split(real_by_sweep, args.blocks, axis=0)
    ])
    samples_path = args.output.with_suffix(".npz")
    np.savez_compressed(
        samples_path,
        positions=np.asarray(sampler.positions),
        layers=np.asarray(sampler.layers),
        measurement_energy_meV=real_by_sweep,
    )
    payload = {
        "definition": "independent random-start re-equilibrated validation chain",
        "saved_final_ensemble": str(samples_path),
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "walkers": args.walkers,
        "burn_sweeps": args.burn_sweeps,
        "measurement_sweeps_after_burn": args.measurement_sweeps,
        "energy_measurements": int(real.size),
        "energy_meV": float(real.mean()),
        "energy_per_particle_meV": float(real.mean() / spec.n_particles),
        "local_energy_variance_meV2": float(real.var(ddof=1)),
        "naive_walker_sem_meV": float(real.std(ddof=1) / np.sqrt(len(real))),
        "blocked_sem_meV": float(block_values.std(ddof=1) / np.sqrt(args.blocks)),
        "blocked_sem_per_particle_meV": float(
            block_values.std(ddof=1) / np.sqrt(args.blocks) / spec.n_particles
        ),
        "mean_burn_position_acceptance": float(np.mean([
            row["position_acceptance"] for row in burn_acceptance
        ])),
        "mean_burn_layer_acceptance": float(np.mean([
            row["layer_acceptance"] for row in burn_acceptance
        ])),
        "mean_measurement_position_acceptance": float(np.mean([
            row["position_acceptance"] for row in measurement_acceptance
        ])),
        "mean_measurement_layer_acceptance": float(np.mean([
            row["layer_acceptance"] for row in measurement_acceptance
        ])),
        "caveat": "one validation chain; repeat independent seeds for a production uncertainty",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
