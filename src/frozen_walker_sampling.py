"""Collect measurement-only walkers with a frozen final QNS wavefunction.

The training checkpoint supplies the final parameters, Markov-chain state and
PRNG key.  This module advances only the Metropolis chains: it never computes
an optimizer update.  Snapshots are stored walker-major so block estimates in
downstream diagnostics keep all correlated times from one chain together.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jax.numpy as jnp
import numpy as np

from .qns27_diagnostics import _checkpoint
from .run_jax_neural_bloch import JaxMetropolis


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict:
    if args.measurement_steps < 1 or args.sweeps_per_step < 1:
        raise ValueError("measurement steps and sweeps per step must be positive")
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    parameters, constants, spec, metadata = _checkpoint(checkpoint)
    with np.load(checkpoint) as payload:
        if not {"positions", "layers", "sampler_key"} <= set(payload.files):
            raise ValueError("checkpoint does not contain a resumable walker state")
        positions = np.asarray(payload["positions"])
        layers = np.asarray(payload["layers"])
        sampler_key = np.asarray(payload["sampler_key"])
        support = (
            np.asarray(payload["gamma_sector_indices"])
            if "gamma_sector_indices" in payload.files else None
        )

    sampler = JaxMetropolis(
        parameters, constants, spec, len(positions), args.proposal_scale,
        args.seed, args.wavefunction_batch,
    )
    if support is not None:
        sampler.set_support(jnp.asarray(support), parameters, refresh=False)
    sampler.positions = jnp.asarray(positions)
    sampler.layers = jnp.asarray(layers)
    sampler.key = jnp.asarray(sampler_key)
    sampler.refresh(parameters)

    started = time.monotonic()
    position_history = []
    layer_history = []
    logpsi_history = []
    acceptance = []
    for measurement_step in range(args.measurement_steps):
        current = sampler.sweep(parameters, args.sweeps_per_step)
        acceptance.append({"measurement_step": measurement_step + 1, **current})
        position_history.append(np.asarray(sampler.positions))
        layer_history.append(np.asarray(sampler.layers))
        logpsi_history.append(np.asarray(sampler.values))
        _write_json(output.with_suffix(".status.json"), {
            "status": "sampling",
            "completed_measurement_steps": measurement_step + 1,
            "target_measurement_steps": args.measurement_steps,
            "elapsed_seconds": time.monotonic() - started,
            "acceptance": acceptance,
        })

    # (time, walker, ...) -> (walker, time, ...) -> flat samples.  Keeping the
    # time history of each independent chain contiguous makes ordinary
    # contiguous block estimates conservative with respect to time
    # autocorrelation.
    position_array = np.stack(position_history, axis=0).swapaxes(0, 1)
    layer_array = np.stack(layer_history, axis=0).swapaxes(0, 1)
    logpsi_array = np.stack(logpsi_history, axis=0).swapaxes(0, 1)
    np.savez_compressed(
        output,
        positions=position_array.reshape(-1, spec.n_particles, 2),
        layers=layer_array.reshape(-1, spec.n_particles),
        logpsi=logpsi_array.reshape(-1),
        walker_id=np.repeat(np.arange(len(positions)), args.measurement_steps),
        measurement_step=np.tile(
            np.arange(1, args.measurement_steps + 1), len(positions)
        ),
    )
    result = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "output": str(output),
        "training_step": int(metadata.get("training_step", 0)),
        "parameters_frozen": True,
        "optimizer_updates": 0,
        "initial_walkers": len(positions),
        "measurement_steps": args.measurement_steps,
        "sweeps_per_step": args.sweeps_per_step,
        "stored_configurations": int(position_array.shape[0] * position_array.shape[1]),
        "proposal_scale": args.proposal_scale,
        "acceptance": acceptance,
        "wall_seconds": time.monotonic() - started,
        "storage_order": "walker-major, then measurement time",
    }
    _write_json(output.with_suffix(".json"), result)
    _write_json(output.with_suffix(".status.json"), result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(__doc__)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--measurement-steps", type=int, default=5)
    result.add_argument("--sweeps-per-step", type=int, default=2)
    result.add_argument("--proposal-scale", type=float, default=0.06)
    result.add_argument("--wavefunction-batch", type=int, default=43)
    result.add_argument("--seed", type=int, default=260914)
    return result


def main() -> None:
    result = run(parser().parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
