"""Coordinate-only S(q) and density diagnostics for frozen QNS walkers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .qns27_diagnostics import (
    _coordinate_observables,
    _physical_rotation_residual,
)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.samples) as payload:
        positions = np.asarray(payload["positions"])
        walker_id = np.asarray(payload["walker_id"])
        measurement_step = np.asarray(payload["measurement_step"])
    if len(positions) != 20640:
        raise ValueError("frozen five-step sample must contain 20,640 configurations")
    if len(np.unique(walker_id)) != 4128:
        raise ValueError("frozen sample must retain 4,128 independent chain labels")
    if not np.array_equal(np.unique(measurement_step), np.arange(1, 6)):
        raise ValueError("frozen sample must contain measurement steps 1 through 5")
    # frozen_walker_sampling stores all five adjacent times from a chain
    # contiguously.  Preserve that order so the eight blocks split by chain,
    # rather than incorrectly treating adjacent snapshots as independent.
    expected_walker = np.repeat(np.arange(4128), 5)
    expected_step = np.tile(np.arange(1, 6), 4128)
    if not (
        np.array_equal(walker_id, expected_walker)
        and np.array_equal(measurement_step, expected_step)
    ):
        raise ValueError("frozen samples are not in walker-major time order")

    coordinate = _coordinate_observables(positions, args.error_blocks)
    np.savez_compressed(output / "coordinate_diagnostics.npz", **coordinate)
    residual, matched = _physical_rotation_residual(
        coordinate["charge_structure_factor"],
        coordinate["structure_q_vectors"],
    )
    result = {
        "status": "complete",
        "samples": str(args.samples.resolve()),
        "parameters_frozen": True,
        "optimizer_updates": 0,
        "unique_walker_chains": 4128,
        "measurement_steps": 5,
        "sweeps_per_step": 2,
        "coordinate_samples": len(positions),
        "storage_order": "walker-major, then measurement time",
        "error_blocks": args.error_blocks,
        "structure_factor_estimator": (
            "Luo full S(q)=<|sum_i exp(i q.r_i)|^2>/N_e for q!=0, "
            "evaluated directly from five measurement-only snapshots of the "
            "step-120 frozen wavefunction; no auxiliary coordinate, q+G "
            "substitution, Bloch-band projection, or C3 average"
        ),
        "structure_factor_grid_points": int(
            len(coordinate["structure_q_vectors"])
        ),
        "structure_factor_projected_density": False,
        "structure_factor_periodic_wigner_seitz": True,
        "structure_factor_c3_matched_physical_points": matched,
        "structure_factor_raw_c3_residual": residual,
        "maximum_nonzero_structure_factor": float(
            coordinate["charge_structure_factor"].max()
        ),
        "maximum_nonzero_density_amplitude": float(
            coordinate["density_fourier_amplitude_per_particle"].max()
        ),
        "wall_seconds": time.monotonic() - started,
    }
    _write_json(output / "coordinate_diagnostics.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(__doc__)
    result.add_argument("--samples", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--error-blocks", type=int, default=8)
    return result


def main() -> None:
    print(json.dumps(run(parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
