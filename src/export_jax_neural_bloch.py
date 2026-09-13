"""Export a JAX neural-Bloch checkpoint for the existing OBDM diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import numpy as np
import torch

from .jax_neural_bloch import (
    JaxNeuralBlochSpec,
    copy_to_torch,
    initialize,
)
from .neural_bloch import NeuralBlochConfig
from .run_neural_bloch import build_wavefunction, save_checkpoint


def run(checkpoint: Path, output: Path, seed: int = 83) -> Path:
    metadata_path = checkpoint.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing checkpoint metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    spec_data = metadata["spec"] if "spec" in metadata else metadata["protocol"]["spec"]
    spec = JaxNeuralBlochSpec(**spec_data)
    initial, _ = initialize(spec, seed)
    expected_flat, unravel = ravel_pytree(initial)
    with np.load(checkpoint) as payload:
        trained_flat = np.asarray(payload["flat_parameters"])
    if trained_flat.shape != np.asarray(expected_flat).shape:
        raise ValueError("checkpoint parameter shape is incompatible with its metadata")
    parameters = unravel(jnp.asarray(trained_flat))

    torch.set_default_dtype(torch.float64)
    config = NeuralBlochConfig(
        width=spec.width,
        message_passing_steps=spec.message_passing_steps,
        determinants=spec.determinants,
        orbital_hidden=spec.orbital_hidden,
        c3_irrep=spec.c3_irrep,
        fixed_gamma_no_m=spec.fixed_gamma_no_m,
        c3_qns=spec.c3_qns,
        outer_c3_projector=spec.outer_c3_projector,
        v4_gamma_projected_m=bool(spec.translation_projected_rank),
    )
    _, wavefunction = build_wavefunction(spec.n_particles, config, seed)
    copy_to_torch(parameters, wavefunction)
    diagnostic_metadata = {
        "seed": seed,
        "n_particles": spec.n_particles,
        "source_jax_checkpoint": str(checkpoint),
        "training_step": int(metadata.get("training_step", 0)),
        "jax_protocol": metadata,
    }
    save_checkpoint(output, wavefunction, config, diagnostic_metadata)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=83)
    args = parser.parse_args()
    print(run(args.checkpoint, args.output, args.seed))


if __name__ == "__main__":
    main()
