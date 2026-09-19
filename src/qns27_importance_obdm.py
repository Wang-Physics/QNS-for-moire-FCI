"""Low-variance Bloch-projected 1-RDM estimator for the 27-cell QNS states.

The auxiliary coordinate is sampled from the PyQMC orbital-mixture density

    p(x) = norb**-1 sum_a |phi_a(x)|**2,

where ``a`` runs over the first five continuum bands at the 27 physical
first-BZ momenta.  The estimator retains the full one-electron replacement
ratio of the real-space many-body wavefunction.  A simultaneously measured
orbital-normalization control, also used by PyQMC, removes finite auxiliary
quadrature drift without normalizing the many-body trace to ``N_e``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from .jax_neural_bloch import logpsi
from .qns27 import cluster, inputs
from .qns27_diagnostics import (
    _checkpoint,
    _five_band_orbitals,
    _first_bz_momenta,
    _orbital_values,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_progress(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)


def _orbital_mixture_density(
    positions: np.ndarray,
    layers: np.ndarray,
    coefficients: np.ndarray,
    momenta: np.ndarray,
    batch: int,
) -> np.ndarray:
    values = _orbital_values(positions, layers, coefficients, momenta, batch)
    return np.mean(np.abs(values) ** 2, axis=1)


def _uniform_rejection_envelope(coefficients: np.ndarray) -> float:
    """Certified bound for p(x)/(uniform joint position-layer density)."""
    orbitals = len(coefficients)
    # |sum_G u_G exp(iq_G.r)| <= sum_G |u_G|.  Uniform sampling has joint
    # density 1/2 because the fractional cell has unit area and two layers.
    layer_bound = np.sum(
        np.sum(np.abs(coefficients), axis=2) ** 2, axis=0
    ) / orbitals
    return float(2.0 * np.max(layer_bound))


def _five_band_c3_sewing_audit() -> dict[str, float]:
    """Check that the five physical first-BZ projectors form C3 orbits."""
    model, _ = inputs()
    coefficients, momenta = _five_band_orbitals()
    vectors = coefficients.reshape(27, 5, model.dim).transpose(0, 2, 1)
    c3_map = np.asarray(cluster().rotation60)
    c3_map = c3_map[c3_map]
    angle = 2.0 * np.pi / 3.0
    rotation = np.asarray([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    minimum_singular = 1.0
    maximum_leakage = 0.0
    maximum_diagonal_magnitude_error = 0.0
    for source, target in enumerate(c3_map):
        wrap_float = (
            momenta[source] @ rotation.T - momenta[target]
        ) @ np.linalg.inv(model.reciprocal)
        wrap = np.rint(wrap_float).astype(int)
        if np.max(np.abs(wrap_float - wrap)) > 1.0e-10:
            raise RuntimeError("noninteger first-BZ reciprocal wrap under C3")
        operator = np.zeros((model.dim, model.dim), dtype=np.complex128)
        for layer, layer_shift in ((0, (1, 0)), (1, (0, -1))):
            for source_g, (first, second) in enumerate(model.g_tuples):
                rotated_g = (-second, first - second)
                target_g = (
                    rotated_g[0] + int(wrap[0]) + layer_shift[0],
                    rotated_g[1] + int(wrap[1]) + layer_shift[1],
                )
                target_g_index = model.g_index.get(target_g)
                if target_g_index is not None:
                    operator[
                        layer * model.n_g + target_g_index,
                        layer * model.n_g + source_g,
                    ] = 1.0
        sewing = vectors[target].conj().T @ operator @ vectors[source]
        singular = np.linalg.svd(sewing, compute_uv=False)
        minimum_singular = min(minimum_singular, float(singular.min()))
        diagonal = np.diag(np.diag(sewing))
        maximum_leakage = max(
            maximum_leakage,
            float(np.linalg.norm(sewing - diagonal) / np.linalg.norm(sewing)),
        )
        maximum_diagonal_magnitude_error = max(
            maximum_diagonal_magnitude_error,
            float(np.max(np.abs(np.abs(np.diag(sewing)) - 1.0))),
        )
    return {
        "minimum_singular_value": minimum_singular,
        "maximum_interband_leakage": maximum_leakage,
        "maximum_diagonal_magnitude_error": maximum_diagonal_magnitude_error,
    }


def generate_auxiliary_cache(
    output: Path,
    count: int,
    draws: int,
    sweeps_per_draw: int,
    seed: int,
    orbital_batch: int,
) -> dict:
    """Generate a stationary PyQMC-style auxiliary-coordinate ensemble.

    The initial pool is an exact rejection sample using a rigorous plane-wave
    triangle-inequality envelope.  Subsequent independent-uniform Metropolis
    proposals preserve the target orbital-mixture density.  Thus no auxiliary
    burn-in approximation enters the observable.
    """
    if count < 1 or draws < 1 or sweeps_per_draw < 1:
        raise ValueError("count, draws, and sweeps_per_draw must be positive")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    coefficients, momenta = _five_band_orbitals()
    envelope = _uniform_rejection_envelope(coefficients)
    rng = np.random.default_rng(seed)
    started = time.monotonic()

    accepted_positions: list[np.ndarray] = []
    accepted_layers: list[np.ndarray] = []
    accepted_density: list[np.ndarray] = []
    accepted_count = 0
    candidates = 0
    while accepted_count < count:
        remaining = count - accepted_count
        batch_count = max(8192, int(np.ceil(1.15 * envelope * remaining)))
        position = rng.random((batch_count, 2))
        layer = rng.integers(0, 2, size=batch_count, dtype=np.int8)
        density = _orbital_mixture_density(
            position, layer, coefficients, momenta, orbital_batch
        )
        probability = 2.0 * density / envelope
        if np.max(probability) > 1.0 + 2.0e-13:
            raise RuntimeError("analytic auxiliary rejection bound was violated")
        keep = rng.random(batch_count) < np.minimum(probability, 1.0)
        take = min(remaining, int(np.count_nonzero(keep)))
        selected = np.flatnonzero(keep)[:take]
        accepted_positions.append(position[selected])
        accepted_layers.append(layer[selected])
        accepted_density.append(density[selected])
        accepted_count += take
        candidates += batch_count

    current_position = np.concatenate(accepted_positions)
    current_layer = np.concatenate(accepted_layers)
    current_density = np.concatenate(accepted_density)
    position_history = np.empty((draws, count, 2), dtype=np.float64)
    layer_history = np.empty((draws, count), dtype=np.int8)
    density_history = np.empty((draws, count), dtype=np.float64)
    accepted_updates = 0
    proposed_updates = 0

    for draw in range(draws):
        if draw:
            for _ in range(sweeps_per_draw):
                proposal_position = rng.random((count, 2))
                proposal_layer = rng.integers(0, 2, size=count, dtype=np.int8)
                proposal_density = _orbital_mixture_density(
                    proposal_position, proposal_layer, coefficients, momenta,
                    orbital_batch,
                )
                accept = rng.random(count) < np.minimum(
                    proposal_density / current_density, 1.0
                )
                current_position[accept] = proposal_position[accept]
                current_layer[accept] = proposal_layer[accept]
                current_density[accept] = proposal_density[accept]
                accepted_updates += int(np.count_nonzero(accept))
                proposed_updates += count
        # Random assignment prevents persistent alignment of an auxiliary
        # chain with one physical walker while preserving the stationary law.
        assignment = rng.permutation(count)
        position_history[draw] = current_position[assignment]
        layer_history[draw] = current_layer[assignment]
        density_history[draw] = current_density[assignment]

    np.savez_compressed(
        output,
        positions=position_history,
        layers=layer_history,
        density=density_history,
        k_points=momenta,
        supercell_matrix=cluster().matrix,
    )
    result = {
        "status": "complete",
        "output": str(output),
        "count": count,
        "draws": draws,
        "sweeps_per_draw": sweeps_per_draw,
        "seed": seed,
        "target_density": "mean_a |phi_a(x)|^2 over 27 k x first 5 bands",
        "initialization": "exact rejection sample from uniform position and layer",
        "rejection_envelope": envelope,
        "initial_rejection_acceptance": count / candidates,
        "independence_metropolis_acceptance": (
            accepted_updates / proposed_updates if proposed_updates else 1.0
        ),
        "mean_inverse_density": float(np.mean(1.0 / density_history)),
        "expected_mean_inverse_density": 2.0,
        "wall_seconds": time.monotonic() - started,
    }
    _write_json(output.with_suffix(".json"), result)
    return result


def _block_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    blocks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if numerator.shape != denominator.shape:
        raise ValueError("numerator and denominator must have equal shapes")
    pieces = [
        piece for piece in np.array_split(np.arange(len(numerator)), min(blocks, len(numerator)))
        if len(piece)
    ]
    mean = numerator.mean(axis=0) / denominator.mean(axis=0)
    block_values = np.stack([
        numerator[piece].mean(axis=0) / denominator[piece].mean(axis=0)
        for piece in pieces
    ])
    sem = (
        block_values.std(axis=0, ddof=1) / np.sqrt(len(block_values))
        if len(block_values) > 1 else np.zeros_like(mean)
    )
    return mean, sem, block_values


def _normalized_with_blocks(
    mean: np.ndarray,
    block_values: np.ndarray,
    particles: int,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = particles * mean / mean.sum()
    block_normalized = particles * block_values / np.sum(
        block_values, axis=1, keepdims=True
    )
    sem = (
        block_normalized.std(axis=0, ddof=1) / np.sqrt(len(block_normalized))
        if len(block_normalized) > 1 else np.zeros_like(normalized)
    )
    return normalized, sem


def run(args: argparse.Namespace) -> dict:
    if args.auxiliary_draws < 1 or args.checkpoint_interval < 1:
        raise ValueError("auxiliary draws and checkpoint interval must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    parameters, constants, spec, metadata = _checkpoint(args.checkpoint)
    coefficients, momenta = _five_band_orbitals()

    with np.load(args.samples) as payload:
        available = len(payload["positions"])
        count = available if args.samples_count <= 0 else min(args.samples_count, available)
        positions = np.asarray(payload["positions"][:count])
        layers = np.asarray(payload["layers"][:count])
        base_logpsi = np.asarray(payload["logpsi"][:count])
        walker_id = np.asarray(payload["walker_id"][:count]) if "walker_id" in payload else None
        measurement_step = (
            np.asarray(payload["measurement_step"][:count])
            if "measurement_step" in payload else None
        )

    with np.load(args.auxiliary_cache) as payload:
        if len(payload["positions"]) < args.auxiliary_draws:
            raise ValueError("auxiliary cache has too few draws")
        if payload["positions"].shape[1] < count:
            raise ValueError("auxiliary cache has too few samples per draw")
        if not np.allclose(payload["k_points"], momenta, rtol=0.0, atol=1.0e-14):
            raise ValueError("auxiliary cache uses a different Bloch grid")
        if not np.array_equal(payload["supercell_matrix"], cluster().matrix):
            raise ValueError("auxiliary cache uses a different supercell")
        proposal_position = np.asarray(
            payload["positions"][:args.auxiliary_draws, :count]
        )
        proposal_layer = np.asarray(
            payload["layers"][:args.auxiliary_draws, :count]
        )
        proposal_density = np.asarray(
            payload["density"][:args.auxiliary_draws, :count]
        )

    function = jax.jit(lambda x, l: logpsi(parameters, x, l, constants, spec))
    audit_count = min(args.wavefunction_batch, count)
    audit_positions = positions[:audit_count]
    audit_layers = layers[:audit_count]
    if audit_count < args.wavefunction_batch:
        audit_positions = np.concatenate([
            audit_positions,
            np.repeat(audit_positions[-1:], args.wavefunction_batch - audit_count, axis=0),
        ])
        audit_layers = np.concatenate([
            audit_layers,
            np.repeat(audit_layers[-1:], args.wavefunction_batch - audit_count, axis=0),
        ])
    audited_logpsi = np.asarray(
        function(jnp.asarray(audit_positions), jnp.asarray(audit_layers))
    )[:audit_count]
    base_logpsi_max_error = float(np.max(np.abs(
        np.exp(audited_logpsi - base_logpsi[:audit_count]) - 1.0
    )))
    if not np.isfinite(base_logpsi_max_error) or base_logpsi_max_error > 2.0e-8:
        raise RuntimeError(
            "saved walkers are incompatible with the checkpoint: "
            f"relative wavefunction error {base_logpsi_max_error:.3e}"
        )

    rng = np.random.default_rng(args.seed)
    orbitals = len(coefficients)
    progress_path = output / "obdm_progress.npz"
    start_draw = 0
    if progress_path.is_file():
        with np.load(progress_path) as progress:
            compatible = all((
                int(progress["count"]) == count,
                int(progress["orbitals"]) == orbitals,
                int(progress["target_draws"]) == args.auxiliary_draws,
                int(progress["seed"]) == args.seed,
            ))
            if not compatible:
                raise ValueError("existing OBDM progress has incompatible settings")
            start_draw = int(progress["completed_draws"])
            numerator_sum = np.asarray(progress["numerator_sum"])
            normalization_sum = np.asarray(progress["normalization_sum"])
            particle_offset = np.asarray(progress["particle_offset"])
    else:
        numerator_sum = np.zeros((count, orbitals), dtype=np.complex128)
        normalization_sum = np.zeros((count, orbitals), dtype=np.float64)
        particle_offset = rng.integers(spec.n_particles, size=count)

    for draw in range(start_draw, args.auxiliary_draws):
        particle = (particle_offset + draw) % spec.n_particles
        for start in range(0, count, args.wavefunction_batch):
            stop = min(start + args.wavefunction_batch, count)
            valid = stop - start
            row = np.arange(start, stop)
            moved_positions = positions[start:stop].copy()
            moved_layers = layers[start:stop].copy()
            moved_positions[np.arange(valid), particle[row]] = proposal_position[draw, row]
            moved_layers[np.arange(valid), particle[row]] = proposal_layer[draw, row]
            if valid < args.wavefunction_batch:
                moved_positions = np.concatenate([
                    moved_positions,
                    np.repeat(moved_positions[-1:], args.wavefunction_batch - valid, axis=0),
                ])
                moved_layers = np.concatenate([
                    moved_layers,
                    np.repeat(moved_layers[-1:], args.wavefunction_batch - valid, axis=0),
                ])
            replaced = np.asarray(function(
                jnp.asarray(moved_positions), jnp.asarray(moved_layers)
            ))[:valid]
            ratio = np.exp(np.conj(replaced - base_logpsi[start:stop]))
            current = _orbital_values(
                positions[row, particle[row]],
                layers[row, particle[row]],
                coefficients, momenta, args.orbital_batch,
            )
            proposed = _orbital_values(
                proposal_position[draw, row], proposal_layer[draw, row],
                coefficients, momenta, args.orbital_batch,
            )
            inverse_density = 1.0 / proposal_density[draw, row, None]
            numerator_sum[start:stop] += (
                spec.n_particles * np.conj(current) * proposed
                * ratio[:, None] * inverse_density
            )
            normalization_sum[start:stop] += (
                np.abs(proposed) ** 2 * inverse_density
            )
        if (
            (draw + 1) % args.checkpoint_interval == 0
            or draw + 1 == args.auxiliary_draws
        ):
            _write_progress(
                progress_path,
                count=np.asarray(count), orbitals=np.asarray(orbitals),
                target_draws=np.asarray(args.auxiliary_draws),
                seed=np.asarray(args.seed), completed_draws=np.asarray(draw + 1),
                numerator_sum=numerator_sum,
                normalization_sum=normalization_sum,
                particle_offset=particle_offset,
            )
        _write_json(output / "status.json", {
            "status": "wavefunction_ratios",
            "resumed_from_auxiliary_draw": start_draw,
            "completed_auxiliary_draws": draw + 1,
            "target_auxiliary_draws": args.auxiliary_draws,
            "elapsed_seconds": time.monotonic() - started,
        })

    numerator_per_sample = numerator_sum.real / args.auxiliary_draws
    normalization_per_sample = normalization_sum / args.auxiliary_draws
    imaginary_residual = float(np.max(np.abs(
        numerator_sum.imag.mean(axis=0) / args.auxiliary_draws
    )))
    diagonal_flat, diagonal_flat_sem, diagonal_blocks_flat = _block_ratio(
        numerator_per_sample, normalization_per_sample, args.error_blocks
    )
    unnormalized_flat = numerator_per_sample.mean(axis=0)
    unnormalized_pieces = [
        piece for piece in np.array_split(
            np.arange(count), min(args.error_blocks, count)
        ) if len(piece)
    ]
    unnormalized_blocks_flat = np.stack([
        numerator_per_sample[piece].mean(axis=0)
        for piece in unnormalized_pieces
    ])
    unnormalized_sem_flat = unnormalized_blocks_flat.std(
        axis=0, ddof=1
    ) / np.sqrt(len(unnormalized_blocks_flat))
    diagonal = diagonal_flat.reshape(27, 5)
    diagonal_sem = diagonal_flat_sem.reshape(27, 5)
    diagonal_blocks = diagonal_blocks_flat.reshape(-1, 27, 5)

    band_weight = diagonal.sum(axis=0) / spec.n_particles
    band_weight_blocks = diagonal_blocks.sum(axis=1) / spec.n_particles
    band_weight_sem = band_weight_blocks.std(axis=0, ddof=1) / np.sqrt(
        len(band_weight_blocks)
    )
    momentum_band1_raw = diagonal[:, 0]
    momentum_band1_raw_sem = diagonal_sem[:, 0]
    momentum_band1_blocks = diagonal_blocks[:, :, 0]
    momentum_first_five_raw = diagonal.sum(axis=1)
    momentum_first_five_blocks = diagonal_blocks.sum(axis=2)
    momentum_first_five_raw_sem = momentum_first_five_blocks.std(
        axis=0, ddof=1
    ) / np.sqrt(len(momentum_first_five_blocks))
    projected_trace_blocks = momentum_first_five_blocks.sum(axis=1)
    projected_trace_sem = float(
        projected_trace_blocks.std(ddof=1) / np.sqrt(len(projected_trace_blocks))
    )
    momentum_band1, momentum_band1_sem = _normalized_with_blocks(
        momentum_band1_raw, momentum_band1_blocks, spec.n_particles
    )
    momentum_first_five, momentum_first_five_sem = _normalized_with_blocks(
        momentum_first_five_raw, momentum_first_five_blocks, spec.n_particles
    )

    normalization_mean = normalization_per_sample.mean(axis=0).reshape(27, 5)
    class_rotation = np.asarray(cluster().rotation60)
    class_rotation = class_rotation[class_rotation]
    band1_c3 = float(np.max(np.abs(
        momentum_band1_raw - momentum_band1_raw[class_rotation]
    )))
    total_c3 = float(np.max(np.abs(
        momentum_first_five_raw - momentum_first_five_raw[class_rotation]
    )))
    band1_c3_difference_blocks = (
        momentum_band1_blocks - momentum_band1_blocks[:, class_rotation]
    )
    total_c3_difference_blocks = (
        momentum_first_five_blocks
        - momentum_first_five_blocks[:, class_rotation]
    )
    band1_c3_difference_sem = band1_c3_difference_blocks.std(
        axis=0, ddof=1
    ) / np.sqrt(len(band1_c3_difference_blocks))
    total_c3_difference_sem = total_c3_difference_blocks.std(
        axis=0, ddof=1
    ) / np.sqrt(len(total_c3_difference_blocks))
    band1_c3_z = float(np.max(np.divide(
        np.abs(momentum_band1_raw - momentum_band1_raw[class_rotation]),
        band1_c3_difference_sem,
        out=np.zeros_like(band1_c3_difference_sem),
        where=band1_c3_difference_sem > 0.0,
    )))
    total_c3_z = float(np.max(np.divide(
        np.abs(momentum_first_five_raw - momentum_first_five_raw[class_rotation]),
        total_c3_difference_sem,
        out=np.zeros_like(total_c3_difference_sem),
        where=total_c3_difference_sem > 0.0,
    )))
    payload = {
        "n_particles": np.asarray(spec.n_particles, dtype=np.int32),
        "one_body_density_diagonal": diagonal,
        "one_body_density_diagonal_sem": diagonal_sem,
        "one_body_density_diagonal_unnormalized_importance": (
            unnormalized_flat.reshape(27, 5)
        ),
        "one_body_density_diagonal_unnormalized_importance_sem": (
            unnormalized_sem_flat.reshape(27, 5)
        ),
        "one_body_density_diagonal_block_estimates": diagonal_blocks,
        "auxiliary_orbital_normalization": normalization_mean,
        "band_weight": band_weight,
        "band_weight_sem": band_weight_sem,
        "momentum_occupation_band1_raw": momentum_band1_raw,
        "momentum_occupation_band1_raw_sem": momentum_band1_raw_sem,
        "momentum_occupation_band1": momentum_band1,
        "momentum_occupation_band1_sem": momentum_band1_sem,
        "momentum_occupation_first_five_raw": momentum_first_five_raw,
        "momentum_occupation_first_five_raw_sem": momentum_first_five_raw_sem,
        "momentum_occupation_first_five_raw_sum_sem": np.asarray(projected_trace_sem),
        "momentum_occupation_first_five": momentum_first_five,
        "momentum_occupation_first_five_sem": momentum_first_five_sem,
        "momentum_k_points": _first_bz_momenta(),
    }
    np.savez_compressed(output / "qns27_diagnostics.npz", **payload)
    unique_walkers = int(len(np.unique(walker_id))) if walker_id is not None else count
    measurement_steps = (
        int(len(np.unique(measurement_step))) if measurement_step is not None else 1
    )
    result = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "samples": str(args.samples),
        "auxiliary_cache": str(args.auxiliary_cache),
        "cells": 27,
        "n_particles": spec.n_particles,
        "coordinate_samples": count,
        "unique_walker_chains": unique_walkers,
        "measurement_steps": measurement_steps,
        "parameters_frozen": True,
        "optimizer_updates": 0,
        "auxiliary_draws_per_sample": args.auxiliary_draws,
        "error_blocks": args.error_blocks,
        "auxiliary_sampling": (
            "PyQMC-style importance sampling from mean_a |phi_a(x)|^2 over "
            "the 27 first-BZ k points and bands 1:5; exact rejection "
            "initialization plus stationary independence-Metropolis updates"
        ),
        "one_body_estimator": (
            "Bloch-projected one-body RDM diagonal from complete one-electron "
            "wavefunction ratios at the 27 physical first-BZ k points; G occurs "
            "only inside each Bloch orbital; PyQMC orbital-normalization control "
            "is applied blockwise; no particle-trace, C3, or ED-profile fit"
        ),
        "sampling": args.sampling_description,
        "band_weight": band_weight.tolist(),
        "band_weight_sem": band_weight_sem.tolist(),
        "projected_trace": float(diagonal.sum()),
        "momentum_occupation_band1_raw_sum": float(momentum_band1_raw.sum()),
        "momentum_occupation_band1_normalized_sum": float(momentum_band1.sum()),
        "momentum_occupation_first_five_raw_sum": float(momentum_first_five_raw.sum()),
        "momentum_occupation_first_five_raw_sum_sem": projected_trace_sem,
        "momentum_occupation_first_five_normalized_sum": float(momentum_first_five.sum()),
        "momentum_occupation_first_five_sum": float(momentum_first_five.sum()),
        "momentum_occupation_band1_raw_c3_residual": band1_c3,
        "momentum_occupation_first_five_raw_c3_residual": total_c3,
        "momentum_occupation_band1_raw_c3_max_z": band1_c3_z,
        "momentum_occupation_first_five_raw_c3_max_z": total_c3_z,
        "auxiliary_orbital_normalization_max_error": float(
            np.max(np.abs(normalization_mean - 1.0))
        ),
        "one_body_imaginary_residual": imaginary_residual,
        "bloch_projector_c3_sewing_audit": _five_band_c3_sewing_audit(),
        "saved_base_wavefunction_relative_error": base_logpsi_max_error,
        "wall_seconds": time.monotonic() - started,
        "spec": asdict(spec),
        "training_step": int(metadata.get("training_step", 0)),
    }
    _write_json(output / "qns27_diagnostics.json", result)
    _write_json(output / "status.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(__doc__)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--samples", type=Path, required=True)
    result.add_argument("--auxiliary-cache", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--samples-count", type=int, default=0)
    result.add_argument("--auxiliary-draws", type=int, default=32)
    result.add_argument("--error-blocks", type=int, default=8)
    result.add_argument("--wavefunction-batch", type=int, default=43)
    result.add_argument("--orbital-batch", type=int, default=512)
    result.add_argument("--checkpoint-interval", type=int, default=4)
    result.add_argument("--seed", type=int, default=250311756)
    result.add_argument(
        "--sampling-description",
        default="independent measurement-only chain with frozen final parameters",
    )
    return result


def auxiliary_parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser("Generate the shared orbital-mixture cache")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--count", type=int, default=20640)
    result.add_argument("--draws", type=int, default=64)
    result.add_argument("--sweeps-per-draw", type=int, default=1)
    result.add_argument("--seed", type=int, default=250311756)
    result.add_argument("--orbital-batch", type=int, default=4096)
    return result


def main() -> None:
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "auxiliary-cache":
        args = auxiliary_parser().parse_args(sys.argv[2:])
        record = generate_auxiliary_cache(
            args.output, args.count, args.draws, args.sweeps_per_draw,
            args.seed, args.orbital_batch,
        )
    else:
        record = run(parser().parse_args())
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
