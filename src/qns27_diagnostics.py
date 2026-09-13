"""Training-chain observables for the completed 27-cell JAX QNS states.

The older :mod:`neural_bloch_diagnostics` path is tied to the 3x3 cluster.
This module keeps the 27-cell C6 supercell and evaluates complete JAX
wavefunction ratios.  The one-body density matrix is projected onto the fixed
continuum Bloch orbitals.  We retain both the first-band occupation, normalized
to N_e for a shape comparison with one-band ED, and the raw sum over the first
five bands.  The full S(q) is the Fourier transform of the sampled real-space
density-pair correlation.  Its primary grid is exactly the 27 physical Bloch
momenta used by ED; a directly evaluated C6-closed 37-vector grid is saved only
as a boundary/rotation audit.  No observable is C3 averaged.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import jax
jax.config.update("jax_enable_x64", True)
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import gaussian_filter

from .c6_ed import cluster_bloch_states
from .jax_neural_bloch import (
    JaxNeuralBlochSpec,
    initialize,
    logpsi,
)
from .qns27 import (
    cluster,
    first_bz_c6_grid,
    first_bz_momenta,
    inputs,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "result" / "data"


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _checkpoint(checkpoint: Path):
    metadata = json.loads(checkpoint.with_suffix(".json").read_text())
    spec_data = metadata.get("spec", metadata.get("protocol", {}).get("spec"))
    if spec_data is None:
        raise ValueError("checkpoint metadata has no JAX specification")
    spec = JaxNeuralBlochSpec(**spec_data)
    if spec.cells != 27:
        raise ValueError("qns27 diagnostics require a 27-cell checkpoint")
    initial, constants = initialize(spec, int(metadata.get("seed", 3184)))
    expected, unravel = ravel_pytree(initial)
    with np.load(checkpoint) as payload:
        trained = np.asarray(payload["flat_parameters"])
    if trained.shape != np.asarray(expected).shape:
        raise ValueError("checkpoint parameter vector has the wrong shape")
    return unravel(jnp.asarray(trained)), constants, spec, metadata


def _five_band_orbitals() -> tuple[np.ndarray, np.ndarray]:
    cache = DATA / "qns27_bloch_5band.npz"
    model, qns_inputs = inputs()
    if not cache.exists():
        energies, vectors, _ = cluster_bloch_states(model, cluster(), 5)
        np.savez_compressed(
            cache,
            energies=energies,
            vectors=vectors,
            matrix=cluster().matrix,
            theta=np.asarray(2.6),
        )
    with np.load(cache) as payload:
        if not np.array_equal(payload["matrix"], cluster().matrix):
            raise ValueError("cached 27-cell Bloch mesh has the wrong supercell")
        vectors = np.asarray(payload["vectors"])
    coefficients = (
        vectors[:, :, :5].transpose(0, 2, 1).reshape(27 * 5, 2, model.n_g)
    )
    momenta = np.repeat(np.asarray(qns_inputs["momenta"]), 5, axis=0)
    return coefficients, momenta


def _orbital_values(
    positions: np.ndarray,
    layers: np.ndarray,
    coefficients: np.ndarray,
    momenta: np.ndarray,
    batch: int,
) -> np.ndarray:
    model, qns_inputs = inputs()
    plane_momenta = momenta[:, None, :] + model.g_vectors[None, :, :]
    result = []
    for start in range(0, len(positions), batch):
        stop = min(start + batch, len(positions))
        cartesian = np.einsum(
            "ac,bc->ba", qns_inputs["supercell_lattice"], positions[start:stop]
        )
        phase = np.einsum("ba,oga->bog", cartesian, plane_momenta)
        selected = coefficients[:, layers[start:stop], :].transpose(1, 0, 2)
        result.append(np.sum(selected * np.exp(1j * phase), axis=-1))
    return np.concatenate(result)


def _block_sem(values: np.ndarray, blocks: int) -> np.ndarray:
    pieces = [piece for piece in np.array_split(values, min(blocks, len(values))) if len(piece)]
    if len(pieces) < 2:
        return np.zeros(values.shape[1:], dtype=float)
    means = np.stack([piece.mean(axis=0) for piece in pieces])
    return means.std(axis=0, ddof=1) / np.sqrt(len(means))


def _filling_normalized(
    per_sample: np.ndarray, particles: int, blocks: int
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a projected momentum distribution to the physical filling.

    This is a conditional distribution within the selected band subspace. Its
    scale is fixed only by N_e, never by the ED momentum profile. Blockwise
    normalization propagates numerator/trace covariance into the SEM.
    """
    mean = per_sample.mean(axis=0)
    normalized = particles * mean / mean.sum()
    pieces = [
        piece for piece in np.array_split(per_sample, min(blocks, len(per_sample)))
        if len(piece)
    ]
    block_values = np.stack([
        particles * piece.mean(axis=0) / piece.mean(axis=0).sum()
        for piece in pieces
    ])
    sem = (
        block_values.std(axis=0, ddof=1) / np.sqrt(len(block_values))
        if len(block_values) > 1 else np.zeros_like(normalized)
    )
    return normalized, sem


def _first_bz_momenta() -> np.ndarray:
    """The 27 crystal-momentum classes, one representative per class."""
    return first_bz_momenta()


def _rotation_permutation(points: np.ndarray, turns: int = 1) -> np.ndarray:
    """Permutation induced by a 120-degree rotation of a closed point grid."""
    angle = 2.0 * np.pi * turns / 3.0
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    rotated = points @ rotation.T
    distances = np.linalg.norm(rotated[:, None, :] - points[None, :, :], axis=2)
    permutation = np.argmin(distances, axis=1)
    if np.max(distances[np.arange(len(points)), permutation]) > 2.0e-12:
        raise ValueError("Fourier grid is not closed under C3")
    if len(np.unique(permutation)) != len(points):
        raise ValueError("C3 point map is not a permutation")
    return permutation


def _density_pair_structure_factor(
    cartesian: np.ndarray,
    q_vectors: np.ndarray,
    batch: int = 128,
) -> np.ndarray:
    """Fourier transform the empirical real-space density-pair measure.

    This is the unbinned estimator N_e^{-1} sum_ij cos[q.(r_i-r_j)].
    Avoiding a real-space histogram makes the Fourier transform exact at the
    requested q points while the sample batching keeps peak memory small.
    """
    particles = cartesian.shape[1]
    result = np.empty((len(cartesian), len(q_vectors)), dtype=float)
    for start in range(0, len(cartesian), batch):
        stop = min(start + batch, len(cartesian))
        current = cartesian[start:stop]
        displacement = current[:, :, None, :] - current[:, None, :, :]
        phase = np.einsum("bija,qa->bijq", displacement, q_vectors)
        result[start:stop] = np.cos(phase).sum(axis=(1, 2)) / particles
    return result


def _coordinate_observables(positions: np.ndarray, blocks: int) -> dict[str, np.ndarray]:
    _, qns_inputs = inputs()
    supercell = np.asarray(qns_inputs["supercell_lattice"])
    cartesian = np.einsum("ac,bnc->bna", supercell, positions)
    # Boundary reciprocal images are distinct continuum density modes.  The
    # C6-closed audit grid explicitly evaluates each physical vector; it never
    # obtains a boundary value by adding a reciprocal lattice vector.
    audit_q_vectors = first_bz_c6_grid()
    audit_full_per_sample = _density_pair_structure_factor(
        cartesian, audit_q_vectors
    )
    # The publication comparison uses exactly the same 27 physical vectors as
    # the ED Bloch mesh.  The larger grid is retained only to audit rotations
    # of boundary modes without ever substituting q+G.
    q_vectors = _first_bz_momenta()
    primary_indices = np.asarray([
        int(np.argmin(np.linalg.norm(audit_q_vectors - point, axis=1)))
        for point in q_vectors
    ])
    if np.max(np.linalg.norm(
        audit_q_vectors[primary_indices] - q_vectors, axis=1
    )) > 2.0e-12:
        raise RuntimeError("ED Bloch q grid is absent from the C6 audit grid")
    full_per_sample = audit_full_per_sample[:, primary_indices]

    # The disconnected term is evaluated from the same density samples.  It
    # vanishes away from Bragg vectors in a translation-invariant state.
    phase = np.einsum("bna,qa->bnq", cartesian, q_vectors)
    rho_q = np.exp(1j * phase).sum(axis=1)
    particles = positions.shape[1]
    full = full_per_sample.mean(axis=0)
    disconnected = np.abs(rho_q.mean(axis=0)) ** 2 / particles
    connected = full - disconnected
    amplitude = np.abs(rho_q.mean(axis=0)) / particles
    full[0] = connected[0] = amplitude[0] = 0.0
    audit_full = audit_full_per_sample.mean(axis=0)
    audit_full[0] = 0.0

    primitive_fraction = np.remainder(
        np.einsum("ac,bnc->bna", np.asarray(cluster().matrix), positions), 1.0
    )
    bins = 64
    density, xedge, yedge = np.histogram2d(
        primitive_fraction[..., 0].ravel(), primitive_fraction[..., 1].ravel(),
        bins=bins, range=((0.0, 1.0), (0.0, 1.0)),
    )
    density = gaussian_filter(density, 1.25, mode="wrap")
    density /= density.mean()
    return {
        "structure_q_vectors": q_vectors,
        "charge_structure_factor_full": full.real,
        "charge_structure_factor_connected": connected.real,
        "charge_structure_factor_full_sem": _block_sem(full_per_sample, blocks),
        "structure_q_vectors_c6_audit": audit_q_vectors,
        "charge_structure_factor_full_c6_audit": audit_full,
        "density_fourier_amplitude_per_particle": amplitude,
        "density_x_fraction": 0.5 * (xedge[:-1] + xedge[1:]),
        "density_y_fraction": 0.5 * (yedge[:-1] + yedge[1:]),
        "charge_density_over_mean": density,
    }


def run(args: argparse.Namespace) -> dict:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    parameters, constants, spec, metadata = _checkpoint(args.checkpoint)
    with np.load(args.samples) as payload:
        count = min(args.samples_count, len(payload["positions"]))
        positions = np.asarray(payload["positions"][:count])
        layers = np.asarray(payload["layers"][:count])
        base_logpsi = np.asarray(payload["logpsi"][:count])
    rng = np.random.default_rng(args.seed)
    draws = args.auxiliary_draws
    particle = rng.integers(spec.n_particles, size=(draws, count))
    proposal_position = rng.random((draws, count, 2))
    proposal_layer = rng.integers(2, size=(draws, count))
    expanded_positions = np.broadcast_to(
        positions[None], (draws, *positions.shape)
    ).copy()
    expanded_layers = np.broadcast_to(layers[None], (draws, *layers.shape)).copy()
    draw_index = np.arange(draws)[:, None]
    sample_index = np.arange(count)[None, :]
    expanded_positions[draw_index, sample_index, particle] = proposal_position
    expanded_layers[draw_index, sample_index, particle] = proposal_layer

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
    base_logpsi_max_error = float(
        np.max(np.abs(np.exp(audited_logpsi - base_logpsi[:audit_count]) - 1.0))
    )
    if not np.isfinite(base_logpsi_max_error) or base_logpsi_max_error > 2.0e-8:
        raise RuntimeError(
            "saved walkers are incompatible with the final checkpoint: "
            f"relative wavefunction error {base_logpsi_max_error:.3e}"
        )
    flat_positions = expanded_positions.reshape(-1, spec.n_particles, 2)
    flat_layers = expanded_layers.reshape(-1, spec.n_particles)
    replaced_parts = []
    total = len(flat_positions)
    for start in range(0, total, args.wavefunction_batch):
        stop = min(start + args.wavefunction_batch, total)
        x = flat_positions[start:stop]
        l = flat_layers[start:stop]
        valid = stop - start
        if valid < args.wavefunction_batch:
            x = np.concatenate([x, np.repeat(x[-1:], args.wavefunction_batch - valid, axis=0)])
            l = np.concatenate([l, np.repeat(l[-1:], args.wavefunction_batch - valid, axis=0)])
        replaced_parts.append(np.asarray(function(jnp.asarray(x), jnp.asarray(l)))[:valid])
        if start == 0 or stop == total or (start // args.wavefunction_batch) % 50 == 0:
            _write_json(output / "status.json", {
                "status": "wavefunction_ratios", "completed": stop, "target": total,
                "elapsed_seconds": time.monotonic() - started,
            })
    replaced = np.concatenate(replaced_parts).reshape(draws, count)
    conjugate_ratio = np.exp(np.conj(replaced - base_logpsi[None]))
    del expanded_positions, expanded_layers, flat_positions, flat_layers, replaced_parts

    coefficients, orbital_momenta = _five_band_orbitals()
    selected_position = positions[sample_index, particle]
    selected_layer = layers[sample_index, particle]
    flat_selected_position = selected_position.reshape(-1, 2)
    flat_selected_layer = selected_layer.reshape(-1)
    flat_proposal_position = proposal_position.reshape(-1, 2)
    flat_proposal_layer = proposal_layer.reshape(-1)
    diagonal_draws = np.empty((draws * count, 27 * 5), dtype=np.complex128)
    for start in range(0, draws * count, args.orbital_batch):
        stop = min(start + args.orbital_batch, draws * count)
        current = _orbital_values(
            flat_selected_position[start:stop], flat_selected_layer[start:stop],
            coefficients, orbital_momenta, args.orbital_batch,
        )
        proposed = _orbital_values(
            flat_proposal_position[start:stop], flat_proposal_layer[start:stop],
            coefficients, orbital_momenta, args.orbital_batch,
        )
        diagonal_draws[start:stop] = (
            2.0 * spec.n_particles * np.conj(current) * proposed
            * conjugate_ratio.reshape(-1)[start:stop, None]
        )
    diagonal_per_sample = diagonal_draws.reshape(draws, count, 27, 5).mean(axis=0)
    diagonal = diagonal_per_sample.mean(axis=0).real
    band_weight_per_sample = diagonal_per_sample.real.sum(axis=1) / spec.n_particles
    band_weight = band_weight_per_sample.mean(axis=0)
    band_weight_sem = _block_sem(band_weight_per_sample, args.error_blocks)
    momentum_band1_per_sample = diagonal_per_sample.real[:, :, 0]
    momentum_band1_raw = momentum_band1_per_sample.mean(axis=0)
    momentum_band1_raw_sem = _block_sem(
        momentum_band1_per_sample, args.error_blocks
    )
    momentum_band1, momentum_band1_sem = _filling_normalized(
        momentum_band1_per_sample, spec.n_particles, args.error_blocks
    )
    momentum_first_five_per_sample = diagonal_per_sample.real.sum(axis=2)
    momentum_first_five = momentum_first_five_per_sample.mean(axis=0)
    momentum_first_five_sem = _block_sem(
        momentum_first_five_per_sample, args.error_blocks
    )
    coordinate = _coordinate_observables(positions, args.error_blocks)

    payload = {
        "n_particles": np.asarray(spec.n_particles, dtype=np.int32),
        "one_body_density_diagonal": diagonal,
        "band_weight": band_weight,
        "band_weight_sem": band_weight_sem,
        "momentum_occupation_band1_raw": momentum_band1_raw,
        "momentum_occupation_band1_raw_sem": momentum_band1_raw_sem,
        "momentum_occupation_band1": momentum_band1,
        "momentum_occupation_band1_sem": momentum_band1_sem,
        "momentum_occupation_first_five": momentum_first_five,
        "momentum_occupation_first_five_sem": momentum_first_five_sem,
        "momentum_k_points": _first_bz_momenta(),
        **coordinate,
    }
    np.savez_compressed(output / "qns27_diagnostics.npz", **payload)
    result = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "samples": str(args.samples),
        "cells": 27,
        "n_particles": spec.n_particles,
        "coordinate_samples": count,
        "auxiliary_draws_per_sample": draws,
        "sampling": "saved final training walkers; no independent validation chain",
        "one_body_estimator": (
            "Bloch-projected one-body density-matrix diagonal from complete "
            "one-particle wavefunction ratios: n_1(k) is conditionally "
            "normalized to N_e for one-band ED comparison, while the raw "
            "n_1(k) and sum over bands 1:5 are also saved; no C3 or ED-profile fit"
        ),
        "structure_factor_estimator": (
            "unbinned Fourier transform of the real-space density-pair "
            "measure at the 27 ED Bloch momentum vectors; a separately "
            "evaluated C6-closed 37-vector grid is retained only as an audit"
        ),
        "band_weight": band_weight.tolist(),
        "band_weight_sem": band_weight_sem.tolist(),
        "projected_trace": float(diagonal.sum()),
        "momentum_occupation_band1_raw_sum": float(momentum_band1_raw.sum()),
        "momentum_occupation_band1_normalized_sum": float(momentum_band1.sum()),
        "momentum_occupation_first_five_sum": float(momentum_first_five.sum()),
        "momentum_occupation_band1_raw_c3_residual": float(np.max(np.abs(
            momentum_band1_raw - momentum_band1_raw[
                cluster().rotation60[cluster().rotation60]
            ]
        ))),
        "structure_factor_raw_c3_residual": float(np.max(np.abs(
            coordinate["charge_structure_factor_full_c6_audit"]
            - coordinate["charge_structure_factor_full_c6_audit"][
                _rotation_permutation(coordinate["structure_q_vectors_c6_audit"])
            ]
        ))),
        "saved_base_wavefunction_relative_error": base_logpsi_max_error,
        "maximum_nonzero_structure_factor": float(coordinate["charge_structure_factor_full"].max()),
        "maximum_nonzero_density_amplitude": float(coordinate["density_fourier_amplitude_per_particle"].max()),
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
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--samples-count", type=int, default=4128)
    result.add_argument("--auxiliary-draws", type=int, default=16)
    result.add_argument("--error-blocks", type=int, default=8)
    result.add_argument("--wavefunction-batch", type=int, default=43)
    result.add_argument("--orbital-batch", type=int, default=128)
    result.add_argument("--seed", type=int, default=4885)
    return result


def main() -> None:
    print(json.dumps(run(parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
