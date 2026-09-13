"""Training-chain observables for the completed 27-cell JAX QNS states.

The older :mod:`neural_bloch_diagnostics` path is tied to the 3x3 cluster.
This module keeps the 27-cell C6 supercell and evaluates complete JAX
wavefunction ratios.  The displayed n(k) is the direct one-body Fourier
estimator in the two layer-resolved G=0 plane waves at the 27 first-BZ
momenta; reciprocal images are not folded into it.  A separate five-band
diagonal is retained only for bare-band populations, avoiding a 135 by 135
matrix for every Monte Carlo configuration.  The full S(q) is evaluated
directly from coordinate-space density modes without band projection.
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
from .jax_neural_bloch import JaxNeuralBlochSpec, initialize, logpsi
from .qns27 import cluster, first_bz_momenta, inputs


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


def _first_bz_momenta() -> np.ndarray:
    """Compatibility wrapper for the authoritative plotted BZ mesh."""
    return first_bz_momenta()


def _coordinate_observables(positions: np.ndarray, blocks: int) -> dict[str, np.ndarray]:
    model, qns_inputs = inputs()
    supercell = np.asarray(qns_inputs["supercell_lattice"])
    cartesian = np.einsum("ac,bnc->bna", supercell, positions)
    # A physical continuum density mode is not periodic under q -> q + G.
    # Use the same first-BZ transfer representatives as the projected ED
    # density operator and as the coordinates used by the report figure.
    q_vectors = _first_bz_momenta()
    phase = np.einsum("bna,qa->bnq", cartesian, q_vectors)
    rho_q = np.exp(1j * phase).sum(axis=1)
    particles = positions.shape[1]
    full_per_sample = np.abs(rho_q) ** 2 / particles
    full = full_per_sample.mean(axis=0)
    connected = full - np.abs(rho_q.mean(axis=0)) ** 2 / particles
    amplitude = np.abs(rho_q.mean(axis=0)) / particles
    full[0] = connected[0] = amplitude[0] = 0.0

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
        "density_fourier_amplitude_per_particle": amplitude,
        "density_x_fraction": 0.5 * (xedge[:-1] + xedge[1:]),
        "density_y_fraction": 0.5 * (yedge[:-1] + yedge[1:]),
        "charge_density_over_mean": density,
    }


def _plane_wave_momentum_occupation(
    selected_position: np.ndarray,
    selected_layer: np.ndarray,
    proposal_position: np.ndarray,
    proposal_layer: np.ndarray,
    conjugate_ratio: np.ndarray,
    particles: int,
    blocks: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate G=0 plane-wave n(k), summed over the two physical layers.

    The proposal coordinate and layer are sampled uniformly. The factor two
    removes the auxiliary-layer sampling probability. Unlike a crystal-
    momentum projector, this estimator does not sum k+G reciprocal images, so
    its 27-point trace is a finite first-BZ window and need not equal N_e.
    """
    _, qns_inputs = inputs()
    displacement = proposal_position - selected_position
    cartesian_displacement = np.einsum(
        "ac,dbc->dba", np.asarray(qns_inputs["supercell_lattice"]), displacement
    )
    phase = np.exp(1j * np.einsum(
        "dba,ka->dbk", cartesian_displacement, _first_bz_momenta()
    ))
    same_layer = selected_layer == proposal_layer
    per_draw = (
        2.0 * particles * same_layer[..., None] * phase
        * conjugate_ratio[..., None]
    )
    per_sample = per_draw.mean(axis=0)
    mean = per_sample.mean(axis=0)
    return (
        mean.real,
        _block_sem(per_sample.real, blocks),
        float(np.max(np.abs(mean.imag))),
    )


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
    momentum_band1 = momentum_band1_per_sample.mean(axis=0)
    momentum_band1_sem = _block_sem(momentum_band1_per_sample, args.error_blocks)
    momentum_plane_wave, momentum_plane_wave_sem, momentum_plane_wave_imaginary_max = (
        _plane_wave_momentum_occupation(
            selected_position, selected_layer, proposal_position, proposal_layer,
            conjugate_ratio, spec.n_particles, args.error_blocks,
        )
    )
    coordinate = _coordinate_observables(positions, args.error_blocks)

    payload = {
        "one_body_density_diagonal": diagonal,
        "band_weight": band_weight,
        "band_weight_sem": band_weight_sem,
        "momentum_occupation_band1": momentum_band1,
        "momentum_occupation_band1_sem": momentum_band1_sem,
        "momentum_occupation_plane_wave": momentum_plane_wave,
        "momentum_occupation_plane_wave_sem": momentum_plane_wave_sem,
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
            "direct G=0 first-BZ plane-wave Fourier estimator summed over both "
            "layers; no reciprocal-image sum; five-band Bloch projection is "
            "retained only for band-weight diagnostics"
        ),
        "band_weight": band_weight.tolist(),
        "band_weight_sem": band_weight_sem.tolist(),
        "projected_trace": float(diagonal.sum()),
        "momentum_occupation_band1_sum": float(momentum_band1.sum()),
        "momentum_occupation_plane_wave_sum": float(momentum_plane_wave.sum()),
        "momentum_occupation_plane_wave_imaginary_max": momentum_plane_wave_imaginary_max,
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
