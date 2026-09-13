"""Training-chain observables for the completed 27-cell JAX QNS states.

The older :mod:`neural_bloch_diagnostics` path is tied to the 3x3 cluster.
This module keeps the 27-cell C6 supercell and evaluates complete JAX
wavefunction ratios.  The one-body density matrix is projected onto the fixed
continuum Bloch orbitals. Both plotted occupations are raw physical
projections: band 1 alone and the first-five-band sum. Conditionally normalized
variants are retained only as diagnostics. The plotted S(q)
is Luo et al.'s full real-space density-pair structure factor evaluated
directly from the saved final walkers at the same 27 physical first-BZ vectors
drawn for ED. A denser continuous-q transform is retained only as an auxiliary
diagnostic. No observable is C3 averaged.
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

from .jax_neural_bloch import (
    JaxNeuralBlochSpec,
    initialize,
    logpsi,
)
from .qns27 import (
    cluster,
    first_bz_momenta,
    inputs,
    uniform_first_bz_grid,
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
    """Bare Bloch orbitals at the 27 physical first-BZ momenta.

    The momentum-class representatives used internally by the finite torus
    need not lie in the first BZ.  Although shifting such a representative by
    a primitive reciprocal vector is exact in a complete plane-wave basis, it
    is not exact after the finite G-shell cutoff.  Solve at the plotted first-
    BZ k itself so that k in exp(i(k+G).r) is exactly the observable momentum.
    """
    cache = DATA / "qns27_bloch_5band_first_bz.npz"
    model, _ = inputs()
    momenta = _first_bz_momenta()
    if not cache.exists():
        energies = np.empty((len(momenta), 5), dtype=float)
        vectors = np.empty((len(momenta), model.dim, 5), dtype=np.complex128)
        for index, momentum in enumerate(momenta):
            values, states = model.solve(momentum)
            energies[index] = values[:5]
            vectors[index] = states[:, :5]
        np.savez_compressed(
            cache,
            energies=energies,
            vectors=vectors,
            k_points=momenta,
            matrix=cluster().matrix,
            theta=np.asarray(2.6),
            plane_wave_shells=np.asarray(model.params.plane_wave_shells),
            momentum_convention=np.asarray("physical-first-bz"),
        )
    with np.load(cache) as payload:
        if not np.array_equal(payload["matrix"], cluster().matrix):
            raise ValueError("cached 27-cell Bloch mesh has the wrong supercell")
        if not np.allclose(payload["k_points"], momenta, rtol=0.0, atol=1e-14):
            raise ValueError("cached Bloch orbitals are not on the plotted BZ mesh")
        if str(payload["momentum_convention"]) != "physical-first-bz":
            raise ValueError("cached Bloch orbitals use the wrong k convention")
        vectors = np.asarray(payload["vectors"])
    coefficients = (
        vectors[:, :, :5].transpose(0, 2, 1).reshape(27 * 5, 2, model.n_g)
    )
    return coefficients, momenta


def _orbital_values(
    positions: np.ndarray,
    layers: np.ndarray,
    coefficients: np.ndarray,
    momenta: np.ndarray,
    batch: int,
) -> np.ndarray:
    model, qns_inputs = inputs()
    n_momenta = len(momenta)
    if coefficients.shape != (n_momenta * 5, 2, model.n_g):
        raise ValueError("Bloch coefficient array has an incompatible shape")
    band_coefficients = coefficients.reshape(n_momenta, 5, 2, model.n_g)
    plane_momenta = momenta[:, None, :] + model.g_vectors[None, :, :]
    result = []
    for start in range(0, len(positions), batch):
        stop = min(start + batch, len(positions))
        cartesian = np.einsum(
            "ac,bc->ba", qns_inputs["supercell_lattice"], positions[start:stop]
        )
        phase = np.einsum("ba,kga->bkg", cartesian, plane_momenta)
        plane_waves = np.exp(1j * phase)
        # The five bands at fixed k share the same exp(i(k+G).r).  Contract
        # that phase once per k and G, then select the particle's layer.  This
        # is algebraically identical to the v3 orbital-by-orbital expression.
        values_by_layer = np.stack([
            np.einsum(
                "bkg,kfg->bkf", plane_waves,
                band_coefficients[:, :, layer, :], optimize=True,
            )
            for layer in range(2)
        ], axis=1)
        row = np.arange(stop - start)
        values = values_by_layer[row, layers[start:stop]]
        result.append(values.reshape(stop - start, n_momenta * 5))
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


def _physical_rotation_residual(
    values: np.ndarray, points: np.ndarray, turns: int = 1
) -> tuple[float, int]:
    """C3 residual only where the rotated physical q is also in the grid.

    The 27 ED class representatives are not closed under a geometric C3
    rotation at the hexagon boundary. Missing partners are excluded rather
    than silently replaced by q+G.
    """
    angle = 2.0 * np.pi * turns / 3.0
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    rotated = np.asarray(points) @ rotation.T
    distances = np.linalg.norm(
        rotated[:, None, :] - np.asarray(points)[None, :, :], axis=2
    )
    partner = np.argmin(distances, axis=1)
    matched = distances[np.arange(len(points)), partner] < 2.0e-12
    if not np.any(matched):
        raise ValueError("Fourier grid has no physical C3 partners")
    residual = np.max(np.abs(
        np.asarray(values)[matched] - np.asarray(values)[partner[matched]]
    ))
    return float(residual), int(np.count_nonzero(matched))


def _density_pair_structure_factor(
    positions: np.ndarray,
    supercell: np.ndarray,
    q_vectors: np.ndarray,
    batch: int = 64,
) -> np.ndarray:
    """Fourier transform the periodic connected density-pair measure.

    Pair displacements are first wrapped into the C6-symmetric Wigner--Seitz
    cell. The uniform-background transform is then subtracted. This makes
    continuous-q sampling independent of the parallelogram coordinate window,
    gives S(0)=0, and reduces exactly to the usual full structure factor at
    nonzero reciprocal vectors of the finite torus.
    """
    first, second = supercell[:, 0], supercell[:, 1]
    vertices = np.asarray([
        (first + second) / 3.0,
        (2.0 * first - second) / 3.0,
        (first - 2.0 * second) / 3.0,
        -(first + second) / 3.0,
        (-2.0 * first + second) / 3.0,
        (-first + 2.0 * second) / 3.0,
    ])
    vertices = vertices[np.argsort(np.arctan2(vertices[:, 1], vertices[:, 0]))]
    following = np.roll(vertices, -1, axis=0)
    area = 0.5 * abs(np.sum(
        vertices[:, 0] * following[:, 1]
        - vertices[:, 1] * following[:, 0]
    ))
    if not np.isclose(area, abs(np.linalg.det(supercell)), rtol=2.0e-14):
        raise RuntimeError("Wigner--Seitz polygon has the wrong area")

    polygon_transform = np.empty(len(q_vectors), dtype=np.complex128)
    for q_index, q_vector in enumerate(q_vectors):
        norm2 = float(q_vector @ q_vector)
        if norm2 < 1.0e-28:
            polygon_transform[q_index] = area
            continue
        integral = 0.0j
        for left, right in zip(vertices, following):
            edge = right - left
            phase_change = float(q_vector @ edge)
            edge_integral = np.exp(1j * (q_vector @ left))
            if abs(phase_change) > 1.0e-12:
                edge_integral *= np.expm1(1j * phase_change) / (1j * phase_change)
            normal_length = np.asarray([edge[1], -edge[0]])
            integral += (
                (q_vector @ normal_length) * edge_integral / (1j * norm2)
            )
        polygon_transform[q_index] = integral
    if np.max(np.abs(polygon_transform.imag)) > 2.0e-11 * area:
        raise RuntimeError("centered Wigner--Seitz transform is not real")

    particles = positions.shape[1]
    background = particles * polygon_transform.real / area
    shifts = np.asarray([
        (first_index, second_index)
        for first_index in (-1, 0, 1)
        for second_index in (-1, 0, 1)
    ])
    result = np.empty((len(positions), len(q_vectors)), dtype=float)
    for start in range(0, len(positions), batch):
        stop = min(start + batch, len(positions))
        current = positions[start:stop]
        fractional = current[:, :, None, :] - current[:, None, :, :]
        candidates = np.einsum(
            "...c,ac->...a", fractional[..., None, :] - shifts, supercell
        )
        squared = np.sum(candidates * candidates, axis=-1)
        nearest = np.argmin(squared, axis=-1)
        displacement = np.take_along_axis(
            candidates, nearest[..., None, None], axis=3
        ).squeeze(3)
        phase = np.einsum("bija,qa->bijq", displacement, q_vectors)
        result[start:stop] = (
            np.cos(phase).sum(axis=(1, 2)) / particles - background
        )
    result[:, 0] = 0.0
    return result


def _coordinate_observables(positions: np.ndarray, blocks: int) -> dict[str, np.ndarray]:
    _, qns_inputs = inputs()
    supercell = np.asarray(qns_inputs["supercell_lattice"])
    cartesian = np.einsum("ac,bnc->bna", supercell, positions)
    # Sample physical q directly on a uniform triangular grid. Periodic pair
    # displacements and the uniform-background subtraction are performed
    # before the transform; no point is obtained through q+G.
    q_vectors = uniform_first_bz_grid()
    structure_per_sample = _density_pair_structure_factor(
        positions, supercell, q_vectors
    )
    structure = structure_per_sample.mean(axis=0)

    # The plotted full S(q) uses the 27 physical vectors drawn for ED.
    ed_q_vectors = _first_bz_momenta()
    ed_full_per_sample = _density_pair_structure_factor(
        positions, supercell, ed_q_vectors
    )
    ed_full = ed_full_per_sample.mean(axis=0)
    ed_full[0] = 0.0
    ed_phase = np.einsum("bna,qa->bnq", cartesian, ed_q_vectors)
    ed_rho_q = np.exp(1j * ed_phase).sum(axis=1)
    particles = positions.shape[1]
    amplitude = np.abs(ed_rho_q.mean(axis=0)) / particles
    amplitude[0] = 0.0

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
        "structure_q_vectors": ed_q_vectors,
        "charge_structure_factor": ed_full.real,
        "charge_structure_factor_sem": _block_sem(
            ed_full_per_sample, blocks
        ),
        "structure_q_vectors_uniform_auxiliary": q_vectors,
        "charge_structure_factor_uniform_auxiliary": structure.real,
        "charge_structure_factor_uniform_auxiliary_sem": _block_sem(
            structure_per_sample, blocks
        ),
        "density_fourier_q_vectors": ed_q_vectors,
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
    draw_index = np.arange(draws)[:, None]
    particle_offset = rng.integers(spec.n_particles, size=(1, count))
    particle = (draw_index + particle_offset) % spec.n_particles
    layer_offset = rng.integers(2, size=(1, count))
    proposal_layer = (draw_index + layer_offset) % 2
    golden = (np.sqrt(5.0) - 1.0) / 2.0
    low_discrepancy = np.column_stack((
        (np.arange(draws) + 0.5) / draws,
        np.remainder((np.arange(draws) + 0.5) * golden, 1.0),
    ))
    proposal_position = np.remainder(
        low_discrepancy[:, None, :] + rng.random((1, count, 2)), 1.0
    )
    expanded_positions = np.broadcast_to(
        positions[None], (draws, *positions.shape)
    ).copy()
    expanded_layers = np.broadcast_to(layers[None], (draws, *layers.shape)).copy()
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
    momentum_first_five_raw = momentum_first_five_per_sample.mean(axis=0)
    momentum_first_five_raw_sem = _block_sem(
        momentum_first_five_per_sample, args.error_blocks
    )
    projected_trace_per_sample = momentum_first_five_per_sample.sum(axis=1)
    projected_trace_sem = float(_block_sem(
        projected_trace_per_sample, args.error_blocks
    ))
    momentum_first_five, momentum_first_five_sem = _filling_normalized(
        momentum_first_five_per_sample, spec.n_particles, args.error_blocks
    )
    coordinate = _coordinate_observables(positions, args.error_blocks)
    structure_c3_residual, structure_c3_matched = _physical_rotation_residual(
        coordinate["charge_structure_factor"],
        coordinate["structure_q_vectors"],
    )

    payload = {
        "n_particles": np.asarray(spec.n_particles, dtype=np.int32),
        "one_body_density_diagonal": diagonal,
        "band_weight": band_weight,
        "band_weight_sem": band_weight_sem,
        "momentum_occupation_band1_raw": momentum_band1_raw,
        "momentum_occupation_band1_raw_sem": momentum_band1_raw_sem,
        "momentum_occupation_band1": momentum_band1,
        "momentum_occupation_band1_sem": momentum_band1_sem,
        "momentum_occupation_first_five_raw": momentum_first_five_raw,
        "momentum_occupation_first_five_raw_sem": momentum_first_five_raw_sem,
        "momentum_occupation_first_five_raw_sum_sem": np.asarray(
            projected_trace_sem
        ),
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
        "auxiliary_sampling": (
            "Cranley-Patterson shifted low-discrepancy positions with balanced "
            "particle and layer indices"
        ),
        "sampling": args.sampling_description,
        "one_body_estimator": (
            "Bloch-projected one-body density-matrix diagonal from complete "
            "one-particle wavefunction ratios and bare orbitals solved directly "
            "at the 27 physical first-BZ k points (G occurs only inside each "
            "Bloch orbital): plotted n_1(k) and the plotted sum over bands 1:5 "
            "are raw physical projections; conditionally normalized variants "
            "are saved only for audit; no C3 or ED-profile fit"
        ),
        "structure_factor_estimator": (
            "Luo full S(q)=<|sum_i exp(i q.r_i)|^2>/N_e for q!=0, "
            "evaluated directly from the supplied final-wavefunction walkers at "
            "the 27 displayed ED first-BZ q vectors; S(0)=0 and no q+G "
            "substitution or Bloch-band projection is applied"
        ),
        "structure_factor_grid_points": int(
            len(coordinate["structure_q_vectors"])
        ),
        "structure_factor_projected_density": False,
        "structure_factor_periodic_wigner_seitz": True,
        "structure_factor_c3_matched_physical_points": structure_c3_matched,
        "band_weight": band_weight.tolist(),
        "band_weight_sem": band_weight_sem.tolist(),
        "projected_trace": float(diagonal.sum()),
        "momentum_occupation_band1_raw_sum": float(momentum_band1_raw.sum()),
        "momentum_occupation_band1_normalized_sum": float(momentum_band1.sum()),
        "momentum_occupation_first_five_raw_sum": float(
            momentum_first_five_raw.sum()
        ),
        "momentum_occupation_first_five_raw_sum_sem": projected_trace_sem,
        "momentum_occupation_first_five_normalized_sum": float(
            momentum_first_five.sum()
        ),
        "momentum_occupation_first_five_sum": float(momentum_first_five.sum()),
        "momentum_occupation_band1_raw_c3_residual": float(np.max(np.abs(
            momentum_band1_raw - momentum_band1_raw[
                cluster().rotation60[cluster().rotation60]
            ]
        ))),
        "structure_factor_raw_c3_residual": structure_c3_residual,
        "saved_base_wavefunction_relative_error": base_logpsi_max_error,
        "maximum_nonzero_structure_factor": float(
            coordinate["charge_structure_factor"].max()
        ),
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
    result.add_argument(
        "--sampling-description",
        default="saved final training walkers; no independent validation chain",
    )
    return result


def main() -> None:
    print(json.dumps(run(parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
