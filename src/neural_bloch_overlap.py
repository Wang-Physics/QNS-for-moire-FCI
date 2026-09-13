"""Coordinate-space overlap of neural Bloch and 1--5-band ED ground states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .run_neural_bloch import DATA, OUTPUT, ROOT, load_checkpoint


def bits_to_occupied(
    basis: np.ndarray, n_orbitals: int, n_particles: int
) -> np.ndarray:
    """Convert canonical Fock bitstrings to occupied-orbital index rows."""
    result = np.empty((len(basis), n_particles), dtype=np.int64)
    for row, state in enumerate(np.asarray(basis, dtype=np.uint64)):
        occupied = [
            orbital
            for orbital in range(n_orbitals)
            if (int(state) >> orbital) & 1
        ]
        if len(occupied) != n_particles:
            raise ValueError("bitstring has the wrong particle number")
        result[row] = occupied
    return result

@torch.no_grad()
def bloch_orbital_values(
    positions: torch.Tensor,
    layers: torch.Tensor,
    n_bands: int,
    continuum,
) -> torch.Tensor:
    cache = np.load(DATA / "fig2_5band_operator_cache.npz")
    mesh = np.load(DATA / "fig2_bloch_mesh.npz")
    vectors = np.asarray(cache["vectors"])[..., :n_bands]
    coefficients = (
        vectors.reshape(9, continuum.dim, n_bands)
        .transpose(0, 2, 1)
        .reshape(9 * n_bands, 2, continuum.n_g)
    )
    momenta = np.repeat(
        np.asarray(mesh["k_points"]).reshape(9, 2), n_bands, axis=0
    )
    coefficient_tensor = torch.as_tensor(
        coefficients, dtype=torch.complex128, device=positions.device
    )
    momentum_tensor = torch.as_tensor(
        momenta, dtype=positions.dtype, device=positions.device
    )
    g_tensor = torch.as_tensor(
        continuum.g_vectors, dtype=positions.dtype, device=positions.device
    )
    reciprocal = np.stack([continuum.b1, continuum.b2])
    primitive = 2.0 * np.pi * np.linalg.inv(reciprocal)
    supercell = torch.as_tensor(
        primitive @ np.diag([3.0, 3.0]), dtype=positions.dtype,
        device=positions.device,
    )
    cartesian = torch.einsum("ac,bnc->bna", supercell, positions)
    plane_momentum = momentum_tensor[:, None] + g_tensor[None]
    phase = torch.einsum("bna,oga->bnog", cartesian, plane_momentum)
    plane_wave = torch.exp(1j * phase)
    selected = coefficient_tensor.permute(1, 0, 2)[layers]
    return torch.sum(selected * plane_wave, dim=-1)


@torch.no_grad()
def ed_coordinate_wavefunction(
    orbital_values: torch.Tensor,
    basis: np.ndarray,
    amplitudes: np.ndarray,
    chunk_size: int,
) -> torch.Tensor:
    batch, particles, orbitals = orbital_values.shape
    occupied = bits_to_occupied(basis, orbitals, particles)
    result = torch.zeros(batch, dtype=torch.complex128)
    for start in range(0, len(occupied), chunk_size):
        selected = torch.as_tensor(
            occupied[start : start + chunk_size], dtype=torch.long,
            device=orbital_values.device,
        )
        count = len(selected)
        matrices = orbital_values[:, :, selected.reshape(-1)]
        matrices = matrices.reshape(batch, particles, count, particles)
        matrices = matrices.permute(0, 2, 1, 3)
        determinants = torch.linalg.det(matrices)
        coefficient = torch.as_tensor(
            amplitudes[start : start + count], dtype=torch.complex128,
            device=orbital_values.device,
        )
        result = result + torch.sum(determinants * coefficient[None], dim=1)
    return result


def block_uncertainty(ratio: np.ndarray, blocks: int = 8) -> tuple[float, float]:
    def estimate(values: np.ndarray) -> float:
        return float(
            abs(np.mean(values)) ** 2
            / max(float(np.mean(np.abs(values) ** 2)), 1.0e-300)
        )

    value = estimate(ratio)
    blocks = min(blocks, max(len(ratio) // 2, 1))
    if blocks <= 1:
        return value, float("nan")
    estimates = []
    for indices in np.array_split(np.arange(len(ratio)), blocks):
        keep = np.ones(len(ratio), dtype=bool)
        keep[indices] = False
        estimates.append(estimate(ratio[keep]))
    estimates = np.asarray(estimates)
    error = np.sqrt((blocks - 1) * np.mean((estimates - estimates.mean()) ** 2))
    return value, float(error)


def run(args: argparse.Namespace) -> dict:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    continuum, wavefunction, metadata = load_checkpoint(args.checkpoint)
    sample_cache = np.load(args.samples)
    positions_all = torch.as_tensor(sample_cache["positions"], dtype=torch.float64)
    layers_all = torch.as_tensor(sample_cache["layers"], dtype=torch.long)
    generator = np.random.default_rng(args.seed)
    permutation = generator.permutation(len(positions_all))
    sample_counts = [int(value) for value in args.samples_per_band.split(",")]
    if len(sample_counts) != 5:
        raise ValueError("--samples-per-band must contain five comma-separated counts")
    records = []
    for n_bands, requested in enumerate(sample_counts, start=1):
        count = min(requested, len(positions_all))
        indices = permutation[:count]
        positions = positions_all[indices]
        layers = layers_all[indices]
        with torch.no_grad():
            logpsi = wavefunction(positions, layers)
            neural_value = torch.exp(logpsi)
            orbital = bloch_orbital_values(
                positions, layers, n_bands, continuum
            )
        state = np.load(DATA / f"fig2_{n_bands}band_ground_state.npz")
        ed_value = ed_coordinate_wavefunction(
            orbital,
            np.asarray(state["basis"]),
            np.asarray(state["amplitudes"]),
            args.configuration_chunk,
        )
        ratio = (ed_value / neural_value).cpu().numpy()
        overlap, uncertainty = block_uncertainty(ratio)
        effective_samples = float(
            abs(np.sum(np.abs(ratio) ** 2)) ** 2
            / max(np.sum(np.abs(ratio) ** 4), 1.0e-300)
        )
        records.append(
            {
                "n_bands": n_bands,
                "ed_energy_meV": float(state["energy_meV"]),
                "ed_energy_per_particle_meV": float(
                    state["energy_meV"] / metadata["n_particles"]
                ),
                "ground_state_overlap_squared": overlap,
                "block_uncertainty": uncertainty,
                "coordinate_samples": count,
                "ratio_effective_samples": effective_samples,
                "ed_sector_dimension": int(len(state["basis"])),
            }
        )
        print(json.dumps(records[-1]), flush=True)
    result = {
        "definition": (
            "|<Psi_ED^(n)|Psi_NB>|^2 estimated from samples distributed as "
            "|Psi_NB|^2; ED states are evaluated in continuous coordinates"
        ),
        "checkpoint": str(args.checkpoint),
        "samples": str(args.samples),
        "records": records,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "neural_bloch_ed_overlaps.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--checkpoint", type=Path, default=OUTPUT / "neural_bloch_best.pt"
    )
    result.add_argument(
        "--samples", type=Path, default=OUTPUT / "neural_bloch_samples.npz"
    )
    result.add_argument(
        "--samples-per-band", default="256,192,96,48,24"
    )
    result.add_argument("--configuration-chunk", type=int, default=2048)
    result.add_argument("--seed", type=int, default=913)
    result.add_argument("--threads", type=int, default=32)
    return result


def main() -> None:
    print(json.dumps(run(parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
