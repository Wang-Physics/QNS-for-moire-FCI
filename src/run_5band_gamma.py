"""Converge the five-band Gamma-sector ground state and nested-space overlaps."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
from scipy.sparse.linalg import eigsh

from .sparse_ed import NativeHamiltonian


ROOT = Path(__file__).resolve().parents[1]


def embed_basis(states: np.ndarray, source_bands: int, target_bands: int) -> np.ndarray:
    embedded = np.zeros(len(states), dtype=np.uint64)
    for source_orbital in range(9 * source_bands):
        momentum = source_orbital // source_bands
        band = source_orbital % source_bands
        target_orbital = momentum * target_bands + band
        occupied = (states >> np.uint64(source_orbital)) & np.uint64(1)
        embedded |= occupied << np.uint64(target_orbital)
    return embedded


def embedded_vector(
    source_basis: np.ndarray,
    source_vector: np.ndarray,
    source_bands: int,
    target_basis: np.ndarray,
    target_bands: int,
) -> np.ndarray:
    mapped = embed_basis(source_basis, source_bands, target_bands)
    locations = np.searchsorted(target_basis, mapped)
    if np.any(locations == len(target_basis)) or np.any(target_basis[locations] != mapped):
        raise RuntimeError("embedded determinant is absent from target momentum sector")
    result = np.zeros(len(target_basis), dtype=np.complex128)
    result[locations] = source_vector
    return result


def projection_weight(basis: np.ndarray, vector: np.ndarray, n_bands: int, kept_bands: int) -> float:
    allowed_mask = np.uint64(0)
    for momentum in range(9):
        for band in range(kept_bands):
            allowed_mask |= np.uint64(1) << np.uint64(momentum * n_bands + band)
    inside = (basis & ~allowed_mask) == 0
    return float(np.sum(np.abs(vector[inside]) ** 2))


def main() -> None:
    start = time.perf_counter()
    data_dir = ROOT / "result" / "data"
    cache = np.load(data_dir / "fig2_5band_operator_cache.npz")
    energies = np.asarray(cache["energies"])
    pair_interaction = np.asarray(cache["pair"])
    pairs_array = np.asarray(cache["pairs"])
    pairs = [(int(a), int(b)) for a, b in pairs_array]
    basis5 = np.asarray(cache["basis"], dtype=np.uint64)
    diagonal = np.asarray(cache["diagonal"], dtype=np.float64)
    state1 = np.load(data_dir / "fig2_1band_ground_state.npz")
    state3 = np.load(data_dir / "fig2_3band_ground_state.npz")
    vector1_in_5 = embedded_vector(state1["basis"], state1["amplitudes"], 1, basis5, 5)
    vector3_in_5 = embedded_vector(state3["basis"], state3["amplitudes"], 3, basis5, 5)
    hamiltonian = NativeHamiltonian(basis5, diagonal, pair_interaction, pairs)
    try:
        values, vectors = eigsh(
            hamiltonian.operator,
            k=4,
            which="SA",
            v0=vector3_in_5,
            tol=1.0e-7,
            maxiter=400,
            ncv=64,
        )
        order = np.argsort(values)
        values, vectors = values[order], vectors[:, order]
        residuals = [
            float(np.linalg.norm(hamiltonian.matvec(vectors[:, index]) - values[index] * vectors[:, index]))
            for index in range(len(values))
        ]
    finally:
        hamiltonian.close()
    ground5 = vectors[:, 0]
    overlap_1_3 = abs(np.vdot(
        embedded_vector(state1["basis"], state1["amplitudes"], 1, state3["basis"], 3),
        state3["amplitudes"],
    )) ** 2
    summary = {
        "n_bands": 5,
        "momentum": [0, 0],
        "sector_dimension": len(basis5),
        "hilbert_dimension": 8145060,
        "energies_meV": values.tolist(),
        "energies_per_hole_meV": (values / 6.0).tolist(),
        "residuals_meV": residuals,
        "overlap_squared": {
            "1_vs_3": float(overlap_1_3),
            "1_vs_5": float(abs(np.vdot(vector1_in_5, ground5)) ** 2),
            "3_vs_5": float(abs(np.vdot(vector3_in_5, ground5)) ** 2),
        },
        "projection_weight": {
            "five_band_ground_in_one_band_subspace": projection_weight(basis5, ground5, 5, 1),
            "five_band_ground_in_three_band_subspace": projection_weight(basis5, ground5, 5, 3),
        },
        "wall_clock_seconds": time.perf_counter() - start,
    }
    np.savez_compressed(
        data_dir / "fig2_5band_ground_state.npz",
        energy_meV=values[0],
        momentum=np.asarray([0, 0], dtype=np.int32),
        basis=basis5,
        amplitudes=ground5,
        excited_energies_meV=values,
    )
    (data_dir / "fig2_5band_gamma_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
