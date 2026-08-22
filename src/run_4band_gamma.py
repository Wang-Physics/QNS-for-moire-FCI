"""Compute the missing four-band Gamma-sector ED point from the validated five-band cache."""

from __future__ import annotations

import json
import os
import time
from itertools import combinations
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
from scipy.sparse.linalg import eigsh

from .run_5band_gamma import embedded_vector
from .sparse_ed import NativeHamiltonian, one_body_diagonal, sector_basis


ROOT = Path(__file__).resolve().parents[1]


def restrict_five_to_four_bands(
    pair5: np.ndarray, pairs5: np.ndarray
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    keep = np.asarray([
        (first % 5 < 4) and (second % 5 < 4) for first, second in pairs5
    ])
    selected = np.flatnonzero(keep)
    pairs4 = [
        ((int(first) // 5) * 4 + int(first) % 5,
         (int(second) // 5) * 4 + int(second) % 5)
        for first, second in pairs5[keep]
    ]
    expected = list(combinations(range(36), 2))
    if pairs4 != expected:
        raise RuntimeError("four-band pair ordering is inconsistent")
    return np.ascontiguousarray(pair5[np.ix_(selected, selected)]), pairs4


def main() -> None:
    start = time.perf_counter()
    data_dir = ROOT / "result" / "data"
    cache = np.load(data_dir / "fig2_5band_operator_cache.npz")
    energies = np.asarray(cache["energies"])
    pair4, pairs4 = restrict_five_to_four_bands(
        np.asarray(cache["pair"]), np.asarray(cache["pairs"])
    )
    basis4 = sector_basis(3, 3, 4, 6, (0, 0))
    diagonal4 = one_body_diagonal(basis4, energies[:, :, :4].reshape(-1))
    state3 = np.load(data_dir / "fig2_3band_ground_state.npz")
    initial = embedded_vector(state3["basis"], state3["amplitudes"], 3, basis4, 4)
    hamiltonian = NativeHamiltonian(basis4, diagonal4, pair4, pairs4)
    try:
        values, vectors = eigsh(
            hamiltonian.operator, k=1, which="SA", v0=initial,
            tol=1.0e-8, maxiter=600, ncv=48,
        )
        order = np.argsort(values)
        values, vectors = values[order], vectors[:, order]
        residual = float(np.linalg.norm(
            hamiltonian.matvec(vectors[:, 0]) - values[0] * vectors[:, 0]
        ))
    finally:
        hamiltonian.close()
    summary = {
        "n_bands": 4,
        "momentum": [0, 0],
        "sector_dimension": len(basis4),
        "hilbert_dimension": 1947792,
        "ground_energy_meV": float(values[0]),
        "ground_energy_per_hole_meV": float(values[0] / 6.0),
        "residual_meV": residual,
        "construction": "exact first-four-band restriction of validated five-band operator",
        "wall_clock_seconds": time.perf_counter() - start,
    }
    np.savez_compressed(
        data_dir / "fig2_4band_ground_state.npz",
        energy_meV=values[0], momentum=np.asarray([0, 0], dtype=np.int32),
        basis=basis4, amplitudes=vectors[:, 0],
    )
    (data_dir / "fig2_4band_gamma_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
