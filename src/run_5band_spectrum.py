"""Five-band momentum spectrum using the three C6-inequivalent 3x3 sectors."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
from scipy.sparse.linalg import eigsh

from .continuum import ContinuumModel, ContinuumParameters
from .multiband_ed import build_pair_interaction
from .run_5band_gamma import embedded_vector
from .sparse_ed import NativeHamiltonian, diagonalize_sparse_sector, one_body_diagonal, sector_basis


ROOT = Path(__file__).resolve().parents[1]
CLASSES = {
    "Gamma": [(0, 0)],
    "K_pair": [(1, 2), (2, 1)],
    "generic_six": [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (2, 2)],
}
REPRESENTATIVES = {"Gamma": (0, 0), "K_pair": (1, 2), "generic_six": (0, 1)}


def solve_sector(
    momentum: tuple[int, int],
    energies: np.ndarray,
    pair5: np.ndarray,
    pairs5: list[tuple[int, int]],
    pair3: np.ndarray,
    pairs3: list[tuple[int, int]],
) -> dict:
    if momentum == (0, 0):
        state = np.load(ROOT / "result" / "data" / "fig2_5band_ground_state.npz")
        values = np.asarray(state["excited_energies_meV"])
        return {"energies_meV": values.tolist(), "residuals_meV": None, "dimension": 905100}
    seed3 = diagonalize_sparse_sector(
        energies, pair3, pairs3, 3, 3, 3, 6, momentum,
        n_eigenvalues=1, tolerance=1.0e-11, max_iterations=1200,
    )
    basis5 = sector_basis(3, 3, 5, 6, momentum)
    diagonal5 = one_body_diagonal(basis5, energies[:, :, :5].reshape(-1))
    initial = embedded_vector(seed3.basis, seed3.vectors[:, 0], 3, basis5, 5)
    hamiltonian = NativeHamiltonian(basis5, diagonal5, pair5, pairs5)
    try:
        values, vectors = eigsh(
            hamiltonian.operator,
            k=3,
            which="SA",
            v0=initial,
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
    return {"energies_meV": values.tolist(), "residuals_meV": residuals, "dimension": len(basis5)}


def main() -> None:
    start = time.perf_counter()
    data_dir = ROOT / "result" / "data"
    cache = np.load(data_dir / "fig2_5band_operator_cache.npz")
    energies = np.asarray(cache["energies"])
    vectors = np.asarray(cache["vectors"])
    pair5 = np.asarray(cache["pair"])
    pairs5 = [(int(a), int(b)) for a, b in np.asarray(cache["pairs"])]
    model = ContinuumModel(ContinuumParameters(plane_wave_shells=8))
    pair3, pairs3, pair3_hermiticity = build_pair_interaction(model, vectors, 3, 3, 3, 5.0, 8)
    representative_results = {}
    for name, momentum in REPRESENTATIVES.items():
        representative_results[name] = solve_sector(
            momentum, energies, pair5, pairs5, pair3, pairs3
        )
        representative_results[name]["representative_momentum"] = list(momentum)
    sectors = []
    for name, momenta in CLASSES.items():
        for momentum in momenta:
            sectors.append({"momentum": list(momentum), "symmetry_class": name,
                            **representative_results[name]})
    sectors.sort(key=lambda value: tuple(value["momentum"]))
    gamma_energy = representative_results["Gamma"]["energies_meV"][0]
    summary = {
        "method": "three C6-inequivalent sectors; symmetry-related momenta replicated",
        "representatives": representative_results,
        "sectors": sectors,
        "ground_energy_meV": gamma_energy,
        "ground_energy_per_hole_meV": gamma_energy / 6.0,
        "paper_fig2_digitized_5band_meV_per_hole": -52.69,
        "difference_from_paper_meV_per_hole": gamma_energy / 6.0 + 52.69,
        "pair3_hermiticity_error": pair3_hermiticity,
        "wall_clock_seconds": time.perf_counter() - start,
    }
    (data_dir / "fig2_5band_momentum_spectrum.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
