"""Symmetry-resolved ED for nu=1/3 and the one-band theta comparison."""

from __future__ import annotations

import json
import os
from itertools import combinations
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np

from .continuum import ContinuumModel, ContinuumParameters
from .multiband_ed import build_pair_interaction, mesh_bloch_states
from .sparse_ed import diagonalize_sparse_sector


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "result" / "data"


def restrict_operator(
    pair5: np.ndarray, pairs5: np.ndarray, n_bands: int
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    keep = np.asarray([
        (first % 5 < n_bands) and (second % 5 < n_bands)
        for first, second in pairs5
    ])
    selected = np.flatnonzero(keep)
    pairs = [
        ((int(first) // 5) * n_bands + int(first) % 5,
         (int(second) // 5) * n_bands + int(second) % 5)
        for first, second in pairs5[keep]
    ]
    expected = list(combinations(range(9 * n_bands), 2))
    if pairs != expected:
        raise RuntimeError("restricted pair ordering is inconsistent")
    return np.ascontiguousarray(pair5[np.ix_(selected, selected)]), pairs


def solve_all_sectors(
    energies: np.ndarray,
    pair: np.ndarray,
    pairs: list[tuple[int, int]],
    n_bands: int,
    n_holes: int,
    eigenvalues_per_sector: int = 6,
) -> tuple[list[dict], dict]:
    records = []
    ground = None
    for k1 in range(3):
        for k2 in range(3):
            sector = diagonalize_sparse_sector(
                energies, pair, pairs, 3, 3, n_bands, n_holes, (k1, k2),
                n_eigenvalues=eigenvalues_per_sector,
                tolerance=1.0e-11, max_iterations=1800,
            )
            records.append({
                "momentum": [k1, k2],
                "dimension": len(sector.basis),
                "energies_meV": sector.energies.tolist(),
                "residual_meV": sector.residual,
            })
            candidate = (float(sector.energies[0]), sector)
            if ground is None or candidate[0] < ground[0]:
                ground = candidate
    assert ground is not None
    energy, sector = ground
    state = {
        "energy_meV": np.asarray(energy),
        "momentum": np.asarray(sector.momentum, dtype=np.int32),
        "basis": sector.basis,
        "amplitudes": sector.vectors[:, 0],
    }
    return records, state


def main() -> None:
    cache = np.load(DATA / "fig2_5band_operator_cache.npz")
    energies3deg = np.asarray(cache["energies"])
    pair5 = np.asarray(cache["pair"])
    pairs5 = np.asarray(cache["pairs"])

    one_third = {"theta_deg": 3.0, "n_holes": 3, "symmetries": {
        "charge_U1": "fixed N_h=3",
        "spin_U1": "fully polarized fixed S_z=3/2",
        "translations": "nine independent (K1,K2) sectors",
    }, "projections": {}}
    for n_bands in range(1, 6):
        pair, pairs = restrict_operator(pair5, pairs5, n_bands)
        records, state = solve_all_sectors(
            energies3deg, pair, pairs, n_bands, 3, eigenvalues_per_sector=6
        )
        energy = float(state["energy_meV"])
        one_third["projections"][str(n_bands)] = {
            "ground_energy_meV": energy,
            "ground_energy_per_hole_meV": energy / 3.0,
            "ground_momentum": state["momentum"].tolist(),
            "sectors": records,
        }
        np.savez_compressed(DATA / f"nu1of3_{n_bands}band_ground_state.npz", **state)
    (DATA / "nu1of3_multiband_summary.json").write_text(
        json.dumps(one_third, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    comparison = {"cluster": "3x3", "dielectric_constant": 5.0,
                  "projection": "one band", "fillings": {}}
    for theta in (2.6, 3.0):
        if theta == 3.0:
            energies = energies3deg[:, :, :1]
            pair, pairs = restrict_operator(pair5, pairs5, 1)
        else:
            model = ContinuumModel(ContinuumParameters(theta_deg=theta, plane_wave_shells=8))
            energies, vectors, _ = mesh_bloch_states(model, 3, 3, 1)
            pair, pairs, _ = build_pair_interaction(model, vectors, 3, 3, 1, 5.0, 8)
        comparison["fillings"][str(theta)] = {}
        for label, n_holes in (("1/3", 3), ("2/3", 6)):
            records, _ = solve_all_sectors(
                energies, pair, pairs, 1, n_holes, eigenvalues_per_sector=6
            )
            ground = min(record["energies_meV"][0] for record in records)
            levels = sorted(
                (energy - ground, record["momentum"])
                for record in records for energy in record["energies_meV"]
            )
            comparison["fillings"][str(theta)][label] = {
                "sectors": records,
                "lowest_six": [
                    {"E_minus_EGS_meV": value, "momentum": momentum}
                    for value, momentum in levels[:6]
                ],
            }
    (DATA / "theta2p6_vs_3_oneband_spectra.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
