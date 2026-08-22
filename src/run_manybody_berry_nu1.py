"""Twisted-boundary non-Abelian Berry curvature of the 27-cell nu=1/3 CDW."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "12")

import numpy as np

from .c6_ed import TriangularSupercell, build_cluster_pair_interaction
from .continuum import ContinuumModel, ContinuumParameters
from .group_sparse_ed import all_group_sector_bases, diagonalize_sparse_basis
from .run_manybody_berry import twisted_bloch_states


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "result" / "data"
CACHE = DATA / "manybody_berry_27cell_nu1of3"
OUTPUT = DATA / "manybody_berry_27cell_nu1of3.json"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def point_path(i: int, j: int, grid: int) -> Path:
    return CACHE / f"grid{grid}_point_{i:02d}_{j:02d}.npz"


def solve_point(
    i: int,
    j: int,
    grid: int,
    model: ContinuumModel,
    cluster: TriangularSupercell,
    bases: list[np.ndarray],
    sectors: list[int],
    force: bool,
) -> None:
    path = point_path(i, j, grid)
    spectral_cut = i == 0 or j == 0
    eigenvalue_count = 4 if spectral_cut else 1
    if path.exists() and not force:
        cached_count = int(np.count_nonzero(np.isfinite(np.load(path)["energies_0"])))
        if cached_count >= eigenvalue_count:
            log(f"cached twist ({i}/{grid}, {j}/{grid}), {cached_count} level(s) per sector")
            return
        log(f"upgrading twist ({i}/{grid}, {j}/{grid}) from {cached_count} to {eigenvalue_count} levels per sector")
    phi1 = 2.0 * np.pi * i / grid
    phi2 = 2.0 * np.pi * j / grid
    log(f"Bloch/projected interaction at twist ({i}/{grid}, {j}/{grid})")
    orbital_energies, vectors = twisted_bloch_states(
        model, cluster, phi1, phi2
    )
    pair, pairs, hermiticity = build_cluster_pair_interaction(
        model, cluster, vectors, 1, dielectric=5.0, reciprocal_shell_count=8
    )
    arrays: dict[str, np.ndarray] = {
        "phi": np.array([phi1, phi2]),
        "bloch_vectors": vectors[:, :, 0],
        "hermiticity": np.asarray(hermiticity),
        "sectors": np.asarray(sectors, dtype=np.int32),
    }
    residuals = []
    for index, sector in enumerate(sectors):
        np.random.seed(130513 + 101 * i + 7 * j + index)
        log(
            f"Lanczos sector {sector:02d}: dim={len(bases[sector]):,}, "
            + ("ground plus nearest same-sector excitation" if spectral_cut else "ground state")
        )
        result = diagonalize_sparse_basis(
            bases[sector],
            orbital_energies,
            pair,
            pairs,
            momentum=(sector, 0),
            n_eigenvalues=eigenvalue_count,
            tolerance=2.0e-9,
            max_iterations=2200,
        )
        padded_energies = np.full(4, np.nan, dtype=float)
        padded_energies[:eigenvalue_count] = result.energies
        arrays[f"energies_{index}"] = padded_energies
        arrays[f"amplitudes_{index}"] = result.vectors[:, 0]
        residuals.append(result.residual)
    arrays["residuals"] = np.asarray(residuals)
    np.savez_compressed(path, **arrays)
    relative = np.concatenate([arrays[f"energies_{x}"] for x in range(3)])
    relative -= np.nanmin(relative)
    log(
        f"saved {path.name}; relative sampled levels="
        + ", ".join(f"{value:.5f}" for value in np.sort(relative))
        + f" meV; max residual={max(residuals):.2e}"
    )


def load_point(i: int, j: int, grid: int) -> dict:
    cache = np.load(point_path(i, j, grid))
    return {
        "phi": np.asarray(cache["phi"], dtype=float),
        "bloch_vectors": np.asarray(cache["bloch_vectors"], dtype=np.complex128),
        "energies": np.asarray([
            np.pad(
                np.asarray(cache[f"energies_{x}"], dtype=float),
                (0, 4 - len(cache[f"energies_{x}"])),
                constant_values=np.nan,
            )
            for x in range(3)
        ], dtype=float),
        "amplitudes": [
            np.asarray(cache[f"amplitudes_{x}"], dtype=np.complex128)
            for x in range(3)
        ],
        "residuals": np.asarray(cache["residuals"], dtype=float),
        "hermiticity": float(cache["hermiticity"]),
    }


def electron_weights(
    basis: np.ndarray, left_bloch: np.ndarray, right_bloch: np.ndarray
) -> np.ndarray:
    orbital_overlap = np.einsum(
        "ij,ij->i", left_bloch.conj(), right_bloch, optimize=True
    )
    if np.min(np.abs(orbital_overlap)) < 1.0e-10:
        raise RuntimeError("adjacent-twist one-particle overlap is singular")
    weights = np.ones(len(basis), dtype=np.complex128)
    for orbital, overlap in enumerate(orbital_overlap):
        occupied = (
            (basis >> np.uint64(orbital)) & np.uint64(1)
        ).astype(bool)
        weights[occupied] *= overlap
    return weights


def bundle_link(
    left: dict, right: dict, bases: list[np.ndarray], sectors: list[int]
) -> tuple[complex, float]:
    determinant = 1.0 + 0.0j
    minimum = 1.0
    for index, sector in enumerate(sectors):
        weights = electron_weights(
            bases[sector], left["bloch_vectors"], right["bloch_vectors"]
        )
        overlap = np.vdot(
            left["amplitudes"][index],
            weights * right["amplitudes"][index],
        )
        determinant *= overlap
        minimum = min(minimum, abs(overlap))
    if abs(determinant) < 1.0e-14:
        raise RuntimeError("CDW bundle link is singular")
    return determinant / abs(determinant), minimum


def assemble(
    grid: int,
    cluster: TriangularSupercell,
    bases: list[np.ndarray],
    sectors: list[int],
) -> dict:
    points = [
        [load_point(i, j, grid) for j in range(grid + 1)]
        for i in range(grid + 1)
    ]
    link1 = np.empty((grid, grid + 1), dtype=np.complex128)
    link2 = np.empty((grid + 1, grid), dtype=np.complex128)
    min_overlap = 1.0
    for i in range(grid):
        for j in range(grid + 1):
            link1[i, j], overlap = bundle_link(
                points[i][j], points[i + 1][j], bases, sectors
            )
            min_overlap = min(min_overlap, overlap)
    for i in range(grid + 1):
        for j in range(grid):
            link2[i, j], overlap = bundle_link(
                points[i][j], points[i][j + 1], bases, sectors
            )
            min_overlap = min(min_overlap, overlap)

    flux = np.empty((grid, grid), dtype=float)
    for i in range(grid):
        for j in range(grid):
            loop = (
                link1[i, j]
                * link2[i + 1, j]
                * np.conj(link1[i, j + 1])
                * np.conj(link2[i, j])
            )
            flux[i, j] = np.angle(loop)
    step = 2.0 * np.pi / grid
    energies = np.asarray(
        [[points[i][j]["energies"] for j in range(grid + 1)]
         for i in range(grid + 1)]
    )
    sorted_levels = np.sort(energies.reshape(grid + 1, grid + 1, 12), axis=2)
    manifold_gap = sorted_levels[:, :, 3] - sorted_levels[:, :, 2]
    residuals = np.asarray(
        [[points[i][j]["residuals"].max() for j in range(grid + 1)]
         for i in range(grid + 1)]
    )
    hermiticity = np.asarray(
        [[points[i][j]["hermiticity"] for j in range(grid + 1)]
         for i in range(grid + 1)]
    )
    chern = float(flux.sum() / (2.0 * np.pi))
    result = {
        "method": (
            "determinant link of the Gamma, K, Kprime CDW ground-state "
            "manifold under boundary twists"
        ),
        "paper_benchmark": "arXiv:2503.13585v3 Supplementary Fig. 5",
        "theta_deg": 2.6,
        "dielectric_constant": 5.0,
        "bands": 1,
        "cluster_cells": 27,
        "supercell_matrix": cluster.matrix.tolist(),
        "filling": "1/3",
        "electron_number": 9,
        "manifold_sectors": sectors,
        "twist_plaquettes_per_direction": grid,
        "twist_step_radians": step,
        "plaquette_flux_radians": flux.tolist(),
        "berry_curvature": (flux / step**2).tolist(),
        "bundle_chern_number": chern,
        "hall_conductance_e2_over_h": chern / 3.0,
        "minimum_sampled_manifold_gap_meV": float(np.nanmin(manifold_gap)),
        "maximum_sampled_manifold_gap_meV": float(np.nanmax(manifold_gap)),
        "minimum_link_overlap": min_overlap,
        "maximum_lanczos_residual_meV": float(residuals.max()),
        "maximum_operator_hermiticity_error": float(hermiticity.max()),
        "sector_resolved_two_lowest_energies_meV": energies[:, :, :, :2].tolist(),
        "sector_resolved_sampled_energies_meV": energies.tolist(),
        "gap_scope_note": (
            "The plotted excitation is the nearest state sampled within "
            "Gamma, K, Kprime; the zero-twist all-sector spectrum is Fig. 3."
        ),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log(
        f"bundle Chern={chern:.8f}; sampled minimum gap="
        f"{np.nanmin(manifold_gap):.6f} meV; minimum link overlap={min_overlap:.3e}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--assemble-only", action="store_true")
    args = parser.parse_args()
    if args.grid < 2:
        raise ValueError("grid must be at least two")
    CACHE.mkdir(parents=True, exist_ok=True)
    cluster = TriangularSupercell.c6_27()
    bases = all_group_sector_bases(
        cluster.numerators, cluster.determinant, cluster.residue_lookup, 1, 9
    )
    sectors = [
        0,
        cluster.index_of_fraction(2, 1, 3),
        cluster.index_of_fraction(1, 2, 3),
    ]
    log(f"CDW manifold sectors={sectors}")
    if not args.assemble_only:
        model = ContinuumModel(
            ContinuumParameters(theta_deg=2.6, plane_wave_shells=8)
        )
        for i in range(args.grid + 1):
            for j in range(args.grid + 1):
                solve_point(
                    i, j, args.grid, model, cluster, bases, sectors, args.force
                )
    assemble(args.grid, cluster, bases, sectors)


if __name__ == "__main__":
    main()
