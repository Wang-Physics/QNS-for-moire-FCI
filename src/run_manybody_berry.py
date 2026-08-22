"""Twisted-boundary many-body Berry curvature for the 27-cell one-band FCI.

The nu=2/3 electron problem is solved exactly as nine holes. At every twist
(phi1, phi2), the three-state Gamma manifold is retained and the fourth state
checks the isolation gap. Gauge-invariant determinant links give the
non-Abelian Fukui-Hatsugai-Suzuki curvature of the full ground-state bundle.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "12")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

from .c6_ed import (
    TriangularSupercell,
    build_cluster_pair_interaction,
    particle_hole_transform_one_band,
)
from .continuum import ContinuumModel, ContinuumParameters
from .group_sparse_ed import all_group_sector_bases, diagonalize_sparse_basis
from .make_tmote2_report_figures import panel_label, style


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "result" / "data"
FIGURES = ROOT / "result" / "figures"
CACHE = DATA / "manybody_berry_27cell_nu2of3"
OUTPUT = DATA / "manybody_berry_27cell_nu2of3.json"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def twisted_bloch_states(
    model: ContinuumModel,
    cluster: TriangularSupercell,
    phi1: float,
    phi2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one-band states after a uniform boundary-twist momentum shift."""
    reduced_twist = np.array([phi1, phi2], dtype=float) / (2.0 * np.pi)
    delta_fraction = np.linalg.solve(cluster.matrix.T, reduced_twist)
    energies = np.empty(cluster.determinant, dtype=float)
    vectors = np.empty((cluster.determinant, model.dim, 1), dtype=np.complex128)
    for index, fraction in enumerate(cluster.fractions):
        shifted = fraction + delta_fraction
        k = shifted[0] * model.b1 + shifted[1] * model.b2
        values, states = model.solve(k)
        energies[index] = values[0]
        vectors[index, :, 0] = states[:, 0]
    return energies, vectors


def point_path(i: int, j: int, grid: int) -> Path:
    return CACHE / f"grid{grid}_point_{i:02d}_{j:02d}.npz"


def solve_point(
    i: int,
    j: int,
    grid: int,
    model: ContinuumModel,
    cluster: TriangularSupercell,
    gamma_basis: np.ndarray,
    force: bool,
) -> None:
    path = point_path(i, j, grid)
    spectral_cut = i == 0 or j == 0
    eigenvalue_count = 8 if spectral_cut else 4
    if path.exists() and not force:
        cached_count = len(np.load(path)["energies"])
        if cached_count >= eigenvalue_count:
            log(f"cached twist ({i}/{grid}, {j}/{grid}), {cached_count} levels")
            return
        log(f"upgrading twist ({i}/{grid}, {j}/{grid}) from {cached_count} to {eigenvalue_count} levels")

    phi1 = 2.0 * np.pi * i / grid
    phi2 = 2.0 * np.pi * j / grid
    log(f"Bloch/projected interaction at twist ({i}/{grid}, {j}/{grid})")
    energies, vectors = twisted_bloch_states(model, cluster, phi1, phi2)
    pair, pairs, hermiticity = build_cluster_pair_interaction(
        model, cluster, vectors, 1, dielectric=5.0, reciprocal_shell_count=8
    )
    constant, hole_energies, hole_pair = particle_hole_transform_one_band(
        energies, pair, pairs
    )
    np.random.seed(260513 + 101 * i + j)
    log(
        f"Lanczos Gamma: dim={len(gamma_basis):,}, {eigenvalue_count} states, "
        f"hermiticity={hermiticity:.2e}"
    )
    result = diagonalize_sparse_basis(
        gamma_basis,
        hole_energies,
        hole_pair,
        pairs,
        momentum=(0, 0),
        n_eigenvalues=eigenvalue_count,
        tolerance=2.0e-9,
        max_iterations=2200,
        energy_constant=constant,
    )
    np.savez_compressed(
        path,
        phi=np.array([phi1, phi2]),
        energies=np.pad(
            result.energies, (0, 8 - eigenvalue_count),
            constant_values=np.nan,
        ),
        amplitudes=result.vectors[:, :3],
        bloch_vectors=vectors[:, :, 0],
        residual=np.asarray(result.residual),
        hermiticity=np.asarray(hermiticity),
    )
    log(
        f"saved {path.name}; relative levels="
        + ", ".join(f"{x-result.energies[0]:.5f}" for x in result.energies)
        + f" meV; residual={result.residual:.2e}"
    )


def configuration_overlap_weights(
    hole_basis: np.ndarray,
    left_bloch: np.ndarray,
    right_bloch: np.ndarray,
) -> np.ndarray:
    """Electron Slater overlaps for complements of the nine-hole bitstates."""
    orbital_overlap = np.einsum(
        "ij,ij->i", left_bloch.conj(), right_bloch, optimize=True
    )
    if np.min(np.abs(orbital_overlap)) < 1.0e-10:
        raise RuntimeError("adjacent-twist one-particle overlap is singular")
    weights = np.full(
        len(hole_basis), np.prod(orbital_overlap), dtype=np.complex128
    )
    for orbital, overlap in enumerate(orbital_overlap):
        occupied_hole = (
            (hole_basis >> np.uint64(orbital)) & np.uint64(1)
        ).astype(bool)
        weights[occupied_hole] /= overlap
    return weights


def determinant_link(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    hole_basis: np.ndarray,
) -> tuple[complex, float]:
    weights = configuration_overlap_weights(
        hole_basis, left["bloch_vectors"], right["bloch_vectors"]
    )
    overlap = left["amplitudes"].conj().T @ (
        weights[:, None] * right["amplitudes"]
    )
    singular_values = np.linalg.svd(overlap, compute_uv=False)
    determinant = np.linalg.det(overlap)
    if abs(determinant) < 1.0e-14:
        raise RuntimeError("ground-manifold determinant link is singular")
    return determinant / abs(determinant), float(singular_values.min())


def load_point(i: int, j: int, grid: int) -> dict[str, np.ndarray]:
    cache = np.load(point_path(i, j, grid))
    raw_energies = np.asarray(cache["energies"], dtype=float)
    energies = np.full(8, np.nan, dtype=float)
    energies[:len(raw_energies)] = raw_energies
    return {
        "phi": np.asarray(cache["phi"], dtype=float),
        "energies": energies,
        "amplitudes": np.asarray(cache["amplitudes"], dtype=np.complex128),
        "bloch_vectors": np.asarray(cache["bloch_vectors"], dtype=np.complex128),
        "residual": np.asarray(cache["residual"], dtype=float),
        "hermiticity": np.asarray(cache["hermiticity"], dtype=float),
    }


def assemble(
    grid: int, hole_basis: np.ndarray, cluster: TriangularSupercell
) -> dict:
    points = [
        [load_point(i, j, grid) for j in range(grid + 1)]
        for i in range(grid + 1)
    ]
    link1 = np.empty((grid, grid + 1), dtype=np.complex128)
    link2 = np.empty((grid + 1, grid), dtype=np.complex128)
    min_singular = 1.0
    for i in range(grid):
        for j in range(grid + 1):
            link1[i, j], singular = determinant_link(
                points[i][j], points[i + 1][j], hole_basis
            )
            min_singular = min(min_singular, singular)
    for i in range(grid + 1):
        for j in range(grid):
            link2[i, j], singular = determinant_link(
                points[i][j], points[i][j + 1], hole_basis
            )
            min_singular = min(min_singular, singular)

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
    curvature = flux / step**2
    all_energies = np.asarray(
        [[points[i][j]["energies"] for j in range(grid + 1)]
         for i in range(grid + 1)]
    )
    gaps = all_energies[:, :, 3] - all_energies[:, :, 2]
    residuals = np.asarray(
        [[float(points[i][j]["residual"]) for j in range(grid + 1)]
         for i in range(grid + 1)]
    )
    hermiticity = np.asarray(
        [[float(points[i][j]["hermiticity"]) for j in range(grid + 1)]
         for i in range(grid + 1)]
    )
    bundle_chern = float(flux.sum() / (2.0 * np.pi))
    result = {
        "method": (
            "non-Abelian determinant-link FHS curvature of the three-state "
            "nu=2/3 Gamma ground manifold under boundary twists"
        ),
        "paper_benchmark": "arXiv:2503.13585v3 Supplementary Fig. 5",
        "theta_deg": 2.6,
        "dielectric_constant": 5.0,
        "bands": 1,
        "cluster_cells": 27,
        "supercell_matrix": cluster.matrix.tolist(),
        "filling": "2/3",
        "electron_number": 18,
        "hole_number": 9,
        "twist_plaquettes_per_direction": grid,
        "twist_step_radians": step,
        "plaquette_flux_radians": flux.tolist(),
        "berry_curvature": curvature.tolist(),
        "bundle_chern_number": bundle_chern,
        "hall_conductance_e2_over_h": bundle_chern / 3.0,
        "minimum_manifold_gap_meV": float(gaps.min()),
        "maximum_manifold_gap_meV": float(gaps.max()),
        "minimum_link_singular_value": min_singular,
        "maximum_lanczos_residual_meV": float(residuals.max()),
        "maximum_operator_hermiticity_error": float(hermiticity.max()),
        "four_lowest_energies_meV": all_energies[:, :, :4].tolist(),
        "sampled_lowest_energies_meV": all_energies.tolist(),
        "note": (
            "The three-state bundle, rather than an individual eigenvector, "
            "is the gauge-invariant object for the quasi-degenerate FCI."
        ),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log(
        f"bundle Chern={bundle_chern:.8f}; sigma_xy={bundle_chern/3:.8f} e^2/h; "
        f"minimum E4-E3={gaps.min():.6f} meV; min link sv={min_singular:.3e}"
    )
    return result


def make_figure(result: dict) -> None:
    grid = int(result["twist_plaquettes_per_direction"])
    curvature = np.asarray(result["berry_curvature"], dtype=float)
    flux = np.asarray(result["plaquette_flux_radians"], dtype=float)
    chern = float(result["bundle_chern_number"])
    centers = (np.arange(grid) + 0.5) / grid
    edges = np.linspace(0.0, 1.0, grid + 1)
    cumulative = np.r_[0.0, np.cumsum(flux.sum(axis=0)) / (2.0 * np.pi)]

    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.85))
    fig.subplots_adjust(left=0.085, right=0.91, bottom=0.20, top=0.86, wspace=0.34)

    scale = max(abs(float(curvature.min())), abs(float(curvature.max())), 1.0e-12)
    mesh = axes[0].pcolormesh(
        edges, edges, curvature.T, cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale),
        shading="flat", rasterized=True,
    )
    xx, yy = np.meshgrid(centers, centers, indexing="ij")
    axes[0].scatter(xx.ravel(), yy.ravel(), s=8, color="white", alpha=0.68, linewidth=0)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel(r"$\phi_1/2\pi$")
    axes[0].set_ylabel(r"$\phi_2/2\pi$")
    axes[0].set_title("many-body Berry curvature", pad=6)
    axes[0].set_xticks([0, 0.5, 1])
    axes[0].set_yticks([0, 0.5, 1])
    cax = fig.add_axes([0.435, 0.22, 0.012, 0.60])
    colorbar = fig.colorbar(mesh, cax=cax)
    colorbar.locator = MaxNLocator(5)
    colorbar.update_ticks()
    colorbar.set_label(r"$\mathcal{F}_{12}$", labelpad=2)
    colorbar.ax.tick_params(length=2, width=0.55, labelsize=6.5)

    axes[1].plot(
        edges, cumulative, color="#1479B8", marker="o", ms=4.2,
        mfc="white", mec="#1479B8", lw=1.35,
    )
    axes[1].axhline(chern, color="#D1495B", lw=0.9, ls="--")
    axes[1].fill_between(edges, 0.0, cumulative, color="#61A5C2", alpha=0.15)
    axes[1].set_xlim(0, 1)
    margin = max(0.15, 0.12 * max(abs(cumulative)))
    axes[1].set_ylim(
        min(0.0, cumulative.min()) - margin,
        max(0.0, cumulative.max()) + margin,
    )
    axes[1].set_xlabel(r"$\phi_2/2\pi$")
    axes[1].set_ylabel(r"$\sum_{\phi_2'\leq\phi_2}\Phi/(2\pi)$")
    axes[1].set_title("integrated bundle curvature", pad=6)
    axes[1].grid(color="0.91", lw=0.5)
    axes[1].text(
        0.04, 0.95,
        rf"$C_{{\rm bundle}}={chern:.3f}$" + "\n"
        + rf"$\sigma_{{xy}}={chern/3:.3f}\,e^2/h$" + "\n"
        + rf"$\min(E_4-E_3)={result['minimum_manifold_gap_meV']:.3f}$ meV",
        transform=axes[1].transAxes, ha="left", va="top", fontsize=7.2,
        bbox={
            "boxstyle": "round,pad=0.28", "facecolor": "white",
            "edgecolor": "0.82", "linewidth": 0.55, "alpha": 0.92,
        },
    )
    panel_label(axes[0], "a", x=-0.16, y=1.05)
    panel_label(axes[1], "b", x=-0.14, y=1.05)
    fig.suptitle(
        r"27-cell, one-band, $\nu=2/3$: non-Abelian ground-manifold topology",
        y=0.965, fontsize=9.2, fontweight="medium",
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig4_manybody_berry_curvature.pdf")
    fig.savefig(FIGURES / "fig4_manybody_berry_curvature.png", dpi=500)
    plt.close(fig)
    log("saved Fig. 4 PDF/PNG")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.grid < 2:
        raise ValueError("grid must be at least two plaquettes per direction")

    CACHE.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    cluster = TriangularSupercell.c6_27()
    log("generating C(27,9) basis and selecting the Gamma block")
    bases = all_group_sector_bases(
        cluster.numerators, cluster.determinant, cluster.residue_lookup, 1, 9
    )
    gamma_basis = bases[0]
    if not args.plot_only:
        model = ContinuumModel(
            ContinuumParameters(theta_deg=2.6, plane_wave_shells=8)
        )
        for i in range(args.grid + 1):
            for j in range(args.grid + 1):
                solve_point(
                    i, j, args.grid, model, cluster, gamma_basis, args.force
                )
    result = assemble(args.grid, gamma_basis, cluster)
    make_figure(result)


if __name__ == "__main__":
    main()
