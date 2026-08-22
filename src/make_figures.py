"""Nature/Science-style figures for the validated continuum-ED benchmark."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.patches import Polygon

from .continuum import (
    ContinuumModel,
    ContinuumParameters,
    hexagonal_geometry_grid,
    path_gamma_k_m_gamma,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "result" / "data"
FIGURES = ROOT / "result" / "figures"
COLORS = {1: "#0072B2", 3: "#D55E00", 5: "#009E73"}
MARKERS = {1: "o", 3: "s", 5: "^"}


def nature_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
    })


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.16, 1.06, label, transform=axis.transAxes, fontsize=9,
              fontweight="bold", va="top")


def geometry_data(model: ContinuumModel, grid_size: int = 35) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_path = DATA / "geometry_bands_1_3_hex.npz"
    if cache_path.exists():
        cache = np.load(cache_path)
        return cache["points"], cache["berry"], cache["metric"]
    points = hexagonal_geometry_grid(model, grid_size)
    records = [model.quantum_geometry(point, 3) for point in points]
    berry = np.asarray([record["berry_curvature"] for record in records]) / model.a_m**2
    metric = np.asarray([record["trace_metric"] for record in records]) / model.a_m**2
    np.savez_compressed(cache_path, points=points, berry=berry, metric=metric)
    return points, berry, metric


def draw_hex_map(
    axis: plt.Axes,
    model: ContinuumModel,
    points: np.ndarray,
    values: np.ndarray,
    cmap: str,
    norm: mpl.colors.Normalize,
) -> mpl.cm.ScalarMappable:
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1])
    levels = (np.geomspace(norm.vmin, norm.vmax, 80) if isinstance(norm, LogNorm)
              else np.linspace(norm.vmin, norm.vmax, 81))
    artist = axis.tricontourf(triangulation, values, levels=levels, cmap=cmap, norm=norm)
    vertices = model.hex_vertices()
    axis.add_patch(Polygon(vertices, closed=True, fill=False, edgecolor="black", linewidth=0.65))
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    return artist


def figure_bands_geometry(model: ContinuumModel) -> None:
    path, distance, ticks = path_gamma_k_m_gamma(model, 55)
    energies = np.asarray([model.solve(point)[0][:10] for point in path])
    points, berry, metric = geometry_data(model)
    fig = plt.figure(figsize=(7.2, 4.25), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, width_ratios=[1.25, 1, 1, 1])
    band_axis = fig.add_subplot(grid[:, 0])
    for band in range(10):
        if band < 3:
            band_axis.plot(distance, energies[:, band], color=COLORS[2 * band + 1], linewidth=1.25,
                           label=f"band {band + 1}")
        else:
            band_axis.plot(distance, energies[:, band], color="0.72", linewidth=0.7, zorder=0)
    for tick in ticks:
        band_axis.axvline(distance[tick], color="0.86", linewidth=0.6, zorder=-1)
    band_axis.set_xticks([distance[tick] for tick in ticks], [r"$\Gamma$", r"$K$", r"$M$", r"$\Gamma$"])
    band_axis.set_xlim(distance[0], distance[-1])
    band_axis.set_ylabel("Hole energy (meV)")
    band_axis.legend(loc="upper left")
    panel_label(band_axis, "a")
    berry_scale = float(np.quantile(np.abs(berry), 0.985))
    metric_positive = metric[metric > 1.0e-8]
    metric_low, metric_high = np.quantile(metric_positive, [0.02, 0.985])
    berry_artist = None
    metric_artist = None
    for band in range(3):
        top = fig.add_subplot(grid[0, band + 1])
        bottom = fig.add_subplot(grid[1, band + 1])
        berry_artist = draw_hex_map(
            top, model, points, np.clip(berry[:, band], -berry_scale, berry_scale), "RdBu_r",
            TwoSlopeNorm(vmin=-berry_scale, vcenter=0.0, vmax=berry_scale),
        )
        metric_artist = draw_hex_map(
            bottom, model, points, np.clip(metric[:, band], metric_low, metric_high), "magma",
            LogNorm(vmin=metric_low, vmax=metric_high),
        )
        top.set_title(f"band {band + 1}")
        if band == 0:
            top.set_ylabel(r"Berry curvature $\Omega/a_M^2$")
            bottom.set_ylabel(r"metric $\mathrm{Tr}\,g/a_M^2$")
            panel_label(top, "b")
            panel_label(bottom, "c")
    assert berry_artist is not None and metric_artist is not None
    fig.colorbar(berry_artist, ax=[fig.axes[index] for index in (1, 3, 5)], fraction=0.035, pad=0.02)
    fig.colorbar(metric_artist, ax=[fig.axes[index] for index in (2, 4, 6)], fraction=0.035, pad=0.02)
    fig.savefig(FIGURES / "bands_and_quantum_geometry.pdf")
    fig.savefig(FIGURES / "bands_and_quantum_geometry.png", dpi=600)
    plt.close(fig)


def load_sector_minima(n_bands: int) -> dict[tuple[int, int], float]:
    if n_bands in (1, 3):
        summary = json.loads((DATA / "fig2_benchmark_summary.json").read_text(encoding="utf-8"))
        sectors = summary["projections"][str(n_bands)]["sectors"]
    else:
        summary = json.loads((DATA / "fig2_5band_momentum_spectrum.json").read_text(encoding="utf-8"))
        sectors = summary["sectors"]
    return {tuple(record["momentum"]): float(record["energies_meV"][0]) for record in sectors}


def band_population_from_state(n_bands: int, n_holes: int = 6) -> np.ndarray:
    """Return additive single-particle band weights <N_b>/N_h."""
    state = np.load(DATA / f"fig2_{n_bands}band_ground_state.npz")
    basis = np.asarray(state["basis"], dtype=np.uint64)
    probability = np.abs(np.asarray(state["amplitudes"])) ** 2
    probability /= probability.sum()
    population = np.zeros(n_bands)
    for band in range(n_bands):
        for momentum in range(9):
            orbital = momentum * n_bands + band
            occupied = ((basis >> np.uint64(orbital)) & np.uint64(1)) == 1
            population[band] += float(probability[occupied].sum())
    return population / float(n_holes)


def ground_energy_convergence() -> np.ndarray:
    summary = json.loads((DATA / "fig2_benchmark_summary.json").read_text(encoding="utf-8"))
    values = {n_bands: summary["projections"][str(n_bands)]["energy_per_hole_meV"]
              for n_bands in (1, 2, 3)}
    four = json.loads((DATA / "fig2_4band_gamma_summary.json").read_text(encoding="utf-8"))
    five = json.loads((DATA / "fig2_5band_gamma_summary.json").read_text(encoding="utf-8"))
    values[4] = four["ground_energy_per_hole_meV"]
    values[5] = five["energies_per_hole_meV"][0]
    return np.asarray([values[index] for index in range(1, 6)])


def figure_spectrum_band_weights() -> None:
    """Plot cutoff convergence, Fig.-5-style spectra, and additive band weights."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.62), constrained_layout=True,
                             gridspec_kw={"width_ratios": [0.92, 1.45, 1.05]})

    axis = axes[0]
    band_cutoff = np.arange(1, 6)
    ground_energy = ground_energy_convergence()
    axis.plot(band_cutoff, ground_energy, color="0.18", linewidth=1.0,
              marker="o", markersize=4.8, markerfacecolor="#CC79A7",
              markeredgecolor="white", markeredgewidth=0.45, zorder=3)
    axis.set_xticks(band_cutoff)
    axis.set_xlabel(r"retained bands $N_b$")
    axis.set_ylabel(r"$E_{\mathrm{GS}}/N_h$ (meV per hole)")
    axis.grid(axis="y", color="0.90", linewidth=0.55)
    panel_label(axis, "a")

    axis = axes[1]
    offsets = {1: -0.20, 3: 0.0, 5: 0.20}
    sector_rows = []
    for n_bands in (1, 3, 5):
        minima = load_sector_minima(n_bands)
        ordered = sorted(minima, key=lambda momentum: momentum[0] + 3 * momentum[1])
        x = np.asarray([k1 + 3 * k2 for k1, k2 in ordered])
        ground = min(minima.values())
        y = np.asarray([minima[momentum] - ground for momentum in ordered])
        axis.plot(x + offsets[n_bands], y, linestyle="none", marker=MARKERS[n_bands],
                  markersize=5.1, color=COLORS[n_bands], markeredgecolor="white",
                  markeredgewidth=0.45, label=rf"{n_bands}-band", zorder=3)
        for momentum, energy in zip(ordered, y):
            sector_rows.append((n_bands, *momentum, energy))
    axis.set_xlabel(r"momentum-sector index $K_1+3K_2$")
    axis.set_ylabel(r"$E_K-E_{\mathrm{GS}}$ (meV)")
    axis.set_xticks(range(9))
    axis.set_xlim(-0.55, 8.55)
    axis.grid(axis="y", color="0.90", linewidth=0.55)
    axis.legend(ncol=1, loc="lower right", bbox_to_anchor=(0.985, 0.035),
                borderaxespad=0.0, handletextpad=0.35, labelspacing=0.3)
    panel_label(axis, "b")

    axis = axes[2]
    population_rows = []
    for n_bands in (1, 3, 5):
        population = band_population_from_state(n_bands)
        band_index = np.arange(1, n_bands + 1)
        axis.plot(band_index, population, color=COLORS[n_bands], linewidth=0.7,
                  marker=MARKERS[n_bands], markersize=5.5, markeredgecolor="white",
                  markeredgewidth=0.5, label=rf"{n_bands}-band ED", zorder=3)
        for band, weight in zip(band_index, population):
            population_rows.append((n_bands, band, weight))
    axis.axhline(0.0, color="0.25", linewidth=0.65)
    axis.set_xlim(0.65, 5.35)
    axis.set_ylim(-0.025, 1.06)
    axis.set_xticks(range(1, 6))
    axis.set_xlabel("band index $b$")
    axis.set_ylabel(r"$w_b=\langle N_b\rangle/N_h$")
    axis.grid(axis="y", color="0.90", linewidth=0.55)
    axis.legend(ncol=1, loc="upper right", bbox_to_anchor=(0.99, 0.99),
                borderaxespad=0.0, handletextpad=0.35, labelspacing=0.3)
    panel_label(axis, "c")
    fig.savefig(FIGURES / "momentum_ed_and_band_weights.pdf")
    fig.savefig(FIGURES / "momentum_ed_and_band_weights.png", dpi=600)

    with (DATA / "momentum_sector_excitation_energies.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["ed_bands", "K1", "K2", "E_minus_EGS_meV"])
        writer.writerows(sector_rows)
    with (DATA / "band_resolved_ground_state_weights.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["ed_bands", "band_index", "weight_Nb_over_Nh"])
        writer.writerows(population_rows)
    with (DATA / "ground_energy_vs_band_cutoff.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["retained_bands", "ground_energy_meV_per_hole"])
        writer.writerows(zip(band_cutoff, ground_energy))
    plt.close(fig)


def occupation_from_state(n_bands: int) -> np.ndarray:
    state = np.load(DATA / f"fig2_{n_bands}band_ground_state.npz")
    basis = np.asarray(state["basis"], dtype=np.uint64)
    probability = np.abs(np.asarray(state["amplitudes"])) ** 2
    occupation = np.zeros(9)
    for momentum in range(9):
        for band in range(n_bands):
            orbital = momentum * n_bands + band
            occupation[momentum] += float(np.sum(probability[((basis >> np.uint64(orbital)) & 1) == 1]))
    return occupation.reshape(3, 3)


def figure_nk_geometry(model: ContinuumModel) -> None:
    points, _, metric = geometry_data(model)
    values = metric[:, 0]
    lower, upper = np.quantile(values[values > 1.0e-8], [0.02, 0.985])
    mesh_positions = []
    for i in range(3):
        for j in range(3):
            mesh_positions.append(model.wrap_to_hexagon(i * model.b1 / 3.0 + j * model.b2 / 3.0))
    mesh_positions = np.asarray(mesh_positions)
    occupations = {n_bands: occupation_from_state(n_bands).reshape(-1) for n_bands in (1, 3, 5)}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), constrained_layout=True)
    artist = None
    for index, n_bands in enumerate((1, 3, 5)):
        axis = axes[index]
        artist = draw_hex_map(axis, model, points, np.clip(values, lower, upper), "magma",
                              LogNorm(vmin=lower, vmax=upper))
        occupation = occupations[n_bands]
        axis.scatter(mesh_positions[:, 0], mesh_positions[:, 1],
                     s=50 + 180 * occupation / max(value.max() for value in occupations.values()),
                     facecolor=COLORS[n_bands], edgecolor="white", linewidth=0.65, alpha=0.92)
        for position, value in zip(mesh_positions, occupation):
            axis.text(position[0], position[1], f"{value:.2f}", ha="center", va="center",
                      fontsize=5.0, color="white", fontweight="bold")
        vertices = model.hex_vertices()
        pad_x = 0.10 * np.ptp(vertices[:, 0])
        pad_y = 0.10 * np.ptp(vertices[:, 1])
        axis.set_xlim(vertices[:, 0].min() - pad_x, vertices[:, 0].max() + pad_x)
        axis.set_ylim(vertices[:, 1].min() - pad_y, vertices[:, 1].max() + pad_y)
        axis.set_title(rf"{n_bands}-band ED;  $\sum_k n(k)={occupation.sum():.1f}$")
        panel_label(axis, chr(ord("a") + index))
    assert artist is not None
    colorbar = fig.colorbar(artist, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label(r"lowest-band $\mathrm{Tr}\,g(k)/a_M^2$")
    fig.savefig(FIGURES / "nk_on_quantum_geometry.pdf")
    fig.savefig(FIGURES / "nk_on_quantum_geometry.png", dpi=600)
    with (DATA / "momentum_occupations_1_3_5.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["i", "j", "kx", "ky", "n_1band", "n_3band", "n_5band"])
        for index, position in enumerate(mesh_positions):
            writer.writerow([index // 3, index % 3, *position,
                             occupations[1][index], occupations[3][index], occupations[5][index]])
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    nature_style()
    model = ContinuumModel(ContinuumParameters(plane_wave_shells=8))
    figure_bands_geometry(model)


if __name__ == "__main__":
    main()
