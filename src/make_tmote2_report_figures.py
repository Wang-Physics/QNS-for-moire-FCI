"""Final Nature/PRL-style figures for the continuum multiband ED report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from .continuum import (
    ContinuumModel,
    ContinuumParameters,
    hexagonal_geometry_grid,
    path_gamma_k_m_gamma,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "result" / "data"
FIGURES = ROOT / "result" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

COLORS = {1: "#0072B2", 3: "#D55E00", 5: "#009E73"}
MARKERS = {1: "o", 3: "s", 5: "^"}
BAND_COLORS = ["#0072B2", "#D55E00", "#009E73"]


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.5,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 7.2,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def panel_label(axis: plt.Axes, label: str, x: float = -0.15, y: float = 1.06) -> None:
    axis.text(
        x, y, label, transform=axis.transAxes, ha="left", va="top",
        fontsize=9.5, fontweight="bold", zorder=20,
    )


def geometry_theta26(grid_size: int = 49) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = DATA / "geometry_theta2p6_bands1_3_hex.npz"
    if path.exists():
        cache = np.load(path)
        return cache["points"], cache["berry"], cache["metric"]
    model = ContinuumModel(ContinuumParameters(theta_deg=2.6, plane_wave_shells=8))
    points = hexagonal_geometry_grid(model, grid_size)
    records = [model.quantum_geometry(point, 3) for point in points]
    berry = np.asarray([record["berry_curvature"] for record in records]) / model.a_m**2
    metric = np.asarray([record["trace_metric"] for record in records]) / model.a_m**2
    np.savez_compressed(path, points=points, berry=berry, metric=metric)
    return points, berry, metric


def hex_contour(
    axis: plt.Axes,
    model: ContinuumModel,
    points: np.ndarray,
    values: np.ndarray,
    cmap: str,
    norm: mpl.colors.Normalize,
    alpha: float = 1.0,
) -> mpl.cm.ScalarMappable:
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1])
    levels = np.linspace(float(norm.vmin), float(norm.vmax), 121)
    artist = axis.tricontourf(
        triangulation, np.clip(values, norm.vmin, norm.vmax), levels=levels,
        cmap=cmap, norm=norm, alpha=alpha, antialiased=True,
    )
    vertices = model.hex_vertices()
    axis.add_patch(
        Polygon(vertices, closed=True, fill=False, edgecolor="0.12", linewidth=0.7)
    )
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    return artist


def set_hex_limits(axis: plt.Axes, model: ContinuumModel, pad: float = 0.07) -> None:
    vertices = model.hex_vertices()
    axis.set_xlim(
        vertices[:, 0].min() - pad * np.ptp(vertices[:, 0]),
        vertices[:, 0].max() + pad * np.ptp(vertices[:, 0]),
    )
    axis.set_ylim(
        vertices[:, 1].min() - pad * np.ptp(vertices[:, 1]),
        vertices[:, 1].max() + pad * np.ptp(vertices[:, 1]),
    )


def figure1() -> None:
    model26 = ContinuumModel(ContinuumParameters(theta_deg=2.6, plane_wave_shells=8))
    model30 = ContinuumModel(ContinuumParameters(theta_deg=3.0, plane_wave_shells=8))
    path26, distance26, ticks26 = path_gamma_k_m_gamma(model26, 60)
    path30, distance30, ticks30 = path_gamma_k_m_gamma(model30, 60)
    bands26 = np.asarray([model26.solve(point)[0][:10] for point in path26])
    bands30 = np.asarray([model30.solve(point)[0][:10] for point in path30])
    points, berry, metric = geometry_theta26()

    berry_scale = float(np.quantile(np.abs(berry), 0.985))
    metric_low = max(0.0, float(np.quantile(metric, 0.01)))
    metric_high = float(np.quantile(metric, 0.985))
    berry_norm = TwoSlopeNorm(vmin=-berry_scale, vcenter=0.0, vmax=berry_scale)
    metric_norm = Normalize(vmin=metric_low, vmax=metric_high)

    fig = plt.figure(figsize=(7.25, 4.25))
    grid = fig.add_gridspec(
        2, 5, width_ratios=[1.10, 1.10, 1.0, 1.0, 1.0],
        left=0.065, right=0.945, bottom=0.09, top=0.93, wspace=0.30, hspace=0.12,
    )
    band_axes = [fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[:, 1])]
    all_bands = np.concatenate([bands26, bands30])
    ypad = 0.04 * np.ptp(all_bands)
    for axis, theta, distance, ticks, bands in (
        (band_axes[0], 2.6, distance26, ticks26, bands26),
        (band_axes[1], 3.0, distance30, ticks30, bands30),
    ):
        for band in range(10):
            color = BAND_COLORS[band] if band < 3 else "0.78"
            width = 1.15 if band < 3 else 0.60
            axis.plot(distance, bands[:, band], color=color, lw=width, zorder=3-band/20)
        for tick in ticks:
            axis.axvline(distance[tick], color="0.88", lw=0.55, zorder=0)
        axis.set_xticks([distance[index] for index in ticks])
        axis.set_xticklabels([r"$\Gamma$", r"$K$", r"$M$", r"$\Gamma$"])
        axis.set_xlim(distance[0], distance[-1])
        axis.set_ylim(all_bands.min() - ypad, all_bands.max() + ypad)
        axis.set_title(rf"$\theta={theta:.1f}^\circ$")
        axis.set_xlabel("moire crystal momentum")
        axis.grid(axis="y", color="0.92", lw=0.45)
    band_axes[0].set_ylabel("single-hole energy (meV)")
    band_axes[1].tick_params(labelleft=False)
    handles = [Line2D([0], [0], color=BAND_COLORS[i], lw=1.4, label=f"band {i+1}")
               for i in range(3)]
    panel_label(band_axes[0], "a", x=-0.23)
    panel_label(band_axes[1], "b", x=-0.20)

    berry_axes = []
    metric_axes = []
    berry_artist = metric_artist = None
    for band in range(3):
        top = fig.add_subplot(grid[0, 2 + band])
        bottom = fig.add_subplot(grid[1, 2 + band])
        berry_artist = hex_contour(
            top, model26, points, berry[:, band], "RdBu_r", berry_norm
        )
        metric_artist = hex_contour(
            bottom, model26, points, metric[:, band], "magma", metric_norm
        )
        set_hex_limits(top, model26)
        set_hex_limits(bottom, model26)
        top.set_title(f"band {band+1}")
        berry_axes.append(top)
        metric_axes.append(bottom)
        panel_label(top, chr(ord("c") + band), x=-0.10, y=1.04)
        panel_label(bottom, chr(ord("f") + band), x=-0.10, y=1.04)
    berry_axes[0].text(
        -0.18, 0.5, r"Berry curvature $\Omega/a_M^2$", transform=berry_axes[0].transAxes,
        rotation=90, ha="center", va="center",
    )
    metric_axes[0].text(
        -0.18, 0.5, r"trace metric $\mathrm{Tr}\,g/a_M^2$", transform=metric_axes[0].transAxes,
        rotation=90, ha="center", va="center",
    )
    assert berry_artist is not None and metric_artist is not None
    cax1 = fig.add_axes([0.957, 0.55, 0.012, 0.31])
    cb1 = fig.colorbar(berry_artist, cax=cax1)
    cb1.set_ticks(np.linspace(-berry_scale, berry_scale, 5))
    cb1.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cb1.ax.tick_params(length=2, width=0.55, labelsize=6.5)
    cax2 = fig.add_axes([0.957, 0.14, 0.012, 0.31])
    cb2 = fig.colorbar(metric_artist, cax=cax2)
    cb2.set_ticks(np.linspace(metric_low, metric_high, 4))
    cb2.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cb2.ax.tick_params(length=2, width=0.55, labelsize=6.5)
    fig.savefig(FIGURES / "fig1_bands_quantum_geometry.pdf")
    fig.savefig(FIGURES / "fig1_bands_quantum_geometry.png", dpi=500)
    plt.close(fig)


def energy_convergence(filling: str) -> np.ndarray:
    if filling == "1/3":
        summary = json.loads((DATA / "nu1of3_multiband_summary.json").read_text())
        return np.asarray(
            [summary["projections"][str(b)]["ground_energy_per_hole_meV"] for b in range(1, 6)]
        )
    summary = json.loads((DATA / "fig2_benchmark_summary.json").read_text())
    four = json.loads((DATA / "fig2_4band_gamma_summary.json").read_text())
    five = json.loads((DATA / "fig2_5band_gamma_summary.json").read_text())
    return np.asarray(
        [summary["projections"][str(b)]["energy_per_hole_meV"] for b in (1, 2, 3)]
        + [four["ground_energy_per_hole_meV"], five["energies_per_hole_meV"][0]]
    )


def sector_records(filling: str, bands: int) -> list[dict]:
    if filling == "1/3":
        summary = json.loads((DATA / "nu1of3_multiband_summary.json").read_text())
        return summary["projections"][str(bands)]["sectors"]
    if bands in (1, 3):
        summary = json.loads((DATA / "fig2_benchmark_summary.json").read_text())
        return summary["projections"][str(bands)]["sectors"]
    return json.loads((DATA / "fig2_5band_momentum_spectrum.json").read_text())["sectors"]


def band_weights(filling: str, bands: int) -> np.ndarray:
    prefix = "nu1of3" if filling == "1/3" else "fig2"
    particles = 3 if filling == "1/3" else 6
    state = np.load(DATA / f"{prefix}_{bands}band_ground_state.npz")
    basis = np.asarray(state["basis"], dtype=np.uint64)
    probability = np.abs(np.asarray(state["amplitudes"])) ** 2
    probability /= probability.sum()
    result = np.zeros(bands)
    for momentum in range(9):
        for band in range(bands):
            orbital = momentum * bands + band
            occupied = ((basis >> np.uint64(orbital)) & np.uint64(1)) == 1
            result[band] += float(probability[occupied].sum())
    return result / particles


def figure2() -> None:
    fig, axes = plt.subplots(
        2, 3, figsize=(7.25, 4.45),
        gridspec_kw={"width_ratios": [0.95, 1.45, 1.08]},
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.10, top=0.86, wspace=0.36, hspace=0.39)
    offsets = {1: -0.17, 3: 0.0, 5: 0.17}
    for row, filling in enumerate(("1/3", "2/3")):
        axis = axes[row, 0]
        cutoffs = np.arange(1, 6)
        axis.plot(
            cutoffs, energy_convergence(filling), color="0.18", lw=1.0,
            marker="o", markersize=4.8, markerfacecolor="#CC79A7",
            markeredgecolor="white", markeredgewidth=0.5,
        )
        axis.set_xticks([1, 3, 5])
        axis.yaxis.set_major_locator(MaxNLocator(4))
        axis.set_xlabel(r"retained bands $N_b$")
        axis.set_ylabel(r"$E_{\rm GS}/N_h$ (meV)")
        axis.grid(axis="y", color="0.91", lw=0.5)

        axis = axes[row, 1]
        ymax = 0.0
        for bands in (1, 3, 5):
            records = sector_records(filling, bands)
            minima = {
                tuple(record["momentum"]): float(record["energies_meV"][0])
                for record in records
            }
            ground = min(minima.values())
            ordered = sorted(minima, key=lambda k: k[0] + 3 * k[1])
            x = np.asarray([k[0] + 3 * k[1] for k in ordered], dtype=float)
            y = np.asarray([minima[k] - ground for k in ordered])
            ymax = max(ymax, float(y.max()))
            axis.scatter(
                x + offsets[bands], y, s=25, marker=MARKERS[bands],
                facecolor=COLORS[bands], edgecolor="white", linewidth=0.5,
                zorder=3,
            )
        axis.set_xlim(-0.55, 8.55)
        axis.set_ylim(-0.18, ymax + 0.08 * max(ymax, 1.0))
        axis.set_xticks([0, 5, 7])
        axis.set_xticklabels([r"$\Gamma$", r"$K'$", r"$K$"])
        axis.yaxis.set_major_locator(MaxNLocator(4))
        axis.set_xlabel(r"$k_1+3k_2$")
        axis.set_ylabel(r"$E_K-E_{\rm GS}$ (meV)")
        axis.grid(axis="y", color="0.91", lw=0.5)

        axis = axes[row, 2]
        for bands in (1, 3, 5):
            weights = band_weights(filling, bands)
            axis.plot(
                np.arange(1, bands + 1), weights, color=COLORS[bands], lw=0.8,
                marker=MARKERS[bands], markersize=5.0, markeredgecolor="white",
                markeredgewidth=0.5,
            )
        axis.set_xlim(0.65, 5.35)
        axis.set_ylim(-0.025, 1.05)
        axis.set_xticks([1, 3, 5])
        axis.yaxis.set_major_locator(MaxNLocator(4))
        axis.set_xlabel("band index $b$")
        axis.set_ylabel(r"$w_b=\langle N_b\rangle/N_h$")
        axis.grid(axis="y", color="0.91", lw=0.5)
        axes[row, 0].text(
            0.96, 0.06, rf"$\nu={filling}$", transform=axes[row, 0].transAxes,
            rotation=0, ha="right", va="bottom", fontsize=8.5, fontweight="bold",
        )
    for column, title in enumerate(
        ("energy convergence", "momentum-sector spectrum", "band-resolved weight")
    ):
        axes[0, column].set_title(title, pad=7, fontweight="medium")
    handles = [
        Line2D([0], [0], color=COLORS[b], marker=MARKERS[b], lw=0.8,
               markersize=4.8, markeredgecolor="white", label=f"{b}-band ED")
        for b in (1, 3, 5)
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.66, 0.985),
        ncol=3, handletextpad=0.35, columnspacing=1.1,
    )
    for index, axis in enumerate(axes.flat):
        panel_label(axis, chr(ord("a") + index), x=-0.18, y=1.10)
    fig.savefig(FIGURES / "fig2_multiband_ed.pdf")
    fig.savefig(FIGURES / "fig2_multiband_ed.png", dpi=500)
    plt.close(fig)


def annotate_hex(axis: plt.Axes, model: ContinuumModel) -> None:
    positions = {
        r"$\Gamma$": np.zeros(2),
        r"$K$": model.wrap_to_hexagon((2 * model.b1 + model.b2) / 3),
        r"$K'$": model.wrap_to_hexagon((model.b1 + 2 * model.b2) / 3),
    }
    scale = np.linalg.norm(model.b1)
    for label, position in positions.items():
        axis.text(
            position[0] + 0.025 * scale, position[1] + 0.025 * scale, label,
            fontsize=6.3, color="0.12", ha="left", va="bottom", zorder=10,
        )


def figure3() -> None:
    model = ContinuumModel(ContinuumParameters(theta_deg=2.6, plane_wave_shells=8))
    spectra = json.loads((DATA / "fig5_27cell_1band_spectrum.json").read_text())
    observables = json.loads((DATA / "fig5_27cell_observables.json").read_text())
    points, berry, metric = geometry_theta26()
    berry_values = berry[:, 0]
    berry_scale = float(np.quantile(np.abs(berry_values), 0.985))
    berry_background_norm = TwoSlopeNorm(
        vmin=-berry_scale, vcenter=0.0, vmax=berry_scale
    )
    band1_trace_deviation = np.maximum(metric[:, 0] - np.abs(berry_values), 0.0)
    trace_background_norm = Normalize(
        vmin=float(np.quantile(band1_trace_deviation, 0.02)),
        vmax=float(np.quantile(band1_trace_deviation, 0.985)),
    )
    (DATA / "fig3_background_audit.json").write_text(json.dumps({
        "theta_deg": 2.6, "band_index": 1,
        "quantity": "max(Tr(g)-abs(Omega),0)",
        "same_background_for_fillings": ["1/3", "2/3"],
        "minimum": float(band1_trace_deviation.min()),
        "maximum": float(band1_trace_deviation.max()),
    }, indent=2) + "\n")
    paper_indices = np.asarray(observables["momentum_indices"], dtype=int)
    k_points = np.asarray(observables["k_points_hex"], dtype=float)
    nk_all = np.concatenate(
        [np.asarray(observables["fillings"][f]["n_k"], dtype=float) for f in ("1/3", "2/3")]
    )
    sq_all = np.concatenate(
        [np.asarray(observables["fillings"][f]["S_q"], dtype=float) for f in ("1/3", "2/3")]
    )
    nk_norm = Normalize(vmin=0.0, vmax=float(np.ceil(nk_all.max() * 10) / 10))
    sq_norm = Normalize(vmin=0.0, vmax=float(np.ceil(sq_all.max() * 10) / 10))

    fig, axes = plt.subplots(
        2, 3, figsize=(7.25, 4.55), gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]}
    )
    fig.subplots_adjust(left=0.09, right=0.93, bottom=0.10, top=0.90, wspace=0.28, hspace=0.33)
    nk_artist = sq_artist = None
    for row, filling in enumerate(("1/3", "2/3")):
        axis = axes[row, 0]
        records = spectra["fillings"][filling]["sectors"]
        ground = min(float(record["energies_meV"][0]) for record in records)
        for record in records:
            radial_sector = int(record["sector_index"])
            x = int(paper_indices[radial_sector])
            y = np.asarray(record["energies_meV"], dtype=float) - ground
            axis.scatter(
                np.full(len(y), x), y, s=11, facecolor="0.65", edgecolor="none",
                alpha=0.75, zorder=2,
            )
        if filling == "1/3":
            highlighted = [(0, 0), (25, 0), (26, 0)]
        else:
            highlighted = [(0, 0), (0, 1), (0, 2)]
        lookup = {int(record["sector_index"]): record for record in records}
        for sector, level in highlighted:
            x = int(paper_indices[sector])
            y = float(lookup[sector]["energies_meV"][level]) - ground
            axis.scatter(
                x, y, s=37, facecolor="#D1495B", edgecolor="white",
                linewidth=0.55, zorder=5,
            )
        axis.set_xlim(-0.7, 26.7)
        axis.set_ylim(-0.20, 9.0 if filling == "1/3" else 8.0)
        axis.set_xticks([0, 9, 18, 26])
        axis.set_xticklabels([r"$\Gamma$", r"$K'$", r"$K$", "26"])
        axis.set_xlabel(r"many-body momentum $k_1+N_1k_2$ ($N_1=3$)")
        axis.set_ylabel(r"$E-E_{\rm GS}$ (meV)")
        axis.grid(axis="y", color="0.91", lw=0.5)

        nk = np.asarray(observables["fillings"][filling]["n_k"], dtype=float)
        axis = axes[row, 1]
        hex_contour(
            axis, model, points, berry_values, "RdBu_r", berry_background_norm, alpha=0.34
        )
        nk_artist = axis.scatter(
            k_points[:, 0], k_points[:, 1], c=nk, cmap="turbo", norm=nk_norm,
            s=22 + 72 * nk / nk_norm.vmax, edgecolor="white", linewidth=0.45,
            zorder=5,
        )
        set_hex_limits(axis, model)
        annotate_hex(axis, model)
        axis.set_title(rf"$n(\mathbf{{k}})$; $\sum_{{\mathbf{{k}}}}n={nk.sum():.0f}$", pad=4)

        sq = np.asarray(observables["fillings"][filling]["S_q"], dtype=float)
        axis = axes[row, 2]
        hex_contour(
            axis, model, points, band1_trace_deviation, "magma", trace_background_norm,
            alpha=0.32,
        )
        sq_artist = axis.scatter(
            k_points[:, 0], k_points[:, 1], c=sq, cmap="cool", norm=sq_norm,
            s=22 + 105 * sq / sq_norm.vmax, edgecolor="white", linewidth=0.45,
            zorder=5,
        )
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        set_hex_limits(axis, model)
        annotate_hex(axis, model)
        axis.set_title(r"$S(\mathbf{q})$ on band-1 $\mathrm{Tr}\,g-|\Omega|$", pad=4)
        axes[row, 0].text(
            -0.25, 0.5, rf"$\nu={filling}$", transform=axes[row, 0].transAxes,
            rotation=90, ha="center", va="center", fontsize=9, fontweight="bold",
        )
    axes[0, 0].set_title("27-cell one-band spectrum", pad=6, fontweight="medium")
    for index, axis in enumerate(axes.flat):
        panel_label(axis, chr(ord("a") + index), x=-0.13, y=1.04)
    assert nk_artist is not None and sq_artist is not None
    cax1 = fig.add_axes([0.642, 0.17, 0.011, 0.60])
    cb1 = fig.colorbar(nk_artist, cax=cax1)
    cb1.locator = MaxNLocator(4)
    cb1.update_ticks()
    cb1.set_label(r"$n(\mathbf{k})$", labelpad=2)
    cb1.ax.tick_params(length=2, width=0.55, labelsize=6.5)
    cax2 = fig.add_axes([0.945, 0.17, 0.011, 0.60])
    cb2 = fig.colorbar(sq_artist, cax=cax2)
    cb2.locator = MaxNLocator(4)
    cb2.update_ticks()
    cb2.set_label(r"$S(\mathbf{q})$", labelpad=2)
    cb2.ax.tick_params(length=2, width=0.55, labelsize=6.5)
    fig.text(
        0.60, 0.035,
        r"weak backgrounds (Fig. 1 palette): band-1 $\Omega$ behind $n(\mathbf{k})$; "
        r"band-1 $\mathrm{Tr}\,g-|\Omega|$ behind both $S(\mathbf{q})$ rows",
        ha="center", va="bottom", fontsize=6.8, color="0.35",
    )
    fig.savefig(FIGURES / "fig3_fig5_spectra_observables.pdf")
    fig.savefig(FIGURES / "fig3_fig5_spectra_observables.png", dpi=500)
    plt.close(fig)


def main() -> None:
    style()
    figure1()
    figure2()
    figure3()


if __name__ == "__main__":
    main()
