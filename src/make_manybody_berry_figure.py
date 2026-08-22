"""Combined flux spectra and many-body Berry curvature for Fig. 4."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .make_tmote2_report_figures import panel_label, style


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "result" / "data"
FIGURES = ROOT / "result" / "figures"


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text())


def line_nu1(axis: plt.Axes, energies: np.ndarray, direction: int) -> None:
    line = energies[:, 0] if direction == 0 else energies[0, :]
    relative = line - np.nanmin(line, axis=(1, 2))[:, None, None]
    x = np.linspace(0.0, 1.0, len(line))
    for sector in range(3):
        axis.scatter(
            x, relative[:, sector, 0], s=24, facecolor="#D1495B",
            edgecolor="white", linewidth=0.45, zorder=5,
        )
        for level in range(1, relative.shape[2]):
            axis.scatter(
                x, relative[:, sector, level], s=13, facecolor="0.58",
                edgecolor="none", alpha=0.62, zorder=2,
            )
    format_spectrum(axis, direction, relative, -1.0 if direction == 0 else -0.12)


def line_nu2(axis: plt.Axes, energies: np.ndarray, direction: int) -> None:
    line = energies[:, 0] if direction == 0 else energies[0, :]
    relative = line - np.nanmin(line, axis=1)[:, None]
    x = np.linspace(0.0, 1.0, len(line))
    for state in range(3):
        axis.scatter(
            x, relative[:, state], s=24, facecolor="#D1495B",
            edgecolor="white", linewidth=0.45, zorder=5,
        )
    for state in range(3, relative.shape[1]):
        axis.scatter(
            x, relative[:, state], s=13, facecolor="0.58",
            edgecolor="none", alpha=0.62, zorder=2,
        )
    format_spectrum(axis, direction, relative, -0.5 if direction == 0 else -0.12)


def format_spectrum(axis: plt.Axes, direction: int, relative: np.ndarray, lower: float) -> None:
    axis.set_xlim(-0.04, 1.04)
    axis.set_xticks([0.0, 0.5, 1.0])
    axis.set_xlabel(
        r"$\phi_1/2\pi$ ($\phi_2=0$)"
        if direction == 0
        else r"$\phi_2/2\pi$ ($\phi_1=0$)"
    )
    axis.set_ylabel(r"$E-E_{\rm GS}$ (meV)")
    upper = np.ceil(float(np.nanmax(relative)) * 2.0) / 2.0 + 0.18
    axis.set_ylim(lower, upper)
    axis.yaxis.set_major_locator(MaxNLocator(4))
    axis.tick_params(direction="out", top=False, right=False, length=3.0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="0.90", lw=0.45, alpha=0.55)


def curvature_panel(
    axis: plt.Axes,
    result: dict,
    filling: str,
) -> None:
    curvature = np.asarray(result["berry_curvature"], dtype=float)
    grid = curvature.shape[0]
    edges = np.linspace(0.0, 1.0, grid + 1)
    centers = (np.arange(grid) + 0.5) / grid
    if filling == "1/3":
        scale = float(np.max(np.abs(curvature)))
        norm = TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale)
        cmap = "RdBu_r"
    else:
        low = max(0.0, float(curvature.min()) - 0.01)
        high = float(curvature.max()) + 0.01
        norm = Normalize(vmin=low, vmax=high)
        cmap = "magma"
    mesh = axis.pcolormesh(
        edges, edges, curvature.T, cmap=cmap, norm=norm,
        shading="flat", rasterized=True,
    )
    xx, yy = np.meshgrid(centers, centers, indexing="ij")
    axis.scatter(
        xx.ravel(), yy.ravel(), s=7, color="white", alpha=0.63, linewidth=0
    )
    axis.set_aspect("equal")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xticks([0, 0.5, 1])
    axis.set_yticks([0, 0.5, 1])
    axis.set_xlabel(r"$\phi_1/2\pi$")
    axis.set_ylabel(r"$\phi_2/2\pi$")
    cax = inset_axes(
        axis, width="5%", height="58%", loc="center right",
        bbox_to_anchor=(0.12, 0.0, 1, 1), bbox_transform=axis.transAxes,
        borderpad=0,
    )
    colorbar = axis.figure.colorbar(mesh, cax=cax)
    colorbar.locator = MaxNLocator(4)
    colorbar.update_ticks()
    colorbar.set_label(r"$\mathcal{F}_{12}$", labelpad=1)
    colorbar.ax.tick_params(length=1.8, width=0.5, labelsize=5.7)
    chern = float(result["bundle_chern_number"])
    gap_key = (
        "minimum_sampled_manifold_gap_meV"
        if filling == "1/3"
        else "minimum_manifold_gap_meV"
    )
    axis.text(
        0.04, 0.96,
        rf"$C_{{\rm bundle}}={chern:.3f}$" + "\n"
        + rf"$\min\Delta={result[gap_key]:.3f}$ meV",
        transform=axis.transAxes, ha="left", va="top", fontsize=6.8,
        bbox={
            "boxstyle": "round,pad=0.24", "facecolor": "white",
            "edgecolor": "0.82", "linewidth": 0.5, "alpha": 0.90,
        },
    )


def main() -> None:
    nu1 = load("manybody_berry_27cell_nu1of3.json")
    nu2 = load("manybody_berry_27cell_nu2of3.json")
    energies1 = np.asarray(
        nu1["sector_resolved_sampled_energies_meV"], dtype=float
    )
    energies2 = np.asarray(nu2["sampled_lowest_energies_meV"], dtype=float)

    style()
    fig, axes = plt.subplots(
        2, 3, figsize=(7.25, 4.65),
        gridspec_kw={"width_ratios": [1.03, 1.03, 0.90]},
    )
    fig.subplots_adjust(
        left=0.092, right=0.94, bottom=0.105, top=0.885,
        wspace=0.34, hspace=0.42,
    )

    line_nu1(axes[0, 0], energies1, 0)
    line_nu1(axes[0, 1], energies1, 1)
    axes[0, 1].set_ylim(axes[0, 0].get_ylim())
    curvature_panel(axes[0, 2], nu1, "1/3")
    line_nu2(axes[1, 0], energies2, 0)
    line_nu2(axes[1, 1], energies2, 1)
    axes[1, 1].set_ylim(axes[1, 0].get_ylim())
    curvature_panel(axes[1, 2], nu2, "2/3")

    axes[0, 0].set_title(r"flux spectrum: vary $\phi_1$", pad=5)
    axes[0, 1].set_title(r"flux spectrum: vary $\phi_2$", pad=5)
    axes[0, 2].set_title("many-body Berry curvature", pad=5)
    for row, filling in enumerate(("1/3", "2/3")):
        axes[row, 0].text(
            -0.26, 0.5, rf"$\nu={filling}$",
            transform=axes[row, 0].transAxes, rotation=90,
            ha="center", va="center", fontsize=9, fontweight="bold",
        )
    for index, axis in enumerate(axes.flat):
        panel_label(axis, chr(ord("a") + index), x=-0.22, y=1.12)
    fig.suptitle(
        r"27-cell one-band ED under twisted boundaries ($\theta=2.6^\circ$, $\epsilon=5$)",
        y=0.965, fontsize=9.2, fontweight="medium",
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig4_manybody_berry_curvature.pdf")
    fig.savefig(FIGURES / "fig4_manybody_berry_curvature.png", dpi=500)
    plt.close(fig)


if __name__ == "__main__":
    main()
