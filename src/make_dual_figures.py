"""Two-row Nature/PRL-style figures for nu=1/3 and 2/3."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon

from .continuum import ContinuumModel, ContinuumParameters
from .make_figures import COLORS, MARKERS, draw_hex_map, geometry_data, nature_style, panel_label


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "result" / "data"
FIGURES = ROOT / "result" / "figures"


def population(path: Path, n_bands: int, n_holes: int) -> np.ndarray:
    state = np.load(path)
    basis = np.asarray(state["basis"], dtype=np.uint64)
    probability = np.abs(np.asarray(state["amplitudes"])) ** 2
    probability /= probability.sum()
    result = np.zeros(n_bands)
    for band in range(n_bands):
        for momentum in range(9):
            orbital = momentum * n_bands + band
            occupied = ((basis >> np.uint64(orbital)) & np.uint64(1)) == 1
            result[band] += float(probability[occupied].sum())
    return result / n_holes


def energy_convergence(filling: str) -> np.ndarray:
    if filling == "1/3":
        summary = json.loads((DATA / "nu1of3_multiband_summary.json").read_text())
        return np.asarray([summary["projections"][str(n)]["ground_energy_per_hole_meV"]
                           for n in range(1, 6)])
    summary = json.loads((DATA / "fig2_benchmark_summary.json").read_text())
    four = json.loads((DATA / "fig2_4band_gamma_summary.json").read_text())
    five = json.loads((DATA / "fig2_5band_gamma_summary.json").read_text())
    return np.asarray([
        summary["projections"][str(n)]["energy_per_hole_meV"] for n in (1, 2, 3)
    ] + [four["ground_energy_per_hole_meV"], five["energies_per_hole_meV"][0]])


def state_path(filling: str, n_bands: int) -> tuple[Path, int]:
    if filling == "1/3":
        return DATA / f"nu1of3_{n_bands}band_ground_state.npz", 3
    return DATA / f"fig2_{n_bands}band_ground_state.npz", 6


def figure_ed() -> None:
    spectra = json.loads((DATA / "theta2p6_vs_3_oneband_spectra.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.55), constrained_layout=True,
                             gridspec_kw={"width_ratios": [0.95, 1.4, 1.05]})
    for row, filling in enumerate(("1/3", "2/3")):
        axis = axes[row, 0]
        cutoffs = np.arange(1, 6)
        axis.plot(cutoffs, energy_convergence(filling), color="0.18", linewidth=1.0,
                  marker="o", markersize=4.7, markerfacecolor="#CC79A7",
                  markeredgecolor="white", markeredgewidth=0.45)
        axis.set_xticks(cutoffs)
        axis.set_xlabel(r"retained bands $N_b$")
        axis.set_ylabel(r"$E_{\rm GS}/N_h$ (meV per hole)")
        axis.grid(axis="y", color="0.90", linewidth=0.55)

        axis = axes[row, 1]
        for theta, color, marker, fill in (
            ("2.6", "black", "o", "black"), ("3.0", "#CC79A7", "s", "none")
        ):
            records = spectra["fillings"][theta][filling]["sectors"]
            ground = min(record["energies_meV"][0] for record in records)
            for number, record in enumerate(records):
                k1, k2 = record["momentum"]
                y = np.asarray(record["energies_meV"]) - ground
                axis.scatter(np.full(len(y), k1 + 3 * k2), y, s=10.5, marker=marker,
                             facecolor=fill, edgecolor=color, linewidth=0.55,
                             label=rf"$\theta={theta}^\circ$" if number == 0 else None)
        axis.set_xlabel(r"momentum-sector index $K_1+3K_2$")
        axis.set_ylabel(r"$E-E_{\rm GS}$ (meV)")
        sector_labels = [r"$\Gamma$", "1", "2", "3", "4", r"$K$", "6", r"$K\prime$", "8"]
        axis.set_xticks(range(9), sector_labels)
        axis.set_ylim(-0.25, 13.5 if filling == "1/3" else 12.0)
        axis.grid(axis="y", color="0.90", linewidth=0.55)
        axis.legend(loc="upper right", handletextpad=0.3, labelspacing=0.25)

        axis = axes[row, 2]
        for n_bands in (1, 3, 5):
            path, n_holes = state_path(filling, n_bands)
            weights = population(path, n_bands, n_holes)
            bands = np.arange(1, n_bands + 1)
            axis.plot(bands, weights, color=COLORS[n_bands], linewidth=0.7,
                      marker=MARKERS[n_bands], markersize=5.1,
                      markeredgecolor="white", markeredgewidth=0.45,
                      label=f"{n_bands}-band ED")
        axis.set_xlim(0.65, 5.35)
        axis.set_ylim(-0.025, 1.06)
        axis.set_xticks(range(1, 6))
        axis.set_xlabel("band index $b$")
        axis.set_ylabel(r"$w_b=\langle N_b\rangle/N_h$")
        axis.grid(axis="y", color="0.90", linewidth=0.55)
        axis.legend(loc="upper right", handletextpad=0.3, labelspacing=0.25)
        axes[row, 0].text(-0.43, 0.5, rf"$\nu={filling}$", transform=axes[row, 0].transAxes,
                          rotation=90, ha="center", va="center", fontsize=9,
                          fontweight="bold")
    for index, axis in enumerate(axes.flat):
        panel_label(axis, chr(ord("a") + index))
    fig.savefig(FIGURES / "dual_filling_ed_diagnostics.pdf")
    fig.savefig(FIGURES / "dual_filling_ed_diagnostics.png", dpi=600)
    plt.close(fig)


def figure_observables(model: ContinuumModel) -> None:
    data = json.loads((DATA / "dual_filling_5band_observables.json").read_text())
    points, _, metric = geometry_data(model)
    values = metric[:, 0]
    low, high = np.quantile(values[values > 1.0e-8], [0.02, 0.985])
    positions = np.asarray([model.wrap_to_hexagon(i * model.b1 / 3 + j * model.b2 / 3)
                            for i in range(3) for j in range(3)])
    spin_all = np.concatenate([np.asarray(data["fillings"][f]["S_zz_q"]).ravel()
                               for f in ("1/3", "2/3")])
    spin_max = max(float(spin_all.max()), 1.0e-12)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.9), constrained_layout=True)
    metric_artist = spin_artist = None
    for row, filling in enumerate(("1/3", "2/3")):
        nk = np.asarray(data["fillings"][filling]["n_k"]).ravel()
        spin = np.asarray(data["fillings"][filling]["S_zz_q"]).ravel()
        axis = axes[row, 0]
        metric_artist = draw_hex_map(axis, model, points, np.clip(values, low, high),
                                     "magma", LogNorm(vmin=low, vmax=high))
        axis.scatter(positions[:, 0], positions[:, 1],
                     s=42 + 145 * nk / max(nk.max(), 1.0e-12), facecolor="#0072B2",
                     edgecolor="white", linewidth=0.65)
        for position, value in zip(positions, nk):
            axis.text(*position, f"{value:.2f}", ha="center", va="center",
                      color="white", fontsize=5, fontweight="bold")
        axis.set_title(rf"5-band $n(k)$; $\sum_k n(k)={nk.sum():.0f}$")
        vertices = model.hex_vertices()
        pad_x = 0.10 * np.ptp(vertices[:, 0])
        pad_y = 0.10 * np.ptp(vertices[:, 1])
        axis.set_xlim(vertices[:, 0].min() - pad_x, vertices[:, 0].max() + pad_x)
        axis.set_ylim(vertices[:, 1].min() - pad_y, vertices[:, 1].max() + pad_y)

        axis = axes[row, 1]
        axis.add_patch(Polygon(model.hex_vertices(), closed=True, fill=False,
                               edgecolor="black", linewidth=0.7))
        spin_artist = axis.scatter(positions[:, 0], positions[:, 1],
                                   s=38 + 210 * spin / spin_max, c=spin, cmap="viridis",
                                   vmin=0, vmax=spin_max, edgecolor="white", linewidth=0.65)
        for position, value in zip(positions, spin):
            axis.text(*position, f"{value:.3f}", ha="center", va="center", fontsize=5)
        axis.set_aspect("equal")
        axis.set_xlim(vertices[:, 0].min() - pad_x, vertices[:, 0].max() + pad_x)
        axis.set_ylim(vertices[:, 1].min() - pad_y, vertices[:, 1].max() + pad_y)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_title(r"spin structure factor $S^{zz}(q)$")
        axes[row, 0].text(-0.26, 0.5, rf"$\nu={filling}$", transform=axes[row, 0].transAxes,
                          rotation=90, ha="center", va="center", fontsize=9,
                          fontweight="bold")
    assert metric_artist is not None and spin_artist is not None
    cb = fig.colorbar(metric_artist, ax=axes[:, 0], fraction=0.027, pad=0.02)
    cb.set_label(r"band-1 $\mathrm{Tr}\,g(k)/a_M^2$")
    cb = fig.colorbar(spin_artist, ax=axes[:, 1], fraction=0.027, pad=0.02)
    cb.set_label(r"$S^{zz}(q)=S_\rho(q)/4$")
    for index, axis in enumerate(axes.flat):
        panel_label(axis, chr(ord("a") + index))
    fig.savefig(FIGURES / "dual_filling_5band_observables.pdf")
    fig.savefig(FIGURES / "dual_filling_5band_observables.png", dpi=600)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    nature_style()
    figure_ed()
    figure_observables(ContinuumModel(ContinuumParameters(theta_deg=3.0, plane_wave_shells=8)))


if __name__ == "__main__":
    main()
