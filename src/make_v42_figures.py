"""End-page 9/27-cell QNS summaries and 27-cell observables for report v4.2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np

from .make_tmote2_report_figures import (
    annotate_hex,
    geometry_theta26,
    hex_contour,
    set_hex_limits,
)
from .qns27 import inputs as qns27_inputs
from .run_neural_bloch import neural_bloch_inputs


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "result/data"
FIG = ROOT / "result/figures"
RELEASE42 = ROOT / "result/release_data/v42"
NINE = RELEASE42 / "nine"
QNS27 = RELEASE42 / "qns27"
OBS27 = QNS27 / "diagnostics"

BLUE = "#28688C"
RED = "#B5423A"
ORANGE = "#E69F00"
MAGENTA = "#CC79A7"
GRAY = "#747A7E"
COLOR = {
    "full": BLUE,
    "gamma": RED,
    "outer0": "#D55E00",
    "outer1": "#7A5195",
    "outer2": ORANGE,
}
LINESTYLE = {
    "full": "-", "gamma": "-", "outer0": (0, (3, 1.5)),
    "outer1": (0, (1.2, 1.2)), "outer2": (0, (5, 1.5)),
}
OUTER_MARKER = {"outer0": "o", "outer1": "s", "outer2": "^"}
ED9 = {"1/3": -37.3932329459257, "2/3": -52.72553897658505}
ED27 = {"1/3": -47.59546253632218, "2/3": -57.45573020653827}


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.2, "axes.labelsize": 7.3, "axes.titlesize": 7.8,
        "axes.linewidth": 0.7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.direction": "out", "ytick.direction": "out",
        "pdf.fonttype": 42, "figure.dpi": 180, "savefig.dpi": 400,
    })


def trace9(filling: str, branch: str) -> list[dict]:
    prefix = "nu1of3" if filling == "1/3" else "nu2of3"
    suffix = {
        "full": "full_m", "gamma": "v4_gamma",
        "outer0": "v4_outer_c3_p0", "outer1": "v4_outer_c3_p1",
        "outer2": "v4_outer_c3_p2",
    }[branch]
    return json.loads((NINE / "traces" / f"{prefix}_{suffix}.json").read_text())


def trace27(filling: str, branch: str) -> list[dict]:
    prefix = "nu1of3" if filling == "1/3" else "nu2of3"
    suffix = {
        "full": "full_m", "gamma": "v4_gamma",
        "outer0": "v4_outer_c3_p0", "outer1": "v4_outer_c3_p1",
        "outer2": "v4_outer_c3_p2",
    }[branch]
    path = QNS27 / "traces" / f"{prefix}_{suffix}.json"
    return json.loads(path.read_text())


def tail(trace: list[dict]) -> tuple[float, float]:
    values = np.asarray([row["energy_per_particle_meV"] for row in trace[-10:]])
    return float(values.mean()), float(values.std())


def selected(trace_loader, filling: str) -> str:
    branches = [f"outer{m}" for m in range(3)]
    return min(branches, key=lambda branch: tail(trace_loader(filling, branch))[0])


def clean_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color=".92", lw=.45, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(length=2.8)


def plot_optimization(axis, trace_loader, filling, branches, ed=None) -> None:
    all_values = []
    for branch in branches:
        record = trace_loader(filling, branch)
        step = np.asarray([row["step"] for row in record])
        energy = np.asarray([row["energy_per_particle_meV"] for row in record])
        accepted = np.asarray([row["update_accepted"] for row in record])
        display_color = ORANGE if branch.startswith("outer") else COLOR[branch]
        axis.plot(step, energy, color=display_color, ls=LINESTYLE[branch], lw=.9)
        axis.scatter(step[accepted], energy[accepted], s=3.2, color=display_color,
                     alpha=.50, edgecolors="none")
        all_values.append(energy)
    axis.axvline(20, color=".55", lw=.6, ls=(0, (2, 2)), zorder=0)
    axis.set_xlim(0, 122)
    values = np.concatenate(all_values)
    minimum = float(values.min())
    axis.set_ylim(minimum - .5, minimum + 10.0)
    axis.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    axis.yaxis.set_major_locator(MaxNLocator(4))
    axis.set_xlabel("NG update")
    axis.set_ylabel(r"$E/N_e$ (meV)")


def plot_outer_trace(axis, trace_loader, filling) -> None:
    for m in range(3):
        branch = f"outer{m}"
        record = trace_loader(filling, branch)
        axis.plot([row["step"] for row in record],
                  [row["energy_per_particle_meV"] for row in record],
                  color=COLOR[branch], lw=.9, label=rf"$m={m}$")
    axis.axvline(20, color=".55", ls=":", lw=.7)
    axis.set_xlim(0, 122)
    axis.set_xlabel("NG update")
    axis.set_ylabel(r"$E/N_e$ (meV)")
    axis.yaxis.set_major_locator(MaxNLocator(4))


def plot_offsets(axis, trace_loader, filling) -> None:
    means, rms = zip(*(tail(trace_loader(filling, f"outer{m}")) for m in range(3)))
    means, rms = np.asarray(means), np.asarray(rms)
    delta = means - means.min()
    bars = axis.bar(np.arange(3), delta, yerr=rms,
                    color=[COLOR[f"outer{m}"] for m in range(3)], width=.60,
                    edgecolor="white", error_kw={"elinewidth": .7, "capsize": 2})
    axis.bar_label(bars, labels=[f"{value:.2f}" for value in delta],
                   padding=2, fontsize=6.2)
    axis.axhline(0, color=".25", lw=.6)
    axis.set_ylim(-.12, max(delta + rms) * 1.18 + .08)
    axis.set_xticks(range(3), [r"$m=0$", r"$m=1$", r"$m=2$"])
    axis.set_ylabel(r"$\Delta E_{\rm tail}/N_e$ (meV)")


def timing9(filling: str) -> np.ndarray:
    key = "nu1of3" if filling == "1/3" else "nu2of3"
    timing = json.loads((NINE / "summary.json").read_text())["timings_seconds"][key]
    return np.asarray([
        timing["ed_5band_gamma"], timing["full_m"], timing["v4_gamma"],
        timing["v4_outer_c3_three_sector_wall"] / 3,
    ])


def timing27(filling: str) -> np.ndarray:
    prefix = "nu1of3" if filling == "1/3" else "nu2of3"
    records = json.loads((QNS27 / "timing.json").read_text())[prefix]
    full, gamma, outer = records["full"], records["gamma"], records["outer"]
    ed = 30.0 if filling == "1/3" else 29.0
    return np.asarray([ed, full["wall_seconds"], gamma["wall_seconds"], outer["wall_seconds"] / 3])


def plot_timing(axis, values: np.ndarray, cells: int) -> None:
    bars = axis.bar(np.arange(4), values, color=[GRAY, BLUE, RED, ORANGE],
                    width=.60, edgecolor="white", zorder=3)
    bars[-1].set_hatch("///")
    for bar, value in zip(bars, values):
        label = f"{value:.2f} s" if value < 60 else (
            f"{value / 60:.1f} min" if value < 3600 else f"{value / 3600:.1f} h"
        )
        axis.annotate(label, (bar.get_x() + bar.get_width() / 2, value),
                      xytext=(0, 3), textcoords="offset points", ha="center",
                      va="bottom", fontsize=5.8)
    ed_label = "ED\n5b, $\\Gamma$" if cells == 9 else "ED\n1b, $\\Gamma$"
    axis.set_xticks(np.arange(4), [ed_label, r"full $M$", r"v4 $P_\Gamma$" + "\n" + r"$[M]$",
                                          "v4 Outer\nper sector"])
    axis.tick_params(axis="x", labelsize=5.8)
    axis.set_yscale("log")
    axis.set_ylim(max(.1, values.min() / 4), values.max() * 3.5)
    axis.set_ylabel("wall time (s, log scale)")


def diagnostic9(filling: str, branch: str):
    prefix = "nu1of3" if filling == "1/3" else "nu2of3"
    suffix = {"full": "full_m", "gamma": "v4_gamma",
              "outer0": "v4_outer_c3_p0", "outer1": "v4_outer_c3_p1",
              "outer2": "v4_outer_c3_p2"}[branch]
    return np.load(NINE / "diagnostics" / f"{prefix}_{suffix}.npz")


def diagnostic27(filling: str, branch: str):
    prefix = "nu1of3" if filling == "1/3" else "nu2of3"
    suffix = {"full": "full_m", "gamma": "v4_gamma",
              "outer0": "v4_outer_c3_p0", "outer1": "v4_outer_c3_p1",
              "outer2": "v4_outer_c3_p2"}[branch]
    return np.load(OBS27 / f"{prefix}_{suffix}.npz")


def plot_band_weights(axis, diagnostic_loader, filling, branches, include_ed=False) -> None:
    x = np.arange(1, 6)
    width = .72 / (len(branches) + int(include_ed))
    start = -.36 + width / 2
    for index, branch in enumerate(branches):
        record = diagnostic_loader(filling, branch)
        display_color = ORANGE if branch.startswith("outer") else COLOR[branch]
        axis.bar(x + start + index * width, 100 * record["band_weight"], width=width,
                 color=display_color, edgecolor="white", lw=.35,
                 yerr=100 * record["band_weight_sem"],
                 error_kw={"elinewidth": .55, "capsize": 1, "ecolor": display_color})
    if include_ed:
        record = diagnostic_loader(filling, "gamma")
        axis.bar(x + start + len(branches) * width,
                 100 * record["five_band_ed_population"], width=width,
                 color="#B9BEC1", edgecolor="white", lw=.35)
    axis.axhline(0, color=".25", lw=.55)
    axis.set_xlim(.5, 5.5)
    axis.set_ylim(-8, 108)
    axis.set_xticks(x)
    axis.set_yticks([0, 50, 100])
    axis.set_xlabel("bare band index")
    axis.set_ylabel("population (%)")


def panel_labels(axes, x: float = -.14) -> None:
    for label, axis in zip("abcdefghijklmnop", np.asarray(axes).flat):
        axis.text(x, 1.04, label, transform=axis.transAxes,
                  fontsize=8.3, fontweight="bold", ha="left", va="bottom")
        clean_axis(axis)


def figure6() -> None:
    fig, axes = plt.subplots(4, 3, figsize=(7.25, 9.25))
    fig.subplots_adjust(left=.075, right=.99, bottom=.06, top=.925, wspace=.43, hspace=.64)
    for row, filling in enumerate(("1/3", "2/3")):
        chosen = selected(trace9, filling)
        branches = ("full", "gamma", chosen)
        plot_optimization(axes[row, 0], trace9, filling, branches, ED9[filling])
        bands = np.arange(1, 6)
        prefix, particles = ("nu1of3", 3) if filling == "1/3" else ("fig2", 6)
        ed_curve = np.asarray([
            float(np.load(DATA / f"{prefix}_{band}band_ground_state.npz")["energy_meV"])
            / particles for band in bands
        ])
        axes[row, 1].plot(bands, ed_curve, color=".22", lw=1.0, marker="o",
                          ms=4.5, mfc=MAGENTA, mec="white", mew=.5)
        for branch in branches:
            mean, rms = tail(trace9(filling, branch))
            display_color = ORANGE if branch.startswith("outer") else COLOR[branch]
            axes[row, 1].axhline(mean, color=display_color, lw=.9, ls=(0, (4, 2)))
            axes[row, 1].axhspan(mean-rms, mean+rms, color=display_color, alpha=.08, lw=0)
        axes[row, 1].set_xticks(bands)
        axes[row, 1].set_xlabel(r"retained bands $N_b$")
        axes[row, 1].set_ylabel(r"$E/N_e$ (meV)")
        plot_band_weights(axes[row, 2], diagnostic9, filling, branches, include_ed=True)
        for col, title in enumerate(("optimization", "multiband energy", "bare-band weight")):
            axes[row, col].set_title(rf"9 cells, $\nu={filling}$: {title}", fontweight="bold")
    for lower, filling in enumerate(("1/3", "2/3"), start=2):
        plot_outer_trace(axes[lower, 0], trace9, filling)
        plot_offsets(axes[lower, 1], trace9, filling)
        plot_timing(axes[lower, 2], timing9(filling), 9)
        axes[lower, 0].set_title(rf"9 cells, $\nu={filling}$: outer sectors", fontweight="bold")
        axes[lower, 1].set_title(rf"9 cells, $\nu={filling}$: sector offsets", fontweight="bold")
        axes[lower, 2].set_title(rf"9 cells, $\nu={filling}$: timing", fontweight="bold")
    handles = [Line2D([0], [0], color=".22", marker="o", mfc=MAGENTA, mec="white", label="multiband ED"),
               Line2D([0], [0], color=BLUE, label=r"full $M$"),
               Line2D([0], [0], color=RED, label=r"v4 $P_\Gamma[M]$"),
               Line2D([0], [0], color=ORANGE, label="lowest outer sector"),
               Patch(facecolor="#B9BEC1", label="5-band ED weight")]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(.53, .985),
               frameon=False, fontsize=6.4, columnspacing=.9, handletextpad=.35)
    panel_labels(axes)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig6_9cell_qns_summary.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig6_9cell_qns_summary.png", bbox_inches="tight")
    plt.close(fig)


def figure7() -> None:
    fig, axes = plt.subplots(4, 3, figsize=(7.25, 9.25))
    fig.subplots_adjust(left=.075, right=.99, bottom=.06, top=.925, wspace=.43, hspace=.64)
    methods = ("full", "gamma", "outer0", "outer1", "outer2")
    method_labels = [r"full $M$", r"$P_\Gamma[M]$", r"$P_0P_\Gamma$",
                     r"$P_1P_\Gamma$", r"$P_2P_\Gamma$"]
    for row, filling in enumerate(("1/3", "2/3")):
        chosen = selected(trace27, filling)
        branches = ("full", "gamma", chosen)
        plot_optimization(axes[row, 0], trace27, filling, branches, ED27[filling])
        means, rms = zip(*(tail(trace27(filling, branch)) for branch in methods))
        x = np.arange(len(methods))
        for xpos, mean, error, branch in zip(x, means, rms, methods):
            display_color = ORANGE if branch.startswith("outer") else COLOR[branch]
            marker = OUTER_MARKER.get(branch, "o")
            axes[row, 1].errorbar(xpos, mean, yerr=error, fmt=marker,
                                  color=display_color,
                                  markeredgecolor="white",
                                  markeredgewidth=.5, markersize=4.8,
                                  elinewidth=.8, capsize=2, zorder=3)
        axes[row, 1].axhline(ED27[filling], color=".25", ls=(0, (4, 2)), lw=1.0,
                             label="one-band ED")
        axes[row, 1].set_xticks(x, method_labels, rotation=20, ha="right")
        axes[row, 1].set_ylabel(r"$E/N_e$ (meV)")
        plot_band_weights(axes[row, 2], diagnostic27, filling, branches, include_ed=False)
        for col, title in enumerate(("optimization", "QNS energy", "bare-band weight")):
            axes[row, col].set_title(rf"27 cells, $\nu={filling}$: {title}", fontweight="bold")
    for lower, filling in enumerate(("1/3", "2/3"), start=2):
        plot_outer_trace(axes[lower, 0], trace27, filling)
        plot_offsets(axes[lower, 1], trace27, filling)
        plot_timing(axes[lower, 2], timing27(filling), 27)
        axes[lower, 0].set_title(rf"27 cells, $\nu={filling}$: outer sectors", fontweight="bold")
        axes[lower, 1].set_title(rf"27 cells, $\nu={filling}$: sector offsets", fontweight="bold")
        axes[lower, 2].set_title(rf"27 cells, $\nu={filling}$: timing", fontweight="bold")
    handles = [Line2D([0], [0], color=".25", ls=(0, (4, 2)), label="one-band ED"),
               Line2D([0], [0], color=BLUE, label=r"full $M$"),
               Line2D([0], [0], color=RED, label=r"v4 $P_\Gamma[M]$"),
               Line2D([0], [0], color=ORANGE, marker=OUTER_MARKER["outer0"],
                      ls="none", label=r"outer $m=0$"),
               Line2D([0], [0], color=ORANGE, marker=OUTER_MARKER["outer1"],
                      ls="none", label=r"outer $m=1$"),
               Line2D([0], [0], color=ORANGE, marker=OUTER_MARKER["outer2"],
                      ls="none", label=r"outer $m=2$")]
    fig.legend(handles=handles, loc="upper center", ncol=6, bbox_to_anchor=(.53, .985),
               frameon=False, fontsize=6.1, columnspacing=.75, handletextpad=.3)
    panel_labels(axes, x=-.19)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig7_27cell_qns_summary.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig7_27cell_qns_summary.png", bbox_inches="tight")
    plt.close(fig)


def figure11() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.985, bottom=.12, top=.88, wspace=.24, hspace=.33)
    for col, (cells, loader) in enumerate(((9, trace9), (27, trace27))):
        for row, filling in enumerate(("1/3", "2/3")):
            axis = axes[row, col]
            chosen = selected(loader, filling)
            for branch in ("full", "gamma", chosen):
                record = loader(filling, branch)
                axis.plot([item["step"] for item in record],
                          [item["relative_residual_norm"] for item in record],
                          color=(ORANGE if branch.startswith("outer") else COLOR[branch]),
                          ls=LINESTYLE[branch], lw=.85)
            axis.axvline(20, color=".55", lw=.6, ls=(0, (2, 2)))
            axis.axhline(.02, color=".35", lw=.65, ls=(0, (2, 2)))
            axis.axhline(.05, color="#D55E00", lw=.65, ls=(0, (3, 2)))
            axis.set_yscale("log")
            axis.set_xlim(0, 122)
            axis.set_ylim(7e-3, .65)
            axis.set_title(rf"{cells} cells, $\nu={filling}$", fontweight="bold")
            axis.set_xlabel("NG update")
            axis.set_ylabel(r"true $r_{\rm rel}$")
    handles = [Line2D([0], [0], color=BLUE, label=r"full $M$"),
               Line2D([0], [0], color=RED, label=r"v4 $P_\Gamma[M]$"),
               Line2D([0], [0], color=ORANGE, label="lowest outer sector"),
               Line2D([0], [0], color=".35", ls=(0, (2, 2)), label="0.02 early stop"),
               Line2D([0], [0], color="#D55E00", ls=(0, (3, 2)), label="0.05 cap accept")]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(.53, .985),
               frameon=False, fontsize=6.3, columnspacing=.8, handletextpad=.35)
    panel_labels(axes)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig11_cg_stability_9cell_27cell.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig11_cg_stability_9cell_27cell.png", bbox_inches="tight")
    plt.close(fig)


def observable_scales():
    ed = json.loads((DATA / "fig5_27cell_observables.json").read_text())
    nk1 = [np.asarray(ed["fillings"][f]["n_k"]) for f in ("1/3", "2/3")]
    nk5 = []
    sq = [np.asarray(ed["fillings"][f]["S_q"]) for f in ("1/3", "2/3")]
    density = []
    for filling in ("1/3", "2/3"):
        for branch in ("full", "gamma", "outer0", "outer1", "outer2"):
            record = diagnostic27(filling, branch)
            nk1.append(np.asarray(record["momentum_occupation_band1"]))
            nk5.append(np.asarray(record["momentum_occupation_first_five"]))
            sq.append(np.asarray(record["charge_structure_factor_full"]))
            density.append(record["charge_density_over_mean"].ravel())
    nk1_norm = Normalize(0.0, float(np.ceil(np.max(np.concatenate(nk1)) * 10) / 10))
    nk5_norm = Normalize(0.0, float(np.ceil(np.max(np.concatenate(nk5)) * 10) / 10))
    sq_norm = Normalize(0.0, float(np.ceil(np.max(np.concatenate(sq)) * 10) / 10))
    rmin, rmax = np.percentile(np.concatenate(density), [1, 99])
    return ed, nk1_norm, nk5_norm, sq_norm, Normalize(float(rmin), float(rmax))


def plot_observable_figure(name: str, rows: list[tuple[str, str, str]]) -> None:
    model, _ = qns27_inputs()
    ed, nk1_norm, nk5_norm, sq_norm, density_norm = observable_scales()
    points, berry, metric = geometry_theta26()
    berry_values = berry[:, 0]
    berry_scale = float(np.quantile(np.abs(berry_values), .985))
    berry_norm = TwoSlopeNorm(vmin=-berry_scale, vcenter=0.0, vmax=berry_scale)
    trace_deviation = np.maximum(metric[:, 0] - np.abs(berry_values), 0.0)
    trace_norm = Normalize(float(np.quantile(trace_deviation, .02)),
                           float(np.quantile(trace_deviation, .985)))
    fig, axes = plt.subplots(len(rows), 4, figsize=(8.8, 2.0 * len(rows) + .8),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.18]})
    if len(rows) == 1:
        axes = axes[None, :]
    # Reserve a shallow header above every column for a shared horizontal
    # colorbar.  This avoids the visually dominant full-height bars used in
    # v4.1 while retaining one fixed normalization per observable.
    fig.subplots_adjust(left=.11, right=.98, bottom=.08, top=.86, wspace=.34, hspace=.22)
    artists = [None, None, None, None]
    for row, (filling, branch, label) in enumerate(rows):
        record = diagnostic27(filling, branch)
        nk1 = np.asarray(record["momentum_occupation_band1"])
        nk5 = np.asarray(record["momentum_occupation_first_five"])
        sq = np.asarray(record["charge_structure_factor_full"])
        record_k = np.asarray(record["momentum_k_points"])
        record_q = np.asarray(record["structure_q_vectors"])
        if record_k.shape != (27, 2) or not np.allclose(
            record_k, record_q, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(
                f"{filling} {branch}: n(k) and S(q) must use the same 27 ED vectors"
            )
        axis = axes[row, 0]
        hex_contour(axis, model, points, berry_values, "RdBu_r", berry_norm, alpha=.34)
        artists[0] = axis.scatter(record_k[:, 0], record_k[:, 1], c=nk5, cmap="turbo",
                                  norm=nk5_norm, s=22 + 72*np.clip(nk5, 0, nk5_norm.vmax)/nk5_norm.vmax,
                                  edgecolor="white", linewidth=.45, zorder=5)
        set_hex_limits(axis, model)
        annotate_hex(axis, model)
        axis.set_title(r"$n_{\rm tot}(\mathbf{k})$ (bands 1--5)", pad=4)
        axis = axes[row, 1]
        hex_contour(axis, model, points, berry_values, "RdBu_r", berry_norm, alpha=.34)
        artists[1] = axis.scatter(record_k[:, 0], record_k[:, 1], c=nk1, cmap="turbo",
                                  norm=nk1_norm, s=22 + 72*np.clip(nk1, 0, nk1_norm.vmax)/nk1_norm.vmax,
                                  edgecolor="white", linewidth=.45, zorder=5)
        set_hex_limits(axis, model)
        annotate_hex(axis, model)
        axis.set_title(r"$n_1(\mathbf{k})$", pad=4)
        axis = axes[row, 2]
        hex_contour(axis, model, points, trace_deviation, "magma", trace_norm, alpha=.32)
        artists[2] = axis.scatter(record_q[:, 0], record_q[:, 1], c=sq, cmap="cool",
                                  norm=sq_norm, s=22 + 105*np.clip(sq, 0, sq_norm.vmax)/sq_norm.vmax,
                                  edgecolor="white", linewidth=.45, zorder=5)
        set_hex_limits(axis, model)
        annotate_hex(axis, model)
        axis.set_title(r"$S_{\rm full}(\mathbf{q})$", pad=4)
        axis = axes[row, 3]
        x = np.asarray(record["density_x_fraction"])
        y = np.asarray(record["density_y_fraction"])
        density = np.tile(np.asarray(record["charge_density_over_mean"]), (2, 2))
        tx, ty = np.concatenate([x, x + 1]), np.concatenate([y, y + 1])
        fx, fy = np.meshgrid(tx, ty, indexing="ij")
        cx, cy = fx + .5 * fy, np.sqrt(3) * fy / 2
        artists[3] = axis.pcolormesh(cx, cy, density, cmap="magma", norm=density_norm,
                                     shading="nearest", rasterized=True)
        axis.plot([0, 2, 3, 1, 0], [0, 0, np.sqrt(3), np.sqrt(3), 0],
                  color="#34383B", lw=.75)
        axis.set_aspect("equal")
        axis.set_xlim(-.14, 3.14)
        axis.set_ylim(-.12, np.sqrt(3)+.12)
        axis.set_xticks([0, 1.5, 3])
        axis.set_yticks([0, np.sqrt(3)], ["0", r"$\sqrt{3}$"])
        axis.set_xlabel(r"$x/a_M$")
        axis.set_ylabel(r"$y/a_M$")
        axis.set_title(r"$\rho(\mathbf{r})/\bar\rho$", pad=4)
        axes[row, 0].text(-.38, .5, rf"$\nu={filling}$" + "\n" + label,
                          transform=axes[row, 0].transAxes, rotation=90,
                          ha="center", va="center", fontsize=7.6, fontweight="bold")
    for axis in axes[:, :3].flat:
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    for axis in axes[:, 3].flat:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    for label, axis in zip("abcdefghijklmnop", axes.flat):
        axis.text(-.11, 1.03, label, transform=axis.transAxes,
                  fontsize=8.5, fontweight="bold")
    for column, (artist, label) in enumerate(zip(
        artists, (r"$n_{\rm tot}(\mathbf{k})$", r"$n_1(\mathbf{k})$",
                  r"$S(\mathbf{q})$", r"$\rho/\bar\rho$"))
    ):
        pos = axes[0, column].get_position()
        cax = fig.add_axes([pos.x0, .925, pos.width, .012])
        colorbar = fig.colorbar(artist, cax=cax, orientation="horizontal")
        colorbar.locator = MaxNLocator(4)
        colorbar.update_ticks()
        colorbar.ax.set_title(label, pad=3, fontsize=7.0)
        colorbar.ax.tick_params(length=2, width=.55, labelsize=6.3, pad=1.5)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=.08)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=.08)
    plt.close(fig)


def observable_figures() -> None:
    chosen1 = selected(trace27, "1/3")
    chosen2 = selected(trace27, "2/3")
    plot_observable_figure("fig8_27cell_qns_observables", [
        ("1/3", "full", r"full $M$"),
        ("1/3", "gamma", r"v4 $P_\Gamma[M]$"),
        ("2/3", "full", r"full $M$"),
        ("2/3", "gamma", r"v4 $P_\Gamma[M]$"),
    ])
    plot_observable_figure("fig9_27cell_qns_observables_nu1of3", [
        ("1/3", f"outer{m}", rf"v4 $P_{m}P_\Gamma[M]$" +
         (" (lowest)" if f"outer{m}" == chosen1 else "")) for m in range(3)
    ])
    plot_observable_figure("fig10_27cell_qns_observables_nu2of3", [
        ("2/3", f"outer{m}", rf"v4 $P_{m}P_\Gamma[M]$" +
         (" (lowest)" if f"outer{m}" == chosen2 else "")) for m in range(3)
    ])


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--skip-observables", action="store_true")
    args = parser.parse_args()
    style()
    figure6()
    if not args.skip_observables:
        figure7()
        observable_figures()
    figure11()


if __name__ == "__main__":
    main()
