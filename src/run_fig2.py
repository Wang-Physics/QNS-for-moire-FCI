"""Reproduce the first three epsilon=5 multiband-ED points in paper Fig. 2."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import matplotlib.pyplot as plt
import numpy as np
import scipy
import yaml

from .continuum import ContinuumModel, ContinuumParameters
from .multiband_ed import build_pair_interaction, mesh_bloch_states
from .sparse_ed import diagonalize_sparse_sector


ROOT = Path(__file__).resolve().parents[1]


def model_from_config(config: dict) -> ContinuumModel:
    values = config["model"]
    return ContinuumModel(
        ContinuumParameters(
            theta_deg=float(values["theta_deg"]),
            a0_nm=float(values["a0_nm"]),
            potential_meV=float(values["potential_meV"]),
            tunneling_meV=float(values["tunneling_meV"]),
            phase_deg=float(values["phase_deg"]),
            mass_me=float(values["mass_me"]),
            plane_wave_shells=int(values["plane_wave_shells"]),
        )
    )


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plot_comparison(summary: dict, path: Path) -> None:
    bands = np.asarray([1, 2, 3])
    calculated = np.asarray([summary["projections"][str(b)]["energy_per_hole_meV"] for b in bands])
    paper = np.asarray([summary["paper_digitized_meV_per_hole"][str(b)] for b in bands])
    uncertainty = float(summary["digitization_uncertainty_meV_per_hole"])
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.25), constrained_layout=True)
    ax = axes[0]
    ax.errorbar(bands, paper, yerr=uncertainty, color="#D55E00", marker="s",
                linestyle="none", capsize=3, label="Paper Fig. 2 (digitized)")
    ax.plot(bands, calculated, "o-", color="#0072B2", linewidth=1.7, label="This work")
    ax.set_xlabel("Number of projected bands")
    ax.set_ylabel(r"$E_0/N_h$ (meV)")
    ax.set_xticks(bands)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    difference = calculated - paper
    ax.axhspan(-uncertainty, uncertainty, color="0.88", label="digitization uncertainty")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.bar(bands, difference, color="#009E73", width=0.58)
    ax.set_xlabel("Number of projected bands")
    ax.set_ylabel("calculation - paper (meV/hole)")
    ax.set_xticks(bands)
    ax.set_ylim(-0.105, 0.105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def run(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    start = time.perf_counter()
    model = model_from_config(config)
    cluster, ed, interaction, validation = (
        config["cluster"], config["ed"], config["interaction"], config["validation"]
    )
    n1, n2 = int(cluster["n1"]), int(cluster["n2"])
    n_holes = int(cluster["n_holes"])
    projections = [int(value) for value in ed["band_projections"]]
    energies, vectors, k_points = mesh_bloch_states(model, n1, n2, max(projections))
    run_dir = ROOT / "runs" / str(config["run_id"])
    data_dir = ROOT / "result" / "data"
    figure_dir = ROOT / "result" / "figures"
    for directory in (run_dir, data_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    np.savez_compressed(data_dir / "fig2_bloch_mesh.npz", energies=energies, vectors=vectors,
                        k_points=k_points, g_tuples=np.asarray(model.g_tuples, dtype=np.int32))
    projection_results: dict[str, dict] = {}
    for n_bands in projections:
        pair_interaction, pairs, pair_hermiticity = build_pair_interaction(
            model, vectors, n1, n2, n_bands, float(interaction["dielectric_constant"]),
            int(interaction["reciprocal_shells"])
        )
        sector_records = []
        saved_ground = None
        for momentum_1 in range(n1):
            for momentum_2 in range(n2):
                momentum = (momentum_1, momentum_2)
                n_eigenvalues = int(ed["gamma_eigenvalues"] if momentum == (0, 0)
                                    else ed["other_sector_eigenvalues"])
                sector = diagonalize_sparse_sector(
                    energies, pair_interaction, pairs, n1, n2, n_bands, n_holes, momentum,
                    n_eigenvalues=n_eigenvalues, tolerance=float(ed["lanczos_tolerance"]),
                    max_iterations=int(ed["max_iterations"])
                )
                sector_records.append({"momentum": list(momentum), "dimension": len(sector.basis),
                                       "energies_meV": sector.energies.tolist(),
                                       "residual_meV": sector.residual})
                candidate = (float(sector.energies[0]), sector)
                if saved_ground is None or candidate[0] < saved_ground[0]:
                    saved_ground = candidate
        assert saved_ground is not None
        ground_energy, ground_sector = saved_ground
        np.savez_compressed(data_dir / f"fig2_{n_bands}band_ground_state.npz",
                            energy_meV=np.asarray(ground_energy),
                            momentum=np.asarray(ground_sector.momentum, dtype=np.int32),
                            basis=ground_sector.basis, amplitudes=ground_sector.vectors[:, 0])
        paper_value = float(validation["paper_fig2_eps5_meV_per_hole"][n_bands])
        calculated_value = ground_energy / n_holes
        projection_results[str(n_bands)] = {
            "n_bands": n_bands,
            "n_orbitals": n1 * n2 * n_bands,
            "hilbert_dimension": math.comb(n1 * n2 * n_bands, n_holes),
            "ground_momentum": list(ground_sector.momentum),
            "ground_energy_meV": ground_energy,
            "energy_per_hole_meV": calculated_value,
            "paper_digitized_meV_per_hole": paper_value,
            "difference_from_paper_meV_per_hole": calculated_value - paper_value,
            "pair_hermiticity_error": pair_hermiticity,
            "max_lanczos_residual_meV": max(item["residual_meV"] for item in sector_records),
            "sectors": sector_records,
        }
        save_json(data_dir / f"fig2_{n_bands}band_sectors.json", sector_records)
    tolerance = float(validation["paper_digitization_tolerance_meV_per_hole"])
    residual_tolerance = float(validation["residual_tolerance_meV"])
    checks = {
        "all_momentum_sectors_scanned": all(len(value["sectors"]) == n1 * n2
                                             for value in projection_results.values()),
        "ground_state_is_gamma": all(value["ground_momentum"] == [0, 0]
                                      for value in projection_results.values()),
        "pair_hermiticity": all(value["pair_hermiticity_error"] < 1.0e-12
                                 for value in projection_results.values()),
        "lanczos_residual": all(value["max_lanczos_residual_meV"] < residual_tolerance
                                for value in projection_results.values()),
        "paper_agreement": all(abs(value["difference_from_paper_meV_per_hole"]) < tolerance
                               for value in projection_results.values()),
        "variational_monotonicity": all(
            projection_results[str(b + 1)]["ground_energy_meV"]
            <= projection_results[str(b)]["ground_energy_meV"] + 1.0e-8
            for b in projections[:-1]
        ),
    }
    summary = {
        "run_id": config["run_id"],
        "status": "validated" if all(checks.values()) else "failed",
        "benchmark": "Luo-Zaklama-Fu_Fig2_epsilon5_first_three_ED_points",
        "parameters": {"theta_deg": float(config["model"]["theta_deg"]),
                       "dielectric_constant_epsilon": float(interaction["dielectric_constant"]),
                       "cluster": f"{n1}x{n2}", "n_cells": n1 * n2, "filling": "2/3",
                       "n_holes": n_holes, "q0_removed": bool(interaction["remove_q0"]),
                       "plane_wave_shells": int(config["model"]["plane_wave_shells"]),
                       "coulomb_reciprocal_shells": int(interaction["reciprocal_shells"])},
        "paper_digitized_meV_per_hole": {str(key): float(value)
                                         for key, value in validation["paper_fig2_eps5_meV_per_hole"].items()},
        "digitization_uncertainty_meV_per_hole": float(validation["digitization_uncertainty_meV_per_hole"]),
        "projections": projection_results,
        "checks": checks,
        "software": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                     "platform": platform.platform()},
        "wall_clock_seconds": time.perf_counter() - start,
    }
    save_json(run_dir / "summary.json", summary)
    save_json(data_dir / "fig2_benchmark_summary.json", summary)
    plot_comparison(summary, figure_dir / "fig2_eps5_ed_comparison.pdf")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs" / "fig2_theta3_eps5_nu2of3.yaml")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    print(json.dumps(run(config_path), indent=2))


if __name__ == "__main__":
    main()
