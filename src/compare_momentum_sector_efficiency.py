"""Compare matched full-M and fixed-Gamma neural-Bloch optimizations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "result" / "data"
RUNS = {
    "1/3": {
        "full_M": DATA / "neural_bloch_jax_1det_100step_nu1of3",
        "fixed_Gamma_no_M": DATA / "neural_bloch_fixed_gamma_no_m_100step_nu1of3",
    },
    "2/3": {
        "full_M": DATA / "neural_bloch_jax_1det_100step",
        "fixed_Gamma_no_M": DATA / "neural_bloch_fixed_gamma_no_m_100step_nu2of3",
    },
}


def trace_metrics(summary: dict, window: int = 20) -> dict:
    trace = summary["trace"]
    tail = trace[-window:]
    steps = np.asarray([row["step"] for row in tail], dtype=float)
    energy = np.asarray([row["energy_per_particle_meV"] for row in tail])
    variance = np.asarray([row["variance_meV2"] for row in tail])
    force = np.asarray([row["force_norm"] for row in tail])
    seconds = np.asarray([row["step_seconds"] for row in trace])
    relative_residual = np.asarray([
        row.get("relative_residual_norm", row["residual_norm"] / row["force_norm"])
        for row in tail
    ])
    slope = float(np.polyfit(steps, energy, 1)[0])
    return {
        "final_step": int(trace[-1]["step"]),
        "step_200_energy_per_particle_meV": float(trace[-1]["energy_per_particle_meV"]),
        "step_200_training_walker_sem_meV": float(trace[-1]["sem_meV"]) / summary["protocol"]["spec"]["n_particles"],
        "last_20_mean_energy_per_particle_meV": float(energy.mean()),
        "last_20_energy_rms_meV": float(np.sqrt(np.mean((energy - energy.mean()) ** 2))),
        "last_20_min_energy_per_particle_meV": float(energy.min()),
        "last_20_slope_meV_per_particle_per_step": slope,
        "last_20_mean_local_energy_variance_meV2": float(variance.mean()),
        "final_local_energy_variance_meV2": float(trace[-1]["variance_meV2"]),
        "last_20_mean_force_norm": float(force.mean()),
        "mean_step_seconds": float(seconds.mean()),
        "last_20_mean_cg_relative_residual": float(relative_residual.mean()),
        "median_step_seconds": float(np.median(seconds)),
        "training_seconds": float(summary["training_seconds"]),
        "parameter_count": int(summary["protocol"]["parameter_count"]),
    }


def main() -> None:
    result = {
        "comparison": "matched 200-step natural-gradient runs (step-100 checkpoints continued for 100 updates)",
        "efficiency_definition": (
            "energy descent per update and wall time, late-window energy, "
            "local-energy variance, and force norm"
        ),
        "fillings": {},
    }
    for filling, methods in RUNS.items():
        result["fillings"][filling] = {}
        for method, directory in methods.items():
            summary = json.loads(
                (directory / "jax_neural_bloch_summary.json").read_text()
            )
            metrics = trace_metrics(summary)
            ed = -37.3932329459257 if filling == "1/3" else -52.72553897658534
            metrics["five_band_ed_meV_per_particle"] = ed
            metrics["last_20_minus_five_band_ed_meV_per_particle"] = (
                metrics["last_20_mean_energy_per_particle_meV"] - ed
            )
            result["fillings"][filling][method] = metrics
    output = DATA / "momentum_sector_efficiency_comparison.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
