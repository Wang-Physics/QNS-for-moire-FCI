"""Run the prescribed independent validation protocol after all four step-120 runs.

This module never participates in optimization.  It waits until all final
checkpoints exist, validates every final state with fresh walkers, selects the
lower validation energy within each filling, and precisely revalidates only
those two selected states with another fresh ensemble.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


RUNS = {
    "nu1of3_full_m": "neural_bloch_adaptive_cg120_20260821_full_m_nu1of3",
    "nu1of3_no_m_gamma": "neural_bloch_adaptive_cg120_20260821_no_m_gamma_nu1of3",
    "nu2of3_full_m": "neural_bloch_adaptive_cg120_20260821_full_m_nu2of3",
    "nu2of3_no_m_gamma": "neural_bloch_adaptive_cg120_20260821_no_m_gamma_nu2of3",
}


def _validate(
    checkpoint: Path,
    output: Path,
    *,
    walkers: int,
    burn_sweeps: int,
    measurement_sweeps: int,
    seed: int,
) -> dict:
    command = [
        sys.executable,
        "-m",
        "src.validate_jax_neural_bloch",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--walkers",
        str(walkers),
        "--burn-sweeps",
        str(burn_sweeps),
        "--measurement-sweeps",
        str(measurement_sweeps),
        "--blocks",
        "8",
        "--seed",
        str(seed),
        "--wavefunction-batch",
        "516" if walkers == 4128 else "256",
        "--local-energy-batch",
        "32",
    ]
    subprocess.run(command, check=True)
    return json.loads(output.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("result/data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("result/data/neural_bloch_adaptive_cg120_20260821_validation"),
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()

    checkpoints = {
        key: args.data_root / directory / "jax_neural_bloch_step_0120.npz"
        for key, directory in RUNS.items()
    }
    while True:
        missing = [key for key, checkpoint in checkpoints.items() if not checkpoint.exists()]
        if not missing:
            break
        print(json.dumps({"event": "waiting_for_step120", "missing": missing}), flush=True)
        time.sleep(args.poll_seconds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    low_cost = {}
    for index, (key, checkpoint) in enumerate(checkpoints.items()):
        output = args.output_dir / f"low_cost_{key}.json"
        low_cost[key] = _validate(
            checkpoint,
            output,
            walkers=1024,
            burn_sweeps=200,
            measurement_sweeps=10,
            seed=9187 + index,
        )
        print(json.dumps({"event": "low_cost_complete", "run": key, **low_cost[key]}), flush=True)

    selected = {}
    for filling in ("nu1of3", "nu2of3"):
        candidates = [key for key in low_cost if key.startswith(filling)]
        selected[filling] = min(
            candidates,
            key=lambda key: low_cost[key]["energy_per_particle_meV"],
        )

    precise = {}
    for index, (filling, key) in enumerate(selected.items()):
        output = args.output_dir / f"precise_{key}.json"
        precise[key] = _validate(
            checkpoints[key],
            output,
            walkers=4128,
            burn_sweeps=300,
            measurement_sweeps=20,
            seed=19187 + index,
        )
        print(json.dumps({"event": "precise_complete", "run": key, **precise[key]}), flush=True)

    summary = {
        "definition": "post-step120 independent validation protocol",
        "selection_rule": "lowest low-cost re-equilibrated validation energy within each filling",
        "training_minimum_used_for_selection": False,
        "low_cost_protocol": {
            "fresh_walkers": 1024,
            "burn_sweeps": 200,
            "measurement_sweeps": 10,
        },
        "precise_protocol": {
            "fresh_walkers": 4128,
            "burn_sweeps": 300,
            "measurement_sweeps": 20,
        },
        "low_cost": low_cost,
        "selected": selected,
        "precise": precise,
    }
    summary_path = args.output_dir / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "validation_protocol_complete", "summary": str(summary_path)}), flush=True)


if __name__ == "__main__":
    main()
