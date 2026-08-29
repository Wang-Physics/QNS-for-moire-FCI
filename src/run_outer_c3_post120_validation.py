"""Independent post-training validation for the three outer-C3 sectors.

The three optimization jobs are allowed to finish without validation-chain
contention.  Once all step-120 checkpoints exist, each sector is evaluated
with a fresh 1,024-walker chain.  Only the sector with the lowest independently
validated energy is then re-evaluated with the prescribed 4,128-walker chain.
No validation sample is reused by optimization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


SECTORS = ("m0", "m1", "m2")


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
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("result/data/outer_c3_projector_nu1of3_120"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("result/data/outer_c3_projector_nu1of3_validation"),
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()

    checkpoints = {
        sector: args.run_root / sector / "jax_neural_bloch_step_0120.npz"
        for sector in SECTORS
    }
    while True:
        missing = [name for name, path in checkpoints.items() if not path.exists()]
        if not missing:
            break
        print(json.dumps({"event": "waiting_for_step120", "missing": missing}), flush=True)
        time.sleep(args.poll_seconds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    low_cost = {}
    for index, sector in enumerate(SECTORS):
        output = args.output_dir / f"low_cost_{sector}.json"
        low_cost[sector] = _validate(
            checkpoints[sector],
            output,
            walkers=1024,
            burn_sweeps=200,
            measurement_sweeps=10,
            seed=28187 + index,
        )
        print(
            json.dumps({"event": "low_cost_complete", "sector": sector, **low_cost[sector]}),
            flush=True,
        )

    selected = min(
        SECTORS,
        key=lambda sector: low_cost[sector]["energy_per_particle_meV"],
    )
    precise_output = args.output_dir / f"precise_{selected}.json"
    precise = _validate(
        checkpoints[selected],
        precise_output,
        walkers=4128,
        burn_sweeps=300,
        measurement_sweeps=20,
        seed=29187,
    )
    print(
        json.dumps({"event": "precise_complete", "sector": selected, **precise}),
        flush=True,
    )

    summary = {
        "definition": "outer-C3 post-step120 independent validation protocol",
        "training_minimum_used_for_selection": False,
        "selection_rule": "lowest fresh-chain low-cost validation energy",
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
        "precise": {selected: precise},
    }
    summary_path = args.output_dir / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps({"event": "validation_protocol_complete", "summary": str(summary_path)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
