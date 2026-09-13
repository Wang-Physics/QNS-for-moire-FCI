"""Collect the cost-effective frozen-step S(q) and density data for v5."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "result/data/qns27_width64_seed3184"
V4 = ROOT / "result/data/v4_qns27_seed3184"
SAMPLES = ROOT / "result/data/v4_qns27_frozen5_samples"
DEST = ROOT / "result/data/v4_qns27_frozen5_coordinate"


def jobs() -> list[dict]:
    result = []
    for filling in ("nu1of3", "nu2of3"):
        result.extend([
            {
                "stem": f"{filling}_full_m",
                "checkpoint": FULL / f"{filling}_full_m/jax_neural_bloch_final.npz",
                "batch": 86,
            },
            {
                "stem": f"{filling}_v4_gamma",
                "checkpoint": V4 / f"{filling}_v4_gamma/jax_neural_bloch_final.npz",
                "batch": 86,
            },
        ])
        for sector in range(3):
            result.append({
                "stem": f"{filling}_v4_outer_c3_p{sector}",
                "checkpoint": V4 / (
                    f"{filling}_v4_outer_c3/p{sector}/"
                    "jax_neural_bloch_final.npz"
                ),
                "batch": 43,
            })
    return result


def _read_complete(path: Path, checks: dict[str, object]) -> bool:
    if not path.is_file():
        return False
    record = json.loads(path.read_text())
    return record.get("status") == "complete" and all(
        record.get(key) == value for key, value in checks.items()
    )


def _gpu_environment(gpu: int) -> dict[str, str]:
    result = dict(os.environ)
    result.pop("LD_LIBRARY_PATH", None)
    result.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="8",
        PYTHONUNBUFFERED="1",
    )
    return result


def sample_one(job: dict, gpu: int) -> dict:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    sample = SAMPLES / f"{job['stem']}.npz"
    metadata = sample.with_suffix(".json")
    if not _read_complete(metadata, {
        "parameters_frozen": True,
        "optimizer_updates": 0,
        "measurement_steps": 5,
        "sweeps_per_step": 2,
        "stored_configurations": 20640,
    }):
        command = [
            str(ROOT / ".venv/bin/python"), "-m", "src.frozen_walker_sampling",
            "--checkpoint", str(job["checkpoint"]), "--output", str(sample),
            "--measurement-steps", "5", "--sweeps-per-step", "2",
            "--wavefunction-batch", str(job["batch"]),
            "--seed", str(41000 + len(job["stem"])),
        ]
        subprocess.run(
            ["taskset", "-c", "0-15" if gpu == 0 else "32-47", *command],
            cwd=ROOT, env=_gpu_environment(gpu), check=True,
        )
    return json.loads(metadata.read_text())


def sample_lane(queue: list[dict], gpu: int) -> list[dict]:
    return [sample_one(job, gpu) for job in queue]


def coordinate_one(item: tuple[int, dict]) -> dict:
    index, job = item
    output = DEST / job["stem"]
    metadata = output / "coordinate_diagnostics.json"
    if not _read_complete(metadata, {
        "parameters_frozen": True,
        "optimizer_updates": 0,
        "unique_walker_chains": 4128,
        "measurement_steps": 5,
        "coordinate_samples": 20640,
        "structure_factor_grid_points": 27,
        "structure_factor_projected_density": False,
        "structure_factor_periodic_wigner_seitz": True,
    }):
        output.mkdir(parents=True, exist_ok=True)
        command = [
            str(ROOT / ".venv/bin/python"), "-m",
            "src.frozen_coordinate_diagnostics",
            "--samples", str(SAMPLES / f"{job['stem']}.npz"),
            "--output-dir", str(output), "--error-blocks", "8",
        ]
        cpu_start = 8 * (index % 4)
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = ""
        subprocess.run(
            ["taskset", "-c", f"{cpu_start}-{cpu_start + 7}", *command],
            cwd=ROOT, env=environment, check=True,
        )
    return json.loads(metadata.read_text())


def main() -> None:
    all_jobs = jobs()
    # Balance measured sampling costs: two 18-particle outer states plus two
    # short 9-particle jobs versus the remaining outer state and six shorter
    # jobs. Both lanes are approximately 94 s on the local A30 pair.
    lane0 = [all_jobs[index] for index in (7, 8, 0, 2)]
    lane1 = [all_jobs[index] for index in (9, 5, 6, 1, 3, 4)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(sample_lane, lane0, 0),
            executor.submit(sample_lane, lane1, 1),
        ]
        sample_records = [record for future in futures for record in future.result()]
    with ThreadPoolExecutor(max_workers=4) as executor:
        coordinate_records = list(executor.map(coordinate_one, enumerate(all_jobs)))
    summary = {
        "status": "complete",
        "route": (
            "five frozen-parameter measurement steps for coordinate-only "
            "S(q) and density; occupation remains the 4128-by-128 baseline"
        ),
        "sampling": sorted(sample_records, key=lambda item: item["output"]),
        "coordinate": sorted(coordinate_records, key=lambda item: item["samples"]),
    }
    DEST.mkdir(parents=True, exist_ok=True)
    temporary = DEST / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    temporary.replace(DEST / "summary.json")
    print(json.dumps({"status": "complete", "states": len(all_jobs)}, indent=2))


if __name__ == "__main__":
    main()
