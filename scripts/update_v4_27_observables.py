"""Run 27-cell final-training-chain observables on two A30 lanes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "result/data/qns27_width64_seed3184"
V4 = ROOT / "result/data/v4_qns27_seed3184"
DEST = ROOT / "result/data/v4_qns27_observables"


def stamp() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def jobs() -> list[dict]:
    result = []
    for filling in ("nu1of3", "nu2of3"):
        result.append({
            "stem": f"{filling}_full_m",
            "source": FULL / f"{filling}_full_m",
        })
        result.append({
            "stem": f"{filling}_v4_gamma",
            "source": V4 / f"{filling}_v4_gamma",
        })
        for sector in range(3):
            result.append({
                "stem": f"{filling}_v4_outer_c3_p{sector}",
                "source": V4 / f"{filling}_v4_outer_c3/p{sector}",
            })
    return result


def run_one(job: dict, gpu: int) -> dict:
    output = DEST / job["stem"]
    completed = output / "qns27_diagnostics.json"
    if completed.exists():
        record = json.loads(completed.read_text())
        if (
            record.get("status") == "complete"
            and "momentum_occupation_band1_normalized_sum" in record
            and "momentum_occupation_first_five_sum" in record
            and "structure_factor_raw_c3_residual" in record
        ):
            return record
    source = job["source"]
    checkpoint = source / "jax_neural_bloch_final.npz"
    samples = source / "jax_neural_bloch_samples.npz"
    if not checkpoint.exists() or not samples.exists():
        raise FileNotFoundError(f"missing completed state for {job['stem']}")
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.pop("LD_LIBRARY_PATH", None)
    environment.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="8",
        PYTHONUNBUFFERED="1",
    )
    command = [
        str(ROOT / ".venv/bin/python"), "-m", "src.qns27_diagnostics",
        "--checkpoint", str(checkpoint), "--samples", str(samples),
        "--output-dir", str(output), "--samples-count", "4128",
        "--auxiliary-draws", "16", "--error-blocks", "8",
        "--wavefunction-batch", "43", "--orbital-batch", "128",
        "--seed", str(4885 + 101 * len(job["stem"])),
    ]
    write_json(output / "job.json", {
        "status": "running", "stem": job["stem"], "gpu": gpu,
        "started": stamp(), "command": command,
    })
    started = time.monotonic()
    with (output / "run.log").open("w") as log:
        result = subprocess.run(
            ["/usr/bin/time", "-v", "-o", str(output / "resource_usage.txt"),
             "taskset", "-c", "0-15" if gpu == 0 else "32-47", *command],
            cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
        )
    record = {
        "status": "complete" if result.returncode == 0 else "failed",
        "stem": job["stem"], "gpu": gpu, "started": stamp(),
        "finished": stamp(), "wall_seconds": time.monotonic() - started,
        "exit_code": result.returncode,
    }
    write_json(output / "job.json", record)
    if result.returncode:
        raise RuntimeError(f"{job['stem']} failed with {result.returncode}")
    return json.loads(completed.read_text())


def lane(queue: list[dict], gpu: int) -> list[dict]:
    return [run_one(job, gpu) for job in queue]


def worker() -> None:
    # Build the shared five-band Bloch cache once before the GPU workers start.
    # This avoids two processes racing to create the same compressed file.
    from src.qns27_diagnostics import _five_band_orbitals
    _five_band_orbitals()
    all_jobs = jobs()
    # Alternate fillings and projector costs across the two lanes.
    lane0 = all_jobs[::2]
    lane1 = all_jobs[1::2]
    started = time.monotonic()
    write_json(DEST / "status.json", {
        "status": "running", "started": stamp(),
        "jobs": [job["stem"] for job in all_jobs],
    })
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(lane, lane0, 0), executor.submit(lane, lane1, 1)]
            records = [record for future in futures for record in future.result()]
        summary = {
            "status": "complete", "finished": stamp(),
            "wall_seconds": time.monotonic() - started,
            "jobs": sorted(records, key=lambda item: item["checkpoint"]),
        }
        write_json(DEST / "summary.json", summary)
        write_json(DEST / "status.json", summary)
    except BaseException as error:
        write_json(DEST / "status.json", {
            "status": "failed", "finished": stamp(),
            "wall_seconds": time.monotonic() - started, "error": repr(error),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker()
        return
    with (DEST / "launcher.log").open("w") as log:
        process = subprocess.Popen(
            [str(ROOT / ".venv/bin/python"), "-m",
             "scripts.update_v4_27_observables", "--worker"],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    write_json(DEST / "status.json", {
        "status": "launched", "pid": process.pid, "started": stamp(),
    })
    print(json.dumps({"status": "launched", "pid": process.pid, "output": str(DEST)}, indent=2))


if __name__ == "__main__":
    main()
