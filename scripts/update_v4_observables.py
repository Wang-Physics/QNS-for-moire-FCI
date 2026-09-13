"""Export and measure all v4 9-cell report states on two GPUs."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GAMMA = ROOT / "result/data/translation_projected_m_rank1_9cell_seed260904"
OUTER = ROOT / "result/data/v4_outer_c3_9cell"
DEST = ROOT / "result/data/v4_report_observables"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.now().astimezone().isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def jobs():
    result = []
    for filling, particles, seed in (
        ("nu1of3", 3, 26090413), ("nu2of3", 6, 26090423)
    ):
        result.append({
            "stem": f"{filling}_gamma", "filling": filling,
            "particles": particles, "seed": seed,
            "checkpoint": GAMMA / filling / "jax_neural_bloch_final.npz",
            "samples": GAMMA / filling / "jax_neural_bloch_samples.npz",
        })
        for sector in range(3):
            result.append({
                "stem": f"{filling}_outer_c3_p{sector}", "filling": filling,
                "particles": particles, "seed": seed, "sector": sector,
                "checkpoint": OUTER / filling / f"p{sector}/jax_neural_bloch_final.npz",
                "samples": OUTER / filling / f"p{sector}/jax_neural_bloch_samples.npz",
            })
    return result


def process_one(job, gpu):
    output = DEST / job["stem"]
    output.mkdir(parents=True, exist_ok=True)
    provenance = output / "provenance.json"
    if provenance.exists():
        previous = json.loads(provenance.read_text())
        if previous.get("status") == "complete":
            return previous
    checkpoint = job["checkpoint"]
    samples = job["samples"]
    if not checkpoint.exists() or not samples.exists():
        raise FileNotFoundError(f"incomplete v4 state for {job['stem']}")
    metadata = json.loads(checkpoint.with_suffix(".json").read_text())
    if metadata.get("training_step") != 120:
        raise ValueError(f"{job['stem']} is not a completed 120-step state")
    with np.load(checkpoint) as c, np.load(samples) as s:
        for key in ("positions", "layers"):
            np.testing.assert_array_equal(c[key], s[key])
        assert s["positions"].shape == (4128, job["particles"], 2)
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu), XLA_PYTHON_CLIENT_PREALLOCATE="false",
        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="8",
        PYTHONUNBUFFERED="1",
    )
    state = output / "state.pt"
    stages = []
    commands = [
        ("export", [
            str(ROOT / ".venv/bin/python"), "-m", "src.export_jax_neural_bloch",
            "--checkpoint", str(checkpoint), "--output", str(state),
            "--seed", str(job["seed"]),
        ]),
        ("observables", [
            str(ROOT / ".venv/bin/python"), "-m", "src.neural_bloch_diagnostics",
            "--checkpoint", str(state), "--samples", str(samples),
            "--output-dir", str(output), "--samples-count", "4128",
            "--auxiliary-draws", "16", "--error-blocks", "8",
            "--wavefunction-batch", "16", "--threads", "8",
            "--seed", str(job["seed"] + 1701 + 101 * job.get("sector", 0)),
            "--device", "cuda",
        ]),
    ]
    for name, command in commands:
        if name == "observables" and (output / "neural_bloch_diagnostics.json").exists():
            raise FileExistsError(f"refusing to overwrite {job['stem']} observables")
        timed = [
            "/usr/bin/time", "-v", "-o", str(output / f"{name}_resource_usage.txt"),
            "taskset", "-c", "0-15" if gpu == 0 else "32-47", *command,
        ]
        started = time.monotonic()
        with (output / f"{name}.log").open("w") as log:
            subprocess.run(
                timed, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, check=True,
            )
        stages.append({"stage": name, "wall_seconds": time.monotonic() - started})
    result = {
        **{k: v for k, v in job.items() if k not in ("checkpoint", "samples")},
        "status": "complete", "gpu": gpu, "finished": stamp(),
        "checkpoint": str(checkpoint), "samples": str(samples),
        "checkpoint_sha256": digest(checkpoint),
        "samples_sha256": digest(samples), "stages": stages,
        "sampling": "saved final training walkers; no fresh validation chain",
    }
    write_json(provenance, result)
    return result


def lane(queue, gpu):
    return [process_one(job, gpu) for job in queue]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if not args.worker:
        DEST.mkdir(parents=True, exist_ok=True)
        with (DEST / "launcher.log").open("w") as log:
            process = subprocess.Popen(
                [str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve()),
                 "--worker"], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        write_json(DEST / "status.json", {
            "status": "launched", "pid": process.pid, "started": stamp()
        })
        print(json.dumps({"status": "launched", "pid": process.pid,
                          "output": str(DEST)}, indent=2))
        return
    all_jobs = jobs()
    lane0 = all_jobs[::2]
    lane1 = all_jobs[1::2]
    start = time.monotonic()
    write_json(DEST / "status.json", {
        "status": "running", "pid": os.getpid(), "started": stamp(),
        "jobs": [job["stem"] for job in all_jobs],
    })
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(lane, lane0, 0), executor.submit(lane, lane1, 1)]
            records = [item for future in futures for item in future.result()]
        summary = {
            "status": "complete", "finished": stamp(),
            "wall_seconds": time.monotonic() - start,
            "jobs": sorted(records, key=lambda item: item["stem"]),
        }
        write_json(DEST / "summary.json", summary)
        write_json(DEST / "status.json", summary)
    except BaseException as error:
        write_json(DEST / "status.json", {
            "status": "failed", "finished": stamp(),
            "wall_seconds": time.monotonic() - start, "error": repr(error),
        })
        raise


if __name__ == "__main__":
    main()
