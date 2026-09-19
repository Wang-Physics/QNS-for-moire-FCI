"""Resume all ten 27-cell QNS states from step 120 to step 240, serially."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "result/data/v5_qns27_continue_to240"
TARGET_STEP = 240
SOURCES = [
    ROOT / "src/qns27.py",
    ROOT / "src/jax_neural_bloch.py",
    ROOT / "src/run_jax_neural_bloch.py",
    Path(__file__).resolve(),
]


def stamp() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): digest(path) for path in SOURCES}


def jobs() -> list[dict]:
    full = ROOT / "result/data/qns27_width64_seed3184"
    projected = ROOT / "result/data/v4_qns27_seed3184"
    result = []
    for filling, particles in (("nu1of3", 9), ("nu2of3", 18)):
        result.append({
            "name": f"{filling}_full_m",
            "output": full / f"{filling}_full_m",
            "particles": particles,
            "minor_chunk": 1024,
            "wave_batch": 86,
            "energy_batch": 4,
            "sr_chunk": 86,
            "projection_args": [],
        })
        result.append({
            "name": f"{filling}_v5_gamma",
            "output": projected / f"{filling}_v4_gamma",
            "particles": particles,
            "minor_chunk": 128,
            "wave_batch": 86,
            "energy_batch": 4,
            "sr_chunk": 86,
            "projection_args": ["--v4-gamma-projected-m"],
        })
        for sector in range(3):
            result.append({
                "name": f"{filling}_v5_outer_c3_p{sector}",
                "output": projected / f"{filling}_v4_outer_c3/p{sector}",
                "particles": particles,
                "minor_chunk": 256,
                "wave_batch": 43,
                "energy_batch": 1,
                "sr_chunk": 43,
                "projection_args": [
                    "--v4-gamma-projected-m", "--outer-c3-projector",
                    "--c3-irrep", str(sector),
                ],
            })
    return result


def checkpoint_step(path: Path) -> int:
    metadata = path.with_suffix(".json")
    if not path.is_file() or not metadata.is_file():
        return -1
    return int(json.loads(metadata.read_text()).get("training_step", -1))


def resume_checkpoint(output: Path) -> tuple[Path, int]:
    candidates = [
        output / "jax_neural_bloch_latest.npz",
        output / "jax_neural_bloch_final.npz",
        *sorted(output.glob("jax_neural_bloch_step_*.npz")),
    ]
    ranked = [(checkpoint_step(path), path) for path in candidates]
    step, path = max(ranked, key=lambda item: item[0])
    if step < 120 or step > TARGET_STEP:
        raise RuntimeError(f"{output}: invalid resume step {step}")
    return path, step


def environment(gpu: int) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        JAX_COMPILATION_CACHE_DIR=str(ROOT / ".jax-cache"),
        OMP_NUM_THREADS="16",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    return env


def command(job: dict, checkpoint: Path, gpu: int) -> list[str]:
    cpu_range = "0-15" if gpu == 0 else "32-47"
    output = job["output"]
    return [
        "/usr/bin/time", "-v", "-o",
        str(output / "continuation_to240_resource_usage.txt"),
        "taskset", "-c", cpu_range,
        str(ROOT / ".venv/bin/python"), "-m", "src.run_jax_neural_bloch",
        "--output-dir", str(output),
        "--resume", str(checkpoint),
        "--cells", "27", "--particles", str(job["particles"]),
        "--width", "64", "--seed", "3184",
        "--backflow-init-scale", "0.01",
        "--samples", "4128", "--steps", str(TARGET_STEP),
        "--burn-sweeps", "20", "--sweeps-per-step", "2",
        "--wavefunction-batch", str(job["wave_batch"]),
        "--local-energy-batch", str(job["energy_batch"]),
        "--sr-chunk", str(job["sr_chunk"]),
        "--minor-chunk", str(job["minor_chunk"]),
        "--checkpoint-interval", "10", "--log-interval", "1",
        "--cg-true-residual-interval", "5",
        "--checkpoint-continuation",
        *job["projection_args"],
    ]


def verify_sources(expected: dict[str, str]) -> None:
    if source_hashes() != expected:
        raise RuntimeError("training source changed after queue launch")


def worker(gpu: int) -> None:
    launcher = json.loads((QUEUE / "launcher.json").read_text())
    expected_hashes = launcher["source_sha256"]
    state = {
        "status": "running",
        "target_step": TARGET_STEP,
        "gpu": gpu,
        "worker_pid": os.getpid(),
        "started": stamp(),
        "completed_jobs": [],
    }
    write_json(QUEUE / "status.json", state)
    metrics = (QUEUE / "gpu_metrics.csv").open("a")
    monitor = subprocess.Popen([
        "nvidia-smi",
        "--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,temperature.gpu",
        "--format=csv", "-l", "30",
    ], stdout=metrics, stderr=subprocess.DEVNULL)
    try:
        for index, job in enumerate(jobs(), start=1):
            verify_sources(expected_hashes)
            checkpoint, start_step = resume_checkpoint(job["output"])
            if start_step == TARGET_STEP:
                record = {
                    "name": job["name"], "status": "already_complete",
                    "start_step": start_step, "target_step": TARGET_STEP,
                    "finished": stamp(),
                }
                state["completed_jobs"].append(record)
                write_json(QUEUE / "status.json", state)
                continue
            args = command(job, checkpoint, gpu)
            record = {
                "name": job["name"], "status": "running", "queue_index": index,
                "output": str(job["output"]), "checkpoint": str(checkpoint),
                "start_step": start_step, "target_step": TARGET_STEP,
                "started": stamp(), "command": args,
            }
            state["current_job"] = record
            write_json(QUEUE / "status.json", state)
            started = time.monotonic()
            with (job["output"] / "continuation_to240.log").open("a") as log:
                process = subprocess.Popen(
                    args, cwd=ROOT, env=environment(gpu), stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT,
                )
                record["pid"] = process.pid
                while process.poll() is None:
                    record["elapsed_seconds"] = time.monotonic() - started
                    _, current_step = resume_checkpoint(job["output"])
                    record["current_step"] = current_step
                    state["updated"] = stamp()
                    write_json(QUEUE / "status.json", state)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
            _, final_step = resume_checkpoint(job["output"])
            record.update(
                exit_code=process.returncode,
                elapsed_seconds=time.monotonic() - started,
                final_step=final_step,
                finished=stamp(),
            )
            valid = process.returncode == 0 and final_step == TARGET_STEP
            record["status"] = "complete" if valid else "failed"
            state["completed_jobs"].append(record)
            state.pop("current_job", None)
            write_json(QUEUE / "status.json", state)
            if not valid:
                raise RuntimeError(f"continuation failed: {job['name']}")
        state.update(status="complete", finished=stamp())
    except BaseException as error:
        state.update(status="failed", error=repr(error), finished=stamp())
        raise
    finally:
        state["updated"] = stamp()
        write_json(QUEUE / "status.json", state)
        monitor.terminate()
        try:
            monitor.wait(timeout=10)
        except subprocess.TimeoutExpired:
            monitor.kill()
            monitor.wait()
        metrics.close()


def launch(gpu: int) -> None:
    if QUEUE.exists():
        raise FileExistsError(f"refusing duplicate queue: {QUEUE}")
    initial = []
    for job in jobs():
        checkpoint, step = resume_checkpoint(job["output"])
        initial.append({
            "name": job["name"], "output": str(job["output"]),
            "checkpoint": str(checkpoint), "training_step": step,
        })
    if len(initial) != 10 or any(row["training_step"] != 120 for row in initial):
        raise RuntimeError("all ten input states must be at step 120")
    QUEUE.mkdir(parents=True)
    record = {
        "status": "launched", "started": stamp(), "cells": 27,
        "gpu": gpu, "serial": True, "job_count": 10,
        "initial_step": 120, "additional_updates_per_state": 120,
        "target_step": TARGET_STEP, "walkers": 4128,
        "burn_sweeps_on_resume": 20, "sweeps_per_update": 2,
        "checkpoint_selection": "highest saved training_step; no energy selection",
        "initial_jobs": initial, "source_sha256": source_hashes(),
    }
    write_json(QUEUE / "launcher.json", record)
    with (QUEUE / "launcher.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", "--gpu", str(gpu)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    record["worker_pid"] = process.pid
    write_json(QUEUE / "launcher.json", record)
    print(json.dumps(record, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if args.worker:
        worker(args.gpu)
    else:
        launch(args.gpu)


if __name__ == "__main__":
    main()
