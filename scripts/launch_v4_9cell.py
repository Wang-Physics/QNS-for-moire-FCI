"""Detached 9-cell v4 single-M exact-Gamma comparison queue."""

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result/data/translation_projected_m_rank1_9cell_seed260904"


def stamp():
    return datetime.now().astimezone().isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_one(particles, seed):
    filling = "nu1of3" if particles == 3 else "nu2of3"
    output = OUT / filling
    output.mkdir(parents=True, exist_ok=False)
    command = [
        "/usr/bin/time", "-v", "-o", str(output / "resource_usage.txt"),
        "taskset", "-c", "0-15", str(ROOT / ".venv/bin/python"),
        "-m", "src.run_jax_neural_bloch", "--output-dir", str(output),
        "--particles", str(particles), "--cells", "9", "--width", "32",
        "--message-passing-steps", "2", "--determinants", "1",
        "--v4-gamma-projected-m", "--samples", "4128",
        "--steps", "120", "--learning-rate", "0.002",
        "--burn-sweeps", "300", "--sweeps-per-step", "2",
        "--proposal-scale", "0.06", "--wavefunction-batch", "258",
        "--local-energy-batch", "32", "--sr-chunk", "258",
        "--sr-damping", "0.01", "--cg-min-iterations", "20",
        "--cg-max-iterations", "60", "--cg-true-tolerance", "0.02",
        "--cg-acceptance-tolerance", "0.05", "--recovery-damping", "0.03",
        "--recovery-learning-rate", "0.001", "--recovery-stable-steps", "2",
        "--checkpoint-interval", "10", "--log-interval", "10",
        "--cg-true-residual-interval", "5", "--seed", str(seed),
        "--adaptive-cg-120-step",
    ]
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    env.update(
        CUDA_VISIBLE_DEVICES="0", XLA_PYTHON_CLIENT_PREALLOCATE="false",
        JAX_COMPILATION_CACHE_DIR=str(ROOT / ".jax-cache"),
        OMP_NUM_THREADS="16", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    record = {
        "status": "running", "particles": particles, "filling": filling,
        "seed": seed, "gpu": 0, "started": stamp(), "command": command,
    }
    started = time.monotonic()
    with (output / "run.log").open("w") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        record["pid"] = process.pid
        while process.poll() is None:
            record["elapsed_seconds"] = time.monotonic() - started
            try:
                trace = json.loads((output / "jax_neural_bloch_trace.json").read_text())
                record["completed_updates"] = len(trace)
                if trace:
                    record["last_energy_meV_per_particle"] = trace[-1]["energy_per_particle_meV"]
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            write(output / "status.json", record)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
    record.update(
        exit_code=process.returncode, wall_seconds=time.monotonic() - started,
        finished=stamp(),
    )
    try:
        summary = json.loads((output / "jax_neural_bloch_summary.json").read_text())
        trace = summary["trace"]
        valid = len(trace) == 120 and all(
            np.isfinite(row["energy_per_particle_meV"]) for row in trace
        )
        tail = np.asarray([row["energy_per_particle_meV"] for row in trace[-10:]])
        record.update(
            tail10_mean_meV_per_particle=float(tail.mean()),
            tail10_rms_meV_per_particle=float(tail.std()),
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        valid = False
    record["status"] = "complete" if process.returncode == 0 and valid else "failed"
    write(output / "status.json", record)
    return record


def worker():
    state = {"status": "running", "started": stamp(), "pid": os.getpid(),
             "completed": []}
    metrics_log = (OUT / "gpu_metrics.csv").open("w")
    monitor = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,"
         "memory.used,power.draw,temperature.gpu", "--format=csv", "-l", "5"],
        stdout=metrics_log, stderr=subprocess.DEVNULL,
    )
    try:
        for particles, seed in ((3, 26090413), (6, 26090423)):
            state["current"] = "nu1of3" if particles == 3 else "nu2of3"
            write(OUT / "status.json", state)
            result = run_one(particles, seed)
            state["completed"].append(result)
            if result["status"] != "complete":
                raise RuntimeError(f"projected-M N={particles} failed")
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        monitor.terminate()
        try:
            monitor.wait(timeout=10)
        except subprocess.TimeoutExpired:
            monitor.kill()
            monitor.wait()
        metrics_log.close()
        state["updated"] = stamp()
        write(OUT / "status.json", state)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker()
        return
    if OUT.exists():
        raise FileExistsError(f"refusing duplicate run: {OUT}")
    OUT.mkdir(parents=True)
    sources = [
        ROOT / "src/jax_neural_bloch.py", ROOT / "src/run_jax_neural_bloch.py",
        ROOT / "tests/test_jax_neural_bloch.py", Path(__file__).resolve(),
    ]
    protocol = {
        "status": "launched", "started": stamp(), "cells": 9,
        "ansatz_version": "v4", "dense_m_count": 1,
        "definition": "(1/9) sum_T det(M D_T Q)",
        "translation_group_order": 9, "walkers": 4128,
        "timing": (
            "per-job /usr/bin/time plus 5-second nvidia-smi utilization, "
            "memory, power and temperature samples"
        ),
        "steps": 120, "gpu": 0,
        "baseline": "result/data/gpu_vmc_adaptive_cg120_seed260904",
        "jobs": [
            {"filling": "1/3", "particles": 3, "seed": 26090413},
            {"filling": "2/3", "particles": 6, "seed": 26090423},
        ],
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
    }
    with (OUT / "launcher.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    protocol["pid"] = process.pid
    write(OUT / "launcher.json", protocol)
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
