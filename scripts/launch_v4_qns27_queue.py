"""Detached 27-cell v4 Gamma and outer-C3 production queue."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result/data/v4_qns27_seed3184"
SOURCES = [
    ROOT / "src/qns27.py",
    ROOT / "src/jax_neural_bloch.py",
    ROOT / "src/run_jax_neural_bloch.py",
    ROOT / "src/run_outer_c3_shared_pool.py",
    ROOT / "src/mixture_sr.py",
    ROOT / "src/active_support.py",
    ROOT / "tests/test_qns27.py",
    Path(__file__).resolve(),
]


def stamp():
    return datetime.now().astimezone().isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes():
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in SOURCES
    }


def production_env(gpu=None):
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    if gpu is None:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.update(
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        JAX_COMPILATION_CACHE_DIR=str(ROOT / ".jax-cache"),
        OMP_NUM_THREADS="16",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    return env


def verify_sources():
    expected = json.loads((OUT / "launcher.json").read_text())["source_sha256"]
    actual = source_hashes()
    if actual != expected:
        raise RuntimeError("source changed after launch; refusing mixed-code queue")


def gamma_job(particles, gpu):
    filling = "nu1of3" if particles == 9 else "nu2of3"
    output = OUT / f"{filling}_v4_gamma"
    output.mkdir(parents=True, exist_ok=False)
    command = [
        "/usr/bin/time", "-v", "-o", str(output / "resource_usage.txt"),
        "taskset", "-c", "0-15" if gpu == 0 else "32-47",
        str(ROOT / ".venv/bin/python"), "-m", "src.run_jax_neural_bloch",
        "--output-dir", str(output),
        "--cells", "27", "--particles", str(particles), "--width", "64",
        "--seed", "3184", "--backflow-init-scale", "0.01",
        "--v4-gamma-projected-m", "--samples", "4128", "--steps", "120",
        "--burn-sweeps", "300", "--sweeps-per-step", "2",
        "--wavefunction-batch", "86", "--sr-chunk", "86",
        "--local-energy-batch", "4", "--minor-chunk", "128",
        "--checkpoint-interval", "10", "--log-interval", "1",
        "--cg-true-residual-interval", "5", "--adaptive-cg-120-step",
    ]
    record = {
        "status": "running", "stage": "v4_gamma", "filling": filling,
        "particles": particles, "gpu": gpu, "started": stamp(),
        "command": command,
    }
    started = time.monotonic()
    with (output / "run.log").open("w") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=production_env(gpu),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        record["pid"] = process.pid
        while process.poll() is None:
            record["elapsed_seconds"] = time.monotonic() - started
            try:
                trace = json.loads((output / "jax_neural_bloch_trace.json").read_text())
                record["completed_updates"] = len(trace)
                if trace:
                    record["last_energy_meV_per_particle"] = trace[-1][
                        "energy_per_particle_meV"
                    ]
                    record["rejected_updates"] = trace[-1][
                        "rejected_updates_cumulative"
                    ]
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            write_json(output / "queue_job.json", record)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
    record.update(
        exit_code=process.returncode,
        wall_seconds=time.monotonic() - started,
        finished=stamp(),
    )
    try:
        summary = json.loads((output / "jax_neural_bloch_summary.json").read_text())
        trace = summary["trace"]
        valid = (
            len(trace) == 120
            and summary["protocol"]["translation_projection"][
                "translation_group_order"
            ] == 27
            and all(np.isfinite(row["energy_per_particle_meV"]) for row in trace)
        )
        tail = np.asarray([row["energy_per_particle_meV"] for row in trace[-10:]])
        record.update(
            tail10_mean_meV_per_particle=float(tail.mean()),
            tail10_rms_meV_per_particle=float(tail.std()),
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        valid = False
    record["status"] = "complete" if process.returncode == 0 and valid else "failed"
    write_json(output / "queue_job.json", record)
    return record


def outer_job(particles):
    filling = "nu1of3" if particles == 9 else "nu2of3"
    output = OUT / f"{filling}_v4_outer_c3"
    output.mkdir(parents=True, exist_ok=False)
    command = [
        "/usr/bin/time", "-v", "-o", str(output / "total_resource_usage.txt"),
        str(ROOT / ".venv/bin/python"), "-m", "src.run_outer_c3_shared_pool",
        "--output", str(output), "--cells", "27",
        "--particles", str(particles), "--seed", "3184",
        "--ansatz-version", "v4",
    ]
    record = {
        "status": "running", "stage": "v4_outer_c3", "filling": filling,
        "particles": particles, "gpus": [0, 1], "started": stamp(),
        "command": command,
    }
    started = time.monotonic()
    with (output / "queue.log").open("w") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=production_env(),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        record["pid"] = process.pid
        while process.poll() is None:
            record["elapsed_seconds"] = time.monotonic() - started
            try:
                record["nested_status"] = json.loads(
                    (output / "status.json").read_text()
                )
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            write_json(output / "queue_job.json", record)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
    record.update(
        exit_code=process.returncode,
        wall_seconds=time.monotonic() - started,
        finished=stamp(),
    )
    try:
        summary = json.loads((output / "summary.json").read_text())
        protocol = json.loads((output / "protocol.json").read_text())
        valid = (
            summary.get("status") == "complete"
            and len(summary.get("branches", [])) == 3
            and all(row.get("status") == "complete" for row in summary["branches"])
            and protocol["ansatz_version"] == "v4"
            and protocol["active_gamma_minors"] is None
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        valid = False
    record["status"] = "complete" if process.returncode == 0 and valid else "failed"
    write_json(output / "queue_job.json", record)
    return record


def worker():
    state = {
        "status": "running", "started": stamp(), "pid": os.getpid(),
        "completed_jobs": [], "full_m_repeated": False,
    }
    write_json(OUT / "status.json", state)
    metrics_log = (OUT / "gpu_metrics.csv").open("w")
    monitor = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,temperature.gpu",
            "--format=csv", "-l", "10",
        ],
        stdout=metrics_log, stderr=subprocess.DEVNULL,
    )
    try:
        verify_sources()
        state["phase"] = "v4_gamma_parallel"
        write_json(OUT / "status.json", state)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(gamma_job, 9, 0),
                pool.submit(gamma_job, 18, 1),
            ]
            records = [future.result() for future in futures]
        state["completed_jobs"].extend(records)
        write_json(OUT / "status.json", state)
        if any(record["status"] != "complete" for record in records):
            raise RuntimeError("v4 Gamma stage failed; outer queue not started")

        for particles in (9, 18):
            verify_sources()
            state["phase"] = (
                "v4_outer_c3_nu1of3" if particles == 9
                else "v4_outer_c3_nu2of3"
            )
            write_json(OUT / "status.json", state)
            record = outer_job(particles)
            state["completed_jobs"].append(record)
            write_json(OUT / "status.json", state)
            if record["status"] != "complete":
                raise RuntimeError("v4 outer stage failed; queue stopped")
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        state["updated"] = stamp()
        write_json(OUT / "status.json", state)
        monitor.terminate()
        try:
            monitor.wait(timeout=20)
        except subprocess.TimeoutExpired:
            monitor.kill()
            monitor.wait()
        metrics_log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker()
        return
    if OUT.exists():
        raise FileExistsError(f"refusing duplicate queue: {OUT}")
    OUT.mkdir(parents=True)
    record = {
        "status": "launched", "started": stamp(), "cells": 27,
        "theta_deg": 2.6, "dielectric": 5, "seed": 3184,
        "width": 64, "walkers": 4128, "updates": 120,
        "translation_group_order": 27,
        "gamma_definition": "(1/27) sum_T det(M D_T Q)",
        "outer_protocol": "same-seed shared-mixture20 then private100",
        "backflow_init_scale": 0.01,
        "gamma_batches": {"wavefunction": 86, "local_energy": 4, "sr": 86},
        "outer_batches": {"wavefunction": 43, "local_energy": 1, "sr": 43},
        "full_m_reference": {
            "rerun": False,
            "nu1of3_wall_seconds": 8544.336311407009,
            "nu2of3_wall_seconds": 39218.19207576802,
        },
        "validation": {
            "unit_tests": "tests.test_qns27: 6 passed",
            "gamma_smoke": "finite local energy and SR update",
            "outer_smoke": "shared pool and all three private branches complete",
            "outer_n18_chunk86": "rejected after measured A30 OOM",
            "outer_n18_chunk43": "finite and complete",
        },
        "queue": [
            "parallel v4 Gamma: nu1of3 on GPU0, nu2of3 on GPU1",
            "v4 Outer-C3 nu1of3 on both GPUs",
            "v4 Outer-C3 nu2of3 on both GPUs",
        ],
        "source_sha256": source_hashes(),
    }
    record["pid"] = None
    write_json(OUT / "launcher.json", record)
    with (OUT / "launcher.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    record["pid"] = process.pid
    write_json(OUT / "launcher.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
