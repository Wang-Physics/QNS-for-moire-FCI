"""Run both 9-cell v4 outer-C3 fillings sequentially in a detached queue."""

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result/data/v4_outer_c3_9cell"


def stamp():
    return datetime.now().astimezone().isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def worker():
    jobs = ((3, 26090413, "nu1of3"), (6, 26090423, "nu2of3"))
    state = {"status": "running", "started": stamp(), "pid": os.getpid(),
             "completed": []}
    write_json(OUT / "status.json", state)
    try:
        for particles, seed, filling in jobs:
            output = OUT / filling
            command = [
                "/usr/bin/time", "-v", "-o", str(output / "total_resource_usage.txt"),
                str(ROOT / ".venv/bin/python"), "-m", "src.run_outer_c3_shared_pool",
                "--output", str(output), "--seed", str(seed),
                "--particles", str(particles), "--cells", "9",
                "--ansatz-version", "v4",
            ]
            output.mkdir(parents=True, exist_ok=False)
            state.update(current=filling, current_seed=seed, command=command)
            write_json(OUT / "status.json", state)
            started = time.monotonic()
            with (output / "launcher.log").open("w") as log:
                process = subprocess.Popen(
                    command, cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                state["child_pid"] = process.pid
                write_json(OUT / "status.json", state)
                process.wait()
            summary_path = output / "summary.json"
            complete = False
            if process.returncode == 0 and summary_path.exists():
                summary = json.loads(summary_path.read_text())
                complete = summary.get("status") == "complete"
            record = {
                "filling": filling, "particles": particles, "seed": seed,
                "exit_code": process.returncode,
                "wall_seconds": time.monotonic() - started,
                "status": "complete" if complete else "failed",
            }
            state["completed"].append(record)
            write_json(OUT / "status.json", state)
            if not complete:
                raise RuntimeError(f"v4 outer-C3 {filling} failed")
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        state["finished"] = stamp()
        write_json(OUT / "status.json", state)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker()
        return
    if OUT.exists():
        raise FileExistsError(f"refusing duplicate queue: {OUT}")
    OUT.mkdir(parents=True)
    sources = [
        Path(__file__).resolve(), ROOT / "src/jax_neural_bloch.py",
        ROOT / "src/run_jax_neural_bloch.py",
        ROOT / "src/run_outer_c3_shared_pool.py", ROOT / "src/mixture_sr.py",
    ]
    with (OUT / "queue.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    record = {
        "status": "launched", "started": stamp(), "pid": process.pid,
        "ansatz_version": "v4", "cells": 9, "walkers": 4128,
        "protocol": "three same-seed sectors; shared mixture20 then private100",
        "seeds": {"nu1of3": 26090413, "nu2of3": 26090423},
        "timing": "per-filling /usr/bin/time plus 10-second nvidia-smi metrics",
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
    }
    write_json(OUT / "launcher.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
