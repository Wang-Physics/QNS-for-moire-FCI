"""Run the nu=2/3 outer-C3 P2 continuation on GPU 1 in the background."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "result/data/v5_qns27_continue_to240"
OUTPUT = ROOT / "result/data/v4_qns27_seed3184/nu2of3_v4_outer_c3/p2"
STATUS = QUEUE / "gpu1_nu2_p2_status.json"
LAUNCHER = QUEUE / "gpu1_nu2_p2_launcher.json"
TARGET_STEP = 240


def stamp() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checkpoint_step(path: Path) -> int:
    metadata = path.with_suffix(".json")
    if not path.is_file() or not metadata.is_file():
        return -1
    return int(json.loads(metadata.read_text()).get("training_step", -1))


def resume_checkpoint() -> tuple[Path, int]:
    candidates = [
        OUTPUT / "jax_neural_bloch_latest.npz",
        OUTPUT / "jax_neural_bloch_final.npz",
        *sorted(OUTPUT.glob("jax_neural_bloch_step_*.npz")),
    ]
    step, path = max(
        ((checkpoint_step(path), path) for path in candidates),
        key=lambda item: item[0],
    )
    return path, step


def command(checkpoint: Path) -> list[str]:
    return [
        "/usr/bin/time", "-v", "-o",
        str(OUTPUT / "continuation_to240_gpu1_resource_usage.txt"),
        "taskset", "-c", "32-47",
        str(ROOT / ".venv/bin/python"), "-m", "src.run_jax_neural_bloch",
        "--output-dir", str(OUTPUT), "--resume", str(checkpoint),
        "--cells", "27", "--particles", "18", "--width", "64",
        "--seed", "3184", "--backflow-init-scale", "0.01",
        "--samples", "4128", "--steps", str(TARGET_STEP),
        "--burn-sweeps", "20", "--sweeps-per-step", "2",
        "--wavefunction-batch", "43", "--local-energy-batch", "1",
        "--sr-chunk", "43", "--minor-chunk", "256",
        "--checkpoint-interval", "10", "--log-interval", "1",
        "--cg-true-residual-interval", "5", "--checkpoint-continuation",
        "--v4-gamma-projected-m", "--outer-c3-projector", "--c3-irrep", "2",
    ]


def worker() -> None:
    checkpoint, start_step = resume_checkpoint()
    if start_step != 120:
        raise RuntimeError(f"P2 must start at step 120, found {start_step}")
    args = command(checkpoint)
    record = {
        "status": "running", "gpu": 1, "name": "nu2of3_v5_outer_c3_p2",
        "worker_pid": os.getpid(), "started": stamp(),
        "start_step": start_step, "target_step": TARGET_STEP,
        "checkpoint": str(checkpoint), "command": args,
    }
    write_json(STATUS, record)
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    env.update(
        CUDA_VISIBLE_DEVICES="1", XLA_PYTHON_CLIENT_PREALLOCATE="false",
        JAX_COMPILATION_CACHE_DIR=str(ROOT / ".jax-cache"),
        OMP_NUM_THREADS="16", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    started = time.monotonic()
    with (OUTPUT / "continuation_to240_gpu1.log").open("a") as log:
        process = subprocess.Popen(
            args, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT,
        )
        record["training_pid"] = process.pid
        while process.poll() is None:
            _, current_step = resume_checkpoint()
            record.update(
                current_step=current_step,
                elapsed_seconds=time.monotonic() - started,
                updated=stamp(),
            )
            write_json(STATUS, record)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
    _, final_step = resume_checkpoint()
    record.update(
        exit_code=process.returncode, final_step=final_step,
        elapsed_seconds=time.monotonic() - started, finished=stamp(),
        status=(
            "complete"
            if process.returncode == 0 and final_step == TARGET_STEP
            else "failed"
        ),
    )
    write_json(STATUS, record)
    if record["status"] != "complete":
        raise RuntimeError("GPU-1 P2 continuation failed")


def launch() -> None:
    if STATUS.exists() or LAUNCHER.exists():
        raise FileExistsError("GPU-1 P2 continuation already launched")
    checkpoint, step = resume_checkpoint()
    if step != 120:
        raise RuntimeError(f"P2 must be at step 120, found {step}")
    record = {
        "status": "launched", "gpu": 1, "serial_queue_gpu": 0,
        "name": "nu2of3_v5_outer_c3_p2", "started": stamp(),
        "start_step": step, "target_step": TARGET_STEP,
        "checkpoint": str(checkpoint),
        "coordination": "GPU0 runs P0 then P1; main queue skips P2 at step 240",
    }
    write_json(LAUNCHER, record)
    with (QUEUE / "gpu1_nu2_p2_launcher.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    record["worker_pid"] = process.pid
    write_json(LAUNCHER, record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker()
    else:
        launch()
