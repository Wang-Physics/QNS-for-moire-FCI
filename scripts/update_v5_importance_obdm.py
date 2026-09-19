"""Recompute all 27-cell occupations with the low-variance OBDM estimator."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "result/data/qns27_width64_seed3184"
V4 = ROOT / "result/data/v4_qns27_seed3184"
SAMPLES = ROOT / "result/data/v4_qns27_frozen5_samples"
DEST = ROOT / "result/data/v5_qns27_importance_obdm/final"
AUXILIARY = ROOT / "result/data/v5_qns27_importance_obdm/auxiliary_20640x64.npz"
DESCRIPTION = (
    "20,640 configurations from five frozen-parameter measurement steps; "
    "4,128 walker chains; two Metropolis sweeps between adjacent stored times"
)


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
        result.extend([
            {
                "stem": f"{filling}_full_m",
                "checkpoint": FULL / f"{filling}_full_m/jax_neural_bloch_final.npz",
            },
            {
                "stem": f"{filling}_v4_gamma",
                "checkpoint": V4 / f"{filling}_v4_gamma/jax_neural_bloch_final.npz",
            },
        ])
        for sector in range(3):
            result.append({
                "stem": f"{filling}_v4_outer_c3_p{sector}",
                "checkpoint": V4 / (
                    f"{filling}_v4_outer_c3/p{sector}/jax_neural_bloch_final.npz"
                ),
            })
    return result


def environment(gpu: int) -> dict[str, str]:
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


def complete(path: Path, draws: int, blocks: int) -> bool:
    if not path.is_file():
        return False
    record = json.loads(path.read_text())
    return all((
        record.get("status") == "complete",
        record.get("training_step") == 240,
        record.get("coordinate_samples") == 20640,
        record.get("unique_walker_chains") == 4128,
        record.get("measurement_steps") == 5,
        record.get("parameters_frozen") is True,
        record.get("optimizer_updates") == 0,
        record.get("auxiliary_draws_per_sample") == draws,
        record.get("error_blocks", blocks) == blocks,
        record.get("auxiliary_sampling", "").startswith("PyQMC-style"),
    ))


def run_one(job: dict, gpu: int, draws: int, blocks: int) -> dict:
    output = DEST / job["stem"]
    metadata = output / "qns27_diagnostics.json"
    if complete(metadata, draws, blocks):
        return json.loads(metadata.read_text())
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(ROOT / ".venv/bin/python"), "-m", "src.qns27_importance_obdm",
        "--checkpoint", str(job["checkpoint"]),
        "--samples", str(SAMPLES / f"{job['stem']}.npz"),
        "--auxiliary-cache", str(AUXILIARY),
        "--output-dir", str(output),
        "--samples-count", "0",
        "--auxiliary-draws", str(draws),
        "--error-blocks", str(blocks),
        "--wavefunction-batch", "86",
        "--orbital-batch", "512",
        "--seed", str(250311756 + 101 * len(job["stem"])),
        "--sampling-description", DESCRIPTION,
    ]
    write_json(output / "job.json", {
        "status": "running", "stem": job["stem"], "gpu": gpu,
        "started": stamp(), "command": command,
    })
    started = time.monotonic()
    with (output / "run.log").open("w") as log:
        process = subprocess.run(
            ["taskset", "-c", "0-15" if gpu == 0 else "32-47", *command],
            cwd=ROOT, env=environment(gpu), stdout=log,
            stderr=subprocess.STDOUT,
        )
    write_json(output / "job.json", {
        "status": "complete" if process.returncode == 0 else "failed",
        "stem": job["stem"], "gpu": gpu, "finished": stamp(),
        "wall_seconds": time.monotonic() - started,
        "exit_code": process.returncode,
    })
    if process.returncode:
        raise RuntimeError(f"{job['stem']} failed with {process.returncode}")
    record = json.loads(metadata.read_text())
    record["error_blocks"] = blocks
    write_json(metadata, record)
    return record


def lane(queue: list[dict], gpu: int, draws: int, blocks: int) -> list[dict]:
    return [run_one(job, gpu, draws, blocks) for job in queue]


def add_symmetry_improved_outer_estimator(stem: str) -> None:
    """Store, but do not substitute for, the raw outer-C3 occupations."""
    if "outer_c3" not in stem:
        return
    from src.qns27 import cluster
    path = DEST / stem / "qns27_diagnostics.npz"
    with np.load(path) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    block_diagonal = payload["one_body_density_diagonal_block_estimates"]
    c3 = np.asarray(cluster().rotation60)
    c3 = c3[c3]
    c3_squared = c3[c3]

    def improve(values: np.ndarray) -> np.ndarray:
        return (values + values[..., c3] + values[..., c3_squared]) / 3.0

    band1_blocks = block_diagonal[:, :, 0]
    total_blocks = block_diagonal.sum(axis=2)
    band1 = improve(payload["momentum_occupation_band1_raw"])
    total = improve(payload["momentum_occupation_first_five_raw"])
    band1_block_improved = improve(band1_blocks)
    total_block_improved = improve(total_blocks)
    payload.update({
        "momentum_occupation_band1_c3_improved": band1,
        "momentum_occupation_band1_c3_improved_sem": (
            band1_block_improved.std(axis=0, ddof=1)
            / np.sqrt(len(band1_block_improved))
        ),
        "momentum_occupation_first_five_c3_improved": total,
        "momentum_occupation_first_five_c3_improved_sem": (
            total_block_improved.std(axis=0, ddof=1)
            / np.sqrt(len(total_block_improved))
        ),
    })
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def worker(draws: int, blocks: int) -> None:
    if not AUXILIARY.is_file():
        raise FileNotFoundError(AUXILIARY)
    by_name = {job["stem"]: job for job in jobs()}
    # The 18-particle outer projectors dominate.  Put two on lane 0 and one
    # plus both cheaper 18-particle unprojected states on lane 1, then fill
    # with the 9-particle jobs.
    lane0_names = [
        "nu2of3_v4_outer_c3_p0", "nu2of3_v4_outer_c3_p2",
        "nu1of3_full_m", "nu1of3_v4_gamma", "nu1of3_v4_outer_c3_p2",
    ]
    lane1_names = [
        "nu2of3_v4_outer_c3_p1", "nu2of3_full_m", "nu2of3_v4_gamma",
        "nu1of3_v4_outer_c3_p0", "nu1of3_v4_outer_c3_p1",
    ]
    lanes = [[by_name[name] for name in lane0_names], [by_name[name] for name in lane1_names]]
    started = time.monotonic()
    write_json(DEST / "status.json", {
        "status": "running", "started": stamp(), "draws": draws,
        "error_blocks": blocks,
    })
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(lane, lanes[gpu], gpu, draws, blocks)
                for gpu in range(2)
            ]
            records = [item for future in futures for item in future.result()]
        # Deterministic single-particle audit is common to every state.  Add it
        # here as well so results completed by an older resumed worker receive
        # the same metadata without rerunning any many-body ratios.
        from src.qns27_importance_obdm import _five_band_c3_sewing_audit
        sewing_audit = _five_band_c3_sewing_audit()
        records = []
        for job in jobs():
            metadata = DEST / job["stem"] / "qns27_diagnostics.json"
            add_symmetry_improved_outer_estimator(job["stem"])
            record = json.loads(metadata.read_text())
            record["bloch_projector_c3_sewing_audit"] = sewing_audit
            if "outer_c3" in job["stem"]:
                record["c3_improved_estimator"] = (
                    "equal-weight orbit average of the three raw momentum "
                    "projectors, stored as an explicit variance-reduced audit; "
                    "raw occupations remain the plotted primary result"
                )
            write_json(metadata, record)
            records.append(record)
        summary = {
            "status": "complete", "finished": stamp(), "draws": draws,
            "error_blocks": blocks, "wall_seconds": time.monotonic() - started,
            "jobs": sorted(records, key=lambda item: item["samples"]),
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
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--error-blocks", type=int, default=32)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.detach:
        DEST.mkdir(parents=True, exist_ok=True)
        with (DEST / "launcher.log").open("a") as log:
            process = subprocess.Popen(
                [str(ROOT / ".venv/bin/python"), "-m",
                 "scripts.update_v5_importance_obdm", "--draws", str(args.draws),
                 "--error-blocks", str(args.error_blocks)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
        print(json.dumps({"pid": process.pid, "status": "launched"}))
        return
    worker(args.draws, args.error_blocks)


if __name__ == "__main__":
    main()
