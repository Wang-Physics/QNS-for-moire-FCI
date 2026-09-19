"""Assemble the compact, figure-reproducible v5 data bundle."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "result/data"
DEST = ROOT / "result/release_data/v5"
NINE = DATA / "local_v4_final"
QNS27 = DATA / "v4_qns27_seed3184"
FULL27 = DATA / "qns27_width64_seed3184"
OBS27 = DATA / "v5_qns27_importance_obdm/final"
FROZEN_SAMPLES = DATA / "v4_qns27_frozen5_samples"
FROZEN_COORD = DATA / "v4_qns27_frozen5_coordinate"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def portable_metadata(value: object) -> object:
    """Replace repository-local absolute paths in copied JSON metadata."""
    if isinstance(value, dict):
        return {key: portable_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_metadata(item) for item in value]
    if isinstance(value, str):
        return value.replace(f"{ROOT}/", "")
    return value


def copy(source: Path, destination: Path, manifest: dict[str, dict]) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".json":
        write_json(destination, portable_metadata(json.loads(source.read_text())))
    else:
        shutil.copy2(source, destination)
    content = destination.read_bytes()
    manifest[str(destination.relative_to(DEST))] = {
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "source": str(source.relative_to(ROOT)),
    }


def main() -> None:
    status = json.loads((OBS27 / "status.json").read_text())
    if status.get("status") != "complete":
        raise RuntimeError("27-cell v5 observables are not complete")

    manifest: dict[str, dict] = {}
    copy(NINE / "summary.json", DEST / "nine/summary.json", manifest)
    for source in sorted((NINE / "traces").glob("*.json")):
        copy(source, DEST / "nine/traces" / source.name, manifest)
    for source in sorted((NINE / "diagnostics").glob("*.npz")):
        copy(source, DEST / "nine/diagnostics" / source.name, manifest)

    continuation_status = json.loads(
        (DATA / "v5_qns27_continue_to240/status.json").read_text()
    )
    continuation_jobs = {
        item["name"]: item for item in continuation_status["completed_jobs"]
    }
    gpu1_p2 = json.loads(
        (DATA / "v5_qns27_continue_to240/gpu1_nu2_p2_status.json").read_text()
    )
    continuation_jobs[gpu1_p2["name"]] = gpu1_p2

    timing = {}
    for filling in ("nu1of3", "nu2of3"):
        stems = [f"{filling}_full_m", f"{filling}_v4_gamma"] + [
            f"{filling}_v4_outer_c3_p{sector}" for sector in range(3)
        ]
        for stem in stems:
            source = OBS27 / stem
            copy(
                source / "qns27_diagnostics.npz",
                DEST / "qns27/diagnostics" / f"{stem}.npz",
                manifest,
            )
            copy(
                source / "qns27_diagnostics.json",
                DEST / "qns27/diagnostics" / f"{stem}.json",
                manifest,
            )
            copy(
                FROZEN_COORD / stem / "coordinate_diagnostics.npz",
                DEST / "qns27/frozen5_coordinate" / f"{stem}.npz",
                manifest,
            )
            copy(
                FROZEN_COORD / stem / "coordinate_diagnostics.json",
                DEST / "qns27/frozen5_coordinate" / f"{stem}.json",
                manifest,
            )
            copy(
                FROZEN_SAMPLES / f"{stem}.json",
                DEST / "qns27/frozen5_sampling" / f"{stem}.json",
                manifest,
            )

        copy(
            FULL27 / f"{filling}_full_m/jax_neural_bloch_trace.json",
            DEST / "qns27/traces" / f"{filling}_full_m.json",
            manifest,
        )
        copy(
            QNS27 / f"{filling}_v4_gamma/jax_neural_bloch_trace.json",
            DEST / "qns27/traces" / f"{filling}_v4_gamma.json",
            manifest,
        )
        for sector in range(3):
            copy(
                QNS27 / (
                    f"{filling}_v4_outer_c3/p{sector}/jax_neural_bloch_trace.json"
                ),
                DEST / "qns27/traces" / f"{filling}_v4_outer_c3_p{sector}.json",
                manifest,
            )

        full = json.loads(
            (FULL27 / f"{filling}_full_m/queue_job.json").read_text()
        )
        gamma = json.loads(
            (QNS27 / f"{filling}_v4_gamma/queue_job.json").read_text()
        )
        outer = json.loads(
            (QNS27 / f"{filling}_v4_outer_c3/summary.json").read_text()
        )
        continuation_prefix = "nu1of3" if filling == "nu1of3" else "nu2of3"
        full_continuation = continuation_jobs[f"{continuation_prefix}_full_m"]
        gamma_continuation = continuation_jobs[f"{continuation_prefix}_v5_gamma"]
        outer_continuations = [
            continuation_jobs[f"{continuation_prefix}_v5_outer_c3_p{sector}"]
            for sector in range(3)
        ]
        for record, extension in (
            (full, full_continuation), (gamma, gamma_continuation)
        ):
            record["wall_seconds_first_120"] = record["wall_seconds"]
            record["continuation_wall_seconds"] = extension["elapsed_seconds"]
            record["wall_seconds"] += extension["elapsed_seconds"]
            record["training_steps"] = 240
        continuation_start = min(
            datetime.fromisoformat(item["started"])
            for item in outer_continuations
        )
        continuation_finish = max(
            datetime.fromisoformat(item["finished"])
            for item in outer_continuations
        )
        outer["wall_seconds_first_120"] = outer["wall_seconds"]
        outer["continuation_wall_seconds"] = (
            continuation_finish - continuation_start
        ).total_seconds()
        outer["wall_seconds"] += outer["continuation_wall_seconds"]
        outer["continuation_gpu_allocated_seconds"] = sum(
            item["elapsed_seconds"] for item in outer_continuations
        )
        outer["training_steps"] = 240
        timing[filling] = {"full": full, "gamma": gamma, "outer": outer}

    write_json(DEST / "qns27/timing.json", portable_metadata(timing))
    timing_content = (DEST / "qns27/timing.json").read_bytes()
    manifest["qns27/timing.json"] = {
        "bytes": len(timing_content),
        "sha256": hashlib.sha256(timing_content).hexdigest(),
        "source": "selected local queue metadata",
    }
    copy(
        FROZEN_COORD / "summary.json",
        DEST / "qns27/frozen5_coordinate/summary.json",
        manifest,
    )
    for source, name in (
        (OBS27 / "summary.json", "production_summary.json"),
        (OBS27.parent / "auxiliary_20640x64.json", "auxiliary_sampling.json"),
    ):
        copy(source, DEST / "qns27/importance_obdm" / name, manifest)
    if (OBS27 / "comparison.json").is_file():
        copy(OBS27 / "comparison.json",
             DEST / "qns27/importance_obdm/comparison.json", manifest)
    write_json(DEST / "manifest.json", {
        "version": "5.0.0",
        "observable_definition": (
            "Bloch-projected n_tot(k) is the raw sum over bands 1:5 and n_1(k) "
            "is the raw physical band-1 projection with sum N_e W_1; normalized "
            "variants are audit-only; the one-body RDM uses 20,640 frozen-chain "
            "configurations, 64 auxiliary draws from the PyQMC orbital-mixture "
            "importance density, and an orbital-normalization control that does "
            "not constrain the many-body trace; k is evaluated directly "
            "at the 27 drawn first-BZ points and G used only inside each Bloch "
            "orbital; plotted S(q) and rho(r) are evaluated from 20,640 "
            "configurations collected in five measurement-only steps with the "
            "step-240 parameters frozen; S(q) is the unbinned Fourier transform "
            "of the real-space density-pair correlation at the same 27 physical "
            "first-BZ vectors drawn for ED; the uniform 91-vector grid is "
            "auxiliary only; no plotted C3 averaging, q+G "
            "substitution, or ED-profile fit"
        ),
        "files": manifest,
    })


if __name__ == "__main__":
    main()
