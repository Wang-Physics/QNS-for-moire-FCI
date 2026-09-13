"""Assemble the compact, figure-reproducible v4.2 data bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "result/data"
DEST = ROOT / "result/release_data/v42"
NINE = DATA / "local_v4_final"
QNS27 = DATA / "v4_qns27_seed3184"
FULL27 = DATA / "qns27_width64_seed3184"
OBS27 = DATA / "v4_qns27_observables"


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
        raise RuntimeError("27-cell v4.2 observables are not complete")

    manifest: dict[str, dict] = {}
    copy(NINE / "summary.json", DEST / "nine/summary.json", manifest)
    for source in sorted((NINE / "traces").glob("*.json")):
        copy(source, DEST / "nine/traces" / source.name, manifest)
    for source in sorted((NINE / "diagnostics").glob("*.npz")):
        copy(source, DEST / "nine/diagnostics" / source.name, manifest)

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
        timing[filling] = {"full": full, "gamma": gamma, "outer": outer}

    write_json(DEST / "qns27/timing.json", portable_metadata(timing))
    timing_content = (DEST / "qns27/timing.json").read_bytes()
    manifest["qns27/timing.json"] = {
        "bytes": len(timing_content),
        "sha256": hashlib.sha256(timing_content).hexdigest(),
        "source": "selected local queue metadata",
    }
    write_json(DEST / "manifest.json", {
        "version": "4.2.0",
        "observable_definition": (
            "direct G=0 layer-summed plane-wave n(k) and full S(q), both at "
            "the exact 27 plotted first-BZ vectors with no reciprocal-image sum"
        ),
        "files": manifest,
    })


if __name__ == "__main__":
    main()
