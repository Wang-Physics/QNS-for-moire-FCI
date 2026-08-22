from __future__ import annotations

import argparse
import json
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_ARTIFACTS = [
    "result/AI_for_Physics.pdf",
    "result/AI_for_Physics.tex",
    "result/neural_bloch_part2.tex",
    "result/figures/fig1_bands_quantum_geometry.pdf",
    "result/figures/fig2_multiband_ed.pdf",
    "result/figures/fig3_fig5_spectra_observables.pdf",
    "result/figures/fig4_manybody_berry_curvature.pdf",
    "result/figures/fig6_neural_bloch_results.pdf",
    "result/figures/fig7_neural_bloch_observables.pdf",
    "result/release_data/validation_summary.json",
]


def validate_publication() -> None:
    missing = [name for name in PUBLICATION_ARTIFACTS if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"missing publication artifacts: {missing}")
    undersized = [
        name for name in PUBLICATION_ARTIFACTS
        if (ROOT / name).suffix in {".pdf", ".npz"}
        and (ROOT / name).stat().st_size < 10_000
    ]
    if undersized:
        raise SystemExit(f"suspiciously small publication artifacts: {undersized}")


def validate_run(run_dir: Path) -> None:
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    required = ["resolved_config.yaml", "manifest.json", "metrics.jsonl", "summary.json"]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise SystemExit(f"missing run artifacts in {run_dir}: {missing}")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    schema = json.loads((ROOT / "results" / "schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(summary, schema)
    if not all(summary["checks"].values()):
        raise SystemExit(f"failed checks: {summary['checks']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", type=Path)
    args = parser.parse_args()
    validate_publication()
    if args.run_dir is not None:
        validate_run(args.run_dir)
    print("validated publication artifacts" + (
        f" and {args.run_dir}" if args.run_dir is not None else ""
    ))


if __name__ == "__main__":
    main()
