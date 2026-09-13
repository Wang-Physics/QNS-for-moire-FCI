"""Validate the compact v4.2 data, Fourier-point fix, figures, and report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.qns27 import cluster, first_bz_c6_grid, first_bz_momenta
from src.qns27_diagnostics import _rotation_permutation


RELEASE = ROOT / "result/release_data/v42"
FIGURES = ROOT / "result/figures"
PDF = ROOT / "result/AI_for_Physics_v4.2.pdf"


def main() -> None:
    manifest = json.loads((RELEASE / "manifest.json").read_text())
    assert manifest["version"] == "4.2.1"
    for relative, metadata in manifest["files"].items():
        content = (RELEASE / relative).read_bytes()
        assert len(content) == metadata["bytes"]
        assert hashlib.sha256(content).hexdigest() == metadata["sha256"]

    plotted = first_bz_momenta()
    audit_grid = first_bz_c6_grid()
    diagnostics = sorted((RELEASE / "qns27/diagnostics").glob("*.npz"))
    assert len(diagnostics) == 10
    nk_c3_residuals = []
    ntotal_c3_residuals = []
    sq_c3_residuals = []
    for path in diagnostics:
        metadata = json.loads(path.with_suffix(".json").read_text())
        with np.load(path) as record:
            k = record["momentum_k_points"]
            q = record["structure_q_vectors"]
            nk = record["momentum_occupation_band1"]
            nk_raw = record["momentum_occupation_band1_raw"]
            nk_total = record["momentum_occupation_first_five"]
            sq = record["charge_structure_factor_full"]
            audit_q = record["structure_q_vectors_c6_audit"]
            audit_sq = record["charge_structure_factor_full_c6_audit"]
            np.testing.assert_allclose(k, plotted, rtol=0.0, atol=1e-14)
            np.testing.assert_allclose(q, plotted, rtol=0.0, atol=1e-14)
            np.testing.assert_allclose(audit_q, audit_grid, rtol=0.0, atol=1e-14)
            assert nk.shape == nk_raw.shape == nk_total.shape == sq.shape == (27,)
            assert audit_sq.shape == (37,)
            assert all(np.all(np.isfinite(value)) for value in (
                nk, nk_raw, nk_total, sq, audit_sq
            ))
            np.testing.assert_allclose(
                nk.sum(), metadata["n_particles"], rtol=0.0, atol=2e-12,
            )
            np.testing.assert_allclose(
                nk_raw.sum(), metadata["n_particles"] * record["band_weight"][0],
                rtol=0.0, atol=2e-10,
            )
            np.testing.assert_allclose(
                nk_total.sum(), record["one_body_density_diagonal"].sum(),
                rtol=0.0, atol=2e-10,
            )
            rotation = _rotation_permutation(audit_q)
            class_rotation = np.asarray(cluster().rotation60)
            class_rotation = class_rotation[class_rotation]
            nk_c3_residuals.append(float(np.max(np.abs(
                nk_raw - nk_raw[class_rotation]
            ))))
            ntotal_c3_residuals.append(float(np.max(np.abs(
                nk_total - nk_total[class_rotation]
            ))))
            sq_c3_residuals.append(float(np.max(np.abs(
                audit_sq - audit_sq[rotation]
            ))))
            for index, point in enumerate(q):
                partner = int(np.argmin(np.linalg.norm(q + point, axis=1)))
                assert np.linalg.norm(q[partner] + point) < 1e-12
                np.testing.assert_allclose(sq[index], sq[partner], atol=2e-14)

    expected_figures = [
        "fig6_9cell_qns_summary.pdf",
        "fig7_27cell_qns_summary.pdf",
        "fig8_27cell_qns_observables.pdf",
        "fig9_27cell_qns_observables_nu1of3.pdf",
        "fig10_27cell_qns_observables_nu2of3.pdf",
        "fig11_cg_stability_9cell_27cell.pdf",
    ]
    for name in expected_figures:
        assert (FIGURES / name).stat().st_size > 10_000

    info = subprocess.run(
        ["pdfinfo", str(PDF)], check=True, text=True, capture_output=True
    ).stdout
    text = subprocess.run(
        ["pdftotext", str(PDF), "-"], check=True, text=True,
        capture_output=True,
    ).stdout
    assert "Pages:           25" in info
    assert "Version 4.2" in text
    assert "Bloch-projected" in text
    assert "37" in text
    assert "band-complete momentum occupation" not in text.lower()
    assert "undefined" not in text.lower()
    print(json.dumps({
        "status": "passed",
        "diagnostics": len(diagnostics),
        "raw_n1_c3_residual_range": [min(nk_c3_residuals), max(nk_c3_residuals)],
        "raw_ntotal_c3_residual_range": [
            min(ntotal_c3_residuals), max(ntotal_c3_residuals)
        ],
        "raw_sq_c3_residual_range": [min(sq_c3_residuals), max(sq_c3_residuals)],
        "manifest_files": len(manifest["files"]),
        "figures": expected_figures,
        "pdf": str(PDF.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
