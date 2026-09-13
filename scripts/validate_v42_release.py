"""Validate the compact v4.2 data, Fourier-point fix, figures, and report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "result/release_data/v42"
FIGURES = ROOT / "result/figures"
PDF = ROOT / "result/AI_for_Physics_v4.2.pdf"


def main() -> None:
    manifest = json.loads((RELEASE / "manifest.json").read_text())
    assert manifest["version"] == "4.2.0"
    for relative, metadata in manifest["files"].items():
        content = (RELEASE / relative).read_bytes()
        assert len(content) == metadata["bytes"]
        assert hashlib.sha256(content).hexdigest() == metadata["sha256"]

    plotted = np.asarray(json.loads(
        (ROOT / "result/data/fig5_27cell_observables.json").read_text()
    )["k_points_hex"])
    diagnostics = sorted((RELEASE / "qns27/diagnostics").glob("*.npz"))
    assert len(diagnostics) == 10
    for path in diagnostics:
        with np.load(path) as record:
            k = record["momentum_k_points"]
            q = record["structure_q_vectors"]
            nk = record["momentum_occupation_plane_wave"]
            sq = record["charge_structure_factor_full"]
            np.testing.assert_allclose(k, plotted, rtol=0.0, atol=1e-14)
            np.testing.assert_allclose(q, plotted, rtol=0.0, atol=1e-14)
            assert nk.shape == sq.shape == (27,)
            assert np.all(np.isfinite(nk)) and np.all(np.isfinite(sq))
            assert int(np.argmax(sq)) in (25, 26)
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
    assert "direct G = 0" in text
    assert "without a reciprocal-image sum" in text
    assert "band-complete momentum occupation" not in text.lower()
    assert "undefined" not in text.lower()
    print(json.dumps({
        "status": "passed",
        "diagnostics": len(diagnostics),
        "manifest_files": len(manifest["files"]),
        "figures": expected_figures,
        "pdf": str(PDF.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
