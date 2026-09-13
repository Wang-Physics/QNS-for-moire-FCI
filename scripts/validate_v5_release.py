"""Validate the compact v5 data, Fourier-point fix, figures, and report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.qns27 import cluster, first_bz_momenta
from src.qns27_diagnostics import _physical_rotation_residual
from src.make_v5_figures import observable_scales


RELEASE = ROOT / "result/release_data/v5"
FIGURES = ROOT / "result/figures"
PDF = ROOT / "result/AI_for_Physics_v5.pdf"


def main() -> None:
    _, n1_scale, total_scale, structure_scale, _ = observable_scales()
    assert (n1_scale.vmin, n1_scale.vmax) == (0.0, 0.8)
    assert (total_scale.vmin, total_scale.vmax) == (0.0, 0.8)
    assert (structure_scale.vmin, structure_scale.vmax) == (0.0, 1.1)
    manifest = json.loads((RELEASE / "manifest.json").read_text())
    assert manifest["version"] == "5.0.0"
    for relative, metadata in manifest["files"].items():
        content = (RELEASE / relative).read_bytes()
        assert len(content) == metadata["bytes"]
        assert hashlib.sha256(content).hexdigest() == metadata["sha256"]

    plotted = first_bz_momenta()
    diagnostics = sorted((RELEASE / "qns27/diagnostics").glob("*.npz"))
    coordinate_diagnostics = sorted(
        (RELEASE / "qns27/frozen5_coordinate").glob("*.npz")
    )
    assert len(diagnostics) == 10
    assert len(coordinate_diagnostics) == 10
    nk_c3_residuals = []
    ntotal_c3_residuals = []
    sq_c3_residuals = []
    for path, coordinate_path in zip(diagnostics, coordinate_diagnostics):
        assert path.name == coordinate_path.name
        metadata = json.loads(path.with_suffix(".json").read_text())
        coordinate_metadata = json.loads(
            coordinate_path.with_suffix(".json").read_text()
        )
        expected_draws = 128
        assert metadata["auxiliary_draws_per_sample"] == expected_draws
        assert metadata["coordinate_samples"] == 4128
        assert metadata["auxiliary_sampling"].startswith("Cranley-Patterson")
        assert "physical first-BZ k points" in metadata["one_body_estimator"]
        assert metadata["momentum_occupation_first_five_raw_sum_sem"] >= 0.0
        assert coordinate_metadata["coordinate_samples"] == 20640
        assert coordinate_metadata["unique_walker_chains"] == 4128
        assert coordinate_metadata["measurement_steps"] == 5
        assert coordinate_metadata["parameters_frozen"] is True
        assert coordinate_metadata["optimizer_updates"] == 0
        assert coordinate_metadata["structure_factor_grid_points"] == 27
        assert coordinate_metadata["structure_factor_projected_density"] is False
        assert coordinate_metadata[
            "structure_factor_c3_matched_physical_points"
        ] == 21
        with np.load(path) as record, np.load(coordinate_path) as coordinate:
            k = record["momentum_k_points"]
            q = coordinate["structure_q_vectors"]
            nk = record["momentum_occupation_band1"]
            nk_raw = record["momentum_occupation_band1_raw"]
            nk_total = record["momentum_occupation_first_five"]
            nk_total_raw = record["momentum_occupation_first_five_raw"]
            sq = coordinate["charge_structure_factor"]
            np.testing.assert_allclose(k, plotted, rtol=0.0, atol=1e-14)
            np.testing.assert_allclose(q, plotted, rtol=0.0, atol=1e-14)
            assert (
                nk.shape == nk_raw.shape == nk_total.shape
                == nk_total_raw.shape == sq.shape == (27,)
            )
            assert all(np.all(np.isfinite(value)) for value in (
                nk, nk_raw, nk_total, nk_total_raw, sq
            ))
            assert np.min(sq) >= -2e-12
            np.testing.assert_allclose(
                nk.sum(), metadata["n_particles"], rtol=0.0, atol=2e-12,
            )
            np.testing.assert_allclose(
                nk_raw.sum(), metadata["n_particles"] * record["band_weight"][0],
                rtol=0.0, atol=2e-10,
            )
            np.testing.assert_allclose(
                nk_total.sum(), metadata["n_particles"],
                rtol=0.0, atol=2e-12,
            )
            np.testing.assert_allclose(
                nk_total_raw.sum(), record["one_body_density_diagonal"].sum(),
                rtol=0.0, atol=2e-10,
            )
            class_rotation = np.asarray(cluster().rotation60)
            class_rotation = class_rotation[class_rotation]
            nk_c3_residuals.append(float(np.max(np.abs(
                nk_raw - nk_raw[class_rotation]
            ))))
            ntotal_c3_residuals.append(float(np.max(np.abs(
                nk_total_raw - nk_total_raw[class_rotation]
            ))))
            residual, matched = _physical_rotation_residual(sq, q)
            assert matched == 21
            sq_c3_residuals.append(residual)
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
    assert "Pages:           27" in info
    assert "Version 5.0.0" in text
    assert "Bloch-projected" in text
    assert "27 physical" in text
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
