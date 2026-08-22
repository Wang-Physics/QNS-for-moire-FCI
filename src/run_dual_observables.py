"""Five-band n(k) and projected structure factors for both fillings."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np

from .continuum import ContinuumModel, ContinuumParameters
from .observables import projected_structure_factor


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "result" / "data"


def momentum_occupation(basis: np.ndarray, amplitudes: np.ndarray) -> np.ndarray:
    probability = np.abs(amplitudes) ** 2
    probability /= probability.sum()
    occupation = np.zeros((3, 3), dtype=float)
    for index in range(9):
        for band in range(5):
            orbital = index * 5 + band
            occupied = ((basis >> np.uint64(orbital)) & np.uint64(1)) == 1
            occupation[index // 3, index % 3] += float(probability[occupied].sum())
    return occupation


def main() -> None:
    model = ContinuumModel(ContinuumParameters(theta_deg=3.0, plane_wave_shells=8))
    cache = np.load(DATA / "fig2_5band_operator_cache.npz")
    vectors = np.asarray(cache["vectors"])
    output = {
        "theta_deg": 3.0,
        "n_bands": 5,
        "structure_factor_definition": "S_rho(q)=<rho_-q rho_q>/N_h for q!=0",
        "spin_relation": "fully polarized: S_zz(q)=S_rho(q)/4 for spin-1/2 operators",
        "fillings": {},
    }
    for label, n_holes, filename in (
        ("1/3", 3, "nu1of3_5band_ground_state.npz"),
        ("2/3", 6, "fig2_5band_ground_state.npz"),
    ):
        state = np.load(DATA / filename)
        basis = np.asarray(state["basis"], dtype=np.uint64)
        amplitudes = np.asarray(state["amplitudes"], dtype=np.complex128)
        nk = momentum_occupation(basis, amplitudes)
        structure = projected_structure_factor(
            model, vectors, basis, amplitudes, 5, n_holes
        )
        output["fillings"][label] = {
            "n_holes": n_holes,
            "momentum": np.asarray(state["momentum"]).tolist(),
            "n_k": nk.tolist(),
            "n_k_sum": float(nk.sum()),
            "S_charge_q": structure.tolist(),
            "S_zz_q": (structure / 4.0).tolist(),
        }
    (DATA / "dual_filling_5band_observables.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
