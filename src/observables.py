"""Projected density observables for symmetry-resolved multiband ED states."""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np

from .continuum import ContinuumModel
from .multiband_ed import density_matrix_for_transfer
from .sparse_ed import sector_basis


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native" / "density_kernel.cpp"
LIBRARY = ROOT / "native" / "libdensity_kernel.so"


def compile_density_kernel() -> Path:
    if LIBRARY.exists() and LIBRARY.stat().st_mtime >= SOURCE.stat().st_mtime:
        return LIBRARY
    subprocess.run([
        "g++", "-O3", "-march=native", "-std=c++17", "-fPIC", "-shared",
        "-fopenmp", str(SOURCE), "-o", str(LIBRARY),
    ], check=True, cwd=ROOT)
    return LIBRARY


class DensityContext:
    def __init__(self, source_basis: np.ndarray) -> None:
        self.source_basis = np.ascontiguousarray(source_basis, dtype=np.uint64)
        self.library = ctypes.CDLL(str(compile_density_kernel()))
        self.library.density_create_context.restype = ctypes.c_void_p
        self.library.density_create_context.argtypes = [
            ctypes.c_int64, ctypes.c_int, ctypes.c_void_p,
        ]
        self.library.density_destroy_context.argtypes = [ctypes.c_void_p]
        self.library.density_apply.argtypes = [
            ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]
        n_orbitals = max(int(state).bit_length() for state in self.source_basis)
        self.context = self.library.density_create_context(
            len(self.source_basis), n_orbitals,
            self.source_basis.ctypes.data_as(ctypes.c_void_p),
        )
        if not self.context:
            raise RuntimeError("density context allocation failed")

    def apply(
        self, target_basis: np.ndarray, operator: np.ndarray, vector: np.ndarray
    ) -> np.ndarray:
        target = np.ascontiguousarray(target_basis, dtype=np.uint64)
        matrix = np.asarray(operator, dtype=np.complex128)
        chunks = [np.flatnonzero(np.abs(matrix[final]) > 1.0e-12).astype(np.int32)
                  for final in range(matrix.shape[0])]
        offsets = np.ascontiguousarray(
            np.cumsum([0] + [len(chunk) for chunk in chunks]), dtype=np.int64
        )
        initials = np.ascontiguousarray(np.concatenate(chunks), dtype=np.int32)
        coefficients = np.ascontiguousarray(
            np.concatenate([matrix[final, chunk] for final, chunk in enumerate(chunks)]),
            dtype=np.complex128,
        )
        source = np.ascontiguousarray(vector, dtype=np.complex128)
        output = np.empty(len(target), dtype=np.complex128)
        self.library.density_apply(
            self.context, len(target), target.ctypes.data_as(ctypes.c_void_p),
            offsets.ctypes.data_as(ctypes.c_void_p),
            initials.ctypes.data_as(ctypes.c_void_p),
            coefficients.ctypes.data_as(ctypes.c_void_p),
            source.ctypes.data_as(ctypes.c_void_p),
            output.ctypes.data_as(ctypes.c_void_p),
        )
        return output

    def close(self) -> None:
        if getattr(self, "context", None):
            self.library.density_destroy_context(self.context)
            self.context = None

    def __del__(self) -> None:
        self.close()


def projected_structure_factor(
    model: ContinuumModel,
    bloch_vectors: np.ndarray,
    basis: np.ndarray,
    amplitudes: np.ndarray,
    n_bands: int,
    n_holes: int,
) -> np.ndarray:
    """Return S_rho(q)=<rho_-q rho_q>/N for the 3x3 momentum mesh."""
    values = np.zeros((3, 3), dtype=float)
    context = DensityContext(basis)
    try:
        for q1 in range(3):
            for q2 in range(3):
                if q1 == 0 and q2 == 0:
                    continue
                du = q1 if q1 <= 1 else q1 - 3
                dv = q2 if q2 <= 1 else q2 - 3
                rho = density_matrix_for_transfer(
                    model, bloch_vectors, 3, 3, n_bands, du, dv, (0, 0)
                )
                target = sector_basis(3, 3, n_bands, n_holes, (q1, q2))
                transformed = context.apply(target, rho, amplitudes)
                values[q1, q2] = float(np.vdot(transformed, transformed).real / n_holes)
    finally:
        context.close()
    return values
