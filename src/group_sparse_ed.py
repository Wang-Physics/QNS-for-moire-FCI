"""Sparse Lanczos helpers for a general finite momentum group."""

from __future__ import annotations

import ctypes
import math
import subprocess
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigsh

from .sparse_ed import NativeHamiltonian, SparseSectorResult, one_body_diagonal


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native" / "group_basis.cpp"
LIBRARY = ROOT / "native" / "libgroup_basis.so"


def compile_group_basis() -> Path:
    if LIBRARY.exists() and LIBRARY.stat().st_mtime >= SOURCE.stat().st_mtime:
        return LIBRARY
    subprocess.run(
        [
            "g++", "-O3", "-march=native", "-std=c++17", "-fPIC", "-shared",
            str(SOURCE), "-o", str(LIBRARY),
        ],
        check=True,
        cwd=ROOT,
    )
    return LIBRARY


def all_group_sector_bases(
    momentum_numerators: np.ndarray,
    determinant: int,
    residue_lookup: np.ndarray,
    n_bands: int,
    n_particles: int,
) -> list[np.ndarray]:
    library = ctypes.CDLL(str(compile_group_basis()))
    library.ed_generate_all_basis.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
    ]
    orbital_x = np.repeat(
        np.asarray(momentum_numerators[:, 0], dtype=np.int32), n_bands
    )
    orbital_y = np.repeat(
        np.asarray(momentum_numerators[:, 1], dtype=np.int32), n_bands
    )
    lookup = np.ascontiguousarray(residue_lookup, dtype=np.int32)
    count = math.comb(len(orbital_x), n_particles)
    states = np.empty(count, dtype=np.uint64)
    sectors = np.empty(count, dtype=np.int32)
    library.ed_generate_all_basis(
        len(orbital_x), n_particles, determinant,
        orbital_x.ctypes.data_as(ctypes.c_void_p),
        orbital_y.ctypes.data_as(ctypes.c_void_p),
        lookup.ctypes.data_as(ctypes.c_void_p),
        states.ctypes.data_as(ctypes.c_void_p),
        sectors.ctypes.data_as(ctypes.c_void_p),
    )
    if np.any(sectors < 0):
        raise RuntimeError("native basis generator produced an invalid momentum sector")
    return [
        np.ascontiguousarray(states[sectors == index])
        for index in range(determinant)
    ]


def diagonalize_sparse_basis(
    basis: np.ndarray,
    orbital_energies: np.ndarray,
    pair_interaction: np.ndarray,
    pairs: list[tuple[int, int]],
    momentum: tuple[int, int],
    n_eigenvalues: int = 1,
    tolerance: float = 1.0e-9,
    max_iterations: int = 1200,
    energy_constant: float = 0.0,
) -> SparseSectorResult:
    diagonal = one_body_diagonal(basis, orbital_energies) + energy_constant
    hamiltonian = NativeHamiltonian(basis, diagonal, pair_interaction, pairs)
    try:
        if len(basis) <= max(32, n_eigenvalues + 2):
            eye = np.eye(len(basis), dtype=np.complex128)
            dense = np.column_stack(
                [hamiltonian.matvec(eye[:, index]) for index in range(len(basis))]
            )
            values, vectors = np.linalg.eigh(0.5 * (dense + dense.conj().T))
            values, vectors = values[:n_eigenvalues], vectors[:, :n_eigenvalues]
        else:
            values, vectors = eigsh(
                hamiltonian.operator,
                k=n_eigenvalues,
                which="SA",
                tol=tolerance,
                maxiter=max_iterations,
                ncv=min(len(basis), max(32, 6 * n_eigenvalues + 12)),
            )
            order = np.argsort(values)
            values, vectors = values[order], vectors[:, order]
        residual = max(
            float(
                np.linalg.norm(
                    hamiltonian.matvec(vectors[:, index])
                    - values[index] * vectors[:, index]
                )
            )
            for index in range(len(values))
        )
    finally:
        hamiltonian.close()
    return SparseSectorResult(momentum, np.asarray(basis), values, vectors, residual)
