"""Momentum-blocked sparse Lanczos ED for the Fig. 2 six-hole benchmark."""

from __future__ import annotations

import ctypes
import subprocess
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh


ROOT = Path(__file__).resolve().parents[1]
NATIVE_SOURCE = ROOT / "native" / "ed_kernel.cpp"
NATIVE_LIBRARY = ROOT / "native" / "libed_kernel.so"


def compile_native_kernel(force: bool = False) -> Path:
    """Compile the small OpenMP matrix-vector kernel when needed."""
    if (
        not force
        and NATIVE_LIBRARY.exists()
        and NATIVE_LIBRARY.stat().st_mtime >= NATIVE_SOURCE.stat().st_mtime
    ):
        return NATIVE_LIBRARY
    command = [
        "g++",
        "-O3",
        "-march=native",
        "-std=c++17",
        "-fPIC",
        "-shared",
        "-fopenmp",
        str(NATIVE_SOURCE),
        "-o",
        str(NATIVE_LIBRARY),
    ]
    subprocess.run(command, check=True, cwd=ROOT)
    return NATIVE_LIBRARY


def orbital_momenta(n1: int, n2: int, n_bands: int) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for i in range(n1):
        for j in range(n2):
            result.extend([(i, j)] * n_bands)
    return result


def sector_basis(
    n1: int,
    n2: int,
    n_bands: int,
    n_particles: int,
    momentum: tuple[int, int],
) -> np.ndarray:
    momenta = orbital_momenta(n1, n2, n_bands)
    states = []
    for occupied in combinations(range(len(momenta)), n_particles):
        total = (
            sum(momenta[o][0] for o in occupied) % n1,
            sum(momenta[o][1] for o in occupied) % n2,
        )
        if total == momentum:
            states.append(sum(1 << o for o in occupied))
    return np.sort(np.asarray(states, dtype=np.uint64))


def one_body_diagonal(basis: np.ndarray, orbital_energies: np.ndarray) -> np.ndarray:
    diagonal = np.zeros(len(basis), dtype=np.float64)
    for orbital, energy in enumerate(np.asarray(orbital_energies).reshape(-1)):
        diagonal += ((basis >> np.uint64(orbital)) & np.uint64(1)) * float(energy)
    return np.ascontiguousarray(diagonal)


@dataclass
class SparseSectorResult:
    momentum: tuple[int, int]
    basis: np.ndarray
    energies: np.ndarray
    vectors: np.ndarray
    residual: float


class NativeHamiltonian:
    """A SciPy LinearOperator backed by a row-parallel C++ matvec."""

    def __init__(
        self,
        basis: np.ndarray,
        diagonal: np.ndarray,
        pair_interaction: np.ndarray,
        pairs: list[tuple[int, int]],
        threshold: float = 1.0e-12,
    ) -> None:
        compile_native_kernel()
        self.basis = np.ascontiguousarray(basis, dtype=np.uint64)
        self.diagonal = np.ascontiguousarray(diagonal, dtype=np.float64)
        self.interaction = np.ascontiguousarray(pair_interaction, dtype=np.complex128)
        self.pair_first = np.ascontiguousarray([p[0] for p in pairs], dtype=np.int32)
        self.pair_second = np.ascontiguousarray([p[1] for p in pairs], dtype=np.int32)
        destination_chunks = [
            np.flatnonzero(np.abs(self.interaction[:, q]) > threshold).astype(np.int32)
            for q in range(len(pairs))
        ]
        self.offsets = np.ascontiguousarray(
            np.cumsum([0] + [len(chunk) for chunk in destination_chunks]), dtype=np.int64
        )
        self.destinations = np.ascontiguousarray(
            np.concatenate(destination_chunks), dtype=np.int32
        )
        self.library = ctypes.CDLL(str(NATIVE_LIBRARY))
        self.library.ed_create_context.restype = ctypes.c_void_p
        self.library.ed_create_context.argtypes = [
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.library.ed_destroy_context.argtypes = [ctypes.c_void_p]
        self.library.ed_matvec.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.context = self.library.ed_create_context(
            len(self.basis),
            len(pairs),
            int(max(self.pair_second)) + 1,
            self.basis.ctypes.data_as(ctypes.c_void_p),
            self.diagonal.ctypes.data_as(ctypes.c_void_p),
            self.pair_first.ctypes.data_as(ctypes.c_void_p),
            self.pair_second.ctypes.data_as(ctypes.c_void_p),
            self.offsets.ctypes.data_as(ctypes.c_void_p),
            self.destinations.ctypes.data_as(ctypes.c_void_p),
            self.interaction.ctypes.data_as(ctypes.c_void_p),
        )
        if not self.context:
            raise RuntimeError("native ED context allocation failed")
        self.operator = LinearOperator(
            shape=(len(self.basis), len(self.basis)),
            dtype=np.dtype(np.complex128),
            matvec=self.matvec,
        )

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        source = np.ascontiguousarray(vector, dtype=np.complex128)
        target = np.empty_like(source)
        self.library.ed_matvec(
            self.context,
            source.ctypes.data_as(ctypes.c_void_p),
            target.ctypes.data_as(ctypes.c_void_p),
        )
        return target

    def close(self) -> None:
        if getattr(self, "context", None):
            self.library.ed_destroy_context(self.context)
            self.context = None

    def __del__(self) -> None:
        self.close()


def diagonalize_sparse_sector(
    single_energies: np.ndarray,
    pair_interaction: np.ndarray,
    pairs: list[tuple[int, int]],
    n1: int,
    n2: int,
    n_bands: int,
    n_particles: int,
    momentum: tuple[int, int],
    n_eigenvalues: int = 1,
    tolerance: float = 1.0e-10,
    max_iterations: int = 1200,
) -> SparseSectorResult:
    basis = sector_basis(n1, n2, n_bands, n_particles, momentum)
    orbital_energies = single_energies[:, :, :n_bands].reshape(-1)
    diagonal = one_body_diagonal(basis, orbital_energies)
    hamiltonian = NativeHamiltonian(basis, diagonal, pair_interaction, pairs)
    try:
        if len(basis) <= max(32, n_eigenvalues + 2):
            dense = np.column_stack(
                [hamiltonian.matvec(np.eye(len(basis), dtype=np.complex128)[:, i]) for i in range(len(basis))]
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
            float(np.linalg.norm(hamiltonian.matvec(vectors[:, index]) - values[index] * vectors[:, index]))
            for index in range(len(values))
        )
    finally:
        hamiltonian.close()
    return SparseSectorResult(momentum, basis, values, vectors, residual)
