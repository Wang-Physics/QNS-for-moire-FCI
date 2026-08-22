"""C3 sewing representation in the projected 3x3 Bloch/Fock basis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .continuum import ContinuumModel, ContinuumParameters


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "result" / "data"


def rotate_momentum(momentum: tuple[int, int]) -> tuple[int, int]:
    """C3: (k1,k2) -> (-k2,k1-k2), modulo the 3x3 mesh."""
    i, j = momentum
    return (-j % 3, (i - j) % 3)


def _plane_wave_rotation(
    model: ContinuumModel, source: tuple[int, int]
) -> tuple[tuple[int, int], np.ndarray]:
    """Return target mesh point and truncated plane-wave C3 map."""
    i, j = source
    raw_i, raw_j = -j, i - j
    target = (raw_i % 3, raw_j % 3)
    wrap = ((raw_i - target[0]) // 3, (raw_j - target[1]) // 3)
    rotation = np.zeros((model.dim, model.dim), dtype=np.complex128)
    # kappa_+ - C3 kappa_+ = b1; kappa_- - C3 kappa_- = -b2.
    for layer, shift in ((0, (1, 0)), (1, (0, -1))):
        for source_g, (m, n) in enumerate(model.g_tuples):
            rotated_g = (-n, m - n)
            target_g = (
                rotated_g[0] + wrap[0] + shift[0],
                rotated_g[1] + wrap[1] + shift[1],
            )
            target_index = model.g_index.get(target_g)
            if target_index is not None:
                rotation[
                    layer * model.n_g + target_index,
                    layer * model.n_g + source_g,
                ] = 1.0
    return target, rotation


@dataclass(frozen=True)
class SingleParticleC3:
    orbital_map: np.ndarray
    sewing_phase: np.ndarray
    maximum_band_leakage: float
    minimum_singular_value: float
    maximum_energy_mismatch_meV: float


def single_particle_c3(n_bands: int) -> SingleParticleC3:
    """Extract diagonal active-band sewing phases from continuum Bloch states."""
    cache = np.load(DATA / "fig2_5band_operator_cache.npz")
    vectors = np.asarray(cache["vectors"])[..., :n_bands]
    energies = np.asarray(cache["energies"])[..., :n_bands]
    model = ContinuumModel(ContinuumParameters(theta_deg=3.0, plane_wave_shells=8))
    orbital_map = np.empty(9 * n_bands, dtype=np.int64)
    sewing = np.empty(9 * n_bands, dtype=np.complex128)
    maximum_leakage = 0.0
    minimum_singular = 1.0
    maximum_energy_mismatch = 0.0
    for i in range(3):
        for j in range(3):
            target, rotation = _plane_wave_rotation(model, (i, j))
            matrix = vectors[target].conj().T @ rotation @ vectors[i, j]
            singular = np.linalg.svd(matrix, compute_uv=False)
            minimum_singular = min(minimum_singular, float(singular.min()))
            diagonal = np.diag(np.diag(matrix))
            maximum_leakage = max(
                maximum_leakage,
                float(np.linalg.norm(matrix - diagonal) / max(np.linalg.norm(matrix), 1e-15)),
            )
            maximum_energy_mismatch = max(
                maximum_energy_mismatch,
                float(np.max(np.abs(energies[i, j] - energies[target]))),
            )
            for band in range(n_bands):
                source_orbital = (i * 3 + j) * n_bands + band
                target_orbital = (target[0] * 3 + target[1]) * n_bands + band
                orbital_map[source_orbital] = target_orbital
                value = matrix[band, band]
                sewing[source_orbital] = value / abs(value)
    return SingleParticleC3(
        orbital_map, sewing, maximum_leakage, minimum_singular,
        maximum_energy_mismatch,
    )


def _sorting_parity(values: list[int]) -> int:
    inversions = sum(
        values[left] > values[right]
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )
    return -1 if inversions % 2 else 1


def rotate_fock_state(
    state: int, orbital_map: np.ndarray, sewing_phase: np.ndarray
) -> tuple[int, complex]:
    occupied = [orbital for orbital in range(len(orbital_map)) if (state >> orbital) & 1]
    mapped = [int(orbital_map[orbital]) for orbital in occupied]
    coefficient = complex(_sorting_parity(mapped))
    for orbital in occupied:
        coefficient *= sewing_phase[orbital]
    target = sum(1 << orbital for orbital in sorted(mapped))
    return target, coefficient


class C3Projector:
    """Sparse Fock-space C3 action and character projectors."""

    def __init__(self, basis: np.ndarray, n_bands: int) -> None:
        self.basis = np.asarray(basis, dtype=np.uint64)
        single = single_particle_c3(n_bands)
        lookup = {int(state): index for index, state in enumerate(self.basis)}
        destination = np.empty(len(self.basis), dtype=np.int64)
        coefficient = np.empty(len(self.basis), dtype=np.complex128)
        for source, state in enumerate(self.basis):
            target, phase = rotate_fock_state(
                int(state), single.orbital_map, single.sewing_phase
            )
            if target not in lookup:
                raise ValueError("C3 leaves the selected many-body momentum sector")
            destination[source] = lookup[target]
            coefficient[source] = phase
        self.destination = destination
        self.coefficient = coefficient
        self.inverse_destination = np.empty_like(destination)
        self.inverse_destination[destination] = np.arange(len(destination))

        self.single_particle_diagnostics = single

        probe = np.ones(len(self.basis), dtype=np.complex128)
        rotated_three = self.rotate(self.rotate(self.rotate(probe)))
        self.cube_error = float(np.max(np.abs(rotated_three - probe)))

    def rotate(self, vector: np.ndarray) -> np.ndarray:
        output = np.empty_like(np.asarray(vector, dtype=np.complex128))
        output[self.destination] = self.coefficient * vector
        return output

    def project(self, vector: np.ndarray, irrep: int) -> np.ndarray:
        root = np.exp(-2j * np.pi * irrep / 3.0)
        first = self.rotate(vector)
        second = self.rotate(first)
        return (vector + root * first + root**2 * second) / 3.0
    def projection_terms(self, irrep: int) -> tuple[np.ndarray, np.ndarray]:
        """Sources and coefficients for each output of the C3 projector."""
        size = len(self.basis)
        sources = np.empty((3, size), dtype=np.int64)
        coefficients = np.empty((3, size), dtype=np.complex128)
        sources[0] = np.arange(size)
        coefficients[0] = 1.0
        sources[1] = self.inverse_destination
        first = self.coefficient[sources[1]]
        sources[2] = self.inverse_destination[sources[1]]
        second = self.coefficient[sources[2]]
        root = np.exp(-2j * np.pi * irrep / 3.0)
        coefficients[1] = root * first
        coefficients[2] = root**2 * first * second
        return sources, coefficients


    def projected_norms(self, vector: np.ndarray) -> np.ndarray:
        return np.asarray([
            np.vdot(self.project(vector, irrep), self.project(vector, irrep)).real
            for irrep in range(3)
        ])

    def commutator_error(self, matvec, seed: int = 123) -> float:
        rng = np.random.default_rng(seed)
        probe = rng.normal(size=len(self.basis)) + 1j * rng.normal(size=len(self.basis))
        probe /= np.linalg.norm(probe)
        difference = matvec(self.rotate(probe)) - self.rotate(matvec(probe))
        scale = max(np.linalg.norm(matvec(probe)), 1e-15)
        return float(np.linalg.norm(difference) / scale)
