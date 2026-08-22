"""C3 sewing and exact Fock-space projection on a general C6 cluster."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .c6_ed import TriangularSupercell
from .continuum import ContinuumModel


def _sorting_parity(values: list[int]) -> int:
    inversions = sum(
        values[left] > values[right]
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )
    return -1 if inversions % 2 else 1


@dataclass(frozen=True)
class ClusterC3Representation:
    orbital_map: np.ndarray
    sewing_phase: np.ndarray
    minimum_sewing_magnitude: float
    maximum_energy_mismatch_meV: float


def match_cached_bloch_gauge(
    current_pair: np.ndarray,
    cached_pair: np.ndarray,
    pairs: list[tuple[int, int]],
    *,
    seed: int = 27,
) -> tuple[np.ndarray, float]:
    """Infer orbital phases relating a cached interaction to current Bloch vectors.

    The matrices obey ``V_current[q,p] = exp(-i theta_q) V_cache[q,p]
    exp(i theta_p)``, with ``theta_(a,b)=delta_a+delta_b``.  Recovering
    ``delta`` lets the continuum sewing matrix act in the historical gauge of
    the ED/QNS benchmark cache.
    """
    current = np.asarray(current_pair, dtype=np.complex128)
    cached = np.asarray(cached_pair, dtype=np.complex128)
    pair_array = np.asarray(pairs, dtype=np.int64)
    if current.shape != cached.shape or current.shape[0] != len(pair_array):
        raise ValueError("incompatible pair matrices for gauge matching")
    magnitude_error = np.linalg.norm(np.abs(current) - np.abs(cached)) / max(
        np.linalg.norm(np.abs(current)), 1.0e-30
    )
    if magnitude_error > 1.0e-9:
        raise RuntimeError("interaction caches are not related by Bloch gauge")
    mask = (np.abs(current) > 1.0e-8) & (np.abs(cached) > 1.0e-8)
    rows, columns = np.nonzero(mask)
    target = current[rows, columns] / cached[rows, columns]
    target /= np.abs(target)
    n_orbitals = int(pair_array.max()) + 1

    def residual(delta: np.ndarray) -> np.ndarray:
        theta = delta[pair_array[:, 0]] + delta[pair_array[:, 1]]
        predicted = np.exp(1j * (-theta[rows] + theta[columns]))
        difference = predicted - target
        return np.concatenate([difference.real, difference.imag])

    initial = np.random.default_rng(seed).uniform(-np.pi, np.pi, n_orbitals)
    fit = least_squares(
        residual, initial, max_nfev=300,
        ftol=1.0e-12, xtol=1.0e-12, gtol=1.0e-12,
    )
    relative_error = float(
        np.linalg.norm(residual(fit.x)) / np.sqrt(2.0 * len(target))
    )
    if relative_error > 1.0e-8:
        raise RuntimeError(f"Bloch-gauge fit failed: {relative_error:.3e}")
    return np.asarray(fit.x), relative_error


def transform_sewing_to_cached_gauge(
    representation: ClusterC3Representation, delta: np.ndarray
) -> ClusterC3Representation:
    """Transform a C3 sewing representation from current to cached gauge."""
    mapping = representation.orbital_map
    phase = (
        np.exp(1j * delta[mapping])
        * representation.sewing_phase
        * np.exp(-1j * delta)
    )
    return ClusterC3Representation(
        orbital_map=mapping,
        sewing_phase=phase,
        minimum_sewing_magnitude=representation.minimum_sewing_magnitude,
        maximum_energy_mismatch_meV=representation.maximum_energy_mismatch_meV,
    )


def cluster_c3_representation(
    model: ContinuumModel,
    cluster: TriangularSupercell,
    vectors: np.ndarray,
    energies: np.ndarray,
    *,
    holes: bool = False,
) -> ClusterC3Representation:
    """Extract the one-band C3 sewing phase on an arbitrary C6 momentum group."""
    if vectors.shape[0] != cluster.determinant or vectors.shape[-1] != 1:
        raise ValueError("the large-cluster projector currently supports one band")
    rotation120 = cluster.rotation60[cluster.rotation60]
    sewing = np.empty(cluster.determinant, dtype=np.complex128)
    minimum_magnitude = 1.0
    maximum_energy_mismatch = 0.0
    for source in range(cluster.determinant):
        target = int(rotation120[source])
        first, second = cluster.fractions[source]
        raw = np.array([-second, first - second])
        wrap_float = raw - cluster.fractions[target]
        wrap = np.rint(wrap_float).astype(int)
        if np.max(np.abs(wrap_float - wrap)) > 1.0e-10:
            raise RuntimeError("noninteger reciprocal wrap under C3")
        rotation = np.zeros((model.dim, model.dim), dtype=np.complex128)
        for layer, layer_shift in ((0, (1, 0)), (1, (0, -1))):
            for source_g, (m, n) in enumerate(model.g_tuples):
                rotated_g = (-n, m - n)
                target_g = (
                    rotated_g[0] + int(wrap[0]) + layer_shift[0],
                    rotated_g[1] + int(wrap[1]) + layer_shift[1],
                )
                target_g_index = model.g_index.get(target_g)
                if target_g_index is not None:
                    rotation[
                        layer * model.n_g + target_g_index,
                        layer * model.n_g + source_g,
                    ] = 1.0
        value = complex(
            vectors[target, :, 0].conj() @ rotation @ vectors[source, :, 0]
        )
        minimum_magnitude = min(minimum_magnitude, abs(value))
        if abs(value) < 1.0e-10:
            raise RuntimeError("singular one-band C3 sewing matrix")
        sewing[source] = value / abs(value)
        maximum_energy_mismatch = max(
            maximum_energy_mismatch,
            abs(float(energies[source, 0] - energies[target, 0])),
        )
    if holes:
        sewing = sewing.conj()
    return ClusterC3Representation(
        orbital_map=np.asarray(rotation120, dtype=np.int64),
        sewing_phase=sewing,
        minimum_sewing_magnitude=float(minimum_magnitude),
        maximum_energy_mismatch_meV=float(maximum_energy_mismatch),
    )


def rotate_fock_state(
    state: int, orbital_map: np.ndarray, sewing_phase: np.ndarray
) -> tuple[int, complex]:
    occupied = [
        orbital for orbital in range(len(orbital_map))
        if (state >> orbital) & 1
    ]
    mapped = [int(orbital_map[orbital]) for orbital in occupied]
    coefficient = complex(_sorting_parity(mapped))
    for orbital in occupied:
        coefficient *= sewing_phase[orbital]
    target = sum(1 << orbital for orbital in sorted(mapped))
    return target, coefficient


class ClusterC3Projector:
    """Exact C3 character projector within one translation sector."""

    def __init__(
        self,
        basis: np.ndarray,
        representation: ClusterC3Representation,
    ) -> None:
        self.basis = np.ascontiguousarray(basis, dtype=np.uint64)
        lookup = {int(state): index for index, state in enumerate(self.basis)}
        destination = np.empty(len(self.basis), dtype=np.int64)
        coefficient = np.empty(len(self.basis), dtype=np.complex128)
        for source, state in enumerate(self.basis):
            target, phase = rotate_fock_state(
                int(state), representation.orbital_map, representation.sewing_phase
            )
            if target not in lookup:
                raise ValueError("C3 leaves the chosen translation sector")
            destination[source] = lookup[target]
            coefficient[source] = phase
        self.destination = destination
        self.coefficient = coefficient
        self.inverse_destination = np.empty_like(destination)
        self.inverse_destination[destination] = np.arange(len(destination))
        self.representation = representation

        probe = np.ones(len(self.basis), dtype=np.complex128)
        self.cube_error = float(
            np.max(np.abs(self.rotate(self.rotate(self.rotate(probe))) - probe))
        )

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

    def commutator_error(self, matvec, seed: int = 2718) -> float:
        rng = np.random.default_rng(seed)
        probe = rng.normal(size=len(self.basis)) + 1j * rng.normal(size=len(self.basis))
        probe /= np.linalg.norm(probe)
        difference = matvec(self.rotate(probe)) - self.rotate(matvec(probe))
        scale = max(np.linalg.norm(matvec(probe)), 1.0e-15)
        return float(np.linalg.norm(difference) / scale)
