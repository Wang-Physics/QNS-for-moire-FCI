"""Plane-wave continuum Hamiltonian for one spin/valley of twisted MoTe2."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.linalg import eigh


HBAR2_OVER_2ME_MEV_NM2 = 38.0998212


def reciprocal_shells(n_shells: int) -> list[tuple[int, int]]:
    """Return zero plus ``n_shells`` radial shells of a triangular RL."""
    extent = max(6, 3 * n_shells)
    grouped: dict[int, list[tuple[int, int]]] = {}
    for m, n in product(range(-extent, extent + 1), repeat=2):
        norm2 = m * m + n * n - m * n
        grouped.setdefault(norm2, []).append((m, n))
    keys = sorted(grouped)[: n_shells + 1]
    vectors = [pair for key in keys for pair in sorted(grouped[key])]
    return sorted(vectors, key=lambda x: (x[0] * x[0] + x[1] * x[1] - x[0] * x[1], x))


@dataclass(frozen=True)
class ContinuumParameters:
    theta_deg: float = 3.0
    a0_nm: float = 0.352
    potential_meV: float = 11.2
    tunneling_meV: float = 13.3
    phase_deg: float = -91.0
    mass_me: float = 0.62
    plane_wave_shells: int = 8


class ContinuumModel:
    def __init__(self, params: ContinuumParameters):
        self.params = params
        theta = np.deg2rad(params.theta_deg)
        self.a_m = params.a0_nm / (2.0 * np.sin(theta / 2.0))
        gmag = 4.0 * np.pi / (np.sqrt(3.0) * self.a_m)
        self.b1 = np.array([gmag, 0.0])
        self.b2 = np.array([-0.5 * gmag, 0.5 * np.sqrt(3.0) * gmag])
        self.reciprocal = np.stack([self.b1, self.b2])
        self.cell_area = np.sqrt(3.0) * self.a_m**2 / 2.0
        self.bz_area = abs(self.b1[0] * self.b2[1] - self.b1[1] * self.b2[0])
        self.kappa_plus = (2.0 * self.b1 + self.b2) / 3.0
        self.kappa_minus = (self.b1 - self.b2) / 3.0
        self.kappas = np.stack([self.kappa_plus, self.kappa_minus])
        self.g_tuples = reciprocal_shells(params.plane_wave_shells)
        self.g_index = {g: i for i, g in enumerate(self.g_tuples)}
        self.g_vectors = np.array([m * self.b1 + n * self.b2 for m, n in self.g_tuples])
        self.n_g = len(self.g_tuples)
        self.dim = 2 * self.n_g
        self.alpha = HBAR2_OVER_2ME_MEV_NM2 / params.mass_me
        self._static = self._build_static()

    def _idx(self, layer: int, g_index: int) -> int:
        return layer * self.n_g + g_index

    def _build_static(self) -> np.ndarray:
        h = np.zeros((self.dim, self.dim), dtype=np.complex128)
        phase = np.deg2rad(self.params.phase_deg)
        potential_directions = [(1, 0), (0, 1), (-1, -1)]
        for layer, phi_l in enumerate((-phase, phase)):
            for ig, (m, n) in enumerate(self.g_tuples):
                col = self._idx(layer, ig)
                for dm, dn in potential_directions:
                    for sign, coeff in ((1, -self.params.potential_meV * np.exp(1j * phi_l)),
                                        (-1, -self.params.potential_meV * np.exp(-1j * phi_l))):
                        target = (m + sign * dm, n + sign * dn)
                        if target in self.g_index:
                            row = self._idx(layer, self.g_index[target])
                            h[row, col] += coeff

        # Interlayer tunneling in the explicit plane-wave convention.
        # Explicit upper-block matrix elements of Reddy et al. use transfers
        # 0,+g2,+g3, with g2=b1+b2 and g3=b2.
        for ig, (m, n) in enumerate(self.g_tuples):
            col = self._idx(1, ig)
            for dm, dn in ((0, 0), (1, 1), (0, 1)):
                target = (m + dm, n + dn)
                if target in self.g_index:
                    row = self._idx(0, self.g_index[target])
                    h[row, col] += self.params.tunneling_meV
                    h[col, row] += self.params.tunneling_meV
        return h

    def hamiltonian(self, k: np.ndarray) -> np.ndarray:
        h = self._static.copy()
        for layer in range(2):
            momenta = k[None, :] + self.g_vectors - self.kappas[layer][None, :]
            kinetic = self.alpha * np.einsum("ij,ij->i", momenta, momenta)
            sl = slice(layer * self.n_g, (layer + 1) * self.n_g)
            h[sl, sl] += np.diag(kinetic)
        return h

    def solve(self, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return eigh(self.hamiltonian(np.asarray(k, dtype=float)), check_finite=False)

    def velocities(self, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vx = np.empty(self.dim, dtype=float)
        vy = np.empty(self.dim, dtype=float)
        for layer in range(2):
            p = k[None, :] + self.g_vectors - self.kappas[layer][None, :]
            sl = slice(layer * self.n_g, (layer + 1) * self.n_g)
            vx[sl] = 2.0 * self.alpha * p[:, 0]
            vy[sl] = 2.0 * self.alpha * p[:, 1]
        return vx, vy

    def quantum_geometry(self, k: np.ndarray, n_bands: int = 10) -> dict[str, np.ndarray]:
        energies, vectors = self.solve(k)
        vx_diag, vy_diag = self.velocities(k)
        vx = (vectors.conj().T * vx_diag[None, :]) @ vectors
        vy = (vectors.conj().T * vy_diag[None, :]) @ vectors
        gxx = np.zeros(n_bands)
        gxy = np.zeros(n_bands)
        gyy = np.zeros(n_bands)
        berry = np.zeros(n_bands)
        for n in range(n_bands):
            de2 = (energies[n] - energies) ** 2
            mask = np.arange(self.dim) != n
            mask &= de2 > 1.0e-18
            qxx = np.sum(vx[n, mask] * np.conj(vx[n, mask]) / de2[mask])
            qyy = np.sum(vy[n, mask] * np.conj(vy[n, mask]) / de2[mask])
            qxy = np.sum(vx[n, mask] * np.conj(vy[n, mask]) / de2[mask])
            gxx[n] = qxx.real
            gyy[n] = qyy.real
            gxy[n] = qxy.real
            berry[n] = -2.0 * qxy.imag
        return {
            "energy": energies[:n_bands],
            "g_xx": gxx,
            "g_xy": gxy,
            "g_yy": gyy,
            "trace_metric": gxx + gyy,
            "berry_curvature": berry,
        }

    def high_symmetry_points(self) -> dict[str, np.ndarray]:
        return {
            "Gamma": np.zeros(2),
            "K": (2.0 * self.b1 + self.b2) / 3.0,
            "M": self.b1 / 2.0,
        }

    def hex_vertices(self) -> np.ndarray:
        candidates = []
        for m, n in product(range(-2, 3), repeat=2):
            if m == 0 and n == 0:
                continue
            g = m * self.b1 + n * self.b2
            candidates.append(g)
        # Voronoi vertices are the six C6 rotations of K.
        k0 = (2.0 * self.b1 + self.b2) / 3.0
        vertices = []
        for angle in np.arange(6) * np.pi / 3.0:
            rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            vertices.append(rot @ k0)
        return np.asarray(vertices)

    def inside_hexagon(self, k: np.ndarray, tolerance: float = 1.0e-12) -> bool:
        k2 = float(k @ k)
        for m, n in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)):
            g = m * self.b1 + n * self.b2
            if k2 > float((k - g) @ (k - g)) + tolerance:
                return False
        return True

    def wrap_to_hexagon(self, k: np.ndarray) -> np.ndarray:
        candidates = [k - m * self.b1 - n * self.b2 for m, n in product(range(-2, 3), repeat=2)]
        return min(candidates, key=lambda x: float(x @ x))


def path_gamma_k_m_gamma(model: ContinuumModel, points_per_segment: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    hs = model.high_symmetry_points()
    nodes = [hs["Gamma"], hs["K"], hs["M"], hs["Gamma"]]
    path: list[np.ndarray] = []
    distances: list[float] = []
    ticks = [0]
    total = 0.0
    for segment, (left, right) in enumerate(zip(nodes[:-1], nodes[1:])):
        ts = np.linspace(0.0, 1.0, points_per_segment, endpoint=(segment == 2))
        for idx, t in enumerate(ts):
            point = (1.0 - t) * left + t * right
            if path:
                total += float(np.linalg.norm(point - path[-1]))
            path.append(point)
            distances.append(total)
        ticks.append(len(path) - 1)
    return np.asarray(path), np.asarray(distances), ticks


def hexagonal_geometry_grid(model: ContinuumModel, grid_size: int) -> np.ndarray:
    vertices = model.hex_vertices()
    xmax = np.max(np.abs(vertices[:, 0]))
    ymax = np.max(np.abs(vertices[:, 1]))
    points = []
    for x in np.linspace(-xmax, xmax, grid_size):
        for y in np.linspace(-ymax, ymax, grid_size):
            k = np.array([x, y])
            if model.inside_hexagon(k):
                points.append(k)
    return np.asarray(points)
