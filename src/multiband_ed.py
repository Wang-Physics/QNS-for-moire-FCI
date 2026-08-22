"""Coulomb projection and translation-resolved few-hole exact diagonalization."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .continuum import ContinuumModel, reciprocal_shells


COULOMB_MEV_NM = 1439.9645478


def mesh_bloch_states(
    model: ContinuumModel, n1: int, n2: int, n_bands: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    energies = np.empty((n1, n2, n_bands), dtype=float)
    vectors = np.empty((n1, n2, model.dim, n_bands), dtype=np.complex128)
    k_points = np.empty((n1, n2, 2), dtype=float)
    for i in range(n1):
        for j in range(n2):
            k = i * model.b1 / n1 + j * model.b2 / n2
            vals, vecs = model.solve(k)
            energies[i, j] = vals[:n_bands]
            vectors[i, j] = vecs[:, :n_bands]
            k_points[i, j] = k
    return energies, vectors, k_points


def shifted_overlap(
    model: ContinuumModel,
    final_vectors: np.ndarray,
    initial_vectors: np.ndarray,
    shift: tuple[int, int],
) -> np.ndarray:
    """Return band matrix <u_final|exp(i g_shift r)|u_initial>."""
    n_bands = initial_vectors.shape[1]
    result = np.zeros((n_bands, n_bands), dtype=np.complex128)
    dm, dn = shift
    for layer in range(2):
        offset = layer * model.n_g
        for ig, (m, n) in enumerate(model.g_tuples):
            target = (m + dm, n + dn)
            jt = model.g_index.get(target)
            if jt is None:
                continue
            left = final_vectors[offset + jt]
            right = initial_vectors[offset + ig]
            result += np.outer(np.conj(left), right)
    return result


def density_matrix_for_transfer(
    model: ContinuumModel,
    vectors: np.ndarray,
    n1: int,
    n2: int,
    n_bands: int,
    du: int,
    dv: int,
    reciprocal_shift: tuple[int, int],
) -> np.ndarray:
    n_orbitals = n1 * n2 * n_bands
    rho = np.zeros((n_orbitals, n_orbitals), dtype=np.complex128)
    qm, qn = reciprocal_shift
    for i in range(n1):
        for j in range(n2):
            raw_i, raw_j = i + du, j + dv
            fi, fj = raw_i % n1, raw_j % n2
            wrap = (raw_i // n1, raw_j // n2)
            form = shifted_overlap(
                model,
                vectors[fi, fj, :, :n_bands],
                vectors[i, j, :, :n_bands],
                (qm + wrap[0], qn + wrap[1]),
            )
            initial0 = (i * n2 + j) * n_bands
            final0 = (fi * n2 + fj) * n_bands
            rho[final0 : final0 + n_bands, initial0 : initial0 + n_bands] = form
    return rho


def build_pair_interaction(
    model: ContinuumModel,
    vectors: np.ndarray,
    n1: int,
    n2: int,
    n_bands: int,
    dielectric: float,
    reciprocal_shell_count: int,
) -> tuple[np.ndarray, list[tuple[int, int]], float]:
    n_orbitals = n1 * n2 * n_bands
    pairs = list(combinations(range(n_orbitals), 2))
    first = np.asarray([x for x, _ in pairs], dtype=int)
    second = np.asarray([y for _, y in pairs], dtype=int)
    interaction = np.zeros((len(pairs), len(pairs)), dtype=np.complex128)
    area = n1 * n2 * model.cell_area
    q_reciprocals = reciprocal_shells(reciprocal_shell_count)
    du_values = range(-(n1 // 2), n1 // 2 + 1)
    dv_values = range(-(n2 // 2), n2 // 2 + 1)
    for du in du_values:
        for dv in dv_values:
            qmesh = du * model.b1 / n1 + dv * model.b2 / n2
            for qm, qn in q_reciprocals:
                q = qmesh + qm * model.b1 + qn * model.b2
                qnorm = float(np.linalg.norm(q))
                if qnorm < 1.0e-13:
                    continue
                rho = density_matrix_for_transfer(
                    model, vectors, n1, n2, n_bands, du, dv, (qm, qn)
                )
                rho_minus = rho.conj().T
                direct = rho[first[:, None], first[None, :]] * rho_minus[
                    second[:, None], second[None, :]
                ]
                exchange = rho[first[:, None], second[None, :]] * rho_minus[
                    second[:, None], first[None, :]
                ]
                interaction += (2.0 * np.pi * COULOMB_MEV_NM / (dielectric * qnorm * area)) * (
                    direct - exchange
                )
    herm_error = np.linalg.norm(interaction - interaction.conj().T) / max(
        np.linalg.norm(interaction), 1.0e-30
    )
    if herm_error < 1.0e-9:
        interaction = 0.5 * (interaction + interaction.conj().T)
    return interaction, pairs, float(herm_error)


def _annihilate(state: int, orbital: int) -> tuple[int, int] | None:
    if not (state >> orbital) & 1:
        return None
    sign = -1 if (state & ((1 << orbital) - 1)).bit_count() % 2 else 1
    return state ^ (1 << orbital), sign


def _create(state: int, orbital: int) -> tuple[int, int] | None:
    if (state >> orbital) & 1:
        return None
    sign = -1 if (state & ((1 << orbital) - 1)).bit_count() % 2 else 1
    return state | (1 << orbital), sign


def _apply_pair(state: int, a: int, b: int, c: int, d: int) -> tuple[int, int] | None:
    sign = 1
    current = state
    for operation, orbital in ((_annihilate, a), (_annihilate, b), (_create, d), (_create, c)):
        outcome = operation(current, orbital)
        if outcome is None:
            return None
        current, local_sign = outcome
        sign *= local_sign
    return current, sign


@dataclass
class SectorResult:
    momentum: tuple[int, int]
    basis: list[int]
    energies: np.ndarray
    vectors: np.ndarray
    hermiticity_error: float
    residual: float


def diagonalize_projection(
    single_energies: np.ndarray,
    pair_interaction: np.ndarray,
    pairs: list[tuple[int, int]],
    n1: int,
    n2: int,
    n_bands: int,
    n_particles: int,
) -> list[SectorResult]:
    n_orbitals = n1 * n2 * n_bands
    orbital_energies = single_energies[:, :, :n_bands].reshape(-1)
    orbital_momenta = []
    for i in range(n1):
        for j in range(n2):
            orbital_momenta.extend([(i, j)] * n_bands)
    pair_index = {pair: p for p, pair in enumerate(pairs)}
    nonzero_by_initial = []
    for p in range(len(pairs)):
        nonzero_by_initial.append(np.flatnonzero(np.abs(pair_interaction[:, p]) > 1.0e-12))

    sectors: dict[tuple[int, int], list[int]] = {}
    for occupied in combinations(range(n_orbitals), n_particles):
        momentum = (
            sum(orbital_momenta[o][0] for o in occupied) % n1,
            sum(orbital_momenta[o][1] for o in occupied) % n2,
        )
        state = sum(1 << o for o in occupied)
        sectors.setdefault(momentum, []).append(state)

    results = []
    for momentum in sorted(sectors):
        basis = sectors[momentum]
        lookup = {state: idx for idx, state in enumerate(basis)}
        h = np.zeros((len(basis), len(basis)), dtype=np.complex128)
        for col, state in enumerate(basis):
            occupied = [o for o in range(n_orbitals) if (state >> o) & 1]
            h[col, col] += sum(orbital_energies[o] for o in occupied)
            for a, b in combinations(occupied, 2):
                p = pair_index[(a, b)]
                for q in nonzero_by_initial[p]:
                    c, d = pairs[int(q)]
                    applied = _apply_pair(state, a, b, c, d)
                    if applied is None:
                        continue
                    new_state, sign = applied
                    row = lookup.get(new_state)
                    if row is not None:
                        h[row, col] += sign * pair_interaction[q, p]
        herm_error = np.linalg.norm(h - h.conj().T) / max(np.linalg.norm(h), 1.0e-30)
        if herm_error < 1.0e-9:
            h = 0.5 * (h + h.conj().T)
        values, vectors = np.linalg.eigh(h)
        residual = 0.0
        for idx in range(min(6, len(values))):
            residual = max(residual, float(np.linalg.norm(h @ vectors[:, idx] - values[idx] * vectors[:, idx])))
        results.append(
            SectorResult(momentum, basis, values, vectors, float(herm_error), residual)
        )
    return results


def collect_low_states(results: list[SectorResult], count: int) -> list[tuple[float, SectorResult, int]]:
    states = []
    for sector in results:
        for index, energy in enumerate(sector.energies[:count]):
            states.append((float(energy), sector, index))
    return sorted(states, key=lambda x: x[0])[:count]


def momentum_occupation(
    states: list[tuple[float, SectorResult, int]],
    n1: int,
    n2: int,
    n_bands: int,
) -> np.ndarray:
    occupation = np.zeros((n1, n2), dtype=float)
    n_orbitals = n1 * n2 * n_bands
    for _, sector, eigen_index in states:
        probabilities = np.abs(sector.vectors[:, eigen_index]) ** 2
        for probability, bitstate in zip(probabilities, sector.basis):
            for orbital in range(n_orbitals):
                if (bitstate >> orbital) & 1:
                    kindex = orbital // n_bands
                    occupation[kindex // n2, kindex % n2] += float(probability)
    return occupation / len(states)
