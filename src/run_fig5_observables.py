"""Ground-manifold n(k) and projected S(q) for the 27-cell Fig. 5 benchmark."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "12")

import numpy as np

from .c6_ed import (
    TriangularSupercell,
    cluster_bloch_states,
    density_matrix_for_cluster_transfer,
    particle_hole_transform_one_band,
)
from .continuum import ContinuumModel, ContinuumParameters
from .group_sparse_ed import all_group_sector_bases, diagonalize_sparse_basis
from .observables import DensityContext


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "result" / "data"
STATE_CACHE = DATA / "fig5_27cell_ground_manifolds.npz"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def paper_momentum_coordinates(cluster: TriangularSupercell) -> np.ndarray:
    """Return the Z3 x Z9 coordinates used for k1+N1*k2 (N1=3).

    For S=[[3,-3],[3,6]], every reciprocal residue is 3(u,v), with
    u+v divisible by three.  The map ((u+v)/3 mod 3, u mod 9) is a
    bijective group homomorphism onto Z3 x Z9.  It places Gamma, K', K at
    indices 0, 9, 18, respectively.
    """
    uv = cluster.numerators // 3
    coordinates = np.column_stack((((uv[:, 0] + uv[:, 1]) // 3) % 3, uv[:, 0] % 9))
    indices = coordinates[:, 0] + 3 * coordinates[:, 1]
    if sorted(indices.tolist()) != list(range(27)):
        raise RuntimeError("paper momentum coordinate map is not bijective")
    return coordinates.astype(np.int32)


def occupation(basis: np.ndarray, amplitudes: np.ndarray, n_orbitals: int) -> np.ndarray:
    probability = np.abs(amplitudes) ** 2
    probability /= probability.sum()
    values = np.empty(n_orbitals, dtype=float)
    for orbital in range(n_orbitals):
        occupied = ((basis >> np.uint64(orbital)) & np.uint64(1)) == 1
        values[orbital] = float(probability[occupied].sum())
    return values


def particle_hole_phase(hole_basis: np.ndarray, n_orbitals: int) -> np.ndarray:
    phase = np.ones(len(hole_basis), dtype=float)
    for orbital in range(1, n_orbitals, 2):
        occupied = ((hole_basis >> np.uint64(orbital)) & np.uint64(1)) == 1
        phase[occupied] *= -1.0
    return phase


def solve_ground_manifolds(
    cluster: TriangularSupercell,
    energies: np.ndarray,
    pair: np.ndarray,
    pairs: list[tuple[int, int]],
    bases: list[np.ndarray],
) -> dict[str, list[dict]]:
    gamma = 0
    kval = cluster.index_of_fraction(2, 1, 3)
    kprime = cluster.index_of_fraction(1, 2, 3)
    one_third = []
    for label, sector in (("Gamma", gamma), ("K", kval), ("Kprime", kprime)):
        log(f"nu=1/3 {label}: dim={len(bases[sector]):,}")
        result = diagonalize_sparse_basis(
            bases[sector], energies[:, 0], pair, pairs, (sector, 0),
            n_eigenvalues=1, tolerance=2.0e-9, max_iterations=1800,
        )
        one_third.append(
            {
                "label": label,
                "sector": sector,
                "basis": result.basis,
                "amplitudes": result.vectors[:, 0],
                "energy": float(result.energies[0]),
                "residual": result.residual,
                "hole_representation": False,
            }
        )

    log("building exact particle-hole Hamiltonian for nu=2/3")
    constant, hole_energies, hole_pair = particle_hole_transform_one_band(
        energies[:, 0], pair, pairs
    )
    result = diagonalize_sparse_basis(
        bases[0], hole_energies, hole_pair, pairs, (0, 0),
        n_eigenvalues=3, tolerance=2.0e-9, max_iterations=1800,
        energy_constant=constant,
    )
    two_thirds = []
    for index in range(3):
        two_thirds.append(
            {
                "label": f"Gamma{index}",
                "sector": 0,
                "basis": result.basis,
                "amplitudes": result.vectors[:, index],
                "energy": float(result.energies[index]),
                "residual": result.residual,
                "hole_representation": True,
            }
        )
    return {"1/3": one_third, "2/3": two_thirds}


def save_manifolds(manifolds: dict[str, list[dict]]) -> None:
    arrays: dict[str, np.ndarray] = {}
    metadata = {}
    for filling_key, filling in (("nu1", "1/3"), ("nu2", "2/3")):
        metadata[filling] = []
        for index, state in enumerate(manifolds[filling]):
            arrays[f"{filling_key}_basis_{index}"] = state["basis"]
            arrays[f"{filling_key}_amplitudes_{index}"] = state["amplitudes"]
            metadata[filling].append(
                {key: value for key, value in state.items() if key not in ("basis", "amplitudes")}
            )
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(STATE_CACHE, **arrays)


def load_manifolds() -> dict[str, list[dict]]:
    cache = np.load(STATE_CACHE)
    metadata = json.loads(str(cache["metadata_json"]))
    result: dict[str, list[dict]] = {}
    for filling_key, filling in (("nu1", "1/3"), ("nu2", "2/3")):
        result[filling] = []
        for index, item in enumerate(metadata[filling]):
            state = dict(item)
            state["basis"] = np.asarray(cache[f"{filling_key}_basis_{index}"], dtype=np.uint64)
            state["amplitudes"] = np.asarray(
                cache[f"{filling_key}_amplitudes_{index}"], dtype=np.complex128
            )
            result[filling].append(state)
    return result


def electron_state(
    state: dict, cluster: TriangularSupercell, bases: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, int]:
    if not state["hole_representation"]:
        return state["basis"], state["amplitudes"], int(state["sector"])
    full = np.uint64((1 << cluster.determinant) - 1)
    hole_basis = np.asarray(state["basis"], dtype=np.uint64)
    basis = np.bitwise_xor(full, hole_basis)
    amplitudes = particle_hole_phase(hole_basis, cluster.determinant) * state["amplitudes"]
    return basis, amplitudes, int(cluster.negation[int(state["sector"])])


def structure_factor_for_state(
    state: dict,
    cluster: TriangularSupercell,
    bases: list[np.ndarray],
    density_operators: list[np.ndarray],
    n_particles: int,
) -> np.ndarray:
    source_basis, amplitudes, source_sector = electron_state(state, cluster, bases)
    values = np.zeros(cluster.determinant, dtype=float)
    context = DensityContext(source_basis)
    full = np.uint64((1 << cluster.determinant) - 1)
    try:
        for transfer in range(1, cluster.determinant):
            target_sector = int(cluster.add_table[source_sector, transfer])
            if state["hole_representation"]:
                target_hole_sector = int(cluster.negation[target_sector])
                target_basis = np.bitwise_xor(full, bases[target_hole_sector])
            else:
                target_basis = bases[target_sector]
            transformed = context.apply(target_basis, density_operators[transfer], amplitudes)
            values[transfer] = float(np.vdot(transformed, transformed).real / n_particles)
    finally:
        context.close()
    return values


def main() -> None:
    np.random.seed(250313585)
    cluster = TriangularSupercell.c6_27()
    model = ContinuumModel(ContinuumParameters(theta_deg=2.6, plane_wave_shells=8))
    energies, bloch_vectors, k_points = cluster_bloch_states(model, cluster, 1)
    operator = np.load(DATA / "fig5_27cell_1band_operator.npz")
    pair = np.asarray(operator["pair"], dtype=np.complex128)
    pairs = [tuple(map(int, item)) for item in np.asarray(operator["pairs"])]
    log("generating C(27,9) basis and translation blocks")
    bases = all_group_sector_bases(
        cluster.numerators, cluster.determinant, cluster.residue_lookup, 1, 9
    )
    if STATE_CACHE.exists():
        log(f"loading {STATE_CACHE.name}")
        manifolds = load_manifolds()
    else:
        manifolds = solve_ground_manifolds(cluster, energies, pair, pairs, bases)
        save_manifolds(manifolds)
        log(f"saved {STATE_CACHE.name}")

    log("constructing projected density transfers")
    density_operators = [
        density_matrix_for_cluster_transfer(
            model, cluster, bloch_vectors, 1, transfer, (0, 0)
        )
        for transfer in range(cluster.determinant)
    ]
    coordinates = paper_momentum_coordinates(cluster)
    output = {
        "benchmark": "arXiv:2503.13585v3 Supplementary Fig. 5 extension",
        "theta_deg": 2.6,
        "dielectric_constant": 5.0,
        "cluster": "27-cell 3sqrt(3) x 3sqrt(3)",
        "bands": 1,
        "momentum_index_definition": "k1 + N1*k2 with N1=3, k1 in Z3, k2 in Z9",
        "structure_factor_definition": "S(q)=<rho(-q)rho(q)>/N for q!=0; S(0)=0 connected",
        "momentum_coordinates": coordinates.tolist(),
        "momentum_indices": (coordinates[:, 0] + 3 * coordinates[:, 1]).tolist(),
        "k_points_hex": k_points.tolist(),
        "fillings": {},
    }
    for filling, n_particles in (("1/3", 9), ("2/3", 18)):
        nk_states = []
        sq_states = []
        state_meta = []
        for state in manifolds[filling]:
            source_basis, amplitudes, _ = electron_state(state, cluster, bases)
            nk_states.append(occupation(source_basis, amplitudes, cluster.determinant))
            log(f"{filling} {state['label']}: computing S(q)")
            sq_states.append(
                structure_factor_for_state(
                    state, cluster, bases, density_operators, n_particles
                )
            )
            state_meta.append(
                {
                    key: value for key, value in state.items()
                    if key not in ("basis", "amplitudes")
                }
            )
        nk = np.mean(nk_states, axis=0)
        sq = np.mean(sq_states, axis=0)
        output["fillings"][filling] = {
            "n_particles": n_particles,
            "manifold_states": state_meta,
            "n_k": nk.tolist(),
            "n_k_sum": float(nk.sum()),
            "S_q": sq.tolist(),
            "S_q_max": float(sq.max()),
            "S_q_max_sector": int(np.argmax(sq)),
        }
    path = DATA / "fig5_27cell_observables.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    log(f"saved {path.name}")


if __name__ == "__main__":
    main()
