"""Paper-aligned Fig. 5 ED and a feasible two-band finite-size check.

Paper reproduction: one band, 27-cell 3sqrt(3) x 3sqrt(3) cluster,
theta=2.6 degrees, epsilon=5, nu=1/3 and 2/3.

Band-mixing check: two bands, 12-cell 2sqrt(3) x 2sqrt(3) C6 cluster.
The 27-cell two-band Hilbert space is combinatorially intractable and is not
mislabelled as an exact calculation here.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "12")

import numpy as np

from .c6_ed import (
    TriangularSupercell,
    build_cluster_pair_interaction,
    cluster_bloch_states,
    particle_hole_transform_one_band,
)
from .continuum import ContinuumModel, ContinuumParameters
from .group_sparse_ed import all_group_sector_bases, diagonalize_sparse_basis


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "result" / "data"
DATA.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def representative_orbits(cluster: TriangularSupercell) -> list[list[int]]:
    """Use geometric C6; plane-wave-cutoff violations are only micro-eV."""
    return cluster.c6_orbits()


def operator_cache(
    name: str,
    cluster: TriangularSupercell,
    n_bands: int,
    reciprocal_shells: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]], float]:
    path = DATA / f"{name}_operator.npz"
    if path.exists():
        cache = np.load(path)
        log(f"loaded {path.name}")
        return (
            np.asarray(cache["energies"]),
            np.asarray(cache["vectors"]),
            np.asarray(cache["k_points"]),
            [tuple(map(int, pair)) for pair in np.asarray(cache["pairs"])],
            float(cache["hermiticity_error"]),
        ) if False else (
            np.asarray(cache["energies"]),
            np.asarray(cache["pair"]),
            np.asarray(cache["k_points"]),
            [tuple(map(int, pair)) for pair in np.asarray(cache["pairs"])],
            float(cache["hermiticity_error"]),
        )
    model = ContinuumModel(
        ContinuumParameters(theta_deg=2.6, plane_wave_shells=8)
    )
    log(f"solving {cluster.determinant} Bloch momenta, {n_bands} band(s)")
    energies, vectors, k_points = cluster_bloch_states(model, cluster, n_bands)
    log("projecting periodic Coulomb interaction")
    pair, pairs, hermiticity_error = build_cluster_pair_interaction(
        model,
        cluster,
        vectors,
        n_bands,
        dielectric=5.0,
        reciprocal_shell_count=reciprocal_shells,
    )
    np.savez_compressed(
        path,
        energies=energies,
        pair=pair,
        pairs=np.asarray(pairs, dtype=np.int32),
        k_points=k_points,
        hermiticity_error=np.asarray(hermiticity_error),
    )
    log(f"saved {path.name}; hermiticity={hermiticity_error:.3e}")
    return energies, pair, k_points, pairs, hermiticity_error


def solve_representatives(
    cluster: TriangularSupercell,
    bases: list[np.ndarray],
    orbital_energies: np.ndarray,
    pair: np.ndarray,
    pairs: list[tuple[int, int]],
    gamma_eigenvalues: int,
    other_eigenvalues: int,
    label: str,
    energy_constant: float = 0.0,
    electron_to_basis_sector: np.ndarray | None = None,
) -> list[dict]:
    records: list[dict | None] = [None] * cluster.determinant
    mapping = (
        np.arange(cluster.determinant, dtype=np.int32)
        if electron_to_basis_sector is None
        else np.asarray(electron_to_basis_sector, dtype=np.int32)
    )
    for orbit in representative_orbits(cluster):
        representative = orbit[0]
        basis_sector = int(mapping[representative])
        count = gamma_eigenvalues if representative == 0 else other_eigenvalues
        log(
            f"{label}: sector {representative:02d}, dim={len(bases[basis_sector]):,}, "
            f"nev={count}, orbit={orbit}"
        )
        result = diagonalize_sparse_basis(
            bases[basis_sector],
            orbital_energies,
            pair,
            pairs,
            momentum=(representative, 0),
            n_eigenvalues=count,
            tolerance=2.0e-9,
            max_iterations=1800,
            energy_constant=energy_constant,
        )
        for sector in orbit:
            records[sector] = {
                "sector_index": sector,
                "representative": representative,
                "momentum_numerator": cluster.numerators[sector].tolist(),
                "momentum_fraction": cluster.centered_fractions[sector].tolist(),
                "dimension": len(bases[int(mapping[sector])]),
                "energies_meV": result.energies.tolist(),
                "residual_meV": result.residual,
                "symmetry_generated": sector != representative,
            }
        log(
            f"{label}: sector {representative:02d} energies "
            + ", ".join(f"{value:.6f}" for value in result.energies)
        )
    return [record for record in records if record is not None]


def diagnostics(
    cluster: TriangularSupercell, records: list[dict], filling: str
) -> dict:
    gamma = 0
    kval = cluster.index_of_fraction(2, 1, 3)
    kprime = cluster.index_of_fraction(1, 2, 3)
    by_sector = {record["sector_index"]: record for record in records}
    ground = min(min(record["energies_meV"]) for record in records)
    all_levels = sorted(
        (energy, record["sector_index"])
        for record in records
        for energy in record["energies_meV"]
    )
    if filling == "1/3":
        manifold = [
            by_sector[gamma]["energies_meV"][0],
            by_sector[kval]["energies_meV"][0],
            by_sector[kprime]["energies_meV"][0],
        ]
        character = "CDW: Gamma, K, K' quasi-degenerate manifold"
    else:
        manifold = by_sector[gamma]["energies_meV"][:3]
        character = "FCI/FQAH: three quasi-degenerate states at Gamma"
    ceiling = max(manifold)
    outside = [energy for energy, sector in all_levels if energy > ceiling + 1.0e-8]
    return {
        "Gamma_sector": gamma,
        "K_sector": kval,
        "Kprime_sector": kprime,
        "ground_energy_meV": ground,
        "manifold_relative_meV": [float(value - ground) for value in manifold],
        "manifold_splitting_meV": float(max(manifold) - min(manifold)),
        "estimated_gap_above_manifold_meV": float(min(outside) - ceiling) if outside else None,
        "phase_signature": character,
    }


def run_paper_27() -> None:
    cluster = TriangularSupercell.c6_27()
    energies, pair, k_points, pairs, herm = operator_cache(
        "fig5_27cell_1band", cluster, 1
    )
    log("generating C(27,9)=4,686,825 states and all translation blocks")
    bases = all_group_sector_bases(
        cluster.numerators, cluster.determinant, cluster.residue_lookup, 1, 9
    )
    log("solving nu=1/3 electron sectors")
    one_third = solve_representatives(
        cluster, bases, energies[:, 0], pair, pairs, 5, 3, "27c 1b nu=1/3"
    )
    log("constructing exact particle-hole Hamiltonian for nu=2/3")
    constant, hole_energies, hole_pair = particle_hole_transform_one_band(
        energies[:, 0], pair, pairs
    )
    two_thirds = solve_representatives(
        cluster,
        bases,
        hole_energies,
        hole_pair,
        pairs,
        5,
        3,
        "27c 1b nu=2/3",
        energy_constant=constant,
        electron_to_basis_sector=cluster.negation,
    )
    output = {
        "status": "paper-aligned one-band reproduction",
        "paper": "arXiv:2503.13585v3 Supplementary Fig. 5",
        "theta_deg": 2.6,
        "dielectric_constant": 5.0,
        "bands": 1,
        "cluster": "27-cell 3sqrt(3) x 3sqrt(3)",
        "supercell_matrix": cluster.matrix.tolist(),
        "charge_U1": True,
        "spin_U1": "fully polarized",
        "translation_sectors": 27,
        "operator_hermiticity_error": herm,
        "k_points": k_points.tolist(),
        "C6_representative_strategy": (
            "one Lanczos solve per geometric C6 orbit; cutoff-level one-body "
            "symmetry mismatch is recorded separately"
        ),
        "one_body_C6_max_error_meV": float(
            np.max(np.abs(energies[:, 0] - energies[cluster.rotation60, 0]))
        ),
        "fillings": {
            "1/3": {
                "n_particles": 9,
                "sectors": one_third,
                "diagnostics": diagnostics(cluster, one_third, "1/3"),
            },
            "2/3": {
                "n_particles": 18,
                "particle_hole_acceleration": "exact 9-hole transform",
                "sectors": two_thirds,
                "diagnostics": diagnostics(cluster, two_thirds, "2/3"),
            },
        },
    }
    path = DATA / "fig5_27cell_1band_spectrum.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    log(f"saved {path.name}")


def run_two_band_12() -> None:
    cluster = TriangularSupercell.c6_12()
    energies, pair, k_points, pairs, herm = operator_cache(
        "large_12cell_2band", cluster, 2
    )
    filling_records = {}
    for filling, n_particles in (("1/3", 4), ("2/3", 8)):
        log(f"generating two-band bases for nu={filling}, N={n_particles}")
        bases = all_group_sector_bases(
            cluster.numerators,
            cluster.determinant,
            cluster.residue_lookup,
            2,
            n_particles,
        )
        records = solve_representatives(
            cluster,
            bases,
            energies.reshape(-1),
            pair,
            pairs,
            5,
            3,
            f"12c 2b nu={filling}",
        )
        filling_records[filling] = {
            "n_particles": n_particles,
            "sectors": records,
            "diagnostics": diagnostics(cluster, records, filling),
        }
    output = {
        "status": "two-band finite-size extension; not the paper's 27-cell Fig. 5",
        "theta_deg": 2.6,
        "dielectric_constant": 5.0,
        "bands": 2,
        "cluster": "12-cell 2sqrt(3) x 2sqrt(3)",
        "supercell_matrix": cluster.matrix.tolist(),
        "translation_sectors": 12,
        "operator_hermiticity_error": herm,
        "k_points": k_points.tolist(),
        "fillings": filling_records,
    }
    path = DATA / "large_12cell_2band_spectrum.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    log(f"saved {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("paper", "two-band", "all"), default="paper"
    )
    args = parser.parse_args()
    np.random.seed(250313585)
    if args.stage in ("paper", "all"):
        run_paper_27()
    if args.stage in ("two-band", "all"):
        run_two_band_12()


if __name__ == "__main__":
    main()
