"""One-body-density-matrix and charge diagnostics for neural Bloch states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from .run_neural_bloch import DATA, OUTPUT, load_checkpoint
from .neural_bloch_overlap import bloch_orbital_values


@torch.no_grad()
def one_body_band_density_matrix(
    wavefunction,
    continuum,
    positions: torch.Tensor,
    layers: torch.Tensor,
    n_bands: int = 5,
    auxiliary_draws: int = 4,
    batch_size: int = 16,
    seed: int = 1701,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float
]:
    r"""Estimate the n_bands-projected rho_(nk,mq) in the fixed bare Bloch basis.

    For every saved configuration, one particle coordinate x is selected and
    an auxiliary coordinate xprime is drawn uniformly from the fractional torus
    and two layers.  The unbiased estimator is

      2 N_e phi_a^*(x) phi_b(xprime) [Psi(xprime,X)/Psi(x,X)]^*.

    Here a=(n,k) and b=(m,q) independently span all 9*n_bands orbitals; no
    translation-diagonal assumption is made.
    """
    if auxiliary_draws <= 0 or batch_size <= 0:
        raise ValueError("auxiliary_draws and batch_size must be positive")
    generator = torch.Generator(device=positions.device)
    generator.manual_seed(seed)
    n_orbitals = 9 * n_bands
    sample_rho = torch.zeros(
        len(positions), n_orbitals, n_orbitals, dtype=torch.complex128
    )
    # Jointly batch all auxiliary replacements for each configuration block.
    # The base projected wavefunction is evaluated once, while every replacement
    # still rebuilds the complete graph, backflow, J/M factors and determinant.
    for start in range(0, len(positions), batch_size):
        stop = min(start + batch_size, len(positions))
        current_positions = positions[start:stop]
        current_layers = layers[start:stop]
        count = stop - start
        base_logpsi = wavefunction(current_positions, current_layers)
        particle = torch.randint(
            wavefunction.n_particles, (auxiliary_draws, count),
            generator=generator, device=positions.device,
        )
        proposal_position = torch.rand(
            auxiliary_draws, count, 2, generator=generator,
            dtype=positions.dtype, device=positions.device,
        )
        proposal_layer = torch.randint(
            0, 2, (auxiliary_draws, count), generator=generator,
            device=positions.device,
        )
        expanded_positions = current_positions.unsqueeze(0).expand(
            auxiliary_draws, -1, -1, -1
        ).clone()
        expanded_layers = current_layers.unsqueeze(0).expand(
            auxiliary_draws, -1, -1
        ).clone()
        draw = torch.arange(auxiliary_draws, device=positions.device)[:, None]
        row = torch.arange(count, device=positions.device)[None, :]
        expanded_positions[draw, row, particle] = proposal_position
        expanded_layers[draw, row, particle] = proposal_layer
        replaced_logpsi = wavefunction(
            expanded_positions.flatten(0, 1), expanded_layers.flatten(0, 1)
        ).reshape(auxiliary_draws, count)
        conjugate_ratio = torch.exp(
            torch.conj(replaced_logpsi - base_logpsi[None])
        )
        selected_position = current_positions[
            torch.arange(count, device=positions.device)[None, :], particle
        ]
        selected_layer = current_layers[
            torch.arange(count, device=positions.device)[None, :], particle
        ]
        current_orbitals = bloch_orbital_values(
            selected_position.flatten(0, 1)[:, None],
            selected_layer.flatten(0, 1)[:, None], n_bands, continuum,
        )[:, 0].reshape(auxiliary_draws, count, n_orbitals)
        proposal_orbitals = bloch_orbital_values(
            proposal_position.flatten(0, 1)[:, None],
            proposal_layer.flatten(0, 1)[:, None], n_bands, continuum,
        )[:, 0].reshape(auxiliary_draws, count, n_orbitals)
        contribution = 2.0 * wavefunction.n_particles * torch.einsum(
            "dba,dbc,db->dbac", torch.conj(current_orbitals),
            proposal_orbitals, conjugate_ratio,
        )
        sample_rho[start:stop] = contribution.mean(0).cpu()
    rho = sample_rho.mean(0)
    rho = 0.5 * (rho + torch.conj(rho.transpose(0, 1)))
    diagonal_per_sample = torch.diagonal(sample_rho, dim1=-2, dim2=-1).real
    band_weights_per_sample = (
        diagonal_per_sample.reshape(len(sample_rho), 9, n_bands).sum(1)
        / wavefunction.n_particles
    )
    blocks = min(8, max(len(sample_rho) // 2, 1))
    if blocks > 1:
        block_means = torch.stack([
            chunk.mean(0) for chunk in torch.tensor_split(
                band_weights_per_sample, blocks
            )
        ])
        weight_sem = block_means.std(0, unbiased=True) / np.sqrt(blocks)
        weight_covariance = torch.cov(block_means.T) / blocks
        trace_sem = (
            wavefunction.n_particles
            * torch.sqrt(torch.sum(weight_covariance))
        )
    else:
        weight_sem = torch.full((n_bands,), float("nan"))
        weight_covariance = torch.full((n_bands, n_bands), float("nan"))
        trace_sem = torch.as_tensor(float("nan"))
    natural_occupations, natural_orbitals = torch.linalg.eigh(rho)
    order = torch.argsort(natural_occupations, descending=True)
    natural_occupations = natural_occupations[order]
    natural_bloch_overlap = torch.abs(natural_orbitals[:, order]) ** 2
    return (
        rho.numpy(), weight_sem.numpy(), natural_occupations.numpy(),
        natural_bloch_overlap.numpy(), weight_covariance.numpy(),
        float(trace_sem),
    )


def five_band_ed_population(n_particles: int = 6) -> np.ndarray:
    if n_particles not in (3, 6):
        raise ValueError("the 3x3 benchmark supports n_particles=3 or 6")
    prefix = "nu1of3" if n_particles == 3 else "fig2"
    state = np.load(DATA / f"{prefix}_5band_ground_state.npz")
    probabilities = np.abs(np.asarray(state["amplitudes"])) ** 2
    population = np.zeros(5)
    for bitstring, probability in zip(state["basis"], probabilities):
        value = int(bitstring)
        while value:
            low = value & -value
            orbital = low.bit_length() - 1
            population[orbital % 5] += probability
            value ^= low
    return population / float(n_particles)


def folded_charge_density(
    positions: np.ndarray,
    bins: int = 64,
    smoothing_bins: float = 1.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    primitive_fraction = np.remainder(3.0 * positions.reshape(-1, 2), 1.0)
    density, xedge, yedge = np.histogram2d(
        primitive_fraction[:, 0],
        primitive_fraction[:, 1],
        bins=bins,
        range=((0.0, 1.0), (0.0, 1.0)),
    )
    density = gaussian_filter(density, smoothing_bins, mode="wrap")
    density = density / np.mean(density)
    x = 0.5 * (xedge[:-1] + xedge[1:])
    y = 0.5 * (yedge[:-1] + yedge[1:])
    return x, y, density


def folded_charge_density_statistics(
    positions: np.ndarray,
    bins: int = 64,
    smoothing_bins: float = 1.25,
    blocks: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Density mean and block SEM, retaining MCMC autocorrelation."""
    x, y, density = folded_charge_density(positions, bins, smoothing_bins)
    chunks = [
        chunk for chunk in np.array_split(positions, min(blocks, len(positions)))
        if len(chunk)
    ]
    block_density = np.stack([
        folded_charge_density(chunk, bins, smoothing_bins)[2] for chunk in chunks
    ])
    if len(block_density) > 1:
        sem = np.std(block_density, axis=0, ddof=1) / np.sqrt(len(block_density))
    else:
        sem = np.zeros_like(density)
    return x, y, density, sem


def projected_momentum_occupation(
    rho: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract n_1(k) and sum_n<=5 n_n(k) from the projected 1-RDM."""
    n_bands = rho.shape[0] // 9
    diagonal = np.real(np.diag(rho)).reshape(9, n_bands)
    return diagonal[:, 0], np.sum(diagonal, axis=1)


def _coordinate_density_modes(
    positions: np.ndarray, continuum,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reciprocal = np.stack([continuum.b1, continuum.b2])
    primitive = 2.0 * np.pi * np.linalg.inv(reciprocal)
    supercell = primitive @ np.diag([3.0, 3.0])
    cartesian = np.einsum("ac,bnc->bna", supercell, positions)
    indices = np.asarray(
        [(i, j) for j in (-1, 0, 1) for i in (-1, 0, 1)], dtype=np.int32
    )
    q_vectors = (
        indices[:, :1] * continuum.b1[None] / 3.0
        + indices[:, 1:] * continuum.b2[None] / 3.0
    )
    phase = np.einsum("bna,qa->bnq", cartesian, q_vectors)
    rho_q = np.sum(np.exp(1j * phase), axis=1)
    return indices, q_vectors, rho_q


def coordinate_structure_factor(
    positions: np.ndarray, continuum,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Connected and raw charge structure factors on the 3x3 torus mesh."""
    indices, q_vectors, rho_q = _coordinate_density_modes(positions, continuum)
    raw = np.mean(np.abs(rho_q) ** 2, axis=0) / positions.shape[1]
    connected = (
        np.mean(np.abs(rho_q) ** 2, axis=0)
        - np.abs(np.mean(rho_q, axis=0)) ** 2
    ) / positions.shape[1]
    zero = np.all(indices == 0, axis=1)
    raw[zero] = 0.0
    connected[zero] = 0.0
    return indices, q_vectors, connected.real, raw.real


def coordinate_density_statistics(
    positions: np.ndarray, continuum, blocks: int = 8,
) -> dict[str, np.ndarray]:
    """Block-covariance errors for rho_q and the full structure factor.

    The error bar is not the two-point function itself. It is the standard
    error of block estimators, whose variance is controlled by the connected
    covariance <rho_q rho_-q> - <rho_q><rho_-q>.
    """
    indices, q_vectors, rho_q = _coordinate_density_modes(positions, continuum)
    particles = positions.shape[1]
    mean = np.mean(rho_q, axis=0)
    covariance = np.mean(np.abs(rho_q) ** 2, axis=0) - np.abs(mean) ** 2
    chunks = [
        chunk for chunk in np.array_split(rho_q, min(blocks, len(rho_q)))
        if len(chunk)
    ]
    block_mean = np.stack([np.mean(chunk, axis=0) for chunk in chunks])
    block_full_structure = np.stack([
        np.mean(np.abs(chunk) ** 2, axis=0) / particles
        for chunk in chunks
    ])
    block_connected_structure = np.stack([
        (
            np.mean(np.abs(chunk) ** 2, axis=0)
            - np.abs(np.mean(chunk, axis=0)) ** 2
        ) / particles
        for chunk in chunks
    ])
    block_amplitude = np.abs(block_mean) / particles
    if len(chunks) > 1:
        full_structure_sem = np.std(block_full_structure, axis=0, ddof=1) / np.sqrt(len(chunks))
        connected_structure_sem = np.std(block_connected_structure, axis=0, ddof=1) / np.sqrt(len(chunks))
        amplitude_sem = np.std(block_amplitude, axis=0, ddof=1) / np.sqrt(len(chunks))
        real_sem = np.std(block_mean.real, axis=0, ddof=1) / np.sqrt(len(chunks))
        imag_sem = np.std(block_mean.imag, axis=0, ddof=1) / np.sqrt(len(chunks))
        complex_mean_sem = np.sqrt(real_sem**2 + imag_sem**2)
    else:
        full_structure_sem = np.zeros(len(indices))
        connected_structure_sem = np.zeros(len(indices))
        amplitude_sem = np.zeros(len(indices))
        complex_mean_sem = np.zeros(len(indices))
    zero = np.all(indices == 0, axis=1)
    full_structure_sem[zero] = 0.0
    connected_structure_sem[zero] = 0.0
    amplitude_sem[zero] = 0.0
    return {
        "indices": indices,
        "q_vectors": q_vectors,
        "rho_q_mean": mean,
        "rho_q_connected_covariance": covariance.real,
        "rho_q_mean_sem": complex_mean_sem,
        "rho_q_amplitude_per_particle": np.abs(mean) / particles,
        "rho_q_amplitude_sem": amplitude_sem,
        "structure_factor_full_sem": full_structure_sem,
        "structure_factor_connected_sem": connected_structure_sem,
        "blocks": np.asarray(len(chunks), dtype=np.int32),
    }


def run(args: argparse.Namespace) -> dict:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diagnostics requested but torch.cuda is unavailable")
    output = args.output_dir
    if not output.is_absolute():
        output = Path(__file__).resolve().parents[1] / output
    output.mkdir(parents=True, exist_ok=True)
    continuum, wavefunction, metadata = load_checkpoint(args.checkpoint)
    wavefunction.to(device)
    wavefunction.eval()
    samples = np.load(args.samples)
    count = min(args.samples_count, len(samples["positions"]))
    positions = torch.as_tensor(
        samples["positions"][:count], dtype=torch.float64, device=device
    )
    layers = torch.as_tensor(
        samples["layers"][:count], dtype=torch.long, device=device
    )
    (
        rho, band_weight_sem, natural_occupations, natural_bloch_overlap,
        band_weight_covariance, trace_sem,
    ) = one_body_band_density_matrix(
        wavefunction, continuum, positions, layers, n_bands=5,
        auxiliary_draws=args.auxiliary_draws,
        batch_size=args.wavefunction_batch,
        seed=args.seed,
    )
    n_bands = rho.shape[0] // 9
    rho_grid = rho.reshape(9, n_bands, 9, n_bands)
    rho_diagonal = np.real(np.diag(rho)).reshape(9, n_bands)
    band_weight = rho_diagonal.sum(axis=0) / wavefunction.n_particles
    rho_momentum_blocks = np.stack([
        rho_grid[k, :, k, :] for k in range(9)
    ])
    momentum_offdiagonal = rho_grid.copy()
    for momentum in range(9):
        momentum_offdiagonal[momentum, :, momentum, :] = 0.0
    offdiagonal_fraction = float(
        np.linalg.norm(momentum_offdiagonal.ravel())
        / max(np.linalg.norm(rho.ravel()), 1.0e-300)
    )
    hermiticity_error = float(np.linalg.norm(rho - rho.conj().T))
    ed = five_band_ed_population(wavefunction.n_particles)
    sample_positions = np.asarray(samples["positions"][:count])
    x, y, density, density_sem = folded_charge_density_statistics(
        sample_positions, bins=args.density_bins,
        smoothing_bins=args.density_smoothing,
        blocks=args.error_blocks,
    )
    nk_band1, nk_first_five = projected_momentum_occupation(rho)
    q_indices, q_vectors, structure, structure_raw = (
        coordinate_structure_factor(sample_positions, continuum)
    )
    density_statistics = coordinate_density_statistics(
        sample_positions, continuum, blocks=args.error_blocks
    )
    np.savez_compressed(
        output / "neural_bloch_diagnostics.npz",
        one_body_density_matrix=rho,
        one_body_density_matrix_projected_5band=rho,
        one_body_density_matrix_k_blocks=rho_momentum_blocks,
        band_weight=band_weight,
        band_weight_sem=band_weight_sem,
        band_weight_covariance=band_weight_covariance,
        one_body_density_matrix_trace_sem=trace_sem,
        natural_occupations=natural_occupations,
        natural_bloch_overlap=natural_bloch_overlap,
        five_band_ed_population=ed,
        density_x_fraction=x,
        density_y_fraction=y,
        charge_density_over_mean=density,
        charge_density_sem_over_mean=density_sem,
        momentum_indices=np.arange(9, dtype=np.int32),
        momentum_k_points=np.asarray(np.load(DATA / "fig2_bloch_mesh.npz")["k_points"]).reshape(9, 2),
        momentum_occupation_band1=nk_band1,
        momentum_occupation_first_five=nk_first_five,
        structure_q_indices=q_indices,
        structure_q_vectors=q_vectors,
        charge_structure_factor=structure,
        charge_structure_factor_full=structure_raw,
        charge_structure_factor_connected=structure,
        charge_structure_factor_raw=structure_raw,
        charge_structure_factor_full_sem=density_statistics["structure_factor_full_sem"],
        charge_structure_factor_connected_sem=(
            density_statistics["structure_factor_connected_sem"]
        ),
        density_fourier_mean=density_statistics["rho_q_mean"],
        density_fourier_connected_covariance=(
            density_statistics["rho_q_connected_covariance"]
        ),
        density_fourier_mean_sem=density_statistics["rho_q_mean_sem"],
        density_fourier_amplitude_per_particle=(
            density_statistics["rho_q_amplitude_per_particle"]
        ),
        density_fourier_amplitude_sem=density_statistics["rho_q_amplitude_sem"],
        error_blocks=density_statistics["blocks"],
    )
    result = {
        "definition": (
            "five-band-projected many-body one-body density matrix "
            "rho_(nk,mq)=<c^dagger_(mq)c_(nk)> in the fixed bare Bloch basis, "
            "estimated from |Psi|^2 samples and full-wavefunction ratios"
        ),
        "checkpoint": str(args.checkpoint),
        "coordinate_samples": count,
        "compute_device": str(device),
        "auxiliary_draws_per_sample": args.auxiliary_draws,
        "one_body_density_matrix_shape": list(rho.shape),
        "projected_bare_band_count": n_bands,
        "projector_uses_fixed_bare_bloch_orbitals": True,
        "replacement_recomputes_complete_neural_ansatz": True,
        "base_wavefunction_cached_per_configuration_batch": True,
        "auxiliary_replacements_jointly_batched": True,
        "trace_is_projected_five_band_trace_not_full_basis_trace": True,
        "one_body_density_matrix_trace": float(np.trace(rho).real),
        "one_body_density_matrix_trace_sem": trace_sem,
        "one_body_density_matrix_hermiticity_error": hermiticity_error,
        "projected_rho_minimum_eigenvalue": float(np.min(natural_occupations)),
        "projected_rho_negative_eigenvalue_weight": float(
            -np.sum(np.minimum(natural_occupations, 0.0))
        ),
        "momentum_offdiagonal_frobenius_fraction": offdiagonal_fraction,
        "band_weight": band_weight.tolist(),
        "band_weight_sem": band_weight_sem.tolist(),
        "band_weight_covariance": band_weight_covariance.tolist(),
        "first_five_band_weight_sum": float(band_weight.sum()),
        "single_band_fidelity": float(band_weight[0]),
        "remote_band_weight": float(1.0 - band_weight[0]),
        "finite_sample_note": (
            "rho is Hermitized but not projected to the positive semidefinite cone; "
            "individual small negative weights are retained as Monte Carlo noise"
        ),
        "natural_occupations": natural_occupations.tolist(),
        "natural_bloch_overlap": natural_bloch_overlap.tolist(),
        "five_band_ed_population": ed.tolist(),
        "five_band_ed_population_sum": float(ed.sum()),
        "charge_density_min_over_mean": float(density.min()),
        "charge_density_max_over_mean": float(density.max()),
        "charge_density_mean_sem_over_mean": float(np.mean(density_sem)),
        "charge_density_max_sem_over_mean": float(np.max(density_sem)),
        "error_blocks": int(density_statistics["blocks"]),
        "density_error_definition": (
            "block standard errors; their covariance is controlled by the "
            "connected density-density correlator, not identified with it"
        ),
        "momentum_occupation_band1": nk_band1.tolist(),
        "momentum_occupation_band1_sum": float(nk_band1.sum()),
        "momentum_occupation_first_five": nk_first_five.tolist(),
        "momentum_occupation_first_five_sum": float(nk_first_five.sum()),
        "charge_structure_factor_full": structure_raw.tolist(),
        "charge_structure_factor_connected": structure.tolist(),
        "charge_structure_factor_raw": structure_raw.tolist(),
        "charge_structure_factor_full_sem": density_statistics["structure_factor_full_sem"].tolist(),
        "charge_structure_factor_connected_sem": density_statistics["structure_factor_connected_sem"].tolist(),
        "density_fourier_amplitude_per_particle": density_statistics["rho_q_amplitude_per_particle"].tolist(),
        "density_fourier_amplitude_sem": density_statistics["rho_q_amplitude_sem"].tolist(),
        "structure_factor_definitions": {
            "full": "S_full(q)=<|rho_q|^2>/N_e; S_full(0) displayed as zero",
            "connected": "S_c(q)=(<|rho_q|^2>-|<rho_q>|^2)/N_e; S_c(0)=0",
            "density_order_parameter": "m_q=|<rho_q>|/N_e",
        },
        "metadata": metadata,
    }
    (output / "neural_bloch_diagnostics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output-dir", type=Path, default=OUTPUT)
    result.add_argument(
        "--checkpoint", type=Path, default=OUTPUT / "neural_bloch_best.pt"
    )
    result.add_argument(
        "--samples", type=Path, default=OUTPUT / "neural_bloch_samples.npz"
    )
    result.add_argument("--samples-count", type=int, default=512)
    result.add_argument("--density-bins", type=int, default=64)
    result.add_argument("--density-smoothing", type=float, default=1.25)
    result.add_argument("--auxiliary-draws", type=int, default=4)
    result.add_argument("--error-blocks", type=int, default=8)
    result.add_argument("--wavefunction-batch", type=int, default=16)
    result.add_argument("--seed", type=int, default=1701)
    result.add_argument("--threads", type=int, default=32)
    result.add_argument(
        "--device", default="cpu",
        help="PyTorch compute device for complete-wavefunction ratio batches",
    )
    return result


def main() -> None:
    print(json.dumps(run(parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
