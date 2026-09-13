"""Train the continuous-coordinate neural Bloch wavefunction on paper Fig. 2."""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .continuum import ContinuumModel, ContinuumParameters
from .continuum_vmc import ContinuumTorusHamiltonian, ContinuousMetropolis
from .natural_gradient import natural_gradient_step
from .neural_bloch import ManyBodyNeuralBloch, NeuralBlochConfig


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "result" / "data"
OUTPUT = DATA / "neural_bloch"


def neural_bloch_inputs(
    theta_deg: float = 3.0,
) -> tuple[ContinuumModel, dict[str, np.ndarray]]:
    model = ContinuumModel(
        ContinuumParameters(
            theta_deg=theta_deg,
            potential_meV=11.2,
            tunneling_meV=13.3,
            phase_deg=-91.0,
            mass_me=0.62,
            plane_wave_shells=8,
        )
    )
    cache = np.load(DATA / "fig2_bloch_mesh.npz")
    if abs(theta_deg - 3.0) > 1.0e-12:
        raise ValueError("the cached 3x3 Bloch mesh is the theta=3 benchmark")
    momenta = np.asarray(cache["k_points"]).reshape(9, 2)
    coefficients = (
        np.asarray(cache["vectors"])[..., 0]
        .reshape(9, model.dim)
        .reshape(9, 2, model.n_g)
    )
    energies = np.asarray(cache["energies"])[..., 0].reshape(9)
    fractions = np.asarray(
        [[i / 3.0, j / 3.0] for i in range(3) for j in range(3)]
    )
    reciprocal = np.stack([model.b1, model.b2])
    primitive = 2.0 * np.pi * np.linalg.inv(reciprocal)
    supercell = primitive @ np.diag([3.0, 3.0])
    return model, {
        "primitive_reciprocal": reciprocal,
        "supercell_lattice": supercell,
        "momentum_fractions": fractions,
        "momenta": momenta,
        "g_vectors": model.g_vectors,
        "bloch_coefficients": coefficients,
        "one_body_energies": energies,
    }


def build_wavefunction(
    n_particles: int,
    config: NeuralBlochConfig,
    seed: int,
) -> tuple[ContinuumModel, ManyBodyNeuralBloch]:
    continuum, inputs = neural_bloch_inputs()
    return continuum, ManyBodyNeuralBloch(
        **inputs,
        n_particles=n_particles,
        config=config,
        seed=seed,
    )


def save_checkpoint(
    path: Path,
    wavefunction: ManyBodyNeuralBloch,
    config: NeuralBlochConfig,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": wavefunction.state_dict(),
            "config": asdict(config),
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[ContinuumModel, ManyBodyNeuralBloch, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = NeuralBlochConfig(**payload["config"])
    n_particles = int(payload["metadata"]["n_particles"])
    continuum, wavefunction = build_wavefunction(
        n_particles, config, int(payload["metadata"]["seed"])
    )
    incompatible = wavefunction.load_state_dict(payload["state_dict"], strict=False)
    allowed_missing = {
        "c3_fractional_rotation", "c3_layer_shifts", "translation_characters"
    }
    if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint is incompatible: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return continuum, wavefunction, payload["metadata"]


def evaluate_walkers(
    wavefunction: ManyBodyNeuralBloch,
    hamiltonian: ContinuumTorusHamiltonian,
    sampler: ContinuousMetropolis,
    count: int,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    indices = torch.randperm(
        sampler.walkers, generator=generator,
        device=sampler.state.positions.device,
    )[: min(count, sampler.walkers)]
    values = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        values.append(
            hamiltonian.local_energy(
                wavefunction, sampler.state.positions[selected],
                sampler.state.layers[selected],
            )
        )
    return torch.cat(values)


def local_energy_in_batches(
    wavefunction: ManyBodyNeuralBloch,
    hamiltonian: ContinuumTorusHamiltonian,
    positions: torch.Tensor,
    layers: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Evaluate local energies without retaining all walker graphs at once."""
    values = []
    for start in range(0, len(positions), batch_size):
        values.append(
            hamiltonian.local_energy(
                wavefunction,
                positions[start : start + batch_size],
                layers[start : start + batch_size],
            ).detach()
        )
    return torch.cat(values)


def run(args: argparse.Namespace) -> dict:
    if args.energy_batch > args.walkers:
        raise ValueError("energy_batch cannot exceed walkers")
    if args.local_energy_batch <= 0 or args.natural_gradient_chunk <= 0:
        raise ValueError("batch and natural-gradient chunk sizes must be positive")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    config = NeuralBlochConfig(
        width=args.width,
        message_passing_steps=args.message_passing_steps,
        determinants=args.determinants,
        orbital_hidden=args.width,
        c3_irrep=args.c3_irrep,
        outer_c3_projector=args.c3_irrep is not None,
        correction_init_scale=args.correction_init_scale,
    )
    continuum, wavefunction = build_wavefunction(args.particles, config, args.seed)
    output = args.output_dir
    if not output.is_absolute():
        output = ROOT / output
    if args.ed_warmstart:
        if args.particles != 6:
            raise ValueError("the cached one-band warm start is for six particles")
        warm = np.load(DATA / "fig2_1band_ground_state.npz")
        warm_amplitudes = np.asarray(warm["amplitudes"])
        order = np.argsort(np.abs(warm_amplitudes))[::-1][: args.determinants]
        wavefunction.initialize_from_fock_expansion(
            np.asarray(warm["basis"])[order], warm_amplitudes[order]
        )
    if args.resume is not None:
        continuum, wavefunction, _ = load_checkpoint(args.resume)
        config = wavefunction.config
    wavefunction.to(device)
    hamiltonian = ContinuumTorusHamiltonian(
        continuum,
        3,
        3,
        dielectric=args.epsilon,
        reciprocal_shell_count=8,
        finite_difference_fraction=args.finite_difference_fraction,
    )
    sampler = ContinuousMetropolis(
        wavefunction,
        hamiltonian,
        walkers=args.walkers,
        position_step_fraction=args.proposal_scale,
        seed=args.seed + 1009,
        device=device,
        wavefunction_batch_size=args.wavefunction_batch,
    )
    if args.initial_samples is not None:
        cached = np.load(args.initial_samples)
        if len(cached["positions"]) < args.walkers:
            raise ValueError("initial sample cache has fewer walkers than requested")
        sampler.state.positions = torch.as_tensor(
            cached["positions"][: args.walkers], dtype=torch.get_default_dtype(),
            device=device,
        ).clone()
        sampler.state.layers = torch.as_tensor(
            cached["layers"][: args.walkers], dtype=torch.long, device=device,
        ).clone()
        sampler.refresh()
    for _ in range(args.burn_sweeps):
        sampler.sweep()
    optimizer = None
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            wavefunction.parameters(), lr=args.learning_rate, betas=(0.9, 0.98)
        )
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 2027)
    trace = []
    best_path = output / "neural_bloch_best.pt"
    initial_path = output / "neural_bloch_initial.pt"
    final_path = output / "neural_bloch_final.pt"
    base_metadata = {
        "seed": args.seed, "n_particles": args.particles,
        "filling": f"{args.particles}/9", "theta_deg": 3.0,
        "epsilon": args.epsilon,
    }
    save_checkpoint(initial_path, wavefunction, config, {
        **base_metadata, "training_step": 0,
    })
    initial_validation = evaluate_walkers(
        wavefunction, hamiltonian, sampler, args.validation_batch,
        args.evaluation_batch, generator,
    )
    best_energy = float(initial_validation.real.mean())
    initial_sem = float(
        initial_validation.real.std(unbiased=True) / np.sqrt(len(initial_validation))
    )
    best_selection_score = best_energy + args.selection_sigma * initial_sem
    save_checkpoint(best_path, wavefunction, config, {
        **base_metadata, "training_step": 0,
        "validation_energy_meV": best_energy,
    })
    if args.steps > 0:
        wavefunction.reference_fast_path = False
        sampler.refresh()
    start_time = time.perf_counter()
    for step in range(args.steps):
        acceptance = sampler.sweep(args.sweeps_per_step)
        indices = torch.randperm(
            args.walkers, generator=generator, device=device
        )[: args.energy_batch]
        positions = sampler.state.positions[indices].clone()
        layers = sampler.state.layers[indices].clone()
        local_energy = local_energy_in_batches(
            wavefunction, hamiltonian, positions, layers, args.local_energy_batch
        )
        median = torch.median(local_energy.real)
        deviation = torch.median(torch.abs(local_energy.real - median)).clamp_min(1.0)
        clipped_real = torch.clamp(
            local_energy.real, median - 6.0 * deviation, median + 6.0 * deviation
        )
        clipped = torch.complex(clipped_real, local_energy.imag)
        optimizer_record: dict[str, float | int | bool] = {}
        if args.optimizer == "natural_gradient":
            diagnostics = natural_gradient_step(
                wavefunction,
                positions,
                layers,
                clipped,
                learning_rate=args.learning_rate,
                damping=args.sr_damping,
                chunk_size=args.natural_gradient_chunk,
                cg_iterations=args.cg_iterations,
                cg_tolerance=args.cg_tolerance,
                maximum_direction_norm=args.maximum_direction_norm,
            )
            optimizer_record = {
                "natural_gradient_force_norm": diagnostics.force_norm,
                "natural_gradient_direction_norm": diagnostics.direction_norm,
                "natural_gradient_residual_norm": diagnostics.residual_norm,
                "natural_gradient_cg_iterations": diagnostics.cg_iterations,
                "natural_gradient_direction_was_clipped": (
                    diagnostics.direction_was_clipped
                ),
            }
        else:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            logpsi = wavefunction(positions, layers)
            centered = clipped - clipped.mean()
            loss = 2.0 * torch.real(torch.mean(centered * torch.conj(logpsi)))
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                wavefunction.parameters(), args.gradient_clip
            )
            optimizer_record = {
                "gradient_norm_before_clip": float(gradient_norm),
                "gradient_was_clipped": bool(gradient_norm > args.gradient_clip),
            }
            optimizer.step()
        sampler.refresh()
        energy = float(local_energy.real.mean())
        record = {
            "step": step + 1,
            "energy_meV": energy,
            "energy_per_particle_meV": energy / args.particles,
            "sem_meV": float(
                local_energy.real.std(unbiased=True) / np.sqrt(len(local_energy))
            ),
            "variance_meV2": float(local_energy.real.var(unbiased=True)),
            **acceptance,
            **optimizer_record,
        }
        trace.append(record)
        if (step + 1) % args.validation_interval == 0:
            sampler.sweep(args.validation_sweeps)
            validation = evaluate_walkers(
                wavefunction, hamiltonian, sampler, args.validation_batch,
                args.evaluation_batch, generator,
            )
            validation_energy = float(validation.real.mean())
            validation_sem = float(
                validation.real.std(unbiased=True) / np.sqrt(len(validation))
            )
            record["validation_energy_meV"] = validation_energy
            record["validation_sem_meV"] = validation_sem
            selection_score = validation_energy + args.selection_sigma * validation_sem
            record["validation_selection_score_meV"] = selection_score
            if selection_score < best_selection_score:
                best_energy = validation_energy
                best_selection_score = selection_score
                save_checkpoint(best_path, wavefunction, config, {
                    **base_metadata, "training_step": step + 1,
                    "validation_energy_meV": validation_energy,
                })
        if (step + 1) % args.log_interval == 0 or step == 0:
            print(json.dumps(record), flush=True)

    save_checkpoint(final_path, wavefunction, config, {
        **base_metadata, "training_step": args.steps,
    })
    # Evaluate either the validation-selected checkpoint or the final iterate.
    evaluation_path = best_path if args.evaluation_checkpoint == "best" else final_path
    continuum, wavefunction, metadata = load_checkpoint(evaluation_path)
    sampler.wavefunction = wavefunction
    sampler.refresh()
    for _ in range(args.evaluation_burn_sweeps):
        sampler.sweep()
    evaluations = []
    for start in range(0, args.walkers, args.evaluation_batch):
        evaluations.append(
            hamiltonian.local_energy(
                wavefunction,
                sampler.state.positions[start : start + args.evaluation_batch],
                sampler.state.layers[start : start + args.evaluation_batch],
            )
        )
    evaluation = torch.cat(evaluations)
    summary = {
        "method": "continuous-coordinate many-body neural Bloch",
        "paper": "arXiv:2503.13585v3",
        "parameters": {
            "theta_deg": 3.0,
            "epsilon": args.epsilon,
            "cluster": "3x3",
            "n_particles": args.particles,
            "plane_wave_shells": 8,
            "coulomb_reciprocal_shells": 8,
            "q0_removed": True,
            "optimizer": args.optimizer,
            "learning_rate": args.learning_rate,
            "walkers": args.walkers,
            "training_steps": args.steps,
            "natural_gradient_samples_per_step": (
                args.energy_batch if args.optimizer == "natural_gradient" else 0
            ),
            "natural_gradient_damping": args.sr_damping,
            "natural_gradient_cg_iterations": args.cg_iterations,
            "natural_gradient_cg_tolerance": args.cg_tolerance,
            "initialization": "LeCunNormal",
            "implementation": "PyTorch matrix-free JVP/VJP",
            "all_parameters_trainable_from_step": 0,
            "checkpoint_selection_sigma": args.selection_sigma,
            "one_band_ed_warmstart": args.ed_warmstart,
            "initial_trial": (
                "interacting one-band ED diagnostic"
                if args.ed_warmstart
                else "band-1 Bloch orbitals with LeCunNormal neural dressing"
            ),
            "resumed_from": None if args.resume is None else str(args.resume),
            "initial_samples": (
                None if args.initial_samples is None else str(args.initial_samples)
            ),
            "config": asdict(config),
            "parameter_count": sum(p.numel() for p in wavefunction.parameters()),
        },
        "initial_validation_energy_meV": float(initial_validation.real.mean()),
        "initial_validation_sem_meV": initial_sem,
        "best_validation_energy_meV": best_energy,
        "best_validation_selection_score_meV": best_selection_score,
        "evaluation_energy_meV": float(evaluation.real.mean()),
        "evaluation_energy_per_particle_meV": float(
            evaluation.real.mean() / args.particles
        ),
        "evaluation_sem_meV": float(
            evaluation.real.std(unbiased=True) / np.sqrt(len(evaluation))
        ),
        "evaluation_variance_meV2": float(evaluation.real.var(unbiased=True)),
        "evaluation_imaginary_mean_meV": float(evaluation.imag.mean()),
        "trace": trace,
        "checkpoint": (
            str(evaluation_path.relative_to(ROOT))
            if evaluation_path.is_relative_to(ROOT)
            else str(evaluation_path)
        ),
        "evaluation_checkpoint": args.evaluation_checkpoint,
        "wall_clock_seconds": time.perf_counter() - start_time,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "neural_bloch_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output / "neural_bloch_samples.npz",
        positions=sampler.state.positions.detach().cpu().numpy(),
        layers=sampler.state.layers.detach().cpu().numpy(),
        logpsi=sampler.state.logpsi.detach().cpu().numpy(),
        local_energy=evaluation.detach().cpu().numpy(),
    )
    return summary


def run_c3_sector_scan(args: argparse.Namespace) -> dict:
    """Optimize all C3 irreps without using an ED symmetry label."""
    if args.ed_warmstart:
        raise ValueError(
            "automatic C3-sector discovery requires --no-ed-warmstart; "
            "otherwise the ED state leaks its irrep into the initialization"
        )
    base_output = args.output_dir
    if not base_output.is_absolute():
        base_output = ROOT / base_output
    sector_summaries = []
    for irrep in range(3):
        sector_args = copy.deepcopy(args)
        sector_args.scan_c3_sectors = False
        sector_args.c3_irrep = irrep
        sector_args.output_dir = base_output / f"c3_m{irrep}"
        sector_summaries.append(run(sector_args))
    scores = [
        summary["evaluation_energy_meV"] + summary["evaluation_sem_meV"]
        for summary in sector_summaries
    ]
    selected = int(np.argmin(scores))
    result = {
        "selection_rule": "minimum independently evaluated E + one SEM",
        "ed_irrep_used_for_training": False,
        "selected_c3_irrep": selected,
        "sector_scores_meV": scores,
        "sectors": sector_summaries,
    }
    base_output.mkdir(parents=True, exist_ok=True)
    (base_output / "c3_sector_scan.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output-dir", type=Path, default=OUTPUT)
    result.add_argument("--particles", type=int, default=6)
    result.add_argument("--resume", type=Path)
    result.add_argument("--initial-samples", type=Path)
    result.add_argument("--epsilon", type=float, default=5.0)
    result.add_argument("--width", type=int, default=32)
    result.add_argument("--message-passing-steps", type=int, default=2)
    result.add_argument("--c3-irrep", type=int, choices=(0, 1, 2))
    result.add_argument("--scan-c3-sectors", action="store_true")
    result.add_argument("--determinants", type=int, default=1)
    result.add_argument(
        "--ed-warmstart", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "diagnostic only: initialize from the interacting one-band ED state; "
            "production starts from the bare band-1 Bloch trial"
        ),
    )
    result.add_argument("--correction-init-scale", type=float)
    result.add_argument("--walkers", type=int, default=4128)
    result.add_argument("--wavefunction-batch", type=int, default=256)
    result.add_argument("--energy-batch", type=int, default=4128)
    result.add_argument("--local-energy-batch", type=int, default=16)
    result.add_argument("--steps", type=int, default=1000)
    result.add_argument("--burn-sweeps", type=int, default=40)
    result.add_argument("--evaluation-burn-sweeps", type=int, default=30)
    result.add_argument("--evaluation-batch", type=int, default=16)
    result.add_argument(
        "--evaluation-checkpoint", choices=("best", "final"), default="best"
    )
    result.add_argument("--sweeps-per-step", type=int, default=2)
    result.add_argument("--proposal-scale", type=float, default=0.06)
    result.add_argument("--finite-difference-fraction", type=float, default=1.5e-3)
    result.add_argument(
        "--optimizer", choices=("natural_gradient", "adam"),
        default="natural_gradient",
    )
    result.add_argument("--learning-rate", type=float, default=2.0e-3)
    result.add_argument("--gradient-clip", type=float, default=20.0)
    result.add_argument("--selection-sigma", type=float, default=1.0)
    result.add_argument("--sr-damping", type=float, default=1.0e-2)
    result.add_argument("--natural-gradient-chunk", type=int, default=16)
    result.add_argument("--cg-iterations", type=int, default=10)
    result.add_argument("--cg-tolerance", type=float, default=1.0e-3)
    result.add_argument("--maximum-direction-norm", type=float, default=100.0)
    result.add_argument("--seed", type=int, default=83)
    result.add_argument("--threads", type=int, default=32)
    result.add_argument("--log-interval", type=int, default=5)
    result.add_argument("--validation-interval", type=int, default=10)
    result.add_argument("--validation-batch", type=int, default=128)
    result.add_argument("--validation-sweeps", type=int, default=4)
    return result


def main() -> None:
    args = parser().parse_args()
    result = run_c3_sector_scan(args) if args.scan_c3_sectors else run(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
