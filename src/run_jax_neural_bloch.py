"""Paper-protocol JAX training for the continuum neural-Bloch state.

The publication-fixed defaults are natural gradient, learning rate 2e-3,
4,128 MCMC samples per update, 1,000 updates, and LeCunNormal
initialization.  Numerical SR regularization and batching are exposed because
they are not specified in arXiv:2503.13585v3.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import numpy as np

from .jax_neural_bloch import (
    JaxNeuralBlochSpec,
    initialize,
    local_energy,
    logpsi,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "result" / "data" / "neural_bloch_jax_production"


def _batched(function, parameters, positions, layers, batch_size):
    """Evaluate a JIT function in fixed-shape batches, padding only the tail."""
    values = []
    for start in range(0, len(positions), batch_size):
        stop = min(start + batch_size, len(positions))
        count = stop - start
        chunk_positions = positions[start:stop]
        chunk_layers = layers[start:stop]
        if count < batch_size:
            pad = batch_size - count
            chunk_positions = jnp.concatenate(
                [chunk_positions, jnp.repeat(chunk_positions[-1:], pad, axis=0)]
            )
            chunk_layers = jnp.concatenate(
                [chunk_layers, jnp.repeat(chunk_layers[-1:], pad, axis=0)]
            )
        values.append(function(parameters, chunk_positions, chunk_layers)[:count])
    return jnp.concatenate(values)


class JaxMetropolis:
    def __init__(
        self,
        parameters,
        constants,
        spec,
        walkers,
        proposal_scale,
        seed,
        batch_size,
    ):
        self.constants = constants
        self.spec = spec
        self.walkers = walkers
        self.proposal_scale = proposal_scale
        self.batch_size = batch_size
        self.key = jax.random.PRNGKey(seed)
        self.key, position_key, layer_key = jax.random.split(self.key, 3)
        self.positions = jax.random.uniform(
            position_key, (walkers, spec.n_particles, 2), dtype=jnp.float64
        )
        self.layers = jax.random.randint(
            layer_key, (walkers, spec.n_particles), 0, 2, dtype=jnp.int32
        )
        self._logpsi = jax.jit(
            lambda p, x, l: logpsi(p, x, l, constants, spec)
        )
        self.values = self.evaluate(parameters, self.positions, self.layers)

    def evaluate(self, parameters, positions, layers):
        return _batched(
            self._logpsi, parameters, positions, layers, self.batch_size
        )

    def refresh(self, parameters):
        self.values = self.evaluate(parameters, self.positions, self.layers)

    def sweep(self, parameters, sweeps=1):
        accepted_position = 0.0
        accepted_layer = 0.0
        indices = jnp.arange(self.walkers)
        for _ in range(sweeps):
            self.key, particle_key, displacement_key, accept_key = jax.random.split(
                self.key, 4
            )
            particles = jax.random.randint(
                particle_key, (self.walkers,), 0, self.spec.n_particles
            )
            displacement = self.proposal_scale * jax.random.normal(
                displacement_key, (self.walkers, 2), dtype=jnp.float64
            )
            candidate_positions = self.positions.at[indices, particles].set(
                jnp.remainder(
                    self.positions[indices, particles] + displacement, 1.0
                )
            )
            candidate_values = self.evaluate(
                parameters, candidate_positions, self.layers
            )
            accept = jnp.log(jax.random.uniform(accept_key, (self.walkers,))) < (
                2.0 * (candidate_values.real - self.values.real)
            )
            self.positions = jnp.where(
                accept[:, None, None], candidate_positions, self.positions
            )
            self.values = jnp.where(accept, candidate_values, self.values)
            accepted_position += float(jnp.mean(accept))

            self.key, particle_key, accept_key = jax.random.split(self.key, 3)
            particles = jax.random.randint(
                particle_key, (self.walkers,), 0, self.spec.n_particles
            )
            candidate_layers = self.layers.at[indices, particles].set(
                1 - self.layers[indices, particles]
            )
            candidate_values = self.evaluate(
                parameters, self.positions, candidate_layers
            )
            accept = jnp.log(jax.random.uniform(accept_key, (self.walkers,))) < (
                2.0 * (candidate_values.real - self.values.real)
            )
            self.layers = jnp.where(
                accept[:, None], candidate_layers, self.layers
            )
            self.values = jnp.where(accept, candidate_values, self.values)
            accepted_layer += float(jnp.mean(accept))
        return {
            "position_acceptance": accepted_position / sweeps,
            "layer_acceptance": accepted_layer / sweeps,
        }


class MatrixFreeNaturalGradient:
    def __init__(self, constants, spec, chunk_size, damping):
        self.constants = constants
        self.spec = spec
        self.chunk_size = chunk_size
        self.damping = damping
        self.unravel = None

        def scores(flat_parameters, positions, layers):
            values = logpsi(
                self.unravel(flat_parameters), positions, layers, constants, spec
            )
            return jnp.concatenate([values.real, values.imag])

        def force_chunk(flat_parameters, positions, layers, centered, total):
            def loss(flat):
                values = logpsi(
                    self.unravel(flat), positions, layers, constants, spec
                )
                return 2.0 * jnp.real(
                    jnp.sum(centered * jnp.conj(values)) / total
                )
            return jax.grad(loss)(flat_parameters)

        def tangent_chunk(flat_parameters, vector, positions, layers):
            return jax.jvp(
                lambda flat: scores(flat, positions, layers),
                (flat_parameters,),
                (vector,),
            )[1]

        def pullback_chunk(flat_parameters, positions, layers, cotangent):
            _, pullback = jax.vjp(
                lambda flat: scores(flat, positions, layers), flat_parameters
            )
            return pullback(cotangent)[0]

        self._force_chunk = jax.jit(force_chunk)
        self._tangent_chunk = jax.jit(tangent_chunk)
        self._pullback_chunk = jax.jit(pullback_chunk)

    def bind(self, parameters):
        flat, self.unravel = ravel_pytree(parameters)
        return flat

    def _chunks(self, positions, layers):
        if len(positions) % self.chunk_size:
            raise ValueError("natural-gradient samples must divide the SR chunk size")
        for start in range(0, len(positions), self.chunk_size):
            stop = start + self.chunk_size
            yield positions[start:stop], layers[start:stop]

    def force(self, flat_parameters, positions, layers, energies):
        centered = energies - jnp.mean(energies)
        result = jnp.zeros_like(flat_parameters)
        for start, (chunk_positions, chunk_layers) in enumerate(
            self._chunks(positions, layers)
        ):
            begin = start * self.chunk_size
            result = result + self._force_chunk(
                flat_parameters,
                chunk_positions,
                chunk_layers,
                centered[begin:begin + self.chunk_size],
                len(positions),
            )
        return result

    def product(self, flat_parameters, vector, positions, layers):
        tangents = []
        real_sum = jnp.asarray(0.0)
        imag_sum = jnp.asarray(0.0)
        for chunk_positions, chunk_layers in self._chunks(positions, layers):
            tangent = self._tangent_chunk(
                flat_parameters, vector, chunk_positions, chunk_layers
            )
            count = len(chunk_positions)
            tangents.append(tangent)
            real_sum = real_sum + jnp.sum(tangent[:count])
            imag_sum = imag_sum + jnp.sum(tangent[count:])
        real_mean = real_sum / len(positions)
        imag_mean = imag_sum / len(positions)
        result = jnp.zeros_like(vector)
        for tangent, (chunk_positions, chunk_layers) in zip(
            tangents, self._chunks(positions, layers)
        ):
            count = len(chunk_positions)
            cotangent = jnp.concatenate([
                (tangent[:count] - real_mean) / len(positions),
                (tangent[count:] - imag_mean) / len(positions),
            ])
            result = result + self._pullback_chunk(
                flat_parameters, chunk_positions, chunk_layers, cotangent
            )
        return result + self.damping * vector

    def direction(
        self,
        flat_parameters,
        positions,
        layers,
        energies,
        iterations,
        tolerance,
        min_iterations=0,
        true_tolerance=None,
    ):
        """Ordinary CG with optional explicit true-residual stopping checks.

        When ``true_tolerance`` is provided, the authoritative residual is
        recomputed after every iteration beginning at ``min_iterations``.
        No preconditioner is used.
        """
        force = self.force(flat_parameters, positions, layers, energies)
        solution = jnp.zeros_like(force)
        residual = force
        search = residual
        residual_squared = jnp.vdot(residual, residual).real
        initial_norm = jnp.maximum(jnp.sqrt(residual_squared), 1.0e-30)
        completed = 0
        true_checks = 0
        true_residual = None
        true_residual_iteration = -1
        termination_reason = "maximum_iterations"
        for iteration in range(iterations):
            product = self.product(
                flat_parameters, search, positions, layers
            )
            denominator = jnp.vdot(search, product).real
            denominator_value = float(denominator)
            if not np.isfinite(denominator_value) or denominator_value <= 0.0:
                termination_reason = "nonpositive_or_nonfinite_curvature"
                break
            alpha = residual_squared / denominator
            solution = solution + alpha * search
            residual = residual - alpha * product
            next_squared = jnp.vdot(residual, residual).real
            completed = iteration + 1
            if true_tolerance is not None and completed >= min_iterations:
                true_residual = force - self.product(
                    flat_parameters, solution, positions, layers
                )
                true_residual_iteration = completed
                true_checks += 1
                true_relative = float(jnp.linalg.norm(true_residual) / initial_norm)
                if true_relative < true_tolerance:
                    residual_squared = next_squared
                    termination_reason = "true_residual_tolerance"
                    break
            elif true_tolerance is None and (
                float(jnp.sqrt(next_squared)) <= tolerance * float(initial_norm)
            ):
                residual_squared = next_squared
                termination_reason = "recursive_residual_tolerance"
                break
            search = residual + (next_squared / residual_squared) * search
            residual_squared = next_squared
        if true_residual_iteration != completed:
            true_residual = force - self.product(
                flat_parameters, solution, positions, layers
            )
            true_checks += 1
        true_norm = jnp.linalg.norm(true_residual)
        return solution, {
            "force_norm": float(jnp.linalg.norm(force)),
            "direction_norm": float(jnp.linalg.norm(solution)),
            "force_dot_direction": float(jnp.vdot(force, solution).real),
            "residual_norm": float(true_norm),
            "relative_residual_norm": float(true_norm / initial_norm),
            "cg_iterations": completed,
            "true_residual_checks": true_checks,
            "cg_termination_reason": termination_reason,
            "preconditioner": "none",
        }


def _save_checkpoint(path, flat_parameters, metadata, sampler=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"flat_parameters": np.asarray(flat_parameters)}
    if sampler is not None:
        payload["positions"] = np.asarray(sampler.positions)
        payload["layers"] = np.asarray(sampler.layers)
        payload["sampler_key"] = np.asarray(sampler.key)
    np.savez_compressed(path, **payload)
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def _five_band_ed_energy_per_particle(particles):
    if particles == 3:
        path = ROOT / "result" / "data" / "nu1of3_5band_ground_state.npz"
    elif particles == 6:
        path = ROOT / "result" / "data" / "fig2_5band_ground_state.npz"
    else:
        return None
    return float(np.load(path)["energy_meV"]) / particles


def _adaptive_update_is_accepted(diagnostics, args):
    positive = (
        np.isfinite(diagnostics["force_dot_direction"])
        and diagnostics["force_dot_direction"] > 0.0
    )
    early = (
        diagnostics["cg_termination_reason"] == "true_residual_tolerance"
        and diagnostics["relative_residual_norm"] < args.cg_true_tolerance
    )
    at_cap = (
        diagnostics["cg_iterations"] >= args.cg_max_iterations
        and diagnostics["relative_residual_norm"] < args.cg_acceptance_tolerance
    )
    return bool(positive and (early or at_cap))


def _adaptive_next_state(update_accepted, recovery_mode, stable_count, args):
    if not update_accepted:
        return True, 0, args.recovery_learning_rate, args.recovery_damping
    if recovery_mode:
        stable_count += 1
        if stable_count >= args.recovery_stable_steps:
            return False, 0, args.learning_rate, args.sr_damping
        return True, stable_count, args.recovery_learning_rate, args.recovery_damping
    return False, 0, args.learning_rate, args.sr_damping


def _ten_step_progress(trace, particles, rejected_total, next_learning_rate, next_damping):
    window = trace[-10:]
    energies = np.asarray([row["energy_per_particle_meV"] for row in window])
    variances = np.asarray([row["variance_meV2"] for row in window])
    iterations = np.asarray([row["cg_iterations"] for row in window])
    residuals = np.asarray([row["relative_residual_norm"] for row in window])
    ed = _five_band_ed_energy_per_particle(particles)
    mean_energy = float(energies.mean())
    return {
        "event": "ten_step_progress",
        "step": int(trace[-1]["step"]),
        "last_10_energy_mean_per_particle_meV": mean_energy,
        "last_10_energy_min_per_particle_meV": float(energies.min()),
        "last_10_local_energy_variance_mean_meV2": float(variances.mean()),
        "latest_local_energy_variance_meV2": float(variances[-1]),
        "last_10_cg_iterations_mean": float(iterations.mean()),
        "last_10_cg_iterations_max": int(iterations.max()),
        "last_10_true_residual_mean": float(residuals.mean()),
        "latest_true_residual": float(residuals[-1]),
        "last_10_rejected_updates": int(sum(not row["update_accepted"] for row in window)),
        "rejected_updates_cumulative": int(rejected_total),
        "current_learning_rate": float(trace[-1]["learning_rate"]),
        "current_damping": float(trace[-1]["sr_damping"]),
        "next_learning_rate": float(next_learning_rate),
        "next_damping": float(next_damping),
        "last_10_position_acceptance_mean": float(np.mean([row["position_acceptance"] for row in window])),
        "last_10_layer_acceptance_mean": float(np.mean([row["layer_acceptance"] for row in window])),
        "five_band_ed_energy_per_particle_meV": ed,
        "last_10_mean_minus_five_band_ed_meV": None if ed is None else mean_energy - ed,
    }


def run(args):
    if args.samples != 4128 and not args.smoke_test:
        raise ValueError("paper-protocol training requires exactly 4128 samples/update")
    allowed_exploratory_steps = args.exploratory_100_step and args.steps == 100
    allowed_additional_100_steps = (
        args.exploratory_additional_100_step
        and args.resume is not None
        and args.steps == 200
    )
    allowed_checkpoint_continuation = (
        args.checkpoint_additional_100_step
        and args.resume is not None
        and args.steps >= 200
    )
    allowed_adaptive_120 = args.adaptive_cg_120_step and args.steps == 120
    if (
        args.steps != 1000
        and not args.smoke_test
        and not allowed_exploratory_steps
        and not allowed_additional_100_steps
        and not allowed_checkpoint_continuation
        and not allowed_adaptive_120
    ):
        raise ValueError(
            "use 1000 paper-target updates, --exploratory-100-step with --steps 100, "
            "--exploratory-additional-100-step with a step-100 --resume and --steps 200, "
            "or --adaptive-cg-120-step with --steps 120 from a fresh initialization"
        )
    if abs(args.learning_rate - 2.0e-3) > 1.0e-15 and not args.smoke_test:
        raise ValueError("paper-protocol learning rate is 2e-3")
    architecture = (
        args.width, args.message_passing_steps, args.determinants
    )
    if architecture != (32, 2, 1) and not args.smoke_test:
        raise ValueError(
            "production architecture is width=32, message-passing-steps=2, "
            "determinants=1"
        )
    if args.outer_c3_projector and not args.fixed_gamma_no_m:
        raise ValueError("production outer C3 sector scan requires --fixed-gamma-no-m")
    if args.samples % args.sr_chunk:
        raise ValueError("samples must be divisible by sr_chunk")
    if args.adaptive_cg_120_step:
        if args.cg_min_iterations != 20 or args.cg_max_iterations != 60:
            raise ValueError("adaptive protocol requires CG min/max = 20/60")
        if abs(args.cg_true_tolerance - 0.02) > 1.0e-15:
            raise ValueError("adaptive protocol requires true residual tolerance 0.02")
        if abs(args.cg_acceptance_tolerance - 0.05) > 1.0e-15:
            raise ValueError("adaptive protocol requires acceptance residual 0.05")
        if args.checkpoint_interval != 10:
            raise ValueError("adaptive protocol requires checkpoints every 10 steps")
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    spec = JaxNeuralBlochSpec(
        n_particles=args.particles,
        width=args.width,
        message_passing_steps=args.message_passing_steps,
        determinants=args.determinants,
        orbital_hidden=args.width,
        c3_irrep=args.c3_irrep,
        fixed_gamma_no_m=args.fixed_gamma_no_m,
        c3_qns=args.c3_qns,
        outer_c3_projector=args.outer_c3_projector,
    )
    parameters, constants = initialize(spec, args.seed)
    natural_gradient = MatrixFreeNaturalGradient(
        constants, spec, args.sr_chunk, args.sr_damping
    )
    flat_parameters = natural_gradient.bind(parameters)
    parameter_count = len(flat_parameters)
    start_step = 0
    resume_positions = None
    resume_layers = None
    resume_sampler_key = None
    resume_optimizer_state = None
    if args.resume is not None:
        with np.load(args.resume) as checkpoint:
            loaded = np.asarray(checkpoint["flat_parameters"])
            if loaded.shape != np.asarray(flat_parameters).shape:
                raise ValueError("resume checkpoint parameter shape does not match")
            flat_parameters = jnp.asarray(loaded)
            if "positions" in checkpoint and "layers" in checkpoint:
                resume_positions = jnp.asarray(checkpoint["positions"])
                resume_layers = jnp.asarray(checkpoint["layers"])
            if "sampler_key" in checkpoint:
                resume_sampler_key = jnp.asarray(checkpoint["sampler_key"])
        metadata_path = args.resume.with_suffix(".json")
        if metadata_path.exists():
            resume_metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            start_step = int(resume_metadata.get("training_step", 0))
            resume_optimizer_state = resume_metadata.get("optimizer_state")
        parameters = natural_gradient.unravel(flat_parameters)
    if args.checkpoint_additional_100_step and not (
        start_step < args.steps <= start_step + 100
    ):
        raise ValueError(
            "--checkpoint-additional-100-step requires a positive continuation "
            "of at most 100 updates"
        )
    if args.exploratory_additional_100_step and not (100 <= start_step < 200):
        raise ValueError(
            "--exploratory-additional-100-step requires a checkpoint with "
            "100 <= training_step < 200"
        )
    sampler = JaxMetropolis(
        parameters,
        constants,
        spec,
        args.samples,
        args.proposal_scale,
        args.seed + 1009,
        args.wavefunction_batch,
    )
    if resume_positions is not None:
        if resume_positions.shape != sampler.positions.shape:
            raise ValueError("resume checkpoint walker shape does not match")
        sampler.positions = resume_positions
        sampler.layers = resume_layers
        if resume_sampler_key is not None:
            sampler.key = resume_sampler_key
        sampler.refresh(parameters)
    local_energy_function = jax.jit(
        lambda p, x, l: local_energy(p, x, l, constants, spec)
    )
    protocol = {
        "paper": "arXiv:2503.13585v3",
        "optimizer": "natural gradient / stochastic reconfiguration",
        "learning_rate": args.learning_rate,
        "mcmc_samples_per_update": args.samples,
        "training_steps": args.steps,
        "initialization": "jax.nn.initializers.lecun_normal",
        "implementation": f"JAX {jax.__version__}",
        "ed_warm_start": False,
        "initial_trial": (
            "fixed mean-field band-1 Bloch coefficients; random trainable h0; "
            "complex LeCunNormal M for full-M"
        ),
        "architecture": "Luo-Fu Appendix B Eqs. (16)-(23)",
        "node_features": "sin/cos(g_a dot r_i), l_i in {-1,+1}; 7 physical + 25 learned = 32",
        "edge_features": "sin/cos(k dot (r_i-r_j)), source l_i; 19 physical + 13 learned = 32",
        "auxiliary_h0": "random trainable shared vectors; N(0,1/32) local convention",
        "message_passing": "2 unshared iterations; literal sum_j m_ij including self; F,G,H,U one-layer GELU; physical features reinserted",
        "momentum_matrix_initialization": "complex LeCunNormal with E|M_gamma_k|^2=1/N_k; absent for no-M",
        "backflow": "complex linear W V_i with no tanh or external scale",
        "orbital_transform": "complex two-layer MLP J_ki({k},V_i), hidden 32, applied exactly once",
        "bloch_coefficients": "fixed mean-field u^l_{1kG}; only r_i -> r_i + delta r_i changes",
        "spec": asdict(spec),
        "parameter_count": parameter_count,
        "sr_damping_unreported_by_paper": args.sr_damping,
        "cg_iterations_unreported_by_paper": args.cg_iterations,
        "cg_tolerance_unreported_by_paper": args.cg_tolerance,
        "adaptive_cg_120_step": args.adaptive_cg_120_step,
        "adaptive_cg": {
            "preconditioner": "none",
            "minimum_iterations": args.cg_min_iterations,
            "maximum_iterations": args.cg_max_iterations,
            "true_residual_check_starts": args.cg_min_iterations,
            "early_stop_true_residual": args.cg_true_tolerance,
            "maximum_iteration_acceptance_residual": args.cg_acceptance_tolerance,
            "recovery_damping": args.recovery_damping,
            "recovery_learning_rate": args.recovery_learning_rate,
            "stable_steps_to_restore": args.recovery_stable_steps,
        } if args.adaptive_cg_120_step else None,
        "validation_during_training": False,
        "smoke_test": args.smoke_test,
        "exploratory_100_step": args.exploratory_100_step,
        "exploratory_additional_100_step": args.exploratory_additional_100_step,
        "checkpoint_additional_100_step": args.checkpoint_additional_100_step,
        "fixed_gamma_no_m": args.fixed_gamma_no_m,
        "c3_qns": args.c3_qns,
        "outer_c3_projector": args.outer_c3_projector,
        "outer_c3_projector_definition": ({
            "raw_state": "unchanged periodic fixed-Gamma no-M Luo-Fu QNS",
            "projector": "P_m=(1/3) sum_a omega^(-ma) C3^a on the complete wavefunction",
            "physical_action": "C3 includes the continuum layer-gauge sewing factor",
            "downstream_use": "projected logpsi is used by local energy, Metropolis, and SR scores",
        } if args.outer_c3_projector else None),
        "c3_qns_definition": ({
            "group_average": (
                "Reynolds average over X, C3 X, and C3^2 X for both complex "
                "backflow displacement and J_ki"
            ),
            "momentum_orbit": (
                "three C3-closed Gamma minors tied by one-band Bloch sewing "
                "phases; this is the constrained w sum"
            ),
            "physical_gauge_compensation": (
                "each particle column is multiplied by exp(i b_l dot delta_r), "
                "where (R^T-I)b_l=a_l"
            ),
            "difference_from_luo_fu": (
                "Luo-Fu Fourier inputs enforce torus periodicity and primitive "
                "translation symmetry but do not by themselves constrain C3"
            ),
        } if args.c3_qns else None),
        "momentum_sector": [0, 0] if args.fixed_gamma_no_m else None,
        "paper_optimization_settings_retained_except_total_steps": (
            args.exploratory_100_step or args.exploratory_additional_100_step
        ),
        "resumed_from": None if args.resume is None else str(args.resume),
        "start_step": start_step,
    }
    if start_step == 0:
        _save_checkpoint(
            output / "jax_neural_bloch_initial.npz", flat_parameters, protocol,
            sampler,
        )

    burn_start = time.perf_counter()
    for sweep in range(args.burn_sweeps):
        acceptance = sampler.sweep(parameters)
        if (sweep + 1) % args.log_interval == 0:
            print(json.dumps({"burn_sweep": sweep + 1, **acceptance}), flush=True)
    burn_seconds = time.perf_counter() - burn_start

    trace_path = output / "jax_neural_bloch_trace.json"
    trace = []
    if start_step and trace_path.exists():
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    start_time = time.perf_counter()
    current_learning_rate = args.learning_rate
    current_damping = args.sr_damping
    recovery_mode = False
    recovery_stable_count = 0
    rejected_updates = 0
    if resume_optimizer_state is not None:
        current_learning_rate = float(resume_optimizer_state.get(
            "next_learning_rate", current_learning_rate
        ))
        current_damping = float(resume_optimizer_state.get(
            "next_sr_damping", current_damping
        ))
        recovery_mode = bool(resume_optimizer_state.get(
            "recovery_mode", recovery_mode
        ))
        recovery_stable_count = int(resume_optimizer_state.get(
            "recovery_stable_count", recovery_stable_count
        ))
        rejected_updates = int(resume_optimizer_state.get(
            "rejected_updates", rejected_updates
        ))
    for step in range(start_step, args.steps):
        step_start = time.perf_counter()
        acceptance = sampler.sweep(parameters, args.sweeps_per_step)
        energies = _batched(
            local_energy_function,
            parameters,
            sampler.positions,
            sampler.layers,
            args.local_energy_batch,
        )
        natural_gradient.damping = current_damping
        if args.adaptive_cg_120_step:
            direction, diagnostics = natural_gradient.direction(
                flat_parameters,
                sampler.positions,
                sampler.layers,
                energies,
                args.cg_max_iterations,
                0.0,
                min_iterations=args.cg_min_iterations,
                true_tolerance=args.cg_true_tolerance,
            )
            update_accepted = _adaptive_update_is_accepted(diagnostics, args)
        else:
            direction, diagnostics = natural_gradient.direction(
                flat_parameters,
                sampler.positions,
                sampler.layers,
                energies,
                args.cg_iterations,
                args.cg_tolerance,
            )
            update_accepted = True

        used_learning_rate = current_learning_rate
        used_damping = current_damping
        recovery_before = recovery_mode
        if update_accepted:
            flat_parameters = flat_parameters - used_learning_rate * direction
            parameters = natural_gradient.unravel(flat_parameters)
            sampler.refresh(parameters)
        else:
            rejected_updates += 1

        if args.adaptive_cg_120_step:
            (
                recovery_mode,
                recovery_stable_count,
                current_learning_rate,
                current_damping,
            ) = _adaptive_next_state(
                update_accepted, recovery_mode, recovery_stable_count, args
            )

        real_energy = np.asarray(energies.real)
        record = {
            "step": step + 1,
            "energy_meV": float(np.mean(real_energy)),
            "energy_per_particle_meV": float(np.mean(real_energy) / args.particles),
            "sem_meV": float(np.std(real_energy, ddof=1) / np.sqrt(args.samples)),
            "variance_meV2": float(np.var(real_energy, ddof=1)),
            "step_seconds": time.perf_counter() - step_start,
            "update_accepted": update_accepted,
            "rejected_updates_cumulative": rejected_updates,
            "learning_rate": used_learning_rate,
            "sr_damping": used_damping,
            "recovery_mode_before_step": recovery_before,
            "recovery_mode_after_step": recovery_mode,
            "recovery_stable_count_after_step": recovery_stable_count,
            "next_learning_rate": current_learning_rate,
            "next_sr_damping": current_damping,
            **acceptance,
            **diagnostics,
        }
        trace.append(record)
        trace_path.write_text(
            json.dumps(trace, indent=2) + "\n", encoding="utf-8"
        )
        optimizer_state = {
            "next_learning_rate": current_learning_rate,
            "next_sr_damping": current_damping,
            "recovery_mode": recovery_mode,
            "recovery_stable_count": recovery_stable_count,
            "rejected_updates": rejected_updates,
        }
        _save_checkpoint(
            output / "jax_neural_bloch_latest.npz",
            flat_parameters,
            {
                **protocol, "training_step": step + 1, "record": record,
                "optimizer_state": optimizer_state,
            },
            sampler,
        )
        if (step + 1) % args.checkpoint_interval == 0:
            _save_checkpoint(
                output / f"jax_neural_bloch_step_{step + 1:04d}.npz",
                flat_parameters,
                {
                    **protocol, "training_step": step + 1, "record": record,
                    "optimizer_state": optimizer_state,
                },
                sampler,
            )
        if args.adaptive_cg_120_step and (step + 1) % 10 == 0:
            print(json.dumps(_ten_step_progress(
                trace, args.particles, rejected_updates,
                current_learning_rate, current_damping,
            )), flush=True)
        elif not args.adaptive_cg_120_step and (
            (step + 1) % args.log_interval == 0 or step == 0
        ):
            print(json.dumps(record), flush=True)

    _save_checkpoint(
        output / "jax_neural_bloch_final.npz",
        flat_parameters,
        {
            **protocol, "training_step": args.steps,
            "optimizer_state": {
                "next_learning_rate": current_learning_rate,
                "next_sr_damping": current_damping,
                "recovery_mode": recovery_mode,
                "recovery_stable_count": recovery_stable_count,
                "rejected_updates": rejected_updates,
            },
        },
        sampler,
    )
    np.savez_compressed(
        output / "jax_neural_bloch_samples.npz",
        positions=np.asarray(sampler.positions),
        layers=np.asarray(sampler.layers),
        logpsi=np.asarray(sampler.values),
    )
    summary = {
        "method": "continuum many-body neural Bloch",
        "protocol": protocol,
        "burn_in_seconds": burn_seconds,
        "training_seconds": time.perf_counter() - start_time,
        "rejected_updates": rejected_updates,
        "validation_during_training": False,
        "trace": trace,
    }
    (output / "jax_neural_bloch_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--particles", type=int, default=6)
    result.add_argument("--resume", type=Path)
    result.add_argument("--width", type=int, default=32)
    result.add_argument("--message-passing-steps", type=int, default=2)
    result.add_argument("--determinants", type=int, default=1)
    result.add_argument(
        "--c3-irrep", type=int, choices=(0, 1, 2), default=None,
        help="exact C3 character sector; benchmark ED gives 0 for nu=1/3 and 2 for nu=2/3",
    )
    result.add_argument(
        "--fixed-gamma-no-m", action="store_true",
        help=(
            "remove the trainable momentum-mixing matrix M and select the "
            "lowest one-body orbital set whose total mesh momentum is Gamma"
        ),
    )
    result.add_argument(
        "--c3-qns", action="store_true",
        help=("internally C3-equivariant no-M Gamma ansatz with a "
              "Bloch-sewing-weighted three-determinant orbit sum"),
    )
    result.add_argument(
        "--outer-c3-projector", action="store_true",
        help=("keep the raw periodic no-M QNS unrestricted internally and apply "
              "P_m to the complete wavefunction; requires --fixed-gamma-no-m "
              "and --c3-irrep"),
    )
    result.add_argument("--samples", type=int, default=4128)
    result.add_argument("--steps", type=int, default=1000)
    result.add_argument("--learning-rate", type=float, default=2.0e-3)
    result.add_argument("--burn-sweeps", type=int, default=300)
    result.add_argument("--sweeps-per-step", type=int, default=2)
    result.add_argument("--proposal-scale", type=float, default=0.06)
    result.add_argument("--wavefunction-batch", type=int, default=258)
    result.add_argument("--local-energy-batch", type=int, default=32)
    result.add_argument("--sr-chunk", type=int, default=258)
    result.add_argument("--sr-damping", type=float, default=1.0e-2)
    result.add_argument("--cg-iterations", type=int, default=10)
    result.add_argument("--cg-tolerance", type=float, default=1.0e-3)
    result.add_argument("--cg-min-iterations", type=int, default=20)
    result.add_argument("--cg-max-iterations", type=int, default=60)
    result.add_argument("--cg-true-tolerance", type=float, default=0.02)
    result.add_argument("--cg-acceptance-tolerance", type=float, default=0.05)
    result.add_argument("--recovery-damping", type=float, default=0.03)
    result.add_argument("--recovery-learning-rate", type=float, default=1.0e-3)
    result.add_argument("--recovery-stable-steps", type=int, default=2)
    result.add_argument("--checkpoint-interval", type=int, default=25)
    result.add_argument("--log-interval", type=int, default=5)
    result.add_argument("--seed", type=int, default=83)
    result.add_argument(
        "--adaptive-cg-120-step",
        action="store_true",
        help=(
            "fresh 120-update run with ordinary CG(20--60), explicit true-"
            "residual stopping, rejected updates, and recovery damping/lr"
        ),
    )
    result.add_argument(
        "--exploratory-100-step",
        action="store_true",
        help=(
            "allow exactly 100 updates while retaining 4128 samples, lr=2e-3, "
            "LeCunNormal, JAX natural gradient, and the production architecture"
        ),
    )
    result.add_argument(
        "--exploratory-additional-100-step",
        action="store_true",
        help=(
            "resume an exploratory step-100 checkpoint and continue it to step 200 "
            "while retaining the paper's other reported optimization settings"
        ),
    )
    result.add_argument(
        "--checkpoint-additional-100-step",
        action="store_true",
        help=(
            "continue any saved production checkpoint for exactly 100 updates "
            "while retaining all other protocol settings"
        ),
    )
    result.add_argument(
        "--smoke-test",
        action="store_true",
        help="permit non-paper sample/step counts for implementation tests only",
    )
    return result


def main():
    summary = run(parser().parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
