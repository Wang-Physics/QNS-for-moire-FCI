"""Stochastic selected-minor support updates for the 27-cell QNS runs."""

from __future__ import annotations

import hashlib
import time

import jax
import jax.numpy as jnp
import numpy as np

from .jax_neural_bloch import active_minor_basis_values
from .qns27 import gamma_indices, inputs


class ActiveSupportController:
    """Keep a fixed sparse wavefunction between periodic support updates.

    Candidate minors are Monte-Carlo proposed from an equal mixture of a
    uniform Gamma-sector distribution and a low-one-body-energy distribution.
    They are screened by their current VMC energy-gradient magnitude.  This is
    a stochastic selected-CI variational update, not a noisy estimator of the
    coherent minor sum inside Metropolis or local-energy evaluations.
    """

    def __init__(
        self,
        spec,
        constants,
        seed,
        candidate_pool=4096,
        replace_fraction=0.25,
        score_walkers=32,
        score_chunk=32,
        proposal_temperature_mev=3.0,
    ):
        if spec.cells != 27 or not spec.fixed_gamma_no_m:
            raise ValueError("active support is only defined for 27-cell Gamma QNS")
        self.spec = spec
        self.constants = constants
        self.all_indices = gamma_indices(spec.n_particles)
        _, model_inputs = inputs()
        self.one_body_sums = np.asarray(
            model_inputs["one_body_energies"][self.all_indices].sum(axis=1)
        )
        weights = np.exp(
            -(self.one_body_sums - self.one_body_sums.min())
            / proposal_temperature_mev
        )
        weights /= weights.sum()
        self.proposal_probability = 0.5 / len(weights) + 0.5 * weights
        self.rng = np.random.default_rng(seed)
        self.candidate_pool = int(candidate_pool)
        self.replace = max(1, int(round(spec.active_gamma_minors * replace_fraction)))
        self.score_walkers = int(score_walkers)
        self.score_chunk = int(score_chunk)
        self._basis = jax.jit(
            lambda p, x, l, ix: active_minor_basis_values(
                p, x, l, constants, spec, ix, self.score_chunk
            )
        )

    @staticmethod
    def support_hash(indices):
        return hashlib.sha256(np.asarray(indices, dtype=np.int32).tobytes()).hexdigest()

    def propose(self, support):
        active = {tuple(row) for row in np.asarray(support)}
        selected = []
        selected_keys = set()
        while len(selected) < self.candidate_pool:
            need = self.candidate_pool - len(selected)
            draws = self.rng.choice(
                len(self.all_indices), size=min(len(self.all_indices), 2 * need),
                replace=False, p=self.proposal_probability,
            )
            for row in self.all_indices[draws]:
                key = tuple(row)
                if key not in active and key not in selected_keys:
                    selected.append(row)
                    selected_keys.add(key)
                    if len(selected) == self.candidate_pool:
                        break
        return np.asarray(selected, dtype=np.int32)

    def update(
        self, parameters, support, positions, layers, energies, step, weights=None
    ):
        started = time.perf_counter()
        support = np.asarray(support, dtype=np.int32)
        candidates = self.propose(support)
        count = min(self.score_walkers, len(positions))
        sample_ids = np.linspace(0, len(positions) - 1, count, dtype=int)
        x = jnp.asarray(np.asarray(positions)[sample_ids])
        l = jnp.asarray(np.asarray(layers)[sample_ids])
        energy = np.asarray(energies)[sample_ids].real
        if weights is None:
            sample_weights = np.full(count, 1.0 / count)
        else:
            sample_weights = np.asarray(weights, dtype=float)[sample_ids]
            sample_weights /= sample_weights.sum()
        centered = energy - np.sum(sample_weights * energy)
        combined = np.concatenate([support, candidates])
        basis = np.asarray(self._basis(
            parameters, x, l, jnp.asarray(combined, dtype=jnp.int32)
        ))
        mixing = np.asarray(
            parameters["sector_mixing_real"][0]
            + 1j * parameters["sector_mixing_imag"][0]
        )
        denominator = basis[:, :len(support)] @ mixing
        safe = np.abs(denominator) > np.finfo(float).tiny
        ratios = np.zeros_like(basis)
        ratios[safe] = basis[safe] / denominator[safe, None]
        gradient = np.sum(
            sample_weights[:, None] * centered[:, None] * np.conj(ratios),
            axis=0,
        )
        contribution = np.abs(mixing) * np.sqrt(
            np.sum(
                sample_weights[:, None]
                * np.abs(ratios[:, :len(support)]) ** 2,
                axis=0,
            )
        )
        retain_count = len(support) - self.replace
        retained_slots = np.argsort(contribution, kind="stable")[-retain_count:]
        candidate_score = np.abs(gradient[len(support):])
        accepted_candidates = np.argsort(
            candidate_score, kind="stable"
        )[-self.replace:]
        new_support = np.concatenate([
            support[retained_slots], candidates[accepted_candidates]
        ])
        retained_mixing = mixing[retained_slots]
        reference = float(np.median(np.abs(retained_mixing)))
        if not np.isfinite(reference) or reference == 0.0:
            reference = float(np.sqrt(np.mean(np.abs(mixing) ** 2)))
        new_gradient = gradient[len(support):][accepted_candidates]
        phase = np.where(
            np.abs(new_gradient) > 0,
            new_gradient / np.maximum(np.abs(new_gradient), np.finfo(float).tiny),
            1.0 + 0.0j,
        )
        new_mixing = -0.05 * reference * phase
        updated_mixing = np.concatenate([retained_mixing, new_mixing])
        updated = dict(parameters)
        updated["sector_mixing_real"] = jnp.asarray(updated_mixing.real[None])
        updated["sector_mixing_imag"] = jnp.asarray(updated_mixing.imag[None])
        record = {
            "step": int(step),
            "old_support_sha256": self.support_hash(support),
            "new_support_sha256": self.support_hash(new_support),
            "support_size": int(len(new_support)),
            "retained": int(retain_count),
            "replaced": int(self.replace),
            "candidate_pool": int(len(candidates)),
            "score_walkers": int(count),
            "candidate_score_min_selected": float(candidate_score[accepted_candidates].min()),
            "candidate_score_max_selected": float(candidate_score[accepted_candidates].max()),
            "retained_contribution_min": float(contribution[retained_slots].min()),
            "seconds": float(time.perf_counter() - started),
            "support_indices": new_support.tolist(),
        }
        return updated, new_support, record
