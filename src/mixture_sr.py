"""Normalized multi-state reweighting and weighted complex-score SR.

Pool samples from equal-size MCMC strata targeting each |psi_m|^2. Unknown
relative normalizers are estimated by the MBAR bridge equations (Shirts and
Chodera, JCP 129, 124105 (2008)). This is a finite-sample estimator, not exact
normalization and not a claim that short persistent chains are equilibrated.
"""
import numpy as np
from scipy.special import logsumexp
import jax.numpy as jnp

from .run_jax_neural_bloch import MatrixFreeNaturalGradient


def mixture_weights(log_density, counts, tolerance=1e-10, max_iterations=10000):
    """Return per-sector normalized importance weights on a stratified pool.

    log_density[k,n] = log |psi_k(x_n)|^2; all samples are evaluated in all
    sectors. Counts refer to proposal strata, never to optimized sector norms.
    """
    logf = np.asarray(log_density, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if (logf.ndim != 2 or counts.shape != (logf.shape[0],)
            or not np.isfinite(logf).all() or not np.isfinite(counts).all()
            or np.any(counts <= 0) or counts.sum() != logf.shape[1]):
        raise ValueError("invalid mixture density or stratum counts")
    # Remove arbitrary amplitudes for numerical stability; restore them in logZ.
    offsets = np.max(logf, axis=1)
    reduced = logf - offsets[:, None]
    z = np.zeros(len(counts))
    for iteration in range(max_iterations):
        denominator = logsumexp(
            np.log(counts)[:, None] + reduced - z[:, None], axis=0
        )
        next_z = logsumexp(reduced - denominator, axis=1)
        next_z -= next_z[0]
        error = float(np.max(np.abs(next_z - z)))
        z = next_z
        if error < tolerance:
            break
    else:
        raise RuntimeError("mixture normalizers failed to converge; no SR update")
    denominator = logsumexp(np.log(counts)[:, None] + reduced - z[:, None], axis=0)
    weights = np.exp(reduced - z[:, None] - denominator)
    normalization_error = float(np.max(np.abs(weights.sum(axis=1) - 1)))
    weights /= weights.sum(axis=1, keepdims=True)
    # Symmetric form of the MBAR overlap matrix, with identical eigenvalues.
    gram = weights @ weights.T
    overlap = np.sqrt(counts[:, None] * counts[None, :]) * gram
    eigenvalues = np.linalg.eigvalsh(overlap)
    gap = float(1 - eigenvalues[-2]) if len(counts) > 1 else 1.0
    if gap < 1e-6:
        raise RuntimeError("disconnected mixture overlap; relative norms unreliable")
    logz = z + offsets
    logz -= logz[0]
    return weights, dict(
        log_relative_norms=logz.tolist(), iterations=iteration + 1,
        fixed_point_error=error, normalization_error=normalization_error,
        importance_ess=(1 / np.sum(weights**2, axis=1)).tolist(),
        overlap_eigenvalues=eigenvalues.tolist(), overlap_spectral_gap=gap,
        ess_note="weight concentration only, not autocorrelation-corrected ESS",
    )


class WeightedNaturalGradient(MatrixFreeNaturalGradient):
    """Sector SR with normalized importance weights and weighted centering.

    Weights and sampled local energies are held fixed for score derivatives.
    The sector objective and metric both have the common factor 1/3 removed;
    damping retains the original per-sector convention.
    """

    def set_weights(self, weights):
        weights = np.asarray(weights, dtype=float)
        if (weights.ndim != 1 or not np.isfinite(weights).all()
                or np.any(weights < 0) or weights.sum() <= 0):
            raise ValueError("invalid SR weights")
        self.weights = jnp.asarray(weights / weights.sum())

    def force(self, flat_parameters, positions, layers, energies):
        if len(self.weights) != len(positions):
            raise ValueError("SR weights/sample count mismatch")
        centered = energies - jnp.sum(self.weights * energies)
        # Reuse the existing force kernel, whose last argument divides by N.
        weighted = centered * self.weights
        result = jnp.zeros_like(flat_parameters)
        for i, (x, l) in enumerate(self._chunks(positions, layers)):
            begin = i * self.chunk_size
            result += self._force_chunk(
                flat_parameters, x, l, weighted[begin:begin+self.chunk_size],
                1.0, self.support_indices,
            )
        return result

    def product(self, flat_parameters, vector, positions, layers):
        if len(self.weights) != len(positions):
            raise ValueError("SR weights/sample count mismatch")
        tangents = []
        real_mean, imag_mean = jnp.asarray(0.0), jnp.asarray(0.0)
        for i, (x, l) in enumerate(self._chunks(positions, layers)):
            t = self._tangent_chunk(
                flat_parameters, vector, x, l, self.support_indices
            )
            w = self.weights[i*self.chunk_size:(i+1)*self.chunk_size]
            tangents.append(t)
            real_mean += jnp.sum(w * t[:self.chunk_size])
            imag_mean += jnp.sum(w * t[self.chunk_size:])
        result = jnp.zeros_like(vector)
        for i, (t, (x, l)) in enumerate(zip(tangents, self._chunks(positions, layers))):
            w = self.weights[i*self.chunk_size:(i+1)*self.chunk_size]
            cotangent = jnp.concatenate([
                w * (t[:self.chunk_size] - real_mean),
                w * (t[self.chunk_size:] - imag_mean),
            ])
            result += self._pullback_chunk(
                flat_parameters, x, l, cotangent, self.support_indices
            )
        return result + self.damping * vector
