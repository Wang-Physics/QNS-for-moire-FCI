# QNS for moire FCI

Version 2.0.0.

This project benchmarks the continuum model and multiband exact diagonalization (ED) of twisted MoTe2 against Luo, Zaklama, and Fu, [arXiv:2503.13585v3](https://arxiv.org/abs/2503.13585), then tests their continuous-coordinate Neural-Bloch variational ansatz at fillings `nu=1/3` and `nu=2/3`.

The current report is [`result/AI_for_Physics.pdf`](result/AI_for_Physics.pdf). Part I contains the ED benchmark; Part II contains the current Neural-Bloch calculation.

## Current numerical result

The controlled QNS comparison uses a `3 x 3` moire cluster, one generalized Slater determinant, width 32, two unshared message-passing iterations, 4,128 persistent walkers, and explicit true-CG-residual acceptance. Energies below are final-ten means of the equilibrated persistent training chain; the parenthesized RMS measures tail stability and is not an independent error bar.

| filling | state | final-ten training `E/N_e` (meV) | five-band ED (meV) | difference (meV) |
|---|---|---:|---:|---:|
| `1/3` | no `M`, total momentum `Gamma` | `-36.7708 (0.0592)` | `-37.39323` | `+0.6224` |
| `1/3` | internal C3-QNS, `m=0` | `-37.1749 (0.0287)` | `-37.39323` | `+0.2183` |
| `1/3` | outer `P0` (lowest outer sector) | `-37.1660 (0.0532)` | `-37.39323` | `+0.2272` |
| `2/3` | no `M`, total momentum `Gamma` | `-52.4147 (0.0578)` | `-52.72554` | `+0.3108` |
| `2/3` | outer `P2` (lowest outer sector) | `-52.7891 (0.0581)` | `-52.72554` | `-0.0636` |

All three outer characters remain available in the compact training traces. Fig. 6 shows only the lowest-training outer sector at each filling; a separate compact chart reports the full `m=0,1,2` tail-energy splitting and rejected-update counts. The complete-ratio 1-RDM estimator jointly batches all auxiliary replacements while recomputing the full projected wavefunction for every replacement.
## Repository layout

- `src/continuum.py`, `src/multiband_ed.py`, `src/sparse_ed.py`: continuum bands and projected ED.
- `src/jax_neural_bloch.py`: double-precision JAX wavefunction and continuum local energy.
- `src/run_jax_neural_bloch.py`: matrix-free natural gradient with ordinary CG.
- `src/neural_bloch_diagnostics.py`: complete-ratio 1-RDM, band projection, density, and two-body `S(q)`.
- `src/make_adaptive120_figures.py`, `src/make_adaptive120_observables.py`: final Figs. 6–7.
- `native/`: C++17 sparse-ED and observable kernels, compiled automatically when needed.
- `result/release_data/`: compact, publication-facing traces and diagnostics for Figs. 6–7.
- `result/figures/`: vector and raster report figures.
- `tests/`: unit and physics-regression tests.

## Installation

Python 3.12 or newer, a C++17 compiler, and a LaTeX installation are recommended.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

JAX is configured for 64-bit calculations by the production code. GPU users should install the platform-specific JAX build before installing the remaining dependencies.

## Tests

```bash
python -m unittest discover -s tests -v
```

Small ED fixtures required by regression tests are versioned under `result/data/`. Large walkers, checkpoints, many-body twist eigenvectors, and raw training directories are intentionally ignored.

## Reproducing the controlled Neural-Bloch runs

The four jobs differ only in particle number and the presence of the fixed-`Gamma`, no-`M` flag. A representative full-`M`, `nu=1/3` job is:

```bash
python -m src.run_jax_neural_bloch \
  --output-dir result/data/neural_bloch_adaptive_cg120_full_m_nu1of3 \
  --particles 3 --width 32 --message-passing-steps 2 --determinants 1 \
  --samples 4128 --steps 120 --learning-rate 2e-3 \
  --burn-sweeps 300 --sweeps-per-step 2 \
  --wavefunction-batch 258 --local-energy-batch 32 --sr-chunk 258 \
  --sr-damping 1e-2 --cg-min-iterations 20 --cg-max-iterations 60 \
  --cg-true-tolerance 0.02 --cg-acceptance-tolerance 0.05 \
  --recovery-damping 0.03 --recovery-learning-rate 1e-3 \
  --checkpoint-interval 10 --log-interval 10 --adaptive-cg-120-step
```

Use `--particles 6` for `nu=2/3`; add `--fixed-gamma-no-m` for the no-`M` state. These are expensive CPU calculations. The paper's reported production setting is 1,000 optimization steps; the public benchmark here deliberately reports the completed 120-update audit.

## C3-QNS extension

`C3-QNS` keeps the no-`M`, fixed-total-`Gamma` construction but makes the
backflow and orbital transformation internally equivariant. A Reynolds
average evaluates the same message-passing network on `X`, `C3 X`, and
`C3^2 X`; vector outputs are rotated back before averaging. The final wave
function is a tied sum of the three C3-related Slater minors. Their relative
coefficients are fixed by the one-band Bloch sewing phases and the requested
many-body C3 character.

The continuum Bloch sewing relation is evaluated at the dressed coordinate
`r + delta_r`, while the physical layer gauge acts at `r`. The implementation
therefore includes the analytic per-column compensation
`exp(i b_l dot delta_r)`, with `(R^T-I)b_l=a_l`. Omitting this factor gives an
order-one character error even when the raw displacement is equivariant.
At the frozen step-116 checkpoint, 512 independent configurations give a maximum character residual `7.12e-5` (95th percentile `1.77e-5`). The Luo-Fu Fourier input encoding remains unchanged; it guarantees
periodicity and primitive translation symmetry, but does not alone guarantee
C3 equivariance.

The frozen experiment in this release uses `nu=1/3` only:

```bash
python -m src.run_jax_neural_bloch \
  --output-dir result/data/c3_qns_adaptive120_nu1of3 \
  --particles 3 --fixed-gamma-no-m --c3-irrep 0 --c3-qns \
  --samples 4128 --steps 120 --burn-sweeps 300 --sweeps-per-step 2 \
  --wavefunction-batch 258 --local-energy-batch 32 --sr-chunk 258 \
  --checkpoint-interval 10 --log-interval 10 --adaptive-cg-120-step
```

The implementation and JAX/PyTorch parity checks are in `tests/test_c3_qns.py`. The stopped run is evaluated from `jax_neural_bloch_step_0116.npz`; the report does not label it as a completed 120-update run.

## Complete-wavefunction outer C3 projection

The new `outer_c3_projector` leaves the periodic no-`M` Luo--Fu network
internally unrestricted and applies `P_m=(1/3) sum_a omega^(-ma) C3^a` only
to the complete generalized determinant. The physical `C3` action includes
the continuum layer-gauge sewing factor. Projected `logpsi` is the single
source used by local energy, Metropolis ratios, and SR logarithmic derivatives.

`tests/test_outer_c3_projector.py` checks all three characters, `P_m^2=P_m`,
`P_m P_n=0`, nonzero norms in all sectors, projected local-energy/Metropolis/SR
paths, and exact recovery of no-`M` when the projector is disabled. Production
preflight values are stored in `result/data/outer_c3_nu1of3_preflight.json`.
The three independent `nu=1/3` sector runs are launched with:

```bash
nohup setsid scripts/run_outer_c3_nu1of3.sh \
  > result/data/outer_c3_projector_nu1of3_120/launcher.log 2>&1 &
```

Each sector uses 4,128 walkers, 120 adaptive-CG updates, and a separate CPU
affinity and checkpoint directory under `outer_c3_projector_nu1of3_120/m0,m1,m2`.
The analogous `nu=2/3` production launchers are
`scripts/run_outer_c3_nu2of3_p0.sh` and
`scripts/run_outer_c3_nu2of3_sector.sh`; all three sectors are stored under
`outer_c3_projector_nu2of3_120/p0,p1,p2`.

## Redrawing the publication figures

Figs. 6–7 can be regenerated without the raw checkpoints:

```bash
python -m src.make_adaptive120_figures
python -m src.make_adaptive120_observables
```

Build the report with:

```bash
cd result
pdflatex -interaction=nonstopmode -halt-on-error AI_for_Physics.tex
pdflatex -interaction=nonstopmode -halt-on-error AI_for_Physics.tex
```

## Data policy

Git tracks source, tests, compact ED fixtures, release traces/diagnostics, vector figures, and the final PDF. It excludes virtual environments, downloaded papers, native shared libraries, raw MCMC ensembles, checkpoints, and large many-body Berry-curvature eigenvector caches. See [`result/README.md`](result/README.md) for details.
