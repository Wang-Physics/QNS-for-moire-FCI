# QNS for moire FCI

Version 3.2 (same-seed shared-pool sector-mixer revision).

This project benchmarks the continuum model and multiband exact diagonalization (ED) of twisted MoTe2 against Luo, Zaklama, and Fu, [arXiv:2503.13585v3](https://arxiv.org/abs/2503.13585), then tests their continuous-coordinate Neural-Bloch variational ansatz at fillings `nu=1/3` and `nu=2/3`.

The current report is [`result/AI_for_Physics.pdf`](result/AI_for_Physics.pdf). Part I contains the ED benchmark; Part II contains the current Neural-Bloch calculation.

## Current numerical result

The controlled QNS comparison uses a `3 x 3` moire cluster, width 32, two
unshared message-passing iterations, 4,128 persistent walkers, and explicit
true-CG-residual acceptance. Version 3 adds a trainable complex determinant
mixer `M_S`: it linearly combines all 12 selected-momentum determinants whose
total momentum is `Gamma`. Version 3.2 initializes the three outer-`C3`
sectors with the same seed, applies 20 independent-parameter updates using a
shared mixture-sampled pool, then continues with 100 private-chain updates.
Energies below are final-ten means of the correlated persistent training
chain; RMS measures tail stability rather than an independent error bar.

| filling | state | final-ten training `E/N_e` (meV) | five-band ED (meV) | difference (meV) |
|---|---|---:|---:|---:|
| `1/3` | no `M` + `M_S`, `Gamma` | `-37.3037 (0.0421)` | `-37.39323` | `+0.0896` |
| `1/3` | outer `P0` + `M_S` (lowest outer sector) | `-37.3436 (0.0343)` | `-37.39323` | `+0.0497` |
| `2/3` | no `M` + `M_S`, `Gamma` | `-52.7666 (0.0569)` | `-52.72554` | `-0.0411` |
| `2/3` | outer `P2` + `M_S` (lowest outer sector) | `-52.8838 (0.0530)` | `-52.72554` | `-0.1583` |

All three outer characters remain available in the compact training traces.
Fig. 6 shows only the lowest-training outer sector at each filling; Fig. 7
reports all three optimization traces, tail-energy splitting, and separate
five-band-ED/QNS timing panels for `nu=1/3` and `nu=2/3`. The complete-ratio
1-RDM estimator jointly batches all auxiliary replacements while recomputing
the full projected wavefunction for every replacement.
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

## Complete-wavefunction outer C3 projection

The `outer_c3_projector` leaves the periodic no-`M` + `M_S` Luo--Fu network
internally unrestricted and applies `P_m=(1/3) sum_a omega^(-ma) C3^a` only
to the complete determinant mixture. The physical `C3` action includes
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
