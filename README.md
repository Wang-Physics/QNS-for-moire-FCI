# QNS for moire FCI

Version 1.0.0.

This project benchmarks the continuum model and multiband exact diagonalization (ED) of twisted MoTe2 against Luo, Zaklama, and Fu, [arXiv:2503.13585v3](https://arxiv.org/abs/2503.13585), then tests their continuous-coordinate Neural-Bloch variational ansatz at fillings `nu=1/3` and `nu=2/3`.

The current report is [`result/AI_for_Physics.pdf`](result/AI_for_Physics.pdf). Part I contains the ED benchmark; Part II contains the current Neural-Bloch calculation.

## Current numerical result

The controlled QNS comparison uses a `3 x 3` moire cluster, one generalized Slater determinant, width 32, two unshared message-passing iterations, 4,128 persistent walkers, 120 attempted natural-gradient updates, and explicit true-CG-residual acceptance.

| filling | selected state | precise validation `E/N_e` (meV) | five-band ED (meV) | difference (meV) |
|---|---|---:|---:|---:|
| `1/3` | no `M`, total momentum `Gamma` | `-36.78317(10)` | `-37.39323` | `+0.61006` |
| `2/3` | no `M`, total momentum `Gamma` | `-52.39973(17)` | `-52.72554` | `+0.32581` |

The present ansatz therefore has not established a variational energy advantage over five-band ED. The report also documents CG truncation, rejected updates, independent validation, full-wavefunction-ratio 1-RDM estimators, and the distinction between full and connected structure factors.

## Repository layout

- `src/continuum.py`, `src/multiband_ed.py`, `src/sparse_ed.py`: continuum bands and projected ED.
- `src/jax_neural_bloch.py`: double-precision JAX wavefunction and continuum local energy.
- `src/run_jax_neural_bloch.py`: matrix-free natural gradient with ordinary CG.
- `src/validate_jax_neural_bloch.py`: independent random-start validation chains.
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

After all four checkpoints exist, `src/run_post120_validation_protocol.py` performs the independent 1,024-walker selection and 4,128-walker precise validation. Validation samples never enter parameter updates.

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
