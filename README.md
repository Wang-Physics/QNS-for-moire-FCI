# QNS for moire FCI

Version 4.2 (single-matrix momentum projection and direct plane-wave observable revision).

This project benchmarks the continuum model and multiband exact diagonalization (ED) of twisted MoTe2 against Luo, Zaklama, and Fu, [arXiv:2503.13585v3](https://arxiv.org/abs/2503.13585), then tests their continuous-coordinate Neural-Bloch variational ansatz at fillings `nu=1/3` and `nu=2/3`.

The current report is [`result/AI_for_Physics.pdf`](result/AI_for_Physics.pdf). Part I contains the ED benchmark; Part II contains the current Neural-Bloch calculation.

## Current numerical result

Version 4 replaces the explicit momentum-combination mixer by one dense
trainable determinant matrix `M`. The exact finite-translation projector is
applied to the complete generalized determinant, so its Cauchy--Binet
expansion contains all and only total-`Gamma` minors. Version 4.2 also corrects
the post-training observables. `n(k)` is the direct plane-wave Fourier
transform of the one-body density matrix at the 27 displayed first-BZ
momenta, summed over the two layers with only `G=0`. `S(q)` is the direct
Fourier transform of the static density--density correlation at those same
physical vectors. Neither observable folds or sums reciprocal images;
Bloch-state projection is used only for bare-band weights.

The nine-cell jobs use width 32 and the 27-cell jobs width 64, with 4,128
persistent walkers and 120 natural-gradient updates. The three outer-`C3`
branches start from the same seed, use 20 shared-mixture-sampling updates, and
then continue with independent parameters and walkers for 100 updates.
Energies are final-ten means of correlated training chains; RMS is a
tail-stability diagnostic rather than an independent uncertainty.

| cells | filling | lowest outer branch | final-ten `E/N_e` (meV) | ED reference |
|---:|---|---|---:|---:|
| 9 | `1/3` | `P0 P_Gamma[M]` | `-37.3215 (0.0244)` | five-band: `-37.39323` |
| 9 | `2/3` | `P2 P_Gamma[M]` | `-52.9124 (0.0616)` | five-band: `-52.72554` |
| 27 | `1/3` | `P0 P_Gamma[M]` | `-47.5342 (0.0347)` | one-band: `-47.59546` |
| 27 | `2/3` | `P1 P_Gamma[M]` | `-60.4155 (0.0391)` | one-band: `-57.45573` |

Figures 6 and 7 summarize the nine- and 27-cell energies, all three outer
branches, bare-band weights and timing. Figures 8--10 compare the 27-cell
direct `G=0` plane-wave `n(k)`, full `S(q)` and folded density. The stored
Fourier vectors are checked point by point against the dots drawn in the
hexagonal BZ, including the five boundary/corner indices that previously
used a different representative by one reciprocal lattice vector.
## Repository layout

- `src/continuum.py`, `src/multiband_ed.py`, `src/sparse_ed.py`: continuum bands and projected ED.
- `src/jax_neural_bloch.py`: double-precision JAX wavefunction and continuum local energy.
- `src/run_jax_neural_bloch.py`: matrix-free natural gradient with ordinary CG.
- `src/neural_bloch_diagnostics.py`: nine-cell complete-ratio 1-RDM and observables.
- `src/qns27.py`, `src/qns27_diagnostics.py`: 27-cell geometry, projected training, and observables.
- `src/assemble_v42_release_data.py`: compact v4.2 figure-data assembly and checksums.
- `src/make_v42_figures.py`: final Figs. 6–11.
- `native/`: C++17 sparse-ED and observable kernels, compiled automatically when needed.
- `result/release_data/` and selected `result/data/` subtrees: compact publication-facing inputs.
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

## Reproducing the v4 runs

The v4 launchers preserve the exact local settings used for the report:

```bash
python scripts/launch_v4_9cell.py
python scripts/launch_v4_outer_c3_9cell_queue.py
python scripts/launch_v4_qns27_queue.py
python scripts/update_v4_27_observables.py
```

The jobs are GPU calculations and write raw walkers/checkpoints below
`result/data/`; those large files are intentionally not versioned. The compact
training traces and diagnostic arrays needed to inspect the published figures
are included.

## Complete-wavefunction outer C3 projection

The `outer_c3_projector` applies `P_m=(1/3) sum_a omega^(-ma) C3^a` to the
complete translation-projected determinant. The physical `C3` action includes
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

Figs. 6–11 can be regenerated from the compact published inputs:

```bash
python -m src.make_v42_figures
```

Build the report with:

```bash
cd result
tectonic AI_for_Physics.tex
```

## Data policy

Git tracks source, tests, compact ED fixtures, release traces/diagnostics, vector figures, and the final PDF. It excludes virtual environments, downloaded papers, native shared libraries, raw MCMC ensembles, checkpoints, and large many-body Berry-curvature eigenvector caches. See [`result/README.md`](result/README.md) for details.
