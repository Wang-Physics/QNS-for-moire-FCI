#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${PROJECT_DIR}/result/data/outer_c3_projector_nu2of3_120/p0"
mkdir -p "${RUN_DIR}"

cd "${PROJECT_DIR}"
exec taskset -c 0-31 env \
  OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 \
  .venv/bin/python -m src.run_jax_neural_bloch \
  --output-dir "${RUN_DIR}" \
  --particles 6 \
  --fixed-gamma-no-m \
  --outer-c3-projector \
  --c3-irrep 0 \
  --samples 4128 \
  --steps 120 \
  --learning-rate 2e-3 \
  --burn-sweeps 300 \
  --sweeps-per-step 2 \
  --proposal-scale 0.06 \
  --wavefunction-batch 258 \
  --local-energy-batch 32 \
  --sr-chunk 258 \
  --sr-damping 1e-2 \
  --cg-min-iterations 20 \
  --cg-max-iterations 60 \
  --cg-true-tolerance 0.02 \
  --cg-acceptance-tolerance 0.05 \
  --recovery-damping 0.03 \
  --recovery-learning-rate 1e-3 \
  --recovery-stable-steps 2 \
  --checkpoint-interval 10 \
  --log-interval 10 \
  --seed 83 \
  --adaptive-cg-120-step
