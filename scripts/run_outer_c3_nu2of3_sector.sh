#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  printf 'usage: %s C3_SECTOR CPU_RANGE\n' "$0" >&2
  exit 2
fi

SECTOR="$1"
CORE_RANGE="$2"
if [[ ! "${SECTOR}" =~ ^[0-2]$ ]]; then
  printf 'C3_SECTOR must be 0, 1, or 2\n' >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${PROJECT_DIR}/result/data/outer_c3_projector_nu2of3_120/p${SECTOR}"
mkdir -p "${RUN_DIR}"

cd "${PROJECT_DIR}"
exec taskset -c "${CORE_RANGE}" env \
  OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
  .venv/bin/python -m src.run_jax_neural_bloch \
  --output-dir "${RUN_DIR}" \
  --particles 6 \
  --fixed-gamma-no-m \
  --outer-c3-projector \
  --c3-irrep "${SECTOR}" \
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
