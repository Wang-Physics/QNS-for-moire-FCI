#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${PROJECT_DIR}/result/data/sector_mixer_v3/no_m"
mkdir -p "${RUN_ROOT}"
fillings=(nu1of3 nu2of3)
particles=(3 6)
cores=(0-31 32-63)
status=0
for slot in 0 1; do
  run_dir="${RUN_ROOT}/${fillings[$slot]}"
  mkdir -p "${run_dir}"
  printf 'starting %s cores=%s\n' "${fillings[$slot]}" "${cores[$slot]}"
  if taskset -c "${cores[$slot]}" env \
    OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 \
    .venv/bin/python -m src.run_jax_neural_bloch \
    --output-dir "${run_dir}" \
    --particles "${particles[$slot]}" \
    --fixed-gamma-no-m \
    --samples 4128 --steps 120 --learning-rate 2e-3 \
    --burn-sweeps 300 --sweeps-per-step 2 --proposal-scale 0.06 \
    --wavefunction-batch 258 --local-energy-batch 32 --sr-chunk 258 \
    --sr-damping 1e-2 --cg-min-iterations 20 --cg-max-iterations 60 \
    --cg-true-tolerance 0.02 --cg-acceptance-tolerance 0.05 \
    --recovery-damping 0.03 --recovery-learning-rate 1e-3 \
    --recovery-stable-steps 2 --checkpoint-interval 10 --log-interval 10 \
    --seed "$((83 + slot * 1000))" --adaptive-cg-120-step \
    >"${run_dir}/run.log" 2>&1; then
    printf 'completed %s\n' "${fillings[$slot]}"
  else
    printf 'failed %s\n' "${fillings[$slot]}"
    status=1
    break
  fi
done
exit "${status}"
