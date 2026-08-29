#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${PROJECT_DIR}/result/data/outer_c3_projector_nu1of3_120"
mkdir -p "${RUN_ROOT}"

sectors=(0 1 2)
cores=("0-19" "20-39" "40-59")
pids=()

for slot in "${!sectors[@]}"; do
  sector="${sectors[$slot]}"
  sector_dir="${RUN_ROOT}/m${sector}"
  mkdir -p "${sector_dir}"
  (
    cd "${PROJECT_DIR}"
    exec taskset -c "${cores[$slot]}" env \
      OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 OPENBLAS_NUM_THREADS=20 \
      .venv/bin/python -m src.run_jax_neural_bloch \
      --output-dir "${sector_dir}" \
      --particles 3 \
      --fixed-gamma-no-m \
      --outer-c3-projector \
      --c3-irrep "${sector}" \
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
  ) >"${sector_dir}/run.log" 2>&1 &
  pids+=("$!")
  printf 'm=%s pid=%s cores=%s\n' "${sector}" "$!" "${cores[$slot]}" | tee "${sector_dir}/pid.txt"
done

printf '%s\n' "${pids[@]}" >"${RUN_ROOT}/worker_pids.txt"
status=0
for slot in "${!pids[@]}"; do
  if wait "${pids[$slot]}"; then
    printf 'completed m=%s pid=%s\n' "${sectors[$slot]}" "${pids[$slot]}"
  else
    code=$?
    printf 'failed m=%s pid=%s exit=%s\n' "${sectors[$slot]}" "${pids[$slot]}" "${code}"
    status=1
  fi
done
exit "${status}"
