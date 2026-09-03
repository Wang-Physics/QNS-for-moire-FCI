#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${PROJECT_DIR}/result/data/sector_mixer_v3/outer_c3"
mkdir -p "${RUN_ROOT}"
fillings=(nu1of3 nu2of3)
particles=(3 6)
base_seeds=(2083 3083)
cores=(0-19 20-39 40-59)

run_common() {
  local out="$1" particles="$2" seed="$3" affinity="$4"
  shift 4
  taskset -c "$affinity" .venv/bin/python -m src.run_jax_neural_bloch \
    --output-dir "$out" --particles "$particles" --fixed-gamma-no-m \
    --samples 4128 --learning-rate 2e-3 \
    --burn-sweeps 300 --sweeps-per-step 2 --proposal-scale 0.06 \
    --wavefunction-batch 258 --local-energy-batch 32 --sr-chunk 258 \
    --sr-damping 1e-2 --cg-min-iterations 20 --cg-max-iterations 60 \
    --cg-true-tolerance 0.02 --cg-acceptance-tolerance 0.05 \
    --recovery-damping 0.03 --recovery-learning-rate 1e-3 \
    --recovery-stable-steps 2 --checkpoint-interval 10 --log-interval 10 \
    --seed "$seed" "$@"
}

for filling_slot in 0 1; do
  filling="${fillings[$filling_slot]}"
  particle_count="${particles[$filling_slot]}"
  base_seed="${base_seeds[$filling_slot]}"
  filling_root="${RUN_ROOT}/${filling}"
  shared_dir="${filling_root}/shared20"
  mkdir -p "$shared_dir"
  printf 'starting %s outer-only shared raw pretraining (20 steps)\n' "$filling"
  OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 \
    run_common "$shared_dir" "$particle_count" "$base_seed" 0-31 \
    --steps 20 --outer-shared-pretrain-20 >"${shared_dir}/run.log" 2>&1

  shared_checkpoint="${shared_dir}/jax_neural_bloch_final.npz"
  test -f "$shared_checkpoint"
  pids=()
  for m in 0 1 2; do
    branch_dir="${filling_root}/m${m}"
    mkdir -p "$branch_dir"
    printf 'starting %s m=%s from shared20 parameters; fresh walkers\n' "$filling" "$m"
    (
      export OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 OPENBLAS_NUM_THREADS=20
      run_common \
        "$branch_dir" "$particle_count" "$((base_seed + 101 + m))" "${cores[$m]}" \
        --steps 100 --outer-branch-100 \
        --outer-c3-projector --c3-irrep "$m" \
        --parameter-init "$shared_checkpoint"
    ) >"${branch_dir}/run.log" 2>&1 &
    pids+=("$!")
    printf '%s m=%s pid=%s cores=%s\n' "$filling" "$m" "$!" "${cores[$m]}" \
      >"${branch_dir}/pid.txt"
  done
  status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  if test "$status" -ne 0; then
    printf 'one or more %s outer branches failed\n' "$filling"
    exit 1
  fi
  printf 'completed all %s outer branches\n' "$filling"
done
