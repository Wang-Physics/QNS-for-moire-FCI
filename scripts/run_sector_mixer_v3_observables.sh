#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

run_one() {
  local source_dir="$1"
  local release_stem="$2"
  local seed="$3"
  local work_dir="result/data/sector_mixer_v3/diagnostics/${release_stem}"
  local checkpoint="${source_dir}/jax_neural_bloch_final.npz"
  local samples="${source_dir}/jax_neural_bloch_samples.npz"

  if [[ ! -f "$checkpoint" || ! -f "$samples" ]]; then
    echo "waiting for completed ensemble: ${source_dir}" >&2
    return 1
  fi

  mkdir -p "$work_dir" result/release_data/diagnostics
  .venv/bin/python -m src.export_jax_neural_bloch \
    --checkpoint "$checkpoint" \
    --output "${work_dir}/state.pt" \
    --seed "$seed"
  .venv/bin/python -m src.neural_bloch_diagnostics \
    --checkpoint "${work_dir}/state.pt" \
    --samples "$samples" \
    --output-dir "$work_dir" \
    --samples-count 4128 \
    --auxiliary-draws 16 \
    --error-blocks 8 \
    --wavefunction-batch 16 \
    --threads 8 \
    --seed "$seed" > "${work_dir}/observables.log" 2>&1
  cp "${work_dir}/neural_bloch_diagnostics.npz" \
    "result/release_data/diagnostics/${release_stem}.npz"
  cp "${work_dir}/neural_bloch_diagnostics.json" \
    "result/release_data/diagnostics/${release_stem}.json"
}

publish_plain_trace() {
  cp "$1/jax_neural_bloch_trace.json" "result/release_data/traces/$2.json"
}

publish_outer_trace() {
  local shared_dir="$1"
  local branch_dir="$2"
  local release_stem="$3"
  local target="result/release_data/traces/${release_stem}.json"
  local temporary="${target}.tmp"
  jq -s '
    (.[0] | map(. + {stage: "shared_raw", stage_step: .step}))
    +
    (.[1] | map(. + {
      stage: "projected",
      stage_step: .step,
      step: (.step + 20)
    }))
  ' "$shared_dir/jax_neural_bloch_trace.json" \
    "$branch_dir/jax_neural_bloch_trace.json" > "$temporary"
  mv "$temporary" "$target"
}

case "${1:-all-ready}" in
  all-ready)
    run_one result/data/sector_mixer_v3/no_m/nu1of3 nu1of3_no_m_gamma 3301
    run_one result/data/sector_mixer_v3/no_m/nu2of3 nu2of3_no_m_gamma 3302
    publish_plain_trace result/data/sector_mixer_v3/no_m/nu1of3 nu1of3_no_m_gamma
    publish_plain_trace result/data/sector_mixer_v3/no_m/nu2of3 nu2of3_no_m_gamma
    for filling in nu1of3 nu2of3; do
      for m in 0 1 2; do
        source="result/data/sector_mixer_v3/outer_c3/${filling}/m${m}"
        [[ -f "${source}/jax_neural_bloch_final.npz" ]] || continue
        if [[ "$filling" == nu1of3 ]]; then seed=$((3310 + m)); else seed=$((3320 + m)); fi
        run_one "$source" "${filling}_outer_c3_p${m}" "$seed"
        publish_outer_trace \
          "result/data/sector_mixer_v3/outer_c3/${filling}/shared20" \
          "$source" "${filling}_outer_c3_p${m}"
      done
    done
    ;;
  one)
    [[ $# -eq 4 ]] || { echo "usage: $0 one SOURCE_DIR RELEASE_STEM SEED" >&2; exit 2; }
    run_one "$2" "$3" "$4"
    ;;
  publish-traces)
    mkdir -p result/release_data/traces
    publish_plain_trace result/data/sector_mixer_v3/no_m/nu1of3 nu1of3_no_m_gamma
    publish_plain_trace result/data/sector_mixer_v3/no_m/nu2of3 nu2of3_no_m_gamma
    for filling in nu1of3 nu2of3; do
      for m in 0 1 2; do
        publish_outer_trace \
          "result/data/sector_mixer_v3/outer_c3/${filling}/shared20" \
          "result/data/sector_mixer_v3/outer_c3/${filling}/m${m}" \
          "${filling}_outer_c3_p${m}"
      done
    done
    ;;
  *)
    echo "usage: $0 [all-ready | publish-traces | one SOURCE_DIR RELEASE_STEM SEED]" >&2
    exit 2
    ;;
esac
