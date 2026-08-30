#!/usr/bin/env bash
# Launch independent fixed-state pure-neural residual rollouts persistently.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 TASK_IDS_CSV GPUS_CSV RESIDUAL_CHECKPOINT OUTPUT_ROOT" >&2
  exit 2
fi

task_ids_csv=$1
gpus_csv=$2
residual_checkpoint=$3
output_root=$4
base=/data2/lty/naive-e-slotssm-runs/oft-plus-book-local-repair-v1
root=/data/lty/code/naive-e-slotssm
python_bin="$root/.conda/envs/openvla-oft/bin/python"
runner="$root/scripts/run_oft_slotssm_unified_episode.py"
gate_checkpoint="${GATE_CHECKPOINT:-$base/gate-v7-dense-trigger-20260809/gate_best.safetensors}"

IFS=',' read -r -a task_ids <<< "$task_ids_csv"
IFS=',' read -r -a gpus <<< "$gpus_csv"
if [[ ${#task_ids[@]} -eq 0 || ${#gpus[@]} -eq 0 ]]; then
  echo "task ids and GPUs must be nonempty CSV lists" >&2
  exit 2
fi
mkdir -p "$output_root"

run_task() {
  local task_id=$1
  local gpu_id=$2
  local task_output="$output_root/task${task_id}"
  CUDA_VISIBLE_DEVICES="$gpu_id" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false \
    "$python_bin" "$runner" \
    --gpu "$gpu_id" --task-suite libero_10 --task-id "$task_id" --episode-id 0 \
    --official-normal-actions --normal-open-loop 8 \
    --max-learned-recoveries 1 --learned-recovery-max-steps 70 --recovery-confirm-steps 2 \
    --adapter-checkpoint "$base/neural-geometry-v8-aux/recovery-head-v8/adapter_best.safetensors" \
    --gate-checkpoint "$gate_checkpoint" --gate-num-classes 2 \
    --candidate-verifier-checkpoint "$base/onpolicy-gate-states-v2/candidate-verifier-v2-onpolicy-causal/macro_verifier_best.safetensors" \
    --neural-recovery-mode --disable-recovery-phase-conditioning --neural-geometry-conditioning \
    --neural-recovery-residual-checkpoint "$residual_checkpoint" \
    --neural-recovery-residual-threshold 0.5 --save-online-images \
    --output-dir "$task_output" > "$output_root/task${task_id}.log" 2>&1
}

# One worker per GPU loads at most one 7B model at a time.  Workers process
# their assigned states sequentially while separate GPUs remain parallel.
for gpu_index in "${!gpus[@]}"; do
  gpu_id=${gpus[$gpu_index]}
  (
    for index in "${!task_ids[@]}"; do
      if (( index % ${#gpus[@]} == gpu_index )); then
        run_task "${task_ids[$index]}" "$gpu_id"
      fi
    done
  ) &
done
wait
