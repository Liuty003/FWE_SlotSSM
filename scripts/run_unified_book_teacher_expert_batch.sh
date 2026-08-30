#!/usr/bin/env bash
# Evaluate a trained unified teacher-distilled recovery adapter on fixed states.
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 TASK_IDS_CSV GPUS_CSV ADAPTER_CHECKPOINT GATE_CHECKPOINT OUTPUT_ROOT" >&2
  exit 2
fi

task_ids_csv=$1
gpus_csv=$2
adapter_checkpoint=$3
gate_checkpoint=$4
output_root=$5
base=/data2/lty/naive-e-slotssm-runs/oft-plus-book-local-repair-v1
root=/data/lty/code/naive-e-slotssm
python="$root/.conda/envs/openvla-oft/bin/python"
runner="$root/scripts/run_oft_slotssm_unified_episode.py"
threshold=${RECOVERY_THRESHOLD:-0.5}

IFS=',' read -r -a task_ids <<< "$task_ids_csv"
IFS=',' read -r -a gpus <<< "$gpus_csv"
mkdir -p "$output_root"

for gpu_index in "${!gpus[@]}"; do
  gpu=${gpus[$gpu_index]}
  (
    for index in "${!task_ids[@]}"; do
      if (( index % ${#gpus[@]} != gpu_index )); then continue; fi
      task=${task_ids[$index]}
      CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false \
        "$python" "$runner" --gpu "$gpu" --task-suite libero_10 --task-id "$task" --episode-id 0 \
        --official-normal-actions --normal-open-loop 8 \
        --max-learned-recoveries 1 --learned-recovery-max-steps 70 --recovery-confirm-steps 2 \
        --adapter-checkpoint "$adapter_checkpoint" --gate-checkpoint "$gate_checkpoint" --gate-num-classes 2 \
        --recovery-probability-threshold "$threshold" \
        --candidate-verifier-checkpoint "$base/onpolicy-gate-states-v2/candidate-verifier-v2-onpolicy-causal/macro_verifier_best.safetensors" \
        --neural-recovery-mode --disable-recovery-phase-conditioning --neural-geometry-conditioning \
        --save-online-images --output-dir "$output_root/task${task}" >"$output_root/task${task}.log" 2>&1
    done
  ) &
done
wait
