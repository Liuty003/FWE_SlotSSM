#!/usr/bin/env bash
# Replay successful teacher-upper trajectories concurrently, one model per GPU.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 TASK_IDS_CSV GPUS_CSV OUTPUT_ROOT" >&2
  exit 2
fi

IFS=',' read -r -a task_ids <<< "$1"
IFS=',' read -r -a gpus <<< "$2"
output_root=$3
root=/data/lty/code/naive-e-slotssm
python="$root/.conda/envs/openvla-oft/bin/python"
collector="$root/scripts/collect_book_upper_teacher_rollouts.py"
mkdir -p "$output_root"

for gpu_index in "${!gpus[@]}"; do
  gpu=${gpus[$gpu_index]}
  (
    for task_index in "${!task_ids[@]}"; do
      if (( task_index % ${#gpus[@]} == gpu_index )); then
        task=${task_ids[$task_index]}
        CUDA_VISIBLE_DEVICES="$gpu" "$python" "$collector" --gpu "$gpu" \
          --task-ids "$task" --output-root "$output_root" \
          >"$output_root/task${task}.log" 2>&1
      fi
    done
  ) &
done
wait
