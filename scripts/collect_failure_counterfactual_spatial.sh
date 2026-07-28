#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export HF_HOME="${HF_HOME:-$(pwd)/artifacts/hf-cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

ROOT="${ROOT:-/data2/lty/naive-e-slotssm-runs/counterfactual-data}"
TASK_IDS="${TASK_IDS:-0 1 2 3 4 5 6 7 8 9}"
EPISODE_IDS="${EPISODE_IDS:-0}"

for task_id in $TASK_IDS; do
  for episode_id in $EPISODE_IDS; do
    output="$ROOT/libero_spatial_t${task_id}_e${episode_id}_h4_dual"
    if [[ -f "$output/metadata.json" ]] && grep -q '"completed": true' "$output/metadata.json"; then
      echo "skip completed $output"
      continue
    fi
    .conda/envs/openvla-a/bin/python experiments/robot/libero/collect_counterfactual_transitions.py \
      --checkpoint artifacts/checkpoints/openvla-7b-finetuned-libero-spatial \
      --task-suite libero_spatial \
      --task-id "$task_id" \
      --episode-id "$episode_id" \
      --wait-steps 20 \
      --max-steps 220 \
      --collect-every 5 \
      --max-forks 20 \
      --branch-steps 4 \
      --gpu-memory-gib 7 \
      --output-dir "$output"
  done
done
