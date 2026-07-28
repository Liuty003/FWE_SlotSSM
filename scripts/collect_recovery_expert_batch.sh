#!/usr/bin/env bash
set -uo pipefail

DATA_ROOT=${DATA_ROOT:-/data2/lty/naive-e-slotssm-runs/recovery-expert-dataset-v2}
ROLLOUT_ROOT=${ROLLOUT_ROOT:-/data2/lty/naive-e-slotssm-runs/recovery-expert-rollouts-v2}
EPISODES=${EPISODES:-"0 1 2 3 4 8 9"}
GPU=${GPU:-0}

mkdir -p "$DATA_ROOT" "$ROLLOUT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

for episode in $EPISODES; do
  .conda/envs/openvla-a/bin/python \
    experiments/robot/libero/run_single_scene_recovery.py \
    --task-id 9 --episode-id "$episode" \
    --output "$ROLLOUT_ROOT/task9-episode${episode}.mp4" \
    --rollout-dir "$ROLLOUT_ROOT/task9-episode${episode}" \
    --expert-dataset-dir "$DATA_ROOT/task9-episode${episode}" \
    --mistake-trigger visual-lift --injection-lift-threshold-px 15 \
    --mistake-steps 7 --mistake-action 0,0,0.8,0,0,0,-1 \
    --mistake-label "LOST GRASP" \
    --failure-detector stage-b-tracked \
    --visual-lift-threshold-px 20 \
    --oracle-regrasp --max-steps 240 \
    --device-map balanced_low_0 --gpu-memory-gib 20
  status=$?
  echo "episode=$episode exit_status=$status"
done
