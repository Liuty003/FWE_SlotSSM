#!/usr/bin/env bash
set -uo pipefail

# Formal collection template. Splits are assigned later by episode, never here.
DATA_ROOT=${DATA_ROOT:-/data2/lty/naive-e-slotssm-runs/recovery-expert-dataset-v3}
ROLLOUT_ROOT=${ROLLOUT_ROOT:-/data2/lty/naive-e-slotssm-runs/recovery-expert-rollouts-v3}
FAILURE_EPISODES=${FAILURE_EPISODES:-"0 1 2 3 4 5 6 7 8 9"}
NORMAL_EPISODES=${NORMAL_EPISODES:-"10 11 12 13 14 15 16 17 18 19"}
GPU=${GPU:-0}

mkdir -p "$DATA_ROOT" "$ROLLOUT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

collect_episode() {
  local episode=$1
  local kind=$2
  local extra=()
  if [[ "$kind" == "normal" ]]; then
    extra+=(--disable-failure-injection)
  fi
  .conda/envs/openvla-a/bin/python \
    experiments/robot/libero/run_single_scene_recovery.py \
    --task-id 9 --episode-id "$episode" \
    --output "$ROLLOUT_ROOT/task9-episode${episode}-${kind}.mp4" \
    --rollout-dir "$ROLLOUT_ROOT/task9-episode${episode}-${kind}" \
    --expert-dataset-dir "$DATA_ROOT/task9-episode${episode}-${kind}" \
    --mistake-trigger visual-lift --injection-lift-threshold-px 15 \
    --mistake-steps 7 --mistake-action 0,0,0.8,0,0,0,-1 \
    --mistake-label "LOST GRASP" \
    --failure-detector stage-b-tracked \
    --visual-lift-threshold-px 20 \
    --oracle-regrasp --max-steps 240 \
    --device-map balanced_low_0 --gpu-memory-gib 20 \
    "${extra[@]}"
  local status=$?
  echo "episode=$episode kind=$kind exit_status=$status"
  return 0
}

for episode in $FAILURE_EPISODES; do
  collect_episode "$episode" failure
done
for episode in $NORMAL_EPISODES; do
  collect_episode "$episode" normal
done
