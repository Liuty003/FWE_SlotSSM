#!/usr/bin/env bash
set -uo pipefail

# Frozen final holdout. These episodes may be audited for schema completeness,
# but must not be used for training, threshold calibration, architecture choice,
# or routine development metrics.
DATA_ROOT=${DATA_ROOT:-/data2/lty/naive-e-slotssm-runs/recovery-expert-final-holdout-v1}
ROLLOUT_ROOT=${ROLLOUT_ROOT:-/data2/lty/naive-e-slotssm-runs/recovery-rollouts-final-holdout-v1}
SHARD_INDEX=${SHARD_INDEX:-0}
NUM_SHARDS=${NUM_SHARDS:-1}
GPU=${GPU:-0}

if (( SHARD_INDEX < 0 || SHARD_INDEX >= NUM_SHARDS )); then
  echo "invalid SHARD_INDEX=$SHARD_INDEX for NUM_SHARDS=$NUM_SHARDS" >&2
  exit 2
fi

mkdir -p "$DATA_ROOT" "$ROLLOUT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

SPECS=(
  "2000 24 524 nominal"
  "2001 25 525 nominal"
  "2100 26 626 lost_early"
  "2200 27 727 lost_late"
  "2201 28 728 lost_late"
  "2300 29 829 missed"
)

collect_one() {
  local dataset_id=$1 init_state=$2 seed=$3 kind=$4
  local detector=stage-b-tracked trigger=visual-lift threshold=15
  local label="LOST GRASP" action="0,0,0.8,0,0,0,-1"
  local extra=()
  case "$kind" in
    nominal) extra+=(--disable-failure-injection) ;;
    lost_early) threshold=10 ;;
    lost_late) threshold=25 ;;
    missed)
      detector=state-height
      trigger=fixed-step
      label="MISSED GRASP"
      action="0.8,0.8,0,0,0,0,1"
      extra+=(--mistake-start 25)
      ;;
    *) echo "unknown collection kind: $kind" >&2; return 2 ;;
  esac

  local stem="task9-final${dataset_id}-init${init_state}-${kind}"
  .conda/envs/openvla-a/bin/python \
    experiments/robot/libero/run_single_scene_recovery.py \
    --task-id 9 --episode-id "$init_state" \
    --dataset-episode-id "$dataset_id" --seed "$seed" \
    --output "$ROLLOUT_ROOT/${stem}.mp4" \
    --rollout-dir "$ROLLOUT_ROOT/${stem}" \
    --expert-dataset-dir "$DATA_ROOT/${stem}" \
    --mistake-trigger "$trigger" --injection-lift-threshold-px "$threshold" \
    --mistake-steps 7 --mistake-action "$action" --mistake-label "$label" \
    --failure-detector "$detector" --visual-lift-threshold-px 20 \
    --oracle-regrasp --max-steps 240 \
    --device-map balanced_low_0 --gpu-memory-gib 20 \
    "${extra[@]}"
  local status=$?
  echo "final_episode=$dataset_id init_state=$init_state seed=$seed kind=$kind exit_status=$status"
  return 0
}

for index in "${!SPECS[@]}"; do
  if (( index % NUM_SHARDS != SHARD_INDEX )); then
    continue
  fi
  read -r dataset_id init_state seed kind <<<"${SPECS[$index]}"
  collect_one "$dataset_id" "$init_state" "$seed" "$kind"
done
