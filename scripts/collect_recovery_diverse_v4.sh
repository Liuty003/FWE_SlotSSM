#!/usr/bin/env bash
set -uo pipefail

# One shard of the formal 24-episode task-9 recovery collection matrix.
# Launch NUM_SHARDS copies with distinct SHARD_INDEX/GPU values.
DATA_ROOT=${DATA_ROOT:-/data2/lty/naive-e-slotssm-runs/recovery-expert-dataset-v4-diverse}
ROLLOUT_ROOT=${ROLLOUT_ROOT:-/data2/lty/naive-e-slotssm-runs/recovery-expert-rollouts-v4-diverse}
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

# dataset_id init_state seed kind
SPECS=(
  "1000 0 100 nominal" "1001 1 101 nominal"
  "1002 2 102 nominal" "1003 3 103 nominal"
  "1004 4 104 nominal" "1005 5 105 nominal"
  "1006 6 106 nominal" "1007 7 107 nominal"
  "1100 8 208 lost_early" "1101 9 209 lost_early"
  "1102 10 210 lost_early" "1103 11 211 lost_early"
  "1104 12 212 lost_early" "1105 13 213 lost_early"
  "1200 14 314 lost_late" "1201 15 315 lost_late"
  "1202 16 316 lost_late" "1203 17 317 lost_late"
  "1204 18 318 lost_late" "1205 19 319 lost_late"
  "1300 20 420 missed" "1301 21 421 missed"
  "1302 22 422 missed" "1303 23 423 missed"
)

collect_one() {
  local dataset_id=$1 init_state=$2 seed=$3 kind=$4
  local detector=stage-b-tracked trigger=visual-lift threshold=15
  local label="LOST GRASP" action="0,0,0.8,0,0,0,-1"
  local extra=()
  case "$kind" in
    nominal)
      extra+=(--disable-failure-injection)
      ;;
    lost_early)
      threshold=10
      ;;
    lost_late)
      threshold=25
      ;;
    missed)
      detector=state-height
      trigger=fixed-step
      label="MISSED GRASP"
      action="0.8,0.8,0,0,0,0,1"
      extra+=(--mistake-start 25)
      ;;
    *)
      echo "unknown collection kind: $kind" >&2
      return 2
      ;;
  esac

  local stem="task9-dataset${dataset_id}-init${init_state}-${kind}"
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
  echo "dataset_episode=$dataset_id init_state=$init_state seed=$seed kind=$kind exit_status=$status"
  return 0
}

for index in "${!SPECS[@]}"; do
  if (( index % NUM_SHARDS != SHARD_INDEX )); then
    continue
  fi
  read -r dataset_id init_state seed kind <<<"${SPECS[$index]}"
  collect_one "$dataset_id" "$init_state" "$seed" "$kind"
done
