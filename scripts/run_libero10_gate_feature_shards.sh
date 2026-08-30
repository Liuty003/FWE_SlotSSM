#!/usr/bin/env bash
set -euo pipefail

GATE_INPUT_ROOT=${GATE_INPUT_ROOT:-/data2/lty/naive-e-slotssm-runs/libero-plus-global-v1/full-audit-v1/libero_10}
GATE_OUTPUT_ROOT=${GATE_OUTPUT_ROOT:-/data2/lty/naive-e-slotssm-runs/libero-plus-global-v1/libero-10-recovery-v1/gate-data-v3}
GATE_BOUNDARY_MANIFEST=${GATE_BOUNDARY_MANIFEST:-/data2/lty/naive-e-slotssm-runs/libero-plus-global-v1/libero-10-recovery-v1/teacher-routes-v3.json}
GATE_NORMAL_MANIFEST=${GATE_NORMAL_MANIFEST:-$GATE_OUTPUT_ROOT/normal-sources.json}
GATE_FEATURE_GPUS=${GATE_FEATURE_GPUS:-1,2,3,4,5,6,7}
GATE_PYTHON=${GATE_PYTHON:-/data/lty/code/naive-e-slotssm/.conda/envs/openvla-oft/bin/python}

mkdir -p "$GATE_OUTPUT_ROOT/logs"
IFS=',' read -r -a gpu_array <<< "$GATE_FEATURE_GPUS"
num_shards=${#gpu_array[@]}
if (( num_shards < 1 )); then
  echo "GATE_FEATURE_GPUS is empty" >&2
  exit 2
fi

run_stage() {
  local stage=$1
  shift
  local pids=()
  for shard in "${!gpu_array[@]}"; do
    local gpu=${gpu_array[$shard]}
    local output="$GATE_OUTPUT_ROOT/${stage}-shard${shard}.safetensors"
    local log="$GATE_OUTPUT_ROOT/logs/${stage}-shard${shard}.log"
    env CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
      "$GATE_PYTHON" scripts/precompute_v4_oft_relation_features.py \
        --input-root "$GATE_INPUT_ROOT" \
        --output "$output" \
        --task-suite libero_10 \
        --native-input-only \
        --decision-stride 8 \
        --recovery-target-horizon 1 \
        --source-shard-index "$shard" \
        --num-source-shards "$num_shards" \
        "$@" > "$log" 2>&1 &
    pids+=("$!")
    printf 'stage=%s shard=%s gpu=%s pid=%s log=%s\n' \
      "$stage" "$shard" "$gpu" "${pids[-1]}" "$log"
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if (( failed )); then
    echo "$stage feature extraction failed; inspect logs" >&2
    exit 1
  fi
  local inputs=()
  for shard in "${!gpu_array[@]}"; do
    inputs+=("$GATE_OUTPUT_ROOT/${stage}-shard${shard}.safetensors")
  done
  "$GATE_PYTHON" scripts/merge_v4_relation_features.py \
    --inputs "${inputs[@]}" \
    --output "$GATE_OUTPUT_ROOT/${stage}-all.safetensors"
}

run_stage failure-boundary \
  --failed-gate-boundary-manifest "$GATE_BOUNDARY_MANIFEST" \
  --failed-gate-boundary-key event_timestep \
  --failure-lead-steps 8 \
  --gate-boundary-window-before 32 \
  --gate-boundary-window-after 32

normal_task_ids=$("$GATE_PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["task_ids_csv"])' \
  "$GATE_NORMAL_MANIFEST")
run_stage normal-balanced --success-only --task-ids "$normal_task_ids"
