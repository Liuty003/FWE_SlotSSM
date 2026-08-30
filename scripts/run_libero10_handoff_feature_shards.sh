#!/usr/bin/env bash
set -euo pipefail

ROOT="${LIBERO10_RECOVERY_ROOT:-/data2/lty/naive-e-slotssm-runs/libero-plus-global-v1/libero-10-recovery-v1}"
HANDOFF_MANIFEST="${1:?usage: $0 HANDOFF_MANIFEST}"
INPUT_ROOT="${HANDOFF_FEATURE_INPUT:-$ROOT/feature-sources-v3}"
OUTPUT_ROOT="${HANDOFF_FEATURE_ROOT:-$ROOT/handoff-features-v3-h1-s1}"
GPU_CSV="${HANDOFF_FEATURE_GPUS:-0,1,2,3,4,5,6,7}"
PYTHON_BIN="${HANDOFF_PYTHON:-/data/lty/code/naive-e-slotssm/.conda/envs/openvla-oft/bin/python}"

if [[ ! -f "$HANDOFF_MANIFEST" ]]; then
  echo "missing handoff manifest: $HANDOFF_MANIFEST" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT/logs"
IFS=',' read -r -a gpus <<< "$GPU_CSV"
num_shards=${#gpus[@]}
if (( num_shards < 1 )); then
  echo "HANDOFF_FEATURE_GPUS is empty" >&2
  exit 2
fi

run_stage() {
  local stage=$1
  shift
  local pids=()
  for shard in "${!gpus[@]}"; do
    local gpu=${gpus[$shard]}
    local output="$OUTPUT_ROOT/${stage}-shard${shard}.safetensors"
    local log="$OUTPUT_ROOT/logs/${stage}-shard${shard}.log"
    env CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
      "$PYTHON_BIN" scripts/precompute_v4_oft_relation_features.py \
        --input-root "$INPUT_ROOT" --output "$output" --task-suite libero_10 \
        --native-input-only --decision-stride 8 --recovery-sampling-stride 1 \
        --recovery-target-horizon 1 --handoff-manifest "$HANDOFF_MANIFEST" \
        --mark-all-recovery --source-shard-index "$shard" --num-source-shards "$num_shards" \
        "$@" > "$log" 2>&1 &
    pids+=("$!")
    printf 'stage=%s shard=%s gpu=%s pid=%s\n' "$stage" "$shard" "$gpu" "${pids[-1]}"
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=$((failed + 1)); fi
  done
  if (( failed > 0 )); then
    echo "$failed $stage feature workers failed" >&2
    exit 1
  fi
  local inputs=()
  for shard in "${!gpus[@]}"; do
    inputs+=("$OUTPUT_ROOT/${stage}-shard${shard}.safetensors")
  done
  "$PYTHON_BIN" scripts/merge_v4_relation_features.py \
    --inputs "${inputs[@]}" --output "$OUTPUT_ROOT/${stage}-all.safetensors"
}

# Strict resume negatives: only teacher states before the certified handoff.
run_stage recovery-prehandoff
# The second extraction adds exactly one NORMAL row at each certified boundary.
run_stage resume-boundary --include-handoff-gate-rows

echo "Certified handoff feature extraction completed: $OUTPUT_ROOT"
