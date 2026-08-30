#!/usr/bin/env bash
set -euo pipefail

RECOVERY_FEATURE_INPUT=${RECOVERY_FEATURE_INPUT:-/data2/lty/naive-e-slotssm-runs/libero-plus-global-v1/libero-10-recovery-v1/feature-sources-v3}
RECOVERY_FEATURE_ROOT=${RECOVERY_FEATURE_ROOT:-/data2/lty/naive-e-slotssm-runs/libero-plus-global-v1/libero-10-recovery-v1/features-recovery-v3-h1-s4}
RECOVERY_FEATURE_GPUS=${RECOVERY_FEATURE_GPUS:-0,1,2,3,4,5,6,7}
RECOVERY_PYTHON=${RECOVERY_PYTHON:-/data/lty/code/naive-e-slotssm/.conda/envs/openvla-oft/bin/python}

mkdir -p "$RECOVERY_FEATURE_ROOT/logs"
IFS=',' read -r -a gpu_array <<< "$RECOVERY_FEATURE_GPUS"
num_shards=${#gpu_array[@]}
if (( num_shards < 1 )); then
  echo "RECOVERY_FEATURE_GPUS is empty" >&2
  exit 2
fi

pids=()
for shard in "${!gpu_array[@]}"; do
  gpu=${gpu_array[$shard]}
  output="$RECOVERY_FEATURE_ROOT/recovery-shard${shard}.safetensors"
  log="$RECOVERY_FEATURE_ROOT/logs/recovery-shard${shard}.log"
  env CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
    "$RECOVERY_PYTHON" scripts/precompute_v4_oft_relation_features.py \
      --input-root "$RECOVERY_FEATURE_INPUT" \
      --output "$output" \
      --task-suite libero_10 \
      --native-input-only \
      --decision-stride 8 \
      --recovery-sampling-stride 4 \
      --recovery-target-horizon 1 \
      --mark-all-recovery \
      --source-shard-index "$shard" \
      --num-source-shards "$num_shards" \
      > "$log" 2>&1 &
  pids+=("$!")
  printf 'shard=%s gpu=%s pid=%s log=%s\n' "$shard" "$gpu" "${pids[-1]}" "$log"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed )); then
  echo "at least one recovery feature shard failed; inspect logs before resuming" >&2
  exit 1
fi

inputs=()
for shard in "${!gpu_array[@]}"; do
  inputs+=("$RECOVERY_FEATURE_ROOT/recovery-shard${shard}.safetensors")
done
"$RECOVERY_PYTHON" scripts/merge_v4_relation_features.py \
  --inputs "${inputs[@]}" \
  --output "$RECOVERY_FEATURE_ROOT/recovery-all.safetensors"
