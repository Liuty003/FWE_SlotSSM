#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/data2/lty/naive-e-slotssm-runs/libero-plus-global-v1/libero-10-recovery-v1}"
FEATURE_ROOT="$ROOT/gate-data-v3/by-family"
MODEL_ROOT="$ROOT/model-v3/gates-by-family"
INIT="$ROOT/model-v3/native-slot-h1-gate/best.safetensors"

families=(
  black-bowl-drawer book-caddy cream-cheese-butter-basket mug-microwave mug-pudding
  soup-cream-cheese-basket soup-tomato-basket stove-moka two-moka-stove two-mugs-two-plates
)
mkdir -p "$MODEL_ROOT"
pids=()
for gpu in 1 2 3 4 5 6 7; do
  (
    # One worker per physical GPU.  Extra families queue behind the first
    # rather than sharing the 15--19 GiB validation peak on one card.
    for index in "${!families[@]}"; do
      if (( index % 7 != gpu - 1 )); then
        continue
      fi
      family="${families[$index]}"
      input="$FEATURE_ROOT/$family"
      output="$MODEL_ROOT/$family"
      mkdir -p "$output"
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python scripts/train_v43_local_recovery.py \
        --train-features "$input/failure-boundary.safetensors" \
        --val-features "$input/failure-boundary.safetensors" \
        --train-normal-features "$input/normal-balanced.safetensors" \
        --val-normal-features "$input/normal-balanced.safetensors" \
        --output-dir "$output" \
        --model-kind local \
        --init-checkpoint "$INIT" \
        --representation slot \
        --steps 1000 --batch-size 32 --lr 0.00005 \
        --eval-every 50 --save-every 500 \
        --lambda-gate 1.0 --lambda-recovery 0.0 --lambda-slot 0.0 \
        --gate-label-smoothing 0.0 --gripper-loss-weight 1.0 \
        --recovery-output-horizon 1 --gate-only-finetune \
        --device cuda --seed "$((20260829 + index))" --allow-in-sample-validation \
        > "$output/train.log" 2>&1
      echo "$family complete" > "$output/COMPLETE"
    done
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=$((failed + 1))
  fi
done
if (( failed > 0 )); then
  echo "$failed family Gate jobs failed" >&2
  exit 1
fi
echo "All ${#families[@]} family Gate jobs completed."
