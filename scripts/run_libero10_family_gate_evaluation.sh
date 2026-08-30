#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/data2/lty/naive-e-slotssm-runs/libero-plus-global-v1/libero-10-recovery-v1}"
FEATURE_ROOT="$ROOT/gate-data-v3/by-family"
MODEL_ROOT="$ROOT/model-v3/gates-by-family"
families=(
  black-bowl-drawer book-caddy cream-cheese-butter-basket mug-microwave mug-pudding
  soup-cream-cheese-basket soup-tomato-basket stove-moka two-moka-stove two-mugs-two-plates
)

pids=()
for gpu in 1 2 3 4 5 6 7; do
  (
    for index in "${!families[@]}"; do
      if (( index % 7 != gpu - 1 )); then
        continue
      fi
      family="${families[$index]}"
      input="$FEATURE_ROOT/$family"
      output="$MODEL_ROOT/$family"
      CUDA_VISIBLE_DEVICES="$gpu" python scripts/evaluate_v43_local_recovery.py \
        --adapter-dir "$output" --checkpoint best.safetensors \
        --features "$input/failure-boundary.safetensors" "$input/normal-balanced.safetensors" \
        --output "$output/offline-gate-eval-t050.json" \
        --batch-size 128 --gate-threshold 0.5 --device cuda \
        > "$output/offline-gate-eval-t050.stdout"
      echo "$family evaluated" > "$output/EVALUATED"
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
  echo "$failed family Gate evaluation workers failed" >&2
  exit 1
fi
echo "All ${#families[@]} family Gate evaluations completed."
