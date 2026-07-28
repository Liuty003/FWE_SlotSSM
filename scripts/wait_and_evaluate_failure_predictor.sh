#!/usr/bin/env bash
set -euo pipefail

TRAIN_SESSION="${TRAIN_SESSION:-failure_action_predictor}"
FINAL_CHECKPOINT="${FINAL_CHECKPOINT:-/data2/lty/naive-e-slotssm-runs/failure-action-predictor-h32-s10k/action-predictor-step-010000.safetensors}"

while tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; do
  sleep 60
done

if [[ ! -f "$FINAL_CHECKPOINT" ]]; then
  echo "Training session ended without final checkpoint: $FINAL_CHECKPOINT" >&2
  exit 1
fi

bash scripts/run_failure_action_predictor_evaluate.sh
