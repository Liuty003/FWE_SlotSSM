#!/usr/bin/env bash
# Train the fixed-34-state unified recovery expert after frozen features exist.
# Teacher anchors/FSM are never inputs to either training invocation below.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 ACTION_FEATURES GATE_POSITIVE_FEATURES OUTPUT_ROOT GPU" >&2
  exit 2
fi

action_features=$1
gate_positive_features=$2
output_root=$3
gpu=$4
root=/data/lty/code/naive-e-slotssm
base=/data2/lty/naive-e-slotssm-runs/oft-plus-book-local-repair-v1
python="$root/.conda/envs/openvla-oft/bin/python"

mkdir -p "$output_root"

# v25 retains the known NORMAL preservation examples for original baseline
# successes.  The fresh positive teacher windows supply task-specific early
# RECOVERY evidence.  Common keys are intentional: no optional privileged
# diagnostic tensor can slip into the gate payload.
"$python" "$root/scripts/merge_recovery_feature_sets.py" \
  --inputs "$base/gate-v25-preservation/gate-train-v25.safetensors" "$gate_positive_features" \
  --output "$output_root/gate-train.safetensors" --common-keys-only

env CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
  "$python" "$root/scripts/train_book_regrasp_v2_recovery.py" \
  --train-features "$action_features" --val-features "$action_features" \
  --output-dir "$output_root/recovery" \
  --init-adapter "$base/neural-geometry-v8-aux/recovery-head-v8/adapter_best.safetensors" \
  --gpu "$gpu" --steps 3000 --batch-size 16 --lr 1e-5 --eval-every 100 \
  --first-step-weight 3 --gripper-threshold-weight 0.15 \
  --mode-conditioned-action-head --disable-phase-conditioning \
  --geometry-conditioned-action-head --geometry-loss-weight 3

env CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
  "$python" "$root/scripts/train_book_regrasp_v2_gate.py" \
  --train-features "$output_root/gate-train.safetensors" \
  --val-features "$output_root/gate-train.safetensors" \
  --output-dir "$output_root/gate" \
  --init-gate-checkpoint "$base/gate-v25-preservation/gate/gate_best.safetensors" \
  --gpu "$gpu" --steps 1200 --batch-size 32 --lr 1e-5 --eval-every 50 --num-classes 2
