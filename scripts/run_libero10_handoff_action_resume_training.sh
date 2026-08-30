#!/usr/bin/env bash
set -euo pipefail

ROOT="${LIBERO10_RECOVERY_ROOT:-/data2/lty/naive-e-slotssm-runs/libero-plus-global-v1/libero-10-recovery-v1}"
FEATURE_ROOT="${HANDOFF_FEATURE_ROOT:-$ROOT/handoff-features-v3-h1-s1}"
MODEL_ROOT="${HANDOFF_MODEL_ROOT:-$ROOT/model-v3}"
PYTHON_BIN="${HANDOFF_TRAIN_PYTHON:-/data/lty/code/naive-e-slotssm/.conda/envs/openvla-oft/bin/python}"
DEVICE="${HANDOFF_TRAIN_DEVICE:-cuda}"

recovery="$FEATURE_ROOT/recovery-prehandoff-all.safetensors"
boundary="$FEATURE_ROOT/resume-boundary-all.safetensors"
gate_init="$MODEL_ROOT/native-slot-h1-gate/best.safetensors"
action_output="$MODEL_ROOT/native-slot-h1-action-handoff"
resume_output="$MODEL_ROOT/native-slot-h1-dual-handoff"
for path in "$recovery" "$boundary" "$gate_init"; do
  if [[ ! -f "$path" ]]; then echo "missing input: $path" >&2; exit 2; fi
done
mkdir -p "$action_output" "$resume_output"

# Only direct_action is updated; the family Gate representation remains exact.
"$PYTHON_BIN" scripts/train_v43_local_recovery.py \
  --train-features "$recovery" --val-features "$recovery" \
  --output-dir "$action_output" --model-kind local --init-checkpoint "$gate_init" \
  --representation slot --steps 4000 --batch-size 32 --lr 0.0001 \
  --eval-every 250 --save-every 1000 --lambda-gate 0 --lambda-recovery 1 --lambda-slot 0 \
  --gripper-loss-weight 4 --recovery-output-horizon 1 --recovery-head-only-finetune \
  --device "$DEVICE" --seed 20260829 --allow-in-sample-validation \
  2>&1 | tee "$action_output/train.log"

# Only resume_head is updated. Gate, representation, and cropped action stay frozen.
"$PYTHON_BIN" scripts/train_v44_resume_head.py \
  --recovery-features "$recovery" --boundary-features "$boundary" \
  --init-checkpoint "$action_output/best.safetensors" --output-dir "$resume_output" \
  --representation slot --recovery-output-horizon 1 --steps 1500 --batch-size 32 \
  --lr 0.0001 --eval-every 100 --save-every 500 --tail-negative-steps 24 \
  --device "$DEVICE" --seed 20260829 \
  2>&1 | tee "$resume_output/train.log"

echo "Handoff-cropped action and shared Resume training completed: $resume_output"
