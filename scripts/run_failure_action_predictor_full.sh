#!/usr/bin/env bash
set -euo pipefail

# Physical GPU 3 is intentionally excluded. GPU 1 is also excluded because it
# is currently occupied; logical CUDA indices are remapped by CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"

SLOT_CHECKPOINT="${SLOT_CHECKPOINT:-/data2/lty/naive-e-slotssm-runs/stage1-adapters/openvla-7b-finetuned-libero-spatial+libero_mem+b1+lr-1e-05--full10tasks-h16-sync6-s20k-eb6-no-gpu3--image_aug--multiview1000--object/pretrained_slots-wo-filters_s16.safetensors}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-/data2/lty/naive-e-slotssm-runs/stage2-refine-residual-next-h32-s2k-sync6-no-gpu3/stage2-step-002000.safetensors}"
OUTPUT_DIR="${OUTPUT_DIR:-/data2/lty/naive-e-slotssm-runs/failure-action-predictor-h32-s10k}"
STEPS="${STEPS:-10000}"

.conda/envs/openvla-a/bin/python scripts/train_failure_action_predictor.py \
  --slot-checkpoint "$SLOT_CHECKPOINT" \
  --stage2-checkpoint "$STAGE2_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --steps "$STEPS" \
  --save-every 500 \
  --log-every 10 \
  --horizon 32 \
  --vision-microbatch 8 \
  --lr 1e-4 \
  --contrastive-weight 0.1 \
  --reserve-gib 2.0 \
  --free-fraction 0.75
