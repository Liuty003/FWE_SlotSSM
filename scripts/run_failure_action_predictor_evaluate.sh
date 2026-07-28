#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"

SLOT_CHECKPOINT="${SLOT_CHECKPOINT:-/data2/lty/naive-e-slotssm-runs/stage1-adapters/openvla-7b-finetuned-libero-spatial+libero_mem+b1+lr-1e-05--full10tasks-h16-sync6-s20k-eb6-no-gpu3--image_aug--multiview1000--object/pretrained_slots-wo-filters_s16.safetensors}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-/data2/lty/naive-e-slotssm-runs/stage2-refine-residual-next-h32-s2k-sync6-no-gpu3/stage2-step-002000.safetensors}"
PREDICTOR_CHECKPOINT="${PREDICTOR_CHECKPOINT:-/data2/lty/naive-e-slotssm-runs/failure-action-predictor-h32-s10k/action-predictor-step-010000.safetensors}"
OUTPUT="${OUTPUT:-/data2/lty/naive-e-slotssm-runs/failure-action-predictor-h32-s10k/heldout-evaluation.json}"
NUM_WINDOWS="${NUM_WINDOWS:-50}"

.conda/envs/openvla-a/bin/python scripts/evaluate_failure_action_predictor.py \
  --slot-checkpoint "$SLOT_CHECKPOINT" \
  --stage2-checkpoint "$STAGE2_CHECKPOINT" \
  --predictor-checkpoint "$PREDICTOR_CHECKPOINT" \
  --output "$OUTPUT" \
  --num-windows "$NUM_WINDOWS" \
  --horizon 32 \
  --vision-microbatch 8 \
  --reserve-gib 2.0 \
  --free-fraction 0.75
