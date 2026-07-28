#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=1
# Physical GPU 3 is intentionally excluded. Prefer otherwise-idle GPU 0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SLOT_CKPT="/data2/lty/naive-e-slotssm-runs/stage1-adapters/openvla-7b-finetuned-libero-spatial+libero_mem+b1+lr-1e-05--full10tasks-h16-sync6-s20k-eb6-no-gpu3--image_aug--multiview1000--object/pretrained_slots-wo-filters_s16.safetensors"
STAGE2_CKPT="/data2/lty/naive-e-slotssm-runs/stage2-full-mamba2-h32-s20k-sync6-no-gpu3/stage2-step-020000.safetensors"
OUTPUT="/data2/lty/naive-e-slotssm-runs/stage2-full-mamba2-h32-s20k-sync6-no-gpu3/evaluation.json"

.conda/envs/openvla-a/bin/python scripts/evaluate_slot_stage2_ssm.py \
  --slot-checkpoint "${SLOT_CKPT}" \
  --stage2-checkpoint "${STAGE2_CKPT}" \
  --output "${OUTPUT}" \
  --num-windows "${NUM_WINDOWS:-20}"
