#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-slot}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"

SLOT_CHECKPOINT="/data2/lty/naive-e-slotssm-runs/stage1-adapters/openvla-7b-finetuned-libero-spatial+libero_mem+b1+lr-1e-05--refine-bce-dice-h16-s2k--image_aug--multiview1000--object/pretrained_slots-wo-filters_s16.safetensors"
OUTPUT="/data2/lty/naive-e-slotssm-runs/stage2-smoke/mamba2-slot-ssm.safetensors"

CUDA_HOME=/usr/local/cuda-12.1 \
LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  .conda/envs/openvla-a/bin/python scripts/smoke_slot_stage2_ssm.py \
    --slot-checkpoint "${SLOT_CHECKPOINT}" \
    --output "${OUTPUT}" \
    --horizon 16 \
    --ssm-backend mamba2
