#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-slot}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SLOT_CHECKPOINT="/data2/lty/naive-e-slotssm-runs/stage1-adapters/openvla-7b-finetuned-libero-spatial+libero_mem+b1+lr-1e-05--refine-bce-dice-h16-s2k--image_aug--multiview1000--object/pretrained_slots-wo-filters_s16.safetensors"
SSM_CHECKPOINT="/data2/lty/naive-e-slotssm-runs/stage2-smoke/mamba2-slot-ssm.safetensors"
OUTPUT="/data2/lty/naive-e-slotssm-runs/stage3-openvla-smoke/slotssm-openvla-action-path.safetensors"

CUDA_HOME=/usr/local/cuda-12.1 \
LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
  .conda/envs/openvla-a/bin/python scripts/smoke_slot_stage3_openvla.py \
    --slot-checkpoint "${SLOT_CHECKPOINT}" \
    --ssm-checkpoint "${SSM_CHECKPOINT}" \
    --output "${OUTPUT}" \
    --horizon 16 \
    --gpu-count 8 \
    --reserve-gib 3 \
    --free-fraction 0.75
