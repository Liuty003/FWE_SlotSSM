#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH=.
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-slot}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"

CHECKPOINT="/data2/lty/naive-e-slotssm-runs/stage1-adapters/openvla-7b-finetuned-libero-spatial+libero_mem+b1+lr-1e-05--refine-bce-dice-h16-s2k--image_aug--multiview1000--object/pretrained_slots-wo-filters_s16.safetensors"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  .conda/envs/openvla-a/bin/python scripts/visualize_slot_stage1.py \
    --checkpoint "${CHECKPOINT}" \
    --output-dir /data2/lty/naive-e-slotssm-runs/stage1-visualization-refined \
    --horizon 16 \
    --num-frames 4
