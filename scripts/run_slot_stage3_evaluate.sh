#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,4,5,6,7}"

.conda/envs/openvla-a/bin/python scripts/evaluate_slot_stage3_openvla.py \
  --slot-checkpoint /data2/lty/naive-e-slotssm-runs/stage1-adapters/openvla-7b-finetuned-libero-spatial+libero_mem+b1+lr-1e-05--full10tasks-h16-sync6-s20k-eb6-no-gpu3--image_aug--multiview1000--object/pretrained_slots-wo-filters_s16.safetensors \
  --stage2-checkpoint /data2/lty/naive-e-slotssm-runs/stage2-refine-residual-next-h32-s2k-sync6-no-gpu3/stage2-step-002000.safetensors \
  --stage3-checkpoint /data2/lty/naive-e-slotssm-runs/stage3-openvla-causal-h32-a16-s2k/stage3-step-002000.safetensors \
  --output /data2/lty/naive-e-slotssm-runs/stage3-openvla-causal-h32-a16-s2k/evaluation.json \
  --num-windows "${NUM_WINDOWS:-20}"
