#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=1
# Physical GPU 3 is excluded; logical devices become physical 0,1,2,4,5,6,7.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,4,5,6,7}"

.conda/envs/openvla-a/bin/python scripts/train_slot_stage3_openvla.py \
  --slot-checkpoint /data2/lty/naive-e-slotssm-runs/stage1-adapters/openvla-7b-finetuned-libero-spatial+libero_mem+b1+lr-1e-05--full10tasks-h16-sync6-s20k-eb6-no-gpu3--image_aug--multiview1000--object/pretrained_slots-wo-filters_s16.safetensors \
  --stage2-checkpoint /data2/lty/naive-e-slotssm-runs/stage2-refine-residual-next-h32-s2k-sync6-no-gpu3/stage2-step-002000.safetensors \
  --output-dir "${OUTPUT_DIR:-/data2/lty/naive-e-slotssm-runs/stage3-openvla-formal-preflight}" \
  --steps "${STEPS:-1}" --save-every "${SAVE_EVERY:-1}" \
  --log-every "${LOG_EVERY:-1}" --horizon 32 --action-chunk 16
