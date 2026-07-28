#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export HF_HOME="${HF_HOME:-$(pwd)/artifacts/hf-cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

EPISODE_IDS="${EPISODE_IDS:-0,1,2,3,4}"
OUTPUT_DIR="${OUTPUT_DIR:-/data2/lty/naive-e-slotssm-runs/counterfactual-slots-spatial-e0e4-h4-physical}"

.conda/envs/openvla-a/bin/python scripts/precompute_counterfactual_slots.py \
  --slot-checkpoint /data2/lty/naive-e-slotssm-runs/stage1-adapters/openvla-7b-finetuned-libero-spatial+libero_mem+b1+lr-1e-05--full10tasks-h16-sync6-s20k-eb6-no-gpu3--image_aug--multiview1000--object/pretrained_slots-wo-filters_s16.safetensors \
  --input-root /data2/lty/naive-e-slotssm-runs/counterfactual-data \
  --output-dir "${OUTPUT_DIR}" \
  --task-ids 0,1,2,3,4,5,6,7,8,9 \
  --episode-ids "${EPISODE_IDS}" \
  --branch-steps 4 \
  --vision-microbatch 16 \
  --reserve-gib 2.0 \
  --free-fraction 0.75
