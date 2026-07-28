#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-slot-eval}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export TOKENIZERS_PARALLELISM=false

CHECKPOINT="/data2/lty/naive-e-slotssm-runs/stage1-adapters/openvla-7b-finetuned-libero-spatial+libero_mem+b1+lr-1e-05--full10tasks-h16-sync6-s20k-eb6-no-gpu3--image_aug--multiview1000--object/pretrained_slots-wo-filters_s16.safetensors"
OUTPUT="/data2/lty/naive-e-slotssm-runs/stage1-multitask-acceptance"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$({ nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits || true; } \
    | awk -F, '$1 != 3 {gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}' \
    | sort -k2 -nr | head -1 | awk '{print $1}')"
fi
export CUDA_VISIBLE_DEVICES
echo "Evaluating Stage-1 on physical GPU ${CUDA_VISIBLE_DEVICES} (GPU 3 excluded)"

.conda/envs/openvla-a/bin/python scripts/evaluate_slot_stage1_multitask.py \
  --checkpoint "${CHECKPOINT}" \
  --data-root /data2/lty/libero-mem-rlds/full \
  --output-dir "${OUTPUT}" \
  --horizon 16 \
  --num-frames 4 \
  --num-tasks 10
