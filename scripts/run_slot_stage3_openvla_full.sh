#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,4,5,6,7}"
mkdir -p logs
exec > >(tee -a logs/stage3_openvla_formal_train.log) 2>&1
echo "=== Formal Stage-3 started at $(date --iso-8601=seconds) ==="

.conda/envs/openvla-a/bin/python scripts/train_slot_stage3_openvla.py \
  --slot-checkpoint /data2/lty/naive-e-slotssm-runs/stage1-adapters/openvla-7b-finetuned-libero-spatial+libero_mem+b1+lr-1e-05--full10tasks-h16-sync6-s20k-eb6-no-gpu3--image_aug--multiview1000--object/pretrained_slots-wo-filters_s16.safetensors \
  --stage2-checkpoint /data2/lty/naive-e-slotssm-runs/stage2-refine-residual-next-h32-s2k-sync6-no-gpu3/stage2-step-002000.safetensors \
  --output-dir /data2/lty/naive-e-slotssm-runs/stage3-openvla-causal-h32-a16-s2k \
  --steps "${STEPS:-2000}" --save-every "${SAVE_EVERY:-100}" \
  --log-every "${LOG_EVERY:-10}" --horizon 32 --action-chunk 16 --lr 1e-4
