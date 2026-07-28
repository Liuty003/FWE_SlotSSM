#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=1
export TF_NUM_INTEROP_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
export OMP_NUM_THREADS=4
# Keep physical GPU 3 excluded, as requested.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,4,5,6,7}"

SLOT_CKPT="${SLOT_CKPT:-/data2/lty/naive-e-slotssm-runs/stage1-adapters/openvla-7b-finetuned-libero-spatial+libero_mem+b1+lr-1e-05--full10tasks-h16-sync6-s20k-eb6-no-gpu3--image_aug--multiview1000--object/pretrained_slots-wo-filters_s16.safetensors}"
OUTPUT_DIR="${OUTPUT_DIR:-/data2/lty/naive-e-slotssm-runs/stage2-full-mamba2-h32-s20k-sync6-no-gpu3}"
STEPS="${STEPS:-20000}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-10}"

test -f "${SLOT_CKPT}"
mkdir -p logs
LOG="${LOG:-logs/stage2_full_h32_sync6_train.log}"
exec > >(tee -a "${LOG}") 2>&1
echo "=== Stage-2 synchronized training started at $(date --iso-8601=seconds) ==="
echo "Stage-2 GPUs: ${CUDA_VISIBLE_DEVICES}; steps=${STEPS}; checkpoint=${SLOT_CKPT}"

.conda/envs/openvla-a/bin/torchrun --standalone --nnodes=1 --nproc-per-node=6 \
  scripts/train_slot_stage2_ssm.py \
  --data-root /data2/lty/libero-mem-rlds/full \
  --slot-checkpoint "${SLOT_CKPT}" \
  --output-dir "${OUTPUT_DIR}" \
  --horizon 32 \
  --steps "${STEPS}" \
  --save-every "${SAVE_EVERY}" \
  --log-every "${LOG_EVERY}" \
  --backend mamba2
