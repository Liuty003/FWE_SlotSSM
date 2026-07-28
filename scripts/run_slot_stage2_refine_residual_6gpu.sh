#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=1
export TF_NUM_INTEROP_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,4,5,6,7}"

SLOT_CKPT="/data2/lty/naive-e-slotssm-runs/stage1-adapters/openvla-7b-finetuned-libero-spatial+libero_mem+b1+lr-1e-05--full10tasks-h16-sync6-s20k-eb6-no-gpu3--image_aug--multiview1000--object/pretrained_slots-wo-filters_s16.safetensors"
STAGE2_CKPT="/data2/lty/naive-e-slotssm-runs/stage2-full-mamba2-h32-s20k-sync6-no-gpu3/stage2-step-020000.safetensors"
OUTPUT_DIR="${OUTPUT_DIR:-/data2/lty/naive-e-slotssm-runs/stage2-refine-residual-next-h32-s2k-sync6-no-gpu3}"
mkdir -p logs
exec > >(tee -a logs/stage2_refine_residual_next_train.log) 2>&1
echo "=== Residual next-head refinement started at $(date --iso-8601=seconds) ==="

.conda/envs/openvla-a/bin/torchrun --standalone --nnodes=1 --nproc-per-node=6 \
  scripts/train_slot_stage2_ssm.py \
  --data-root /data2/lty/libero-mem-rlds/full \
  --slot-checkpoint "${SLOT_CKPT}" \
  --load-stage2 "${STAGE2_CKPT}" \
  --output-dir "${OUTPUT_DIR}" \
  --horizon 32 --steps "${STEPS:-2000}" --save-every 500 --log-every 10 \
  --lr 5e-5 --backend mamba2 --residual-next --reset-next-head --next-only \
  --window-weight 0 --next-weight 1 --contrast-weight 0.1
