#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-slot-ddp7}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6,7}"

DATA_ROOT="${DATA_ROOT:-/data2/lty/libero-mem-rlds/full}"
SLOT_CKPT="${SLOT_CKPT:-/data2/lty/naive-e-slotssm-runs/stage1-adapters/openvla-7b-finetuned-libero-spatial+libero_mem+b4+lr-1e-05--full10tasks-h16-s31k-ga4--image_aug--multiview1000--object/pretrained_slots-wo-filters_s16.safetensors}"

if [[ ! -f "${SLOT_CKPT}" ]]; then
  echo "Missing step-1000 Stage-1 checkpoint: ${SLOT_CKPT}" >&2
  exit 1
fi

mkdir -p logs
LOG="${LOG:-logs/stage1_full_ddp7_train.log}"
exec > >(tee -a "${LOG}") 2>&1

echo "=== Seven-GPU Stage-1 DDP training started at $(date --iso-8601=seconds) ==="
echo "Physical GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Initialization checkpoint: ${SLOT_CKPT}"
echo "Effective global batch: 7 (1 sample/rank, no accumulation)"

.conda/envs/openvla-a/bin/torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=7 \
  vla-scripts/finetune-1_slotvla_slot-wo-filter.py \
  --vla_path artifacts/checkpoints/openvla-7b-finetuned-libero-spatial \
  --data_root_dir "${DATA_ROOT}" \
  --dataset_name libero_mem \
  --run_root_dir /data2/lty/naive-e-slotssm-runs \
  --adapter_tmp_dir /data2/lty/naive-e-slotssm-runs/stage1-adapters \
  --load_slot_path "${SLOT_CKPT}" \
  --batch_size 1 \
  --horizon_size 16 \
  --number_of_slots 16 \
  --max_steps 17150 \
  --save_steps 1000 \
  --learning_rate 1e-5 \
  --grad_accumulation_steps 1 \
  --image_aug true \
  --shuffle_buffer_size 10000 \
  --modality_key 1000 \
  --run_id_note full10tasks-h16-ddp7-s17150-eb7
