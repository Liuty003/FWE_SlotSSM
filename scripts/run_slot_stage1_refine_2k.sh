#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-slot}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"

SLOT_CKPT="/data2/lty/naive-e-slotssm-runs/stage1-adapters/openvla-7b-finetuned-libero-spatial+libero_mem+b1+lr-2e-05--train-h16-s1k--image_aug--multiview1000--object/pretrained_slots-wo-filters_s16.safetensors"

if [[ ! -f "${SLOT_CKPT}" ]]; then
  echo "Missing Stage-1 1k checkpoint: ${SLOT_CKPT}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  .conda/envs/openvla-a/bin/torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=1 \
    vla-scripts/finetune-1_slotvla_slot-wo-filter.py \
    --vla_path artifacts/checkpoints/openvla-7b-finetuned-libero-spatial \
    --data_root_dir /data2/lty/libero-mem-rlds \
    --dataset_name libero_mem \
    --run_root_dir /data2/lty/naive-e-slotssm-runs \
    --adapter_tmp_dir /data2/lty/naive-e-slotssm-runs/stage1-adapters \
    --load_slot_path "${SLOT_CKPT}" \
    --batch_size 1 \
    --horizon_size 16 \
    --number_of_slots 16 \
    --max_steps 2000 \
    --save_steps 200 \
    --learning_rate 1e-5 \
    --grad_accumulation_steps 1 \
    --image_aug true \
    --shuffle_buffer_size 1000 \
    --modality_key 1000 \
    --run_id_note refine-bce-dice-h16-s2k
