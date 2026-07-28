#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-slot}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"

# One process is intentional: each DDP rank initially reads the 15 GB OpenVLA
# checkpoint on CPU before the unused language model is released.
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
    --batch_size 1 \
    --horizon_size 4 \
    --number_of_slots 16 \
    --max_steps 1 \
    --save_steps 1 \
    --learning_rate 2e-5 \
    --grad_accumulation_steps 1 \
    --image_aug false \
    --shuffle_buffer_size 32 \
    --modality_key 1000 \
    --run_id_note smoke
