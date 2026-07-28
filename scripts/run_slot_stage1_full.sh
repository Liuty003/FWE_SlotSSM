#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-slot}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"

DATA_ROOT="${DATA_ROOT:-/data2/lty/libero-mem-rlds/full}"
SLOT_CKPT="${SLOT_CKPT:-/data2/lty/naive-e-slotssm-runs/stage1-adapters/openvla-7b-finetuned-libero-spatial+libero_mem+b1+lr-1e-05--refine-bce-dice-h16-s2k--image_aug--multiview1000--object/pretrained_slots-wo-filters_s16.safetensors}"

if [[ ! -f "${DATA_ROOT}/libero/rlds_datasets/libero_mem/1.0.0/dataset_info.json" ]]; then
  echo "Missing full LIBERO-Mem RLDS dataset under ${DATA_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${SLOT_CKPT}" ]]; then
  echo "Missing Stage-1 initialization checkpoint: ${SLOT_CKPT}" >&2
  exit 1
fi

# A caller can pin a card explicitly.  Otherwise select the card reporting the
# most free memory at launch time, which is safer on the shared 8-GPU server.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$({ nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits || true; } \
    | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')"
fi
if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  echo "Could not select a CUDA device; set CUDA_VISIBLE_DEVICES explicitly." >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES

echo "Training Stage-1 Slot on physical GPU(s): ${CUDA_VISIBLE_DEVICES}"
echo "Full RLDS root: ${DATA_ROOT}"
echo "Initialization checkpoint: ${SLOT_CKPT}"

.conda/envs/openvla-a/bin/torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=1 \
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
  --max_steps 31000 \
  --save_steps 1000 \
  --learning_rate 1e-5 \
  --grad_accumulation_steps 4 \
  --image_aug true \
  --shuffle_buffer_size 10000 \
  --modality_key 1000 \
  --run_id_note full10tasks-h16-s31k-ga4
