#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export LIBERO_MEM_DATA_DIR="${LIBERO_MEM_DATA_DIR:-/data2/lty/libero-mem-data/LIBERO-Mem}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-/data2/lty/libero-mem-rlds/full/libero/rlds_datasets}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"

echo "Source HDF5 directory: ${LIBERO_MEM_DATA_DIR}"
echo "Destination TFDS root: ${TFDS_DATA_DIR}"

INFO="${TFDS_DATA_DIR}/libero_mem/1.0.0/dataset_info.json"
if [[ -f "${INFO}" ]]; then
  EPISODES="$(jq '[.splits[] | .shardLengths[] | tonumber] | add' "${INFO}")"
  if [[ "${EPISODES}" == "961" ]]; then
    echo "Full LIBERO-Mem RLDS already contains 961 episodes; skipping rebuild."
    exit 0
  fi
fi

.conda/envs/openvla-a/bin/tfds build \
  tools/LIBERO_Mem_rlds_builder \
  --data_dir "${TFDS_DATA_DIR}" \
  --overwrite
