#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs
LOG="${LOG:-logs/stage1_full_pipeline.log}"
exec > >(tee -a "${LOG}") 2>&1

echo "=== Full Stage-1 pipeline started at $(date --iso-8601=seconds) ==="

bash scripts/build_libero_mem_rlds_full.sh

INFO="/data2/lty/libero-mem-rlds/full/libero/rlds_datasets/libero_mem/1.0.0/dataset_info.json"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-961}"
EPISODES="$(jq '[.splits[] | .shardLengths[] | tonumber] | add' "${INFO}")"
if [[ "${EPISODES}" != "${EXPECTED_EPISODES}" ]]; then
  echo "Refusing to train: expected ${EXPECTED_EPISODES} episodes, found ${EPISODES}." >&2
  exit 1
fi

echo "Verified ${EPISODES} full-dataset episodes. Starting formal Stage-1 training."
bash scripts/run_slot_stage1_full.sh
