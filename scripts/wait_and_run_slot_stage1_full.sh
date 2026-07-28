#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

INFO="/data2/lty/libero-mem-rlds/full/libero/rlds_datasets/libero_mem/1.0.0/dataset_info.json"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-961}"

echo "Waiting for full LIBERO-Mem RLDS conversion to finish..."
while [[ ! -f "${INFO}" ]]; do
  sleep 30
done

EPISODES="$(jq '[.splits[] | .shardLengths[] | tonumber] | add' "${INFO}")"
if [[ "${EPISODES}" != "${EXPECTED_EPISODES}" ]]; then
  echo "Refusing to train: expected ${EXPECTED_EPISODES} episodes, found ${EPISODES}." >&2
  exit 1
fi

echo "Verified ${EPISODES} full-dataset episodes. Starting formal Stage-1 training."
exec bash scripts/run_slot_stage1_full.sh
