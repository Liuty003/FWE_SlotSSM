#!/usr/bin/env bash
set -euo pipefail

COLLECTION_SESSION="${COLLECTION_SESSION:-counterfactual_spatial_e1e4}"
POLL_SECONDS="${POLL_SECONDS:-30}"

echo "Waiting for tmux session ${COLLECTION_SESSION} to finish..."
while tmux has-session -t "${COLLECTION_SESSION}" 2>/dev/null; do
  sleep "${POLL_SECONDS}"
done

echo "Collection finished; starting Slot and physical-label precomputation."
EPISODE_IDS="${EPISODE_IDS:-0,1,2,3,4}" \
OUTPUT_DIR="${OUTPUT_DIR:-/data2/lty/naive-e-slotssm-runs/counterfactual-slots-spatial-e0e4-h4-physical}" \
bash scripts/run_precompute_counterfactual_slots.sh
