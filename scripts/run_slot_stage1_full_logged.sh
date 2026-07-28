#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs
LOG="${LOG:-logs/stage1_full_train.log}"
exec > >(tee -a "${LOG}") 2>&1

echo "=== Formal Stage-1 training started at $(date --iso-8601=seconds) ==="
exec bash scripts/run_slot_stage1_full.sh
