#!/usr/bin/env bash
# Collect one deterministic shard of feedback-FSM branches from a JSONL fork manifest.
set -uo pipefail

if [ "$#" -ne 6 ]; then
  echo "usage: $0 FORK_MANIFEST WORKER_INDEX NUM_WORKERS OUTPUT_ROOT PYTHON_BIN LIBERO_ROOT" >&2
  exit 2
fi

fork_manifest=$1
worker_index=$2
num_workers=$3
output_root=$4
python_bin=$5
libero_root=$6

mapfile -t fork_dirs < <(
  "$python_bin" - "$fork_manifest" "$worker_index" "$num_workers" <<'PY'
import json
import sys

for index, line in enumerate(open(sys.argv[1])):
    if line.strip() and index % int(sys.argv[3]) == int(sys.argv[2]):
        print(json.loads(line)["fork_dir"])
PY
)

mkdir -p "$output_root"
overall=0
for fork_dir in "${fork_dirs[@]}"; do
  task_name=$(basename "$(dirname "$fork_dir")")
  step_name=$(basename "$fork_dir")
  branch_root="$output_root/$task_name/$step_name"
  if [ -f "$branch_root/fsm_xyp0_p0/metadata.json" ]; then
    printf '%s skipped_existing\n' "$fork_dir" >> "$output_root/worker-${worker_index}.status"
    continue
  fi
  "$python_bin" scripts/collect_book_regrasp_fsm_branches.py \
    --fork-dir "$fork_dir" \
    --output-root "$branch_root" \
    --libero-root "$libero_root"
  code=$?
  printf '%s %s\n' "$fork_dir" "$code" >> "$output_root/worker-${worker_index}.status"
  if [ "$code" -ne 0 ]; then
    overall=1
  fi
done
exit "$overall"
