#!/usr/bin/env bash
# Run a deterministic shard of a LIBERO manifest with the official OFT runner.
set -uo pipefail

if [ "$#" -lt 6 ] || [ "$#" -gt 7 ]; then
  echo "usage: $0 MANIFEST WORKER_INDEX NUM_WORKERS GPU OUTPUT_DIR PYTHON_BIN [TEACHER_STATES_DIR]" >&2
  exit 2
fi

manifest=$1
worker_index=$2
num_workers=$3
gpu=$4
output_dir=$5
python_bin=$6
teacher_states_dir=${7:-}

mapfile -t task_ids < <(
  "$python_bin" - "$manifest" "$worker_index" "$num_workers" <<'PY'
import json
import sys

records = json.load(open(sys.argv[1]))
worker_index = int(sys.argv[2])
num_workers = int(sys.argv[3])
for i, record in enumerate(records):
    if i % num_workers == worker_index:
        print(record["task_id"])
PY
)

mkdir -p "$output_dir"
overall=0
for task_id in "${task_ids[@]}"; do
  if [ -f "$output_dir/libero_10-task${task_id}-episode0.json" ]; then
    printf '%s skipped_existing\n' "$task_id" >> "$output_dir/worker-${worker_index}.status"
    continue
  fi
  extra_args=()
  if [ -n "$teacher_states_dir" ]; then
    extra_args=(--save-teacher-states-dir "$teacher_states_dir" --save-teacher-state-every 8 --save-teacher-images)
  fi
  "$python_bin" scripts/run_openvla_oft_single_episode.py \
    --checkpoint /data2/lty/models/openvla-7b-oft-finetuned-libero-plus-mixdata \
    --libero-root /data/lty/code/LIBERO-plus \
    --task-suite libero_10 \
    --task-id "$task_id" \
    --episode-id 0 \
    --gpu "$gpu" \
    --output-dir "$output_dir" \
    "${extra_args[@]}"
  code=$?
  printf '%s %s\n' "$task_id" "$code" >> "$output_dir/worker-${worker_index}.status"
  if [ "$code" -ne 0 ]; then
    overall=1
  fi
done
exit "$overall"
