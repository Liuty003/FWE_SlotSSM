#!/usr/bin/env bash
# Launch a reproducible LIBERO-Plus Robot Initial States sweep for levels 1--4.
# One OpenVLA-OFT+ process is kept active per GPU; each worker evaluates its
# assigned tasks serially so the 7B policy is never loaded concurrently on a GPU.

set -uo pipefail

OUTPUT_DIR="${1:?usage: $0 OUTPUT_DIR}"
LIBERO_ROOT="/data/lty/code/LIBERO-plus"
CLASSIFICATION_JSON="$LIBERO_ROOT/libero/libero/benchmark/task_classification.json"
CHECKPOINT="/data2/lty/models/openvla-7b-oft-finetuned-libero-plus-mixdata"
RUNNER="/data/lty/code/naive-e-slotssm/scripts/run_openvla_oft_single_episode.py"
PYTHON="/data/lty/code/naive-e-slotssm/.conda/envs/openvla-oft/bin/python"
CONFIG_DIR="/data2/lty/naive-e-slotssm-runs/libero-plus/config"

mkdir -p "$OUTPUT_DIR/logs"
for level in 1 2 3 4; do
    mkdir -p "$OUTPUT_DIR/level${level}"
done

mapfile -t entries < <(
    for level in 1 2 3 4; do
        jq -r --argjson level "$level" \
            '.libero_10[] | select(.category == "Robot Initial States" and .difficulty_level == $level) | "\($level):\(.id - 1)"' \
            "$CLASSIFICATION_JSON" | sort -t: -k2,2n
    done
)

printf '%s\n' "${entries[@]}" > "$OUTPUT_DIR/logs/task_manifest_level1-4.txt"
printf '%s START total=%s workers=7\n' "$(date -Iseconds)" "${#entries[@]}" >> "$OUTPUT_DIR/logs/scheduler.log"

run_worker() {
    local gpu="$1"
    local log="$OUTPUT_DIR/logs/gpu${gpu}.log"
    local idx entry level task status

    for idx in "${!entries[@]}"; do
        if (( idx % 7 != gpu - 1 )); then
            continue
        fi
        entry="${entries[$idx]}"
        level="${entry%%:*}"
        task="${entry##*:}"
        printf '%s START level=%s task=%s gpu=%s\n' "$(date -Iseconds)" "$level" "$task" "$gpu" >> "$log"
        LIBERO_CONFIG_PATH="$CONFIG_DIR" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false \
            "$PYTHON" "$RUNNER" \
            --gpu "$gpu" --libero-root "$LIBERO_ROOT" --checkpoint "$CHECKPOINT" \
            --task-suite libero_10 --task-id "$task" --episode-id 0 --open-loop-steps 8 \
            --output-dir "$OUTPUT_DIR/level${level}" >> "$log" 2>&1
        status=$?
        printf '%s END level=%s task=%s gpu=%s status=%s\n' "$(date -Iseconds)" "$level" "$task" "$gpu" "$status" >> "$log"
    done
}

for gpu in 1 2 3 4 5 6 7; do
    run_worker "$gpu" &
done
wait
printf '%s COMPLETE\n' "$(date -Iseconds)" >> "$OUTPUT_DIR/logs/scheduler.log"
