#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-/data2/lty/naive-e-slotssm-runs}"
SHARD_INDEX="${SHARD_INDEX:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
GPU_ID="${GPU_ID:-0}"
COLLECTION_TAG="${COLLECTION_TAG:-v1}"

ROLLOUT_ROOT="$RUN_ROOT/recovery-candidate-rollouts-$COLLECTION_TAG"
EXPERT_ROOT="$RUN_ROOT/recovery-candidate-expert-$COLLECTION_TAG"
CANDIDATE_ROOT="$RUN_ROOT/recovery-candidates-$COLLECTION_TAG"
mkdir -p "$ROLLOUT_ROOT" "$EXPERT_ROOT" "$CANDIDATE_ROOT"
cd "$REPO_ROOT"

# dataset_id init_state split
SPECS=(
  "9100 9 train"
  "9101 10 train"
  "9102 11 validation"
  "9103 0 train"
)

collect_one() {
  local dataset_id="$1"
  local init_state="$2"
  local split="$3"
  local stem="task9-dataset${dataset_id}-init${init_state}"
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  MUJOCO_GL=egl \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
    .conda/envs/openvla-a/bin/python \
    experiments/robot/libero/run_single_scene_recovery.py \
    --task-id 9 --episode-id "$init_state" \
    --dataset-episode-id "$dataset_id" --seed 0 \
    --output "$ROLLOUT_ROOT/${stem}.mp4" \
    --rollout-dir "$ROLLOUT_ROOT/${stem}" \
    --expert-dataset-dir "$EXPERT_ROOT/${stem}" \
    --candidate-dataset-dir "$CANDIDATE_ROOT/${stem}" \
    --candidate-split "$split" \
    --candidate-max-forks 12 --candidate-collect-every 8 \
    --candidate-branch-steps 4 \
    --mistake-trigger visual-lift --injection-lift-threshold-px 15 \
    --mistake-steps 7 --mistake-action "0,0,0,0,0,0,-1" \
    --mistake-label "LOST GRASP" \
    --failure-detector stage-b-tracked --visual-lift-threshold-px 20 \
    --oracle-regrasp \
    --learned-recovery-checkpoint \
      "$RUN_ROOT/recovery-residual-smoke-v6-geometry/recovery-residual-mlp.safetensors" \
    --learned-phase-checkpoint \
      "$RUN_ROOT/recovery-phase-predictor-v2-dagger/recovery-phase-predictor.safetensors" \
    --visual-regrasp-checkpoint \
      "$RUN_ROOT/visual-regrasp-verifier-v10-temporal/visual-regrasp-verifier.safetensors" \
    --visual-regrasp-confirm-steps 3 \
    --visual-regrasp-settle-height-m 0.04 \
    --visual-3d-audit "$RUN_ROOT/recovery-visual-3d-audit-v2/report.json" \
    --max-steps 230 --device-map balanced_low_0 --gpu-memory-gib 20
  local status=$?
  echo "dataset_id=$dataset_id init_state=$init_state split=$split status=$status"
  # A failed task can still contain valid matched candidate supervision.
  return 0
}

for index in "${!SPECS[@]}"; do
  if (( index % NUM_SHARDS != SHARD_INDEX )); then
    continue
  fi
  read -r dataset_id init_state split <<<"${SPECS[$index]}"
  collect_one "$dataset_id" "$init_state" "$split"
done
