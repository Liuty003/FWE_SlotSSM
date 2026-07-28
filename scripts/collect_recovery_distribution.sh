#!/usr/bin/env bash
set -euo pipefail

ROOT=/data2/lty/naive-e-slotssm-runs/recovery-distribution-v1
SLOT=/data2/lty/naive-e-slotssm-runs/stage1-adapters/openvla-7b-finetuned-libero-spatial+libero_mem+b1+lr-1e-05--full10tasks-h16-sync6-s20k-eb6-no-gpu3--image_aug--multiview1000--object/pretrained_slots-wo-filters_s16.safetensors
ADV=/data2/lty/naive-e-slotssm-runs/task-aware-advantage-scorer-e0e4-h4/advantage-best-gated.safetensors
TARGET=/data2/lty/naive-e-slotssm-runs/target-state-estimator-failure-aware-v3/target-state-best.safetensors

mkdir -p "$ROOT"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for EPISODE in 5 6; do
  .conda/envs/openvla-a/bin/python experiments/robot/libero/run_advantage_openvla_online_stable.py \
    --model artifacts/checkpoints/openvla-7b-finetuned-libero-spatial \
    --slot-checkpoint "$SLOT" --advantage-checkpoint "$ADV" --target-state-checkpoint "$TARGET" \
    --task-suite libero_spatial --task-id 9 --episode-id "$EPISODE" --max-steps 220 \
    --device-map balanced --intervention-warmup 999 --enable-target-recovery \
    --recovery-mode replan --recovery-max-steps 80 --lift-threshold .8 --lift-confirm-steps 3 \
    --target-approach-m .10 --target-departure-m .16 --max-recovery-attempts 1 \
    --save-slot-trace --output "$ROOT/task9-episode${EPISODE}.mp4" \
    --rollout-dir "$ROOT/task9-episode${EPISODE}"
done
