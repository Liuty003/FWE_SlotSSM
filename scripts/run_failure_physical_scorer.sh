#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"

.conda/envs/openvla-a/bin/python scripts/train_failure_physical_scorer.py \
  --data-dir /data2/lty/naive-e-slotssm-runs/counterfactual-slots-spatial-e0e4-h4-physical \
  --output-dir /data2/lty/naive-e-slotssm-runs/failure-physical-scorer-ranking-e0e4-h4 \
  --train-tasks 0,1,2,3,4,5,6,7 \
  --train-episodes 0,1,2,3 \
  --val-task 8 \
  --val-episodes 0,1,2,3,4 \
  --steps 5000 \
  --batch-forks 16 \
  --lr 2e-4 \
  --physical-weight 1.0 \
  --interaction-bce-weight 0.5 \
  --failure-bce-weight 0.5 \
  --ranking-weight 0.5 \
  --ranking-temperature-m 0.002 \
  --log-every 20 \
  --eval-every 200 \
  --save-every 500
