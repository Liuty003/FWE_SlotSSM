#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
.conda/envs/openvla-a/bin/python scripts/train_task_aware_advantage_scorer.py \
  --data-dir /data2/lty/naive-e-slotssm-runs/counterfactual-slots-spatial-e0e4-h4-physical \
  --progress-dir /data2/lty/naive-e-slotssm-runs/counterfactual-task-progress-e0e4-h4 \
  --output-dir /data2/lty/naive-e-slotssm-runs/task-aware-advantage-scorer-e0e4-h4 \
  --train-tasks 0,1,2,3,4,5,6,7 --train-episodes 0,1,2,3 \
  --val-task 8 --val-episodes 0,1,2,3,4 --steps 5000 --batch-forks 16 --lr 2e-4 \
  --advantage-weight 1.0 --beat-nominal-weight 0.5 --ranking-weight 1.0 \
  --physical-weight 0.25 --classification-weight 0.2 --beat-margin-m 0.001 \
  --eval-every 200 --save-every 500 --log-every 20
