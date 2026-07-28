# Unified Recovery Package

This package is the new implementation path for the shared Recovery SlotSSM.
The existing `experiments/robot/libero/run_single_scene_recovery.py` remains a
teacher, data collector, and modular baseline.

The data contract separates every transition into:

- `online_inputs`: an explicit allowlist of quantities available to the deployed
  policy;
- `labels`: training targets and observable rollout outcomes;
- `teacher_only`: simulator state that may generate labels but must never enter
  policy inputs;
- `provenance`: source and conversion limitations.

Current conversion outputs:

```text
/data2/lty/naive-e-slotssm-runs/recovery-unified-sequences-v1
/data2/lty/naive-e-slotssm-runs/recovery-unified-online-traces-v1
```

The first contains seven legacy expert episodes with aligned RGB, depth, and
target masks. The second contains selected online traces for episodes 10, 11,
and 13. Those traces provide richer temporal and hard-negative labels but do
not contain aligned per-step RGB/depth files, so they must not be presented as
full visual training episodes.

Run the contract tests with:

```bash
.conda/envs/openvla-a/bin/python -m unittest \
  recovery.tests.test_data_contract -v
```
