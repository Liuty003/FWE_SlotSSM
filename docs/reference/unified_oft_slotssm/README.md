# 当前主线：OpenVLA-OFT + Failure-Aware SlotSSM

此目录是当前有效设计的唯一入口。模型以官方 OpenVLA-OFT 为冻结主策略，使用
SlotSSM 时序 slots 经 residual cross-attention 生成 recovery feature，并通过
normal head、recovery head 与三状态 gate 实现统一部署。

阅读和执行顺序：

1. [NEXT_STEPS.md](NEXT_STEPS.md)：接下来实际要做的流程、验收与优先级。
2. [HANDOFF_2026-08-06_UNIFIED_OFT_SLOTSSM.md](HANDOFF_2026-08-06_UNIFIED_OFT_SLOTSSM.md)：当前可复现资产、有效 checkpoint 和已知边界。
3. [OPENVLA_OFT_SLOTSSM_UNIFIED_RECOVERY_PLAN.md](OPENVLA_OFT_SLOTSSM_UNIFIED_RECOVERY_PLAN.md)：模型接口、数据协议和完整评测规范。
4. [OFT_NATURAL_FAILURE_AUDIT_2026-08-06.md](OFT_NATURAL_FAILURE_AUDIT_2026-08-06.md)：最新 OFT Task7/8 失败筛选结果。

代码入口：

- `recovery/models/openvla_oft_slotssm_adapter.py`
- `scripts/run_oft_slotssm_unified_episode.py`
- `scripts/precompute_oft_recovery_window_features.py`
- `scripts/train_oft_slotssm_recovery_smoke.py`
- `scripts/train_oft_slotssm_gate_smoke.py`

`../legacy_openvla/` 保存 2026-08-05 前的 OpenVLA、候选宏、人工触发和早期复现资料，
只供追溯，不作为当前实现或实验结论的依据。
