# 运行资产清理记录（2026-08-30）

## 保留原则

- 保留 `/data2/lty/naive-e-slotssm-runs/libero-plus/oft-plus-robot-levels1-5-v1`，以及同目录下拆分保存的 level 1–4、level 5 基线。
- 保留 `libero-plus-global-v1/full-audit-v1` 的 10,030 条稠密审计底座。
- 保留 `libero-plus-global-v1/libero-10-recovery-v1` 当前 Gate、Recovery、teacher、handoff 和模型资产。
- 保留最终 checkpoint、配置、报告、最终完整评测及其成功/失败视频。
- 保留 simulator-verified 成功 teacher、最终训练数据、task fork、manifest 和论文复现所需 provenance。
- 保留失败的最终评测报告；它们是严格成功率和失败分析的必要证据。

## 第一轮删除范围

删除 37 个精确子目录，删除前合计 `258,594,365,440` bytes（`240.84 GiB`）。这些目录均属于以下类别：

- 已被当前 V4.4.2 H=1 数据取代的黑碗 stride-1 旧特征；
- 黑碗旧 DAgger 特征和已被 detailed-v1 取代的 phase-v1；teacher、source、最终 detailed 模型和评测仍保留；
- 明确标记为 `failed-rgbd`、`failed-nominal-rgbd` 或 `aborted-partials` 的失败原始采集；
- Book34 已被最终 r2、保留 r6/r7 和 task-specific selector 取代的 V4.2/r3/r4/r5 特征；
- V3 action-aware Book 的大体积旧特征；原始轨迹、canonical teacher、模型和成功 placement search 仍保留；
- 旧 book-local-repair 的重复 features/on-policy gate cache；teacher、FSM 分支、checkpoint、报告和评测仍保留。

## 清理后复核

- 根目录清理前：`1,494,938,648,576` bytes。
- 根目录清理后：`1,236,344,283,136` bytes。
- 实际释放：`258,594,365,440` bytes（`240.84 GiB`）。
- `/data2` 文件系统占用率由 42% 降为 41%。
- 指定的 `oft-plus-robot-levels1-5-v1` 已验证存在。
- Black Bowl 最终评测的 44 个 `v4.mp4` 已验证存在。
- Mug + Pudding `online-eval` 下的 51 个 `v4.mp4` 已验证存在。
- 三个历史任务最终 checkpoint、Book34 selector、OFT+ task forks、成功 teacher source 报告和 LIBERO-10 当前 H=1 action checkpoint 均已验证存在。

## 明确保留的论文主资产

- 全局 LIBERO-Plus：`/data2/lty/naive-e-slotssm-runs/libero-plus-global-v1`
- 指定 OFT+ 基线：`/data2/lty/naive-e-slotssm-runs/libero-plus/oft-plus-robot-levels1-5-v1`
- Book34 最终资产：`/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book`
- Mug + Pudding 最终资产：`/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/mug-pudding/counterfactual-factory-unified-v1`
- Black Bowl 最终资产：`/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/black-bowl-bottom-drawer/detailed-v1`
- 旧 Book teacher/FSM/checkpoint：`/data2/lty/naive-e-slotssm-runs/oft-plus-book-local-repair-v1`
- OFT+ task forks：`/data2/lty/naive-e-slotssm-runs/oft-plus-robot-global-v1`

更细的最终 checkpoint、成绩和视频路径仍以 `ARTIFACT_INDEX.md` 为准。
