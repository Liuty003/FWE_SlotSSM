# 下一步：统一 SlotSSM 失败检测与恢复闭环

更新：2026-08-03。

## 目标与边界

第一版只解决两类可恢复桌面失败：漏抓/空夹，以及目标物在运输中掉回可达桌面后的重新抓取。
不把翻倒扶正、不可达掉落、长程重新规划或微波炉关门混入第一版。

最终第一版的在线闭环是：

```text
持续角色 Slot + SlotSSM memory
  -> 单标量风险 p_fail
  -> <RECOVER> 通用重抓策略产生 3 个候选
  -> 动作条件短视界 future / 同一风险模型排序
  -> 仅执行 1--2 步，重新观测并验证
  -> 恢复后交还冻结 OpenVLA 主策略
```

OpenVLA 暂保持冻结；当前 slot 不会伪装成已经接入 OpenVLA 内部 cross-attention。
恢复 adapter、风险模型和 dynamics 读取 slot/视觉/proprio，teacher simulator state 仅用于标签。

## 当前可用资产

- `libero10-natural-baseline-v1`：100 条自然 OpenVLA rollout，其中 51 条失败；是主评测基线。
- Task6：ep1 空夹、ep6 边缘滑脱已有视觉触发、可观测几何候选和成功恢复视频。
- Task9：ep5 运输掉落、ep6 空抓已有候选分支；当前候选几何仍有 privileged teacher 成分。
- `natural-recovery-training-v1`：4 个自然失败状态、16 条匹配候选分支、视频、动作宏和终态 teacher outcome。
- Task6 ep7 t=211：新增一个独立的自然漏抓/边缘滑脱状态，4 条对齐分支；3 条稳定夹持并抬升约 2.5--2.7 cm，1 条重抓后脱落。它为候选风险排序提供同一状态下的正、负结果。
- Task9 ep1 t=180：新增不稳定抓取/碰撞后掉落状态，4 条对齐分支；其中 1 条重新稳定抓起并抬升约 2.9 cm，3 条未能保持夹持。完整 Task9 的放入微波炉、关门尚不计入这批分支的恢复成功。
- `RecoverySlotSSM` / unified candidate head：已有共享时序 memory 的最小代码，但数据量不足以证明泛化。

## 阶段 1：正式恢复序列采集（已完成，2026-08-03）

升级 `collect_natural_recovery_training.py` 及 task6/task9 branch collector，使每个候选的每一步保存：

- 对齐 RGB、depth、目标/角色 mask；
- proprio、主策略 nominal action、实际 recovery action、上一步动作；
- candidate/group/state identity 与动作块位置；
- 非特权在线输入的有效性标记；
- 仅 teacher 的 contact、attached、lifted、slip、终态 outcome。

先写入 raw branch transition，再在离线 replay 中计算每一步冻结 OpenVLA nominal action
并转换为 schema-v2；不得把 recovery action 冒充为 counterfactual observation 的 nominal
action。数据按**源 episode**而非随机帧划分；同时保留候选宏和 MP4。slot 与 SlotSSM
memory 不在采集时硬编码，而对该对齐数据进行可复现 enrichment。

验收已通过：4 组/16 分支、880 步；全部 RGB/depth/mask 路径存在，全部 mask 可见；
raw `online_observation` 未含 teacher pose/contact。产物：

```text
/data2/lty/naive-e-slotssm-runs/natural-recovery-sequences-v1
```

## 阶段 2：Slot / memory enrichment 与风险数据（进行中）

1. 对阶段 1 transition 批量计算 grounded role slots 与连续 SlotSSM memory；
2. 写入当前 slot、next slot 和可观测 target-to-EEF relation；
3. 用自然成功 rollout 作风险负例，用自然失败窗口和恢复分支作正例；
4. 训练单标量 `p_fail`，重点报告 episode 级误触发、提前量和恢复窗口召回，不用帧准确率掩盖问题。

先在 Task6 开发集完成，保留独立 episode 做最终测试；随后才迁移至 Task9。

当前已使用冻结的 Task6 三角色 slot encoder 提取 12 条 Task6 恢复分支（600 步，源
episode 为 1、6、7），每条得到`[T, 3, 96]` role slots及角色几何。提取工具支持合并多个
raw root，同时在 metadata 中保留来源；Task9 ep1 的四条 raw branch 已采集并审计，待使用其
对应的视觉角色 encoder 做同构 enrichment。这只是 enrichment，不用这些分支反过来训练 slot
encoder。

已完成三个不夸大的 dynamics smoke：一步预测不如 identity（应弃用）；16-step action
block 在 ep1 训练、ep6 验证时仅比 identity 低约 3--5%；扩展后使用 ep1+ep6 训练、完全
留出 ep7 验证（272 / 136 个 transition），最佳点总误差比 identity 低约 6.5%（cosine
约 5.8%，Smooth-L1 约 8.2%）。这说明动作块中存在可学习的 slot effect，但只有 3 个
源失败状态，仍远远不足以接入风险模型。下一步继续扩展**独立自然失败状态**，并保留每个
状态内的成功、未抓起和再脱落候选；不在这些高度相关分支上训练单标量风险头或 Recovery
LoRA。

## 阶段 3：动作条件 future 与统一候选风险

训练短视界 `D_psi(slot_memory, candidate_action, proprio)`，预测 candidate 后对象状态/关系。
风险模型同时评估当前状态和 predicted future；候选按最小风险选择。

第一版固定 3 个候选、每次只执行 1--2 个控制步，再观察和重评，避免一次执行整段手工宏。
对照必须包含：无恢复、direct retry、随机候选、当前独立 value scorer、预测风险选择。

## 阶段 4：通用 Recovery Adapter

冻结 OpenVLA 与共享视觉/slot 主干，只训练 `<RECOVER>` adapter/LoRA，学习：

```text
回到目标上方 -> 对齐 -> 下探 -> 闭爪 -> 稳定抬升
```

行为克隆为主，attached/lifted 作为辅助监督。仅当阶段 1--3 数据和风险门槛通过后开始；16 条分支只能做接口 smoke test，不能作为正式训练集。

## 阶段 5：闭环评测与迁移

1. Task6 自然失败上冻结开发方案，报告失败检测和最终任务成功率；
2. Task9 用视觉角色 slot 替换 privileged candidate geometry，评测掉落和空抓；
3. 只证明“通用重新抓取”跨任务可复用，不将 Task9 的运输、插入、关门等新技能算作恢复策略成功。

## 当前明确不做

- 不修改 OpenVLA 内部结构来声称 slot cross-attention 已完成；
- 不把 simulator pose/contact 当作在线输入；
- 不用同一失败状态的 train/validation 数字主张泛化；
- 不在数据量极小时解冻 SlotSSM 或训练长程端到端策略；
- 不把 Task9 的 oracle 门状态机计入学习化恢复结果。
