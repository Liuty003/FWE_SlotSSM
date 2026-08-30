# OpenVLA-OFT + SlotSSM：后续执行流程

更新：2026-08-06。本文是当前唯一执行流程。它只覆盖 OpenVLA-OFT action
feature 与 SlotSSM 状态融合的统一三头模型；早期 OpenVLA、候选宏、人工触发和
特权在线状态方案均不在本流程内。

## 0. 固定问题定义与边界

部署时的输入只能是任务语言、主/腕 RGB、robot proprio、历史观测和历史实际执行
动作。模型输出为 `NORMAL / RECOVERY / RECOVERY_COMPLETE`，以及 normal/recovery
两个 `8 x 7` OFT 归一化 action chunk。

```text
OFT action hidden H_act [56, 4096] ─────────────► frozen normal head
             │
             └─ cross-attend SlotSSM Z_t [16, 512] ─► H_rec ─► recovery head

mean(H_act), mean(Z_t), proprio, action history ───────────────► 3-state gate
```

- normal head 从官方 OFT checkpoint 加载并保持冻结；它始终读取原始 `H_act`。
- recovery head 深拷贝 normal head 后初始化，读取 `H_rec`。
- cross-attention 输出投影零初始化，所以开始训练前 recovery 输出与 normal 输出相同。
- simulator pose/contact/segmentation 和人工选定 failure timestep 仅能离线生成标签、
  分支 teacher 与审计结果，不能成为线上输入或触发条件。
- action-head 监督和环境执行必须严格区分：训练 target 使用 OFT normalized action；
  环境执行前才调用 `_unnormalize_actions(...)`。

## 1. 先完成 OFT 自然失败筛选

**目标：** 选出至少一个具有多个可复现自然失败的 OFT 任务，作为主训练与评测任务。

1. 用官方 OFT 录制 Task7 ep4--9、Task8 ep0--9；每回合保存 MP4、metadata 和完整
   `steps.jsonl`。
2. 汇总每 task 的成功率、失败 episode、失败类型和视频路径；按行为而不是 simulator
   内部原因标注，例如 transport collision、stagnation、empty acquisition、placement
   failure。
3. 选择有多个自然失败且可恢复的任务。Task9 ep4 可保留为 collision case study，
   但 OFT 当前 9/10 的结果不足以单独作为成功率提升主实验。

**验收：** 每个 episode 可重放、可审计；主任务的选择来自 OFT 而非旧 OpenVLA 基线。

## 2. 完善线上 controller 的因果状态机

**目标：** normal chunk 缓存不能使视觉/SlotSSM 状态停在旧帧；gate 也不能逐帧
argmax 后直接抖动切换。

1. 每一个环境 observation 都更新 Stage-1 slots 与 persistent SlotSSM，即使当前动作
   来自已缓存的 normal chunk；OFT action head 可按 chunk 频率运行，但时序状态不可
   漏帧。
2. gate 输入扩展为 pooled `H_act`、`Z_t`、当前 proprio 与历史实际执行动作，并明确
   训练/部署时的历史长度和归一化方式。
3. 实现显式状态机：
   - `NORMAL`：执行 normal chunk；若 gate 确认 `RECOVERY`，立即清空剩余 chunk；
   - `RECOVERY`：只执行 recovery chunk 的首个动作，随后强制重观测；
   - `RECOVERY_COMPLETE`：连续确认若干帧后才交回 `NORMAL`；
   - gate 低置信度或状态不一致：reobserve/abstain，不盲目发 recovery action。
4. 为 gate decision、置信度、执行/截断的 action、状态转换原因写入 rollout log。

**验收：** 关闭/恒定 NORMAL gate 时，OFT 轨迹与官方 baseline 一致；开启 state machine
时每个控制步的 SlotSSM state 都对应当前 observation；recovery 从不执行超过规定的开环步数。

## 3. 采集 failure-to-recovery 数据

**目标：** 获得可按源 episode 隔离的、真实 OFT 格式的训练样本。

1. 为选定主任务的官方 OFT recorder 补齐失败轨迹保存：每步主/腕 RGB、proprio、OFT
   raw/planned action chunk、environment executed action、语言、时间戳和 MP4。
2. 从每个失败轨迹中选择多个由可观察现象定义的 fork 点（如接近入口、首次碰撞、持续
   无进展），而不是固定人工 failure timestep。
3. 对每个 fork 离线运行多个 teacher branch。保存成功和失败 branch、完整执行动作、
   恢复完成/任务成功标签；接触、pose、是否进入目标区域仅写入 teacher 字段。
4. 同时采集独立 OFT nominal successful episodes，作为 NORMAL gate 样本和正常能力
   回归集。
5. 在 manifest 中绑定所有同一 source episode/state 的 fork 与 candidates；split 的最小
   单位是 source episode/failure group，绝不随机拆 window。

**验收：** 每条训练窗口都可追溯原始 rollout、fork 与 action 空间；线上字段通过
privileged-leakage audit；train/val/test 没有 source episode 泄漏。

## 4. 特征预计算、训练与离线评估

**训练顺序：**

1. 用冻结的官方 OFT、Stage-1 和 Stage-2 为完整轨迹预计算真实 `H_act` 与因果 `Z_t`。
2. 先训练 recovery head + cross-attention：只模仿成功 branch 的 normalized recovery
   chunks；失败 branch 不作为动作 teacher。
3. 以 nominal、recovery 和完成窗口训练三状态 gate；可加入 future/outcome 辅助损失。
4. 先使用 teacher-gate 的序列训练，再逐步加入 predicted-gate rollout，避免训练只依赖
   完美切换时刻。
5. OFT backbone 与 normal head 在第一轮均冻结。只有上述闭环通过后，才评估最后层 LoRA
   或 normal branch 的 gated fusion。

建议的第一轮损失：

```text
L = lambda_gate * CE(gate)
  + is_successful_recovery * lambda_action * SmoothL1(A_rec, A_teacher)
  + lambda_normal * consistency(A_normal, frozen_OFT)
  + lambda_outcome * future_or_outcome_loss
```

**离线验收（必须按 held-out episode 报告）：**

- gate：每类 precision/recall/F1、nominal false-positive rate、detection delay；
- recovery：translation/rotation/gripper 的 action-scale 审计，以及短闭环 replay；
- normal：官方 OFT 与 frozen normal head 的逐值一致性和 nominal 轨迹回归；
- checkpoint：adapter-only 保存/加载后输出一致。

## 5. 从 shadow mode 到在线闭环

1. **Shadow：** 执行官方 OFT 动作，只记录 gate 和 recovery 建议；先验证正常成功
   trajectory 的误接管率。
2. **Teacher-forced replay：** 按真实观测序列滚动 recovery head，检查跨步偏移、夹爪
   时序和安全边界。
3. **Hard routing：** gate 确认 RECOVERY 后截断 normal chunk；每次 recovery 单步执行，
   重新观察、更新 `Z_t`、再次判定。
4. **退出：** 只有连续 `RECOVERY_COMPLETE` 才回到 normal；低置信度维持重观测/保守停止。

**验收：** 触发、动作和退出均仅由部署可见输入决定；视频与 JSON log 能完整还原每次模式
切换。

## 6. 正式评测与消融

对每个初始状态配对运行：官方 OFT、OFT direct retry、完整 unified model。报告端到端
成功率及置信区间、自然失败条件下恢复成功率、额外步数、碰撞/卡死率、正常成功 episode
被破坏比例，以及 gate 指标。视频必须同时包含 baseline failure 和 unified recovery。

必做消融：无 Mamba2（当前帧 slots）、无 slots（仅 OFT feature）、无 cross-attention、
recovery head 不复制 normal head、以及不截断 normal chunk。所有结论都以 OFT baseline
为参照，不能引用旧 OpenVLA 结果替代。

## 当前优先级

1. 完成 Task7/8 OFT failure audit；
2. 修正 runner 的逐 observation SlotSSM 更新和 gate state machine；
3. 为选定主任务录制可 fork 的 OFT failure trajectories；
4. 再启动 episode-disjoint 的 recovery/gate 训练与 shadow evaluation。
