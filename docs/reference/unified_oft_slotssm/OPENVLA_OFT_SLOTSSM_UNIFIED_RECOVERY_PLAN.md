# OpenVLA-OFT + SlotSSM 统一失败恢复模型实施计划

更新日期：2026-08-05

## 1. 正式技术决策

后续主线确定为一个可联合训练、单 checkpoint 部署的模型：

> OpenVLA-OFT 负责通用任务表征和正常动作；SlotSSM 提供持续对象级时序状态；动作特征通过 cross-attention 读取 SlotSSM；主策略头、恢复策略头和 failure gate 共同决定最终动作。

当前外部脚本中的固定触发、privileged 几何、强制候选和 simulator contact 只保留为数据采集 teacher、调试工具和可行性上界，不能作为最终在线方法。

## 2. 第一版模型结构

```text
当前双相机图像 + 语言 + proprio
                │
                ▼
          OpenVLA-OFT backbone
                │
 action hidden tokens H_act [B,56,4096]
                │                         图像历史 + proprio + 历史动作
                │                                      │
                │                                      ▼
                │                       persistent SlotSSM [B,16,512]
                │                                      │
                ├──────── Q      K,V ──────────────────┘
                │          Cross-Attention
                │                 │
                │             H_recovery
                │                 │
      ┌─────────┴─────────┬───────┴──────────┐
      ▼                   ▼                  ▼
OFT normal head     recovery action head   failure gate
      │                   │                  │
 A_normal [8,7]      A_recovery [8,7]   NORMAL/RECOVERY/DONE
      └───────────────────┴──────────────────┘
                          │
                    hard action routing
```

推荐的残差融合为：

\[
\Delta H=\operatorname{CrossAttn}(Q=H_{act},K=W_sS_t,V=W_sS_t),
\qquad H_{rec}=H_{act}+\alpha\Delta H.
\]

第一版中：

- OFT 对 LIBERO 每次产生 8 个 7 维动作，因此 action-token hidden state 为 `56×4096`；
- 现有 Stage-1 持续对象表示为 16 个 512 维 slot，cross-attention 直接以其为 K/V；
- `normal head` 直接加载并冻结 OpenVLA-OFT action head，读取原始 `H_act`；
- `recovery head` 从 normal head 权重复制初始化，读取 `H_rec`；
- `failure gate` 读取 pooled `H_act`、SlotSSM 状态、proprio 和历史动作；
- 在线采用硬切换，不对正常动作和恢复动作做数值平均；
- 恢复头预测 8×7 短 action chunk，但每次只执行前 1～2 步，然后重新观察和规划；
- gate 使用 `NORMAL / RECOVERY / RECOVERY_COMPLETE` 三状态和多帧确认，避免频繁抖动。

第一版不让 normal head 读取随机初始化的融合特征，从而保证接入新模块时正常策略行为不变。只有在恢复闭环通过后，才评估零初始化 gated adapter 是否也能提升正常动作。

## 3. 为什么先取消在线多候选系统

第一版恢复头直接学习成功的局部恢复 action chunk，不在在线阶段运行“候选生成器 + 独立评分器 + 独立验证器”多个模型。系统通过短视野闭环实现：

```text
检测失败 → 预测短恢复 chunk → 执行1～2步 → 重新观察
          ↑                              │
          └──────── 未恢复则继续 ────────┘
```

已有仿真候选分支仍然有用，但改为：

- 自动发现成功恢复轨迹；
- 为 recovery head 提供动作监督；
- 为 future-outcome 辅助损失提供成功/失败标签；
- 测量一个失败状态是否存在可恢复动作，而不是成为最终在线系统的人工选择环节。

如果确定性 L1 recovery head 出现模式平均或覆盖不足，再升级为统一 recovery decoder 内部的 K-mode proposal queries 或 diffusion head；这不是第一版阻塞项。

## 4. 数据协议

必须重新统一为 OpenVLA-OFT 格式，每个时刻至少包含：

- 主视角 RGB；
- 腕部 RGB；
- proprio；
- 语言任务；
- OpenVLA-OFT nominal action chunk；
- 实际执行动作；
- persistent role slots 和 SlotSSM 历史；
- causal failure state：`NORMAL / RECOVERY / RECOVERY_COMPLETE`；
- 恢复动作 chunk；
- 任务结果和恢复结果标签。

Simulator object pose、contact、segmentation ID 和成功 predicate 只允许生成 teacher label 或离线审计，禁止进入在线输入。训练、验证、测试必须按 `(task_id, episode/state_id)` 隔离；同一失败状态 fork 出的所有候选必须位于同一个 split。

现有 Task0、Task3、Task6、Task9 自然失败和候选分支可作为种子数据，但独立失败来源仍不足。需要在多个 initial states 上自动采集正常轨迹、自然失败和恢复分支，并保留两类样本：

- nominal replay：防止 gate 误触发和正常性能退化；
- recovery replay：训练失败进入、恢复动作和恢复退出。

## 5. 分阶段实施与验收

### Phase A：复现 OpenVLA-OFT 基线

1. 将官方 OpenVLA-OFT 代码和 LIBERO-10 checkpoint 接入当前环境；大文件放在 `/data2/lty/models`。
2. 验证双相机、proprio、动作归一化、8×7 chunk 和 chunk 中断接口。
3. 在 Task0、3、6、9 上跑相同 initial states，保存 metadata、steps 和 MP4。
4. 得到正式 OFT baseline 成功率；不能继续使用原始 OpenVLA 成功率代替 OFT baseline。

验收：checkpoint 推理可复现、动作尺度正确、视频稳定、结果可重复。

### Phase B：实现统一模型骨架

1. 提取 OFT 最后一层 action hidden tokens。
2. 将 persistent SlotSSM 输出投影到 OFT hidden dimension。
3. 实现零初始化 residual cross-attention adapter。
4. 加载并冻结 OFT normal head。
5. 复制 normal head 初始化 recovery head。
6. 实现三状态 failure gate 和 hard router。
7. 将所有新增模块保存为一个统一 checkpoint。

验收：初始化时 `A_normal` 与原 OFT 输出数值一致；cross-attention 有限值、可反传；关闭 gate 时在线轨迹不变。

### Phase C：训练 gate 和 recovery head

第一阶段冻结 OFT backbone 和 normal head，只训练：

- Slot projector / 必要的 SlotSSM 层；
- cross-attention adapter；
- recovery head；
- failure gate。

联合损失：

\[
\mathcal L=
\lambda_g\mathcal L_{gate}
+y_{rec}\lambda_a\lVert A_{rec}-A^*\rVert_1
+\lambda_n\mathcal L_{normal\ consistency}
+\lambda_f\mathcal L_{future/outcome}.
\]

训练先使用 teacher gate，再逐步加入 predicted-gate rollout，避免只在正确阶段输入下有效。必要时只对 OFT 最后若干层使用 LoRA，不全量微调 7B backbone。

验收：

- held-out failure precision、recall、F1 和检测延迟合格；
- held-out nominal false-positive rate 足够低；
- recovery action 的平移、旋转、夹爪和阶段性方向审计通过；
- normal action consistency 基本不退化；
- 所有线上输入通过 privileged leakage audit。

### Phase D：Shadow mode 与安全在线闭环

1. gate 先只记录建议，不修改动作，评估正常轨迹误触发。
2. recovery head 做 teacher-forced sequential replay，检查累积偏移和下降阶段安全性。
3. 在线启用 hard gate：发生失败时立即截断剩余 nominal chunk。
4. recovery 模式每次只执行 1～2 步，重新观察、更新 slots 和 SlotSSM。
5. 连续多帧 `RECOVERY_COMPLETE` 后才交回 normal head。
6. 恢复置信度过低时 abstain/reobserve，禁止盲目执行。

验收：首先在允许任务内过拟合的 Task3 空抓和掉落上达到自主闭环；触发时刻、恢复动作和退出判断均不能人工指定或读取 simulator state。

### Phase E：多任务量化评测

主对照：

1. OpenVLA-OFT；
2. OpenVLA-OFT 直接重试；
3. OpenVLA-OFT + failure-conditioned language prompt；
4. OpenVLA-OFT + 规则恢复；
5. 完整 OpenVLA-OFT + Failure-Aware SlotSSM。

主要指标：

- 端到端任务成功率及置信区间；
- 自然失败条件下的恢复成功率；
- gate precision/recall/F1、误接管率和检测延迟；
- 正常成功 episode 被破坏的比例；
- 恢复额外步数、推理延迟、碰撞和强制停止率；
- 按空抓、滑落、阶段停滞、放置失败分类的结果。

关键消融：

- 去掉 SlotSSM，只用当前帧；
- 去掉对象 slots，只用全局 OFT 特征；
- 去掉 cross-attention，简单拼接；
- recovery head 不从 normal head 初始化；
- 不截断 chunk / 不采用短视野闭环；
- 二状态 gate 与三状态 gate。

## 6. 当前代码可复用部分

- `prismatic/models/action_heads.py`：已有 OFT 风格 L1 / diffusion continuous action heads；
- `scripts/train_slot_stage3_openvla.py`：已有 SlotSSM、action queries 和连续 action path 原型；
- `scripts/smoke_slot_stage3_openvla.py`：已有 Slot token 注入 OpenVLA 的 smoke；
- Task0/3/6/9 的自然 rollout、恢复分支和视频：可用于数据转换、teacher label 和回归测试。

这些代码只作为实现材料。当前 Stage-3 token injection 不是本计划要求的 action-feature residual cross-attention；当前 Task3/9 开发视频也不是自主模型证据。

## 7. 立即执行顺序

1. 确认并下载适用于 LIBERO-10 的 OpenVLA-OFT checkpoint；
2. 在当前服务器复现 OFT Task3 与 Task0 baseline；
3. 建立 OFT observation/action-chunk 数据转换器并审计现有数据缺项；
4. 实现 `OFT action features ↔ SlotSSM` residual cross-attention；
5. 实现冻结 normal head、克隆 recovery head、三状态 gate；
6. 完成初始化等价性、梯度、checkpoint round-trip 单元测试；
7. 自动采集足量独立 recovery windows；
8. 启动第一轮 gate + recovery head 训练；
9. 通过离线和 shadow gate 后运行 Task3 自主闭环；
10. 扩展 Task0、6、9并形成成功率对比表。

### 2026-08-05 执行状态

- 已下载官方 `openvla-7b-oft-finetuned-libero-10` checkpoint 到 `/data2/lty/models/`；
- 已建立隔离的 `openvla-oft` 环境并固定官方双向注意力 Transformers fork；
- 已完成可指定 task/episode 的 OFT 单回合 runner；
- LIBERO-10 Task3 Episode0 已由官方 OFT 成功完成，视频与 metadata 已保存；
- 已实现 frozen normal head、复制初始化 recovery head、Slot cross-attention、三状态 gate 和 hard router；
- 已通过单元测试及真实 OFT action-head 兼容测试：初始化时 normal/recovery 输出均与原动作头逐值一致；
- 已在同一真实 LIBERO 双相机观测上完成 Stage-1/OFT 特征桥接：OFT vision patches 为 `1×512×2176`，现有 Stage-1 checkpoint 输出 `1×16×512` slots，OFT action hidden 为 `1×56×4096`，统一 adapter 前向全部为有限值；
- 已接入真实 Stage-2 Mamba2 checkpoint：`32` 帧 patch/slot 序列产生 `32×16×512` 时序状态和 next-slot 预测，并已送入统一 adapter。

Phase B 的核心特征链已完成。当前下一项是审计并转换恢复数据为统一 OFT 格式，随后固化统一在线 runner，并启动 gate/recovery 训练。

### 首批双相机恢复数据（2026-08-05）

- 已升级 `BranchTrajectoryWriter`：新采集分支同时保存主视角 RGB、腕视角 RGB、proprio、执行动作和仅离线使用的 teacher outcome；
- 已从 Task6 的 4 个独立自然失败源重采集 6 条恢复分支（4 条末端稳定 regrasp，2 条末端失败）；
- 已转换为 66 个因果的 `8×7` OFT 恢复动作监督窗口，输出位于 `/data2/lty/naive-e-slotssm-runs/oft-dual-recovery-seed-v1/oft-recovery-windows.jsonl`；
- 已以冻结的官方 OFT 和现有 Stage-1/Stage-2 checkpoint 预计算 66 组真实 `56×4096` OFT action hidden 与 `16×512` SlotSSM state；这不是合成 token；
- 已做完 OFT 动作空间审计：action head 使用归一化动作，而环境执行使用反归一化动作；已将缓存 target 转为 action-head 标准空间并保留原文件备份。早期 v1 恢复 checkpoint 因尺度错误仅作排错记录，禁止在线使用；
- 已完成修正后的 recovery-only smoke training：恢复动作头和 slot cross-attention 在 44 个末端稳定 regrasp 窗口上训练 400 步，完整正样本集损失为 `0.0362`。有效 checkpoint/report 位于 `/data2/lty/naive-e-slotssm-runs/oft-slotssm-recovery-head-smoke-v2/`；
- 已采集 Task6 ep0--3 的 4 条原始 OFT 成功双相机轨迹（945 步），转换为 116 个 NORMAL 窗口；与 66 个恢复窗口联合训练后，统一 v2 adapter 位于 `/data2/lty/naive-e-slotssm-runs/oft-slotssm-unified-smoke-v2/`。其训练窗口混淆矩阵为对角，但尚未做 episode-independent 验证；
- 已实现不读取 simulator failure state 的在线 LIBERO runner：每一观测均产生 OFT action hidden、当前 slot 与因果 Mamba2 state，由 learned gate 选择 normal/recovery branch；RECOVERY 分支强制单步重观测。Task6 ep7 的 4 个真实策略步 smoke 已跑通，gate 自主输出 4 次 NORMAL，视频为 `/data2/lty/naive-e-slotssm-runs/oft-slotssm-unified-online-smoke-v1/task6-episode7.mp4`。当前实现每帧均完整重算 OFT，正确但慢，完整评测前需加入合理缓存或降低 gate query 频率。
- 此训练刻意没有训练 gate：当前数据没有 NORMAL 标签，若硬训会把 gate 错误地学成总是恢复。下一阶段必须先收集并合并正常双相机轨迹，再以独立 episode 划分训练/验证 gate。
- 该集合只够 recovery-head overfit/smoke，不能用于宣称可靠 failure gate。gate 仍需要补充 nominal 双相机轨迹和更多彼此独立的失败源。

## 8. 成功标准与停止条件

这条路线成立的最低证据是：

- 正常策略成功率无显著下降；
- failure gate 在 held-out 自然轨迹上有低误报和足够召回；
- Task3 等任务中，不使用 privileged online input、不固定触发、不强制恢复动作仍能完成恢复；
- 完整系统相对 OpenVLA-OFT baseline 显著提高端到端成功率。

如果自动恢复仍依赖 privileged pose、人工触发或人工选定动作，只能报告为 oracle-assisted upper bound，不能称为统一自主恢复模型。如果 OFT baseline 在标准 LIBERO 接近饱和，则转向更难的自然失败任务或鲁棒性评测，而不是人为破坏已经稳定成功的轨迹。
