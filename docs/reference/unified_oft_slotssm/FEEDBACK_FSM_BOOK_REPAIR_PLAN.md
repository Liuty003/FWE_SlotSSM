# 自动反馈 FSM 修复数据计划（book → caddy）

## 目的

不使用人类遥操作。当前固定 teacher 分支只会执行一次预先算好的
「回退—对齐—下探—闭爪—抬升」动作序列，导致同一个 fork 的数据近乎重复。
新的 offline feedback FSM 在每个动作后读取仿真真值作为 *teacher-only* 反馈：
它会根据当前夹爪/书本相对位置重对齐，根据接触判定是否闭爪成功，未抓住时回退并换
横向接近偏移后重试。

在线模型依旧只接收 RGB、wrist RGB、语言、proprio、先前执行动作；任何 object pose、
contact、FSM phase 都不得作为线上输入。

## FSM

`BACKOFF → ALIGN → DESCEND → CLOSE → LIFT_VERIFY`：

- `BACKOFF`：张开并上移到书本上方的安全高度；
- `ALIGN`：根据每步真值相对位姿闭环调节 x/y；
- `DESCEND`：闭环下探至抓取高度；
- `CLOSE`：闭爪并以接触验证；
- `LIFT_VERIFY`：抓住后抬升验证。未抓住则至多一次 lateral retry。

每个 branch 的 `teacher_only.recovery_phase` 是实际状态机阶段，清单构建器优先使用它，
不再把变长重试轨迹错标为固定时间阶段。

## 数据策略

1. 每个 OFT+ 自然失败 fork 执行中心和四个横向偏移 FSM；保留成功/失败 branch outcome。
2. 成功 branch 用于 action、phase 与 visual geometry 监督；失败 branch 用于 verifier/风险标签，
   不作 action cloning 正例。
3. source 级划分保持不变；同一 fork 的所有 FSM/旧 teacher branch 必须留在同一 split。
4. 先收集 task504/511 冒烟；通过 replay 与自动 lift 验证后扩展到全部 9 个 book source。
5. 训练时混合旧固定 teacher 和 FSM teacher，但采样按 source/策略平衡，避免模型只学直线接近。

## 2026-08-09 候选选择审计

多时间点采集完成 18 个 fork、90 条 branch，其中 58 条 verified；连同 v3
终端 fork 后，共得到 27 个 matched group / 135 branches（85 个 verified）。严格的
task504/511 hold-out 上，初版视觉 macro verifier 选中 5/6 个可恢复 candidate，而
matched uniform baseline 的期望为 3.2/6。

这个数字**不能**作为 SlotSSM 候选选择收益：它在所有 6 个 hold-out group 都选择
固定的 `x=+25 mm` 接近宏，而该固定规则恰好也是 5/6。因此当前 FSM 候选集合存在
全局偏置，并没有要求模型利用图像或时序状态。

下一轮只接受满足下列条件的候选数据：

1. 每个状态至少两个候选的结果不同；
2. 不同 state 的最优 candidate 不能被一条固定宏覆盖；
3. 报告必须同时给出最佳固定宏、uniform、oracle 及 learned selector；
4. 测试源在 checkpoint 完全选定后才读取。

这使 candidate verifier 成为一个可证伪的视觉恢复模块，而不是把 teacher 的固定
几何偏置包装成学习效果。

## 2026-08-09 端到端闭环过拟合基线

为实现当前的工程目标（在这组 book→caddy 失败状态上超过原始 OFT+），在线执行器已
从只做 `LIFT_VERIFY` 扩展成完整的 proprio 闭环宏：

`BACKOFF → ALIGN → DESCEND → CLOSE → LIFT → TRANSPORT → PLACE → OPEN → RETREAT`。

它的书本锚点和 caddy 放置锚点均由离线成功/teacher 轨迹拟合；在线仅读取 EEF
position/quaternion，不读取仿真物体位姿、接触或成功标志。这个版本是**任务族内的
过拟合工程基线**，不是 SlotSSM 泛化结论。

在原始 OFT+ 均失败的 9 条稳定 source（494、501、504、506、510、511、513、521、523）
上，oracle 在原失败时刻触发该完整宏后，得到 8/9 成功：

- 成功：494、501、506、510、511、513、521、523；
- 失败：504。视频显示它完成再抓和搬运，但书的放置姿态/格子仍未满足任务判定。

下一步不能把 8/9 误报为学习式 SlotSSM 的收益：应以这批完整成功轨迹补齐
`regrasp → transport → place` 数据，再训练视觉门控、目标/姿态预测和放置验证器；504
则作为姿态恢复与格子选择的针对性难例。

## 2026-08-09 去除手工书锚点：patch-query 几何头

使用 policy-visible 的当前 512 个视觉 patch 与 OFT action-token 历史训练
patch-query 头，预测 `book position - EEF position`。其 source 内验证集平均 L2 为
14.35 mm（逐轴 MAE 6.69 / 6.68 / 7.81 mm）。在线时它替代书本世界坐标锚点；caddy
的 EEF 放置锚点和 oracle 触发暂时仍保留，因而这是一个隔离书定位模块的消融，不能
称作全自主结果。

在 501、506、510、511、513 五条原始 OFT+ 失败 source 上，oracle 触发的在线结果为
3/5：501、510、511 成功；506、513 失败。506 在下探阶段的视觉 z 估计不稳定，执行
49 步下探（成功轨迹为 18 步），形成侧边弱接触，随后运输掉书。513 已完成抓取、开爪
和撤离，但改变的夹持相对位姿使固定 EEF 放置锚点把书留在 caddy 前方而非格内。

结论：视觉头已足以替代手工书位置锚点并把 0/5 OFT+ 失败子集提升为 3/5，但下一轮
监督的关键不再是中心位置，而是 grasp quality / 书相对夹爪姿态，以及能随该姿态
调整的视觉放置目标。不得把这 3/5 与使用两个手工锚点得到的 8/9 混为同一结果。

为此，FSM 数据采集器已扩展为在 `teacher_only` 中记录书本 quaternion，并经 action
manifest 保留为姿态监督；它不进入在线 observation。扩充时要覆盖成功夹持、侧边弱
夹持和掉书三类后抓取轨迹，训练相对夹持姿态/抓取质量头与放置 residual，而非继续
单纯降低书中心位置误差。

## 下一步执行顺序

1. 从 506/513 的多个失败 fork 以及 501/510/511 对照 fork 采集 feedback-FSM branch；
   每帧只在 `teacher_only` 保存书 pose、`attached` 与相对 fork 的抬升量。
2. 用 CLOSE/LIFT 阶段训练可见输入（双相机、SlotSSM 状态、EEF proprio）的 grasp-quality
   head。在线若低置信度，不进入 transport，而是回退并请求一次局部重抓。
3. 用抓稳后的书相对 EEF pose 训练 placement residual，修正当前固定 caddy EEF anchor；
   506 验证重抓，513 验证相对夹持姿态导致的格内放置偏差。
4. 最终只保留 learned trigger、learned grasp-quality decision、视觉几何/placement
   residual；逐项消除手工书锚点、固定 phase 时长和固定 EEF 放置锚点。

## 2026-08-09 官方 Robot 初态全集协议

最终工程目标固定为：在 LIBERO-Plus `Robot Initial States` 的同一 book→back-caddy
任务族上，用一套共享恢复器使最终成功数严格超过 OpenVLA-OFT+。这一集合包含 34 个
BDDL 配置（task id 491--524），每个配置按官方单 trial 协议评测 `episode_id=0`。完整
任务清单保存在
`/data2/lty/naive-e-slotssm-runs/oft-plus-book-local-repair-v1/official-robot-initial-state-book-34-manifest.json`。

冻结的 OFT+ 基线为 11/34 成功（32.35%）和 23/34 失败；详情与每条视频见
`robot-initial-state-book-34-oft-plus-baseline-v1/summary.json`。此前 9 条稳定失败 source
属于该 34 条的子集，只能作为开发数据，不能单独作为最终成功率。

执行顺序：

1. 用原始 OFT+ 回放全部 23 条失败，并在正常动作的每 8 个 policy step 保存 teacher-only
   MuJoCo fork、双相机、EEF/gripper 和动作历史；对 OFT+ 成功的 11 条也保存同格式的
   nominal windows，作为 gate 的非触发样本。
2. 从自动筛选出的失败临界点采集同一套 `BACKOFF → ALIGN → DESCEND → CLOSE →
   LIFT_VERIFY` teacher 分支。teacher 可读 book/caddy pose 与接触来生成标签，但线上输入
   仍严格限制为 RGB、wrist RGB、proprio、动作历史和 SlotSSM 状态。
3. 在全部 source 上训练共享 gate、grasp-quality 和 relative grasp/placement residual。
   不允许按 task id 或 initstate 使用单独的阈值、锚点、状态机或 checkpoint；固定安全阶段
   骨架仅作为共享执行约束。
4. 在线策略先复现 OFT+ 正常动作；只在 gate 高置信时介入一次恢复。验收同时优化
   `OFT+ fail → repair success`，并限制 `OFT+ success → repair failure` 为零或最小。
5. 每轮在冻结的 34 条上逐条配对，报告总成功数、success/failure transition table、每个
   difficulty level 的结果和 MP4；只有总成功数大于 11 才接受为下一轮候选。
