# Failure-Aware SlotSSM跨任务恢复方法

更新时间：2026-07-31（Asia/Shanghai）

## 1. 文档目的

本文把Task6中已经验证有效的恢复方法整理成可迁移流程，供Task9及后续LIBERO任务
复用。现阶段允许每个任务重新采集数据、训练检测头和候选评分器，不要求零样本泛化；
需要保持一致的是闭环结构、数据接口和评测方式。

Task6已经得到三类完整视频结果：

- 空夹：闭爪后物体没有随夹爪抬升，系统重新抓取；
- 边缘夹持后滑脱：物体相对夹爪持续漂移，确认脱离后重新抓取；
- 接近/抓取失败：视觉相对几何触发，生成候选并选择重抓动作。

这些结果证明了方法可行，但不能被表述为“一个统一SlotSSM权重已经跨任务泛化”。
目前在线链路仍包含角色Slot、时序特征、显式门控、候选控制器和学习式评分器。后续
统一模型应逐步替代这些teacher模块，而不是在迁移新任务前强行完成。

## 2. 可迁移的核心闭环

```text
OpenVLA产生正常动作
        ↓
每帧提取任务角色Slot并更新时序状态
        ↓
判断当前子目标、动作阶段和关系进度
        ↓
正常推进 ───────────────→ 继续OpenVLA
        ↓ 异常
确认失败类型并冻结危险动作
        ↓
从当前状态生成多个恢复候选
        ↓
在同一失败快照分叉执行，获得候选结果
        ↓
训练评分器预测候选成功概率/未来进度
        ↓
在线选择一个候选并执行
        ↓
重新观察并验证恢复
        ↓
成功：交还OpenVLA；失败：有限次重规划或安全终止
```

这里最重要的不是某一条手写规则，而是四项机制：

1. **持续状态**：不是只看失败瞬间，而是保存目标、夹爪和目标区域的连续Slot状态；
2. **关系验证**：闭爪、移动或释放动作发生后，验证预期物体关系是否真的改变；
3. **同状态候选比较**：从完全相同的失败快照测试多个动作，避免用不同episode的结果
   错误比较恢复策略；
4. **恢复后再验证**：不能因为宏动作执行完就宣布恢复，必须重新观察物体并验证任务关系。

## 3. 每个任务需要定义的最小规范

迁移到新任务前，新建一份任务配置，至少定义以下内容。

### 3.1 任务角色

角色名称不应绑定具体实例编号，推荐使用：

```text
manipulated_object：当前被操作物体
end_effector：夹爪/末端执行器
goal_region：盘子、篮子、微波炉内部等目标区域
articulated_part：门、抽屉、旋钮等可选角色
distractor：可能被碰撞或混淆的重要物体
```

每帧记录每个角色的可见性、mask/中心、深度或相对几何、置信度和Slot embedding。
角色暂时不可见时保留时序状态，同时降低观测置信度，不能立即把旧位置当作当前真值。

### 3.2 阶段和预期关系

把语言任务拆成有限阶段，并为每个阶段写出动作后的可验证关系。例如：

| 阶段 | 动作意图 | 可观察验证 |
|---|---|---|
| approach | 接近物体 | 夹爪—物体距离持续减小 |
| acquire | 闭爪获取 | 物体随夹爪抬升且相对漂移小 |
| transport | 搬运 | 物体—目标距离减小，抓取关系稳定 |
| place/insert | 放置或插入 | `On`/`In`关系的视觉代理成立 |
| release | 松爪 | 物体留在目标区域而非随夹爪离开 |
| articulate | 关门等操作 | 可动部件达到目标状态 |

阶段标签可以使用仿真真值离线生成，但在线触发只能依赖部署时可获得的视觉、机器人
本体状态和模型输出。

### 3.3 失败类型

优先定义少量可恢复、可观察且动作含义不同的失败：

```text
empty_acquisition       空夹
edge_slip               边缘夹持/滑脱
transport_drop          搬运途中掉落
no_progress             长时间没有有效推进
placement_shortfall     放置位置接近但谓词未完成
insertion_shortfall     对准入口但插入不足
wrong_target            操作错误物体或目标区域
collision_instability   碰撞导致物体倾倒/偏移
```

不要只用单帧距离阈值。至少结合动作事件、短时趋势、可见性和阶段门控，避免把正常停顿
或遮挡误判为失败。

## 4. Task6中已验证有效的方法

### 4.1 动作事件对齐

先从OpenVLA动作流定位闭爪、抬升、运输和释放事件，再在事件后的固定观察窗中检查
Slot关系变化。Task6的有效例子是：

- 闭爪后约15帧检查目标是否抬升；
- 抬升不足且相对漂移很小，判断为空夹；
- 抬升不足且相对漂移明显，继续观察到约30帧，确认边缘滑脱；
- 对仍有脆弱接触的状态不立即分叉，等待物体稳定，避免MuJoCo数值不稳定。

这些帧数和距离阈值是Task6校准值，迁移时必须重新标定，不能直接视为通用常数。

### 4.2 视觉相对几何

候选动作以角色Slot之间的相对关系生成，不使用固定世界坐标：

```text
目标位移 = f(物体Slot位置 - 夹爪Slot位置,
             物体尺度/轴向不确定性,
             当前阶段,
             候选参数)
```

Task6实践表明，视觉位置可能沿某个轴系统性高估。因此候选库不只改变整体动作幅度，
还应独立覆盖横向偏移、纵向偏移、下降深度和接近方向。恢复控制必须限制速度、旋转和
工作空间；激进腕部旋转在开发中造成过数值不稳定，已从最终候选库删除。

### 4.3 同快照分叉与候选标签

在失败确认后的稳定仿真状态保存快照，对每个候选执行：

```text
恢复宏动作
→ 重新观察
→ 检查物体是否附着/关系是否改善
→ 必要时短暂续跑OpenVLA
→ 记录最终任务success和中间安全指标
```

候选标签至少包含：

- 最终任务是否成功；
- 恢复后目标关系是否成立；
- 相对距离改善量；
- 是否发生碰撞、出界或数值异常；
- 动作长度和额外耗时；
- 候选参数及失败前时序状态。

训练/验证应按失败快照或episode划分，不能把同一个快照的不同候选分散到训练集和验证集。

### 4.4 学习式候选评分

评分器输入为失败前的时序状态、当前关系特征和候选动作参数，输出候选成功概率或预期
未来进度。Task6使用冻结时序表示训练宏动作排序器，在线选择得分最高且通过安全约束的
候选。

早期数据较少时可以训练任务专用评分器，甚至允许一定过拟合以获得阶段性视频；但必须
保留独立episode做验证，并报告候选最优命中率、成功候选命中率和相对oracle regret。

### 4.5 恢复后验证与控制权交接

恢复后需要同时满足：

1. 目标物体确实随夹爪移动，或目标空间关系确实成立；
2. 连续若干帧关系稳定；
3. 没有危险碰撞、物体出界或Slot低置信度；
4. OpenVLA接管后没有立即撤销恢复成果。

只有通过验证才交还OpenVLA。若失败，可在预算内重新生成候选；超过最大恢复次数后安全
停止并保存完整轨迹，不能无限循环。

## 5. 迁移到新任务的执行流程

### Step A：自然基线

1. 使用原始OpenVLA运行全部标准初始状态；
2. 每条保存MP4、动作、机器人状态、Slot观测和仿真评测结果；
3. 人工审计失败视频，选择2–3类自然、可恢复、视觉清楚的失败；
4. 优先选择“基本策略已经接近成功”的episode，而不是严重出界或基础能力完全缺失的情况。

### Step B：离线标签与时序数据

1. 定义任务角色、阶段和关系；
2. 用仿真状态产生离线teacher标签；
3. 生成角色Slot序列；
4. 标注动作事件、失败起点、确认时刻和最后可恢复状态；
5. 检查标签在视频上是否符合人的观察。

仿真真值只用于标注、分析和最终评测。在线演示必须另存一份部署可用字段清单，防止
privileged state泄漏。

### Step C：Oracle恢复上限

先用真值定位失败并执行任务专用恢复，确认：

- 当前失败状态在剩余步数内确实可恢复；
- 候选动作空间覆盖了至少一个成功动作；
- 恢复后OpenVLA能继续完成任务。

Oracle失败说明候选控制或任务分解尚不够，不应立即训练检测器。

### Step D：视觉检测和shadow审计

把真值触发替换为Slot时序检测器，但暂不改变机器人动作。对成功与失败episode统一运行，
统计触发时刻、漏检、误检和失败类型混淆。

只有在成功episode上的误触发可接受后，才允许检测器在线接管。

### Step E：候选分叉与评分器

1. 在多个稳定失败快照上执行候选分叉；
2. 候选必须覆盖几何估计误差，而不是只在单一方向微调；
3. 删除危险、数值不稳定或物理上无意义的候选；
4. 按快照划分数据并训练任务专用评分器；
5. 与随机选择、固定候选和oracle最优比较。

### Step F：完整闭环与视频

按同一命令运行任意episode：

```text
OpenVLA → SlotSSM/时序检测 → 候选生成 → 学习评分
        → 恢复执行 → 视觉验证 → OpenVLA续跑
```

视频应显示统一字幕：当前阶段、Slot关系趋势、失败类型、触发原因、候选得分、所选动作、
恢复验证和最终任务结果。展示视频不能隐藏失败前过程，也不能只截取恢复成功片段。

### Step G：正式对比

至少报告：

- 原始OpenVLA成功率；
- 加检测但不恢复的结果；
- 完整恢复闭环成功率；
- 失败检出率、漏检率和正常轨迹误触发率；
- 恢复尝试成功率；
- 平均恢复次数和额外动作步数；
- 按失败类型拆分的结果。

阶段性任务专用结果应明确写成“within-task trained/calibrated”，不能称为跨任务泛化。

## 6. Task9的具体映射

推荐先做Task9 episode4/5的“微波炉入口插入不足”：

| 通用角色/模块 | Task9实例 |
|---|---|
| manipulated_object | 黄白杯 |
| goal_region | 微波炉内部heating region |
| articulated_part | 微波炉门 |
| 主要失败 | `insertion_shortfall` |
| 关系进度 | 杯子中心/可见轮廓相对入口和内部区域的连续变化 |
| 候选维度 | 入口横向对准、高度、推进深度、释放时机 |
| 恢复验证 | 杯子进入内部且稳定，夹爪退出后杯子不被带出 |
| 后续控制 | 交还OpenVLA执行关门；必要时再建立门阶段恢复 |

第一版可以重新训练Task9专用角色头、时序检测头和候选评分器。迁移成功的标准不是复用
Task6权重，而是无需改变核心闭环便能处理新的`In`关系和新的失败类型。

## 7. 实现接口建议

后续应将Task6脚本中的任务特例逐步收敛为以下接口：

```python
TaskSpec.roles(observation) -> role_queries
TaskSpec.stage(slot_history, action_history) -> stage
TaskSpec.relations(slot_history) -> relation_features
FailureHead(slot_history, action_history, stage) -> failure_type, confidence
CandidateGenerator(failure_type, relation_features) -> candidates
CandidateScorer(temporal_state, candidates) -> scores
RecoveryController(candidate) -> action_chunk
RecoveryVerifier(slot_history, task_spec) -> recovered, confidence
```

Task专用内容放入配置、角色定义和候选参数，不应复制一整套在线执行器。统一模型完成后，
`FailureHead`、`CandidateScorer`和`RecoveryVerifier`可以共享Recovery SlotSSM状态；
在此之前保持同样的输入输出接口，能够避免Task6经验变成不可维护的单episode规则。

## 8. 常见错误

- 在不同初始状态上执行不同候选，然后把成功率当成候选优劣；
- 在线读取物体世界坐标、任务谓词或仿真接触真值；
- 用固定episode步数触发，却声称是视觉失败检测；
- 看到夹爪闭合就认为抓取成功，没有验证物体是否随动；
- 在物体仍有脆弱接触时保存/恢复快照，导致物理状态不稳定；
- 只在失败episode调阈值，不检查正常episode误触发；
- 恢复宏动作完成后立即结束视频，没有验证最终任务成功；
- 将任务专用校准结果表述成跨任务或零样本泛化。

## 9. 当前可参考实现与结果

关键脚本：

```text
experiments/robot/libero/run_task6_candidate_recovery_demo.py
scripts/build_task6_execution_labels.py
scripts/collect_task6_failure_candidate_branches.py
scripts/train_task6_macro_candidate_scorer.py
```

Task6代表视频：

```text
/data2/lty/naive-e-slotssm-runs/task6-natural-ep1-empty-grasp-recovery-v2-auto-scored/task6-episode1.mp4
/data2/lty/naive-e-slotssm-runs/task6-natural-ep6-edge-slip-recovery-v6-auto-scored/task6-episode6.mp4
/data2/lty/naive-e-slotssm-runs/task6-candidate-recovery-online-v23-unified-auto-visual-scored/task6-episode3.mp4
```

详细实验沿革、数据路径和限制见`HANDOFF.md`第20章；自然基线见
`LIBERO10_NATURAL_FAILURE_AUDIT.md`。
