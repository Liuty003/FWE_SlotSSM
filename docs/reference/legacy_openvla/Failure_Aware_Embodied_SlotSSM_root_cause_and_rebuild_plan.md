# Failure-Aware Embodied-SlotSSM 根因分析与重构计划

## 1. 当前结论

目前已经验证 OpenVLA 无头仿真、Stage-1 Slot、Stage-2 Mamba2、Stage-3 动作注入、counterfactual 数据和在线 Failure detector 等工程链路能够运行。但是，当前原型还不是可靠的 Embodied-SlotSSM 闭环系统。

核心问题是：

> Slot 尚未被证明能够稳定表示任务指定对象；Stage-2 尚未真正进入在线时序判断；后续模型可能学习了轨迹相关性，而不是对象状态与动作结果之间的因果关系。

## 2. 已观察到的关键失败

### 2.1 Advantage scorer 破坏正常轨迹

task 9 / episode 0 中，原始 OpenVLA 109 步成功。早期 Failure-Aware 版本在第 0 步把 Z 动作从 `+0.357` 修改为 `+0.007`，随后选择错误目标并失败：

- 220 步内干预 122 次；
- 其中 115 次为 `translate_z_minus`；
- 目标碗全程位移 0 cm；
- 离线一步分支排序有效，但在线连续干预发生严重分布偏移。

### 2.2 离线 Target-State 指标具有误导性

只使用正常轨迹训练时，task 9 离线目标位置误差最低约 0.78 cm。但在 episode 6 空抓轨迹中：

- 在线目标距离平均误差约 30.7 cm；
- 目标碗全程没有移动；
- 模型却曾以 97% 以上概率误判目标已经抬起。

这说明模型可能依赖关节位置、正常轨迹阶段和常见目标区域等捷径，而没有真正绑定目标对象。

### 2.3 检测改善后，恢复仍然失败

Target-State Estimator v3 加入 episode 5、6 的失败状态，并在 episode 7 验证：

- 失败轨迹目标距离误差约 2.48 cm；
- “未抬起”识别准确率约 95.9%；
- 正常轨迹目标距离误差约 1.44 cm。

它能发现“找到碗但没有抓起，随后准备去盘子”的错误，但恢复策略仍失败：

- 直接使用预测目标向量控制时，方向在恢复分布上预测反号，误差从约 10.6 cm 扩大到 37.4 cm；
- 重新提示 OpenVLA 时，机械臂与目标距离从 17.9 cm 增至 30.2 cm；
- 两种恢复方式都不能可靠重定位发生位移的碗。

## 3. 根因分析

### 3.1 Stage-1 Slot 缺少目标语义

Stage-1 主要使用 bbox、mask、objectness 等类别无关监督。它没有显式学习：

- 哪个 Slot 是任务指定目标；
- “柜子上的黑碗”与另一个黑碗的区别；
- 空间关系语言对应哪个 Slot；
- 某个 Slot 在下一帧是否仍然表示同一个对象。

task 9 同时包含两个外观近似的黑碗，正好暴露了这个问题。

### 3.2 在线 Slot 没有时序跟踪

当前在线路径每帧调用：

```python
tokenizer(patches, past_slots=None)
```

每一帧 Slot 都重新初始化，Slot identity 可能任意切换。碗发生位移后，系统无法保证继续跟踪同一个对象。

### 3.3 Stage-2 与在线用途脱节

Stage-2 已完成 next-slot 预测训练，但尚未证明：

- 预测 Slot 对应同一个对象；
- 能识别接触、抓取、抬起和掉落；
- 在失败及恢复分布上优于复制上一帧 Slot；
- 在线 Failure detector 实际使用了 Stage-2 状态。

目前 Target-State Estimator 主要使用单帧 Slot，Mamba2 没有真正进入 Failure-Aware 决策。

### 3.4 有效数据规模很小

v3 的 1209 个状态来自少量高度相关轨迹。episode 5、6 的 440 帧本质上只有两条独立失败轨迹。4000 步、batch 64 相当于重复遍历数据约 212 次，容易记忆轨迹。

### 3.5 验证类别不平衡

episode 7 的 220 帧全部是 lifted 负样本。只报告 accuracy 会夸大性能。后续必须同时报告 precision、recall、F1、成功抓取漏报率、失败误报率和检测延迟。

### 3.6 语言编码过弱

当前小模型使用哈希词袋，不能充分表达 `on the wooden cabinet`、`between the plate and the ramekin`、`next to the ramekin` 等关系，也难以区分两个同类对象。

### 3.7 Counterfactual 目标与在线闭环不一致

counterfactual 数据只回答从正常轨迹分叉、执行候选动作 4 步后的短期 progress。它没有覆盖连续干预、空抓、目标位移、滑落、阶段错误和恢复控制器产生的状态。

### 3.8 恢复数据不是专家示范

错误恢复轨迹只能覆盖状态分布，不能提供正确恢复动作。恢复策略需要仿真 oracle 生成专家动作，并通过行为克隆和 DAgger 训练。

## 4. 暂停的工作

在表示审计完成前暂停：

- 调整 XYZ 恢复增益；
- 修改恢复提示词；
- 堆叠更多单帧 probability 规则；
- 在 episode 7 上反复调参；
- 在目标 Slot 未通过定量评测前上线 Recovery Action Head；
- 将同一 episode 同时用于开发和最终测试。

## 5. 重构阶段

### Stage A：Slot 表示审计

目标：证明 Stage-1 Slot 是否真的表示并跟踪任务目标。

需要完成：

1. Slot mask 与仿真 instance segmentation 的 IoU；
2. 目标碗是否被单一 Slot 稳定覆盖；
3. 两个黑碗是否被不同 Slot 分离；
4. 连续帧 Slot identity switch rate；
5. 目标移动后，目标 Slot 是否随对象移动；
6. 成功、空抓和恢复轨迹分别评测；
7. 输出可视化视频和定量 JSON 报告。

必须进行的 probe/消融：

- joints-only；
- Slot-only；
- 原始视觉 patch；
- Slot + joints；
- 打乱 Slot；
- 打乱语言；
- 遮挡目标碗。

若 joints-only 与 Slot 模型表现接近，说明当前模型主要利用轨迹捷径。

### Stage B：目标语义 Slot

若 Stage A 不通过，则重新训练 Stage-1：

- 使用仿真 instance segmentation 提供对象 mask；
- 对目标碗、干扰碗、盘子、机械臂提供对象身份监督；
- 训练语言到 target Slot 的显式匹配；
- 加入空间关系监督；
- 加入跨帧同一对象一致性损失。

期望输出：

```text
target_slot
goal_slot
robot_slot
distractor_slots
```

### Stage C：真正接入时序 SlotSSM

在线维护：

```text
past_slots -> current_slots -> SlotSSM hidden state
```

训练 approach、contact、gripper close、grasp established、lifted、slip/drop、target lost、placed 等事件标签，并与复制上一帧、单帧 MLP、GRU/LSTM 基线比较。

### Stage D：Failure Detector

输入 target Slot 历史、robot Slot/关节历史、动作历史、SlotSSM hidden state和任务阶段；输出抓取成功、空抓、滑落、错目标、阶段转换许可和 OOD/不确定性。

### Stage E：Recovery Policy

使用仿真 oracle 生成：

```text
重新定位目标 -> 抓取位姿 -> 闭合夹爪 -> 抬升 -> 验证目标随动
```

训练顺序：Behavior Cloning、在线采样、oracle 重标注、DAgger 迭代、留出 episode 验证。

### Stage F：严格评测

按 episode 隔离 train、validation 和从未用于调参的 test。报告原始 OpenVLA 成功率、成功轨迹破坏率、failure precision/recall/F1、恢复尝试率、恢复成功率和最终任务成功率。

## 6. 阶段门槛

### Stage A 通过条件

- 目标 Slot mask IoU 明显高于随机和最大区域基线；
- 两个黑碗能被不同 Slot 分离；
- 目标 Slot identity switch rate 足够低；
- Slot probe 显著优于 joints-only；
- 目标移动后仍能跟踪对象，而不是停留在旧位置。

### Stage C 通过条件

- SlotSSM 在事件预测上显著优于单帧模型和复制基线；
- 在失败轨迹和移动目标轨迹上保持性能；
- 在线路径真实加载并使用 Stage-2 checkpoint。

### Stage E 通过条件

- 恢复动作在留出 episode 上通过离线评测；
- 在线恢复不产生无界运动；
- 在多个独立失败 episode 上提高成功率；
- 不降低原本成功 episode 的成功率。

## 7. 当前工作位置

当前停止旧路径的恢复控制实验，正式回到 Stage A。

下一项工作：

> 对 Stage-1 Slot 进行目标覆盖、对象分离、时序 identity 和 shortcut probe 的定量审计。

## 8. Stage A 初始审计发现

已有输出 `/data2/lty/naive-e-slotssm-runs/stage1-multitask-acceptance/metrics.json` 可作为 Stage A 的历史基线：

- 10 个 LIBERO-Mem 任务平均 bbox IoU：0.714；
- 平均 mask IoU：0.546；
- objectness precision：0.999；
- objectness recall：0.970；
- 平均 identity consistency：0.966；
- 总 identity switches：27。

但是，包含三个碗和多个盘子的第 9 个任务明显最差：

- bbox IoU：0.503；
- mask IoU：0.357；
- objectness recall：0.888；
- identity consistency：0.879；
- 16 帧内 identity switches：9。

该任务的匹配结果显示 bowl 1、bowl 2、bowl 3 分配给不同 Slot，但跨帧稳定性显著下降。这与当前 LIBERO Spatial task 9 中两个黑碗导致的目标漂移现象一致。

这份历史评测仍不能作为当前 Stage A 的最终结论，因为：

- 它评测的是 LIBERO-Mem，不是当前在线 LIBERO Spatial task 9；
- 每个任务只选取一个 16 帧窗口；
- identity 根据逐帧 bbox 最大 IoU 推断，没有直接评测任务语言对应的 target Slot；
- 没有覆盖空抓、目标被推动和恢复分布。

下一步将基于 `SegmentationRenderEnv` 在当前 task 9 上采集实例级 segmentation，并分别审计成功、空抓和移动目标轨迹。

## 9. Stage A task 9 实例级审计结果（2026-07-22）

已完成 LIBERO Spatial task 9 的确定性动作回放与实例级 Slot 审计：

- episode 0 成功轨迹：109 帧；
- episode 6 空抓轨迹：220 帧；
- episode 7 目标被碰动及错误恢复分布：220 帧；
- 共 549 帧，EEF 与对象位置的最大回放误差均低于 `6e-8 m`；
- RGB 与实例 segmentation 经相同双轴翻转后像素对齐；
- 正式 Stage-1 checkpoint 的 tokenizer 及 mask-head 权重均已确认加载。

聚合结果：

| 指标 | independent (`past_slots=None`) | tracked（传递上一帧 Slot） |
|---|---:|---:|
| 目标碗 mean best soft IoU | 0.0036 | 0.0060 |
| 目标碗 mean best hard IoU | 0.0031 | 0.0052 |
| 干扰碗 mean best soft IoU | 0.1457 | 0.0775 |
| target identity switches | 72 / 546 | 94 / 546 |
| target switch rate | 13.2% | 17.2% |
| target/distractor collision rate | 20.6% | 51.4% |

分轨迹结果也一致：tracked 没有修复目标身份，反而把 episode 0/6/7 的黑碗碰撞率分别提高到 39.4%、52.9% 和 55.9%。目标碗在 episode 6 的 mean best soft IoU 仅为 0.00017。

结论：Stage A 的目标覆盖、同类对象分离和时序 identity 三项核心门槛均未通过。单纯把上一帧 Slot 作为初始化不能解决问题，并会放大错误身份的持续传播。当前正式决策应进入 Stage B，重新训练具有实例身份、目标语言匹配和跨帧一致性监督的 Stage-1；在新 Stage-1 通过同一审计前，不进入 Stage C，也不恢复在线 Recovery 控制实验。

审计代码与产物：

- `scripts/preflight_slot_segmentation_task9.py`
- `scripts/audit_slot_stage1_task9.py`
- `/data2/lty/naive-e-slotssm-runs/stage-a-task9-segmentation-preflight`
- `/data2/lty/naive-e-slotssm-runs/stage-a-task9-slot-audit`
