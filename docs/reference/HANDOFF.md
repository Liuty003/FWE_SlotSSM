# naive-e-slotssm / Failure-Aware Embodied-SlotSSM 交接文档

更新时间：2026-07-30（Asia/Shanghai）

本文写给一个完全没有聊天上下文的新会话。请先完整阅读本文，再阅读文末列出的核心设计文档。不要直接从最新的恢复控制实验继续堆规则。

## 1. 用户目标

用户最初希望在 8×RTX 4090 无头服务器上复现 Embodied-SlotSSM，并先跑通最小 VLA 仿真结果。随后目标扩展为：

1. 以 OpenVLA LIBERO Spatial checkpoint 为基础；
2. 复现 Embodied-SlotSSM 的 Stage-1/2/3；
3. 对比接入 SlotSSM 前后的动作与仿真表现；
4. 实现用户提出的 Failure-Aware Embodied-SlotSSM；
5. 识别空抓、错目标、滑落、错误阶段转换，并在失败后恢复；
6. 最终在 LIBERO 无头仿真中提高任务成功率，同时不破坏原本成功的 OpenVLA 轨迹。

当前已经完成从 Stage A/B 表示审计到 task9 lost-grasp 学习恢复闭环的可行性验证。
最重要的新决定是：停止继续增加独立恢复模块和手工规则，将现有模块化系统固化为
数据采集器/teacher，下一阶段直接训练共享时序状态的统一 Recovery SlotSSM。

## 2. 环境与机器

- 仓库：`/data/lty/code/naive-e-slotssm`
- Conda 环境：`/data/lty/code/naive-e-slotssm/.conda/envs/openvla-a`
- 大容量输出根目录：`/data2/lty/naive-e-slotssm-runs`
- OpenVLA 本地模型：`artifacts/checkpoints/openvla-7b-finetuned-libero-spatial`
- 仿真：LIBERO + MuJoCo，服务器无头，使用 EGL。
- 常用环境变量：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

- 用户明确说明：现在所有 GPU（0–7）都可以使用。
- GPU 被其他用户共享使用。不要假设空闲显存恒定，启动前用 `nvidia-smi` 检查。
- 7B OpenVLA 通常用 Accelerate `device_map=balanced` 和每卡约 `7GiB` 上限稳定加载。
- 当前无我们的训练、采集或在线 rollout 进程在运行。不要误杀其他用户进程。

## 3. 重要设计文档

建议按此顺序阅读：

1. `Embodied-SlotSSM_paper_only_reproduction_spec.md`：论文复现说明。
2. `Embodied-SlotSSM_reproduction_progress.md`：原始复现过程与阶段进展。
3. `Failure_Aware_Embodied_SlotSSM.md`：用户的 Failure-Aware 想法。
4. `Failure_Aware_Embodied_SlotSSM_implementation_review.md`：实现评审与早期计划。
5. `Failure_Aware_Embodied_SlotSSM_root_cause_and_rebuild_plan.md`：根因分析与 Stage A–F 重构计划。
6. `CURRENT_PROGRESS_2026-07-24.md`：当前能力、产物、边界和下一步的集中总结。
7. `recovery_expert_dataset_schema.md`：已有恢复 expert 数据格式。
8. `daily_progress/2026-07-21.md`：早期详细工作记录。
9. `LIBERO10_NATURAL_FAILURE_AUDIT.md`：100条标准自然基线、17条重点失败分类
   和当前主任务选择。

## 4. 当前系统由什么组成

仓库中保留两条不同性质的系统。论文复现链路是：

```text
冻结 OpenVLA 7B
  + Stage-1 Slot tokenizer（16×512）
  + Stage-2 Mamba2 / SlotSSM
  + Stage-3 Slot 注入动作路径
  + Task-Aware Advantage Scorer
  + Target-State Estimator v3
```

最新成功视频使用的 Failure-Aware 模块化恢复链路是：

```text
冻结 OpenVLA 7B
  + Stage-B target selector / mask tracker
  + predicted-mask motion failure detector
  + mask-depth visual 3D estimator
  + learned phase predictor
  + learned geometry residual controller
  + visual temporal regrasp verifier
  + hand-written VISUAL SETTLE
  + gripper sign safety override
```

这两条链路尚未统一。虽然 Stage-2/3 已训练并跑通，最新恢复闭环没有使用 Stage-2
SlotSSM 时序状态；它使用多个独立小模型和规则。下一阶段的任务正是用一个共享
Recovery SlotSSM 将第二条链路合并，而不是继续扩展模块数量。

## 5. 已完成的基础复现

### 5.1 OpenVLA 基线

已在无头 LIBERO Spatial 中跑通 OpenVLA。关键脚本：

- `experiments/robot/libero/run_openvla_baseline.py`
- `experiments/robot/libero/screen_openvla_episodes.py`

严格独立运行的 task 9 / episode 0：

- 任务：`pick up the black bowl on the wooden cabinet and place it on the plate`
- 109 步成功。
- 输出：`/data2/lty/naive-e-slotssm-runs/openvla-online-control/task9-episode0`

严格独立运行的 task 9 / episode 6：

- 220 步失败；
- OpenVLA 去空气中抓取，然后前往盘子松开；
- 两个碗位移均为 0；
- 输出：`/data2/lty/naive-e-slotssm-runs/openvla-standalone-repeat-v1/task9-episode6`

### 5.2 Stage-1 Slot tokenizer

正式 checkpoint：

```text
/data2/lty/naive-e-slotssm-runs/stage1-adapters/
openvla-7b-finetuned-libero-spatial+libero_mem+b1+lr-1e-05--
full10tasks-h16-sync6-s20k-eb6-no-gpu3--image_aug--multiview1000--object/
pretrained_slots-wo-filters_s16.safetensors
```

实际为一条连续路径，上述换行仅为可读性；文件约 53MB。

Stage-1 使用 bbox、GIoU、objectness 和 mask 等监督训练，已经完成 smoke、refine 和正式训练，并做过可视化。

### 5.3 Stage-2 Mamba2 / SlotSSM

正式 checkpoint：

```text
/data2/lty/naive-e-slotssm-runs/stage2-refine-residual-next-h32-s2k-sync6-no-gpu3/
stage2-step-002000.safetensors
```

文件约 34MB。已跑通 Mamba2 前向、反向、loss 和 checkpoint。曾修复 `FlashMHA` 不是 type 导致的 `isinstance` 错误，以及模型并行时 CPU tensor 被 Triton/Mamba2 访问的问题。

### 5.4 Stage-3 Slot/OpenVLA 动作路径

正式 checkpoint：

```text
/data2/lty/naive-e-slotssm-runs/stage3-openvla-causal-h32-a16-s2k/
stage3-step-002000.safetensors
```

文件约 32MB。已证明 Slot token 能影响动作输出，梯度有限且可回传。但未经充分闭环验证，不应直接让 Stage-3 替代原始 OpenVLA 动作。

## 6. Counterfactual 与 Failure-Aware 已完成工作

### 6.1 Counterfactual 数据

完成 10 个 LIBERO Spatial task × 5 episode 的 counterfactual 数据，约 960 fork states。主要数据：

- 原始分支：`/data2/lty/naive-e-slotssm-runs/counterfactual-data`
- Slot + 物理标签：`/data2/lty/naive-e-slotssm-runs/counterfactual-slots-spatial-e0e4-h4-physical`
- Task progress：`/data2/lty/naive-e-slotssm-runs/counterfactual-task-progress-e0e4-h4`

每个 fork 包含 nominal 和 9 个动作扰动分支，每个分支最多 4 步。

### 6.2 Physical Failure Scorer

checkpoint：

```text
/data2/lty/naive-e-slotssm-runs/failure-physical-scorer-e0e4-h4/scorer-best.safetensors
```

离线能预测短期物理效果，但动作选择不能稳定提高任务进展。

### 6.3 Task-Aware Advantage Scorer

checkpoint：

```text
/data2/lty/naive-e-slotssm-runs/task-aware-advantage-scorer-e0e4-h4/
advantage-best-gated.safetensors
```

离线 task 9 结果曾显示 gated progress 比 nominal 高约 2.12mm，但在线连续闭环失败。

早期在线 task 9 / episode 0：

- 220 步失败；
- 干预 122 次（55.5%）；
- 115 次选择 `translate_z_minus`；
- 第 0 步把 OpenVLA Z 动作从 `+0.357` 改成 `+0.007`；
- 导致选择错误目标/抓取失败。

结果：`/data2/lty/naive-e-slotssm-runs/advantage-online-full/task9-episode0`

之后增加 warmup、cooldown、方向保护、小扰动和高门槛。在 episode 0 中 0 次干预、109 步成功，但这只证明“不破坏”，不证明有恢复能力。

## 7. Target-State Estimator

代码：`scripts/train_target_state_estimator.py`

输入：

- 16×512 Slot；
- 7 维关节；
- 哈希词袋语言。

输出：

- target - EEF 三维向量；
- goal - target 三维向量；
- lifted logit。

### 7.1 重要版本与结论

v1 训练 task 0–8、验证 task 9，明显过拟合。正常轨迹误差约 7–10cm。

v2 加入 task 9 正常 episode，离线 episode 4 目标位置误差低至约 0.78cm，但在 episode 6 空抓在线轨迹上误差约 30.7cm，并曾错误输出 97% 以上 lifted probability。这证明离线指标走捷径。

v3 加入 on-policy 失败 episode 5、6，episode 7 留出验证。

最佳 checkpoint：

```text
/data2/lty/naive-e-slotssm-runs/target-state-estimator-failure-aware-v3/
target-state-best.safetensors
```

最佳离线指标：

- 正常 episode 4：target relative MAE 约 1.25cm；
- 失败 episode 7：target distance MAE 约 2.48cm；
- 失败 episode 7 未抬起识别约 95.9%。

在线 episode 7 中，它成功发现碗没有真正抬起：

- 碗最大移动约 7cm，但最大抬升仅 0.41cm；
- OpenVLA 开始离开目标并去盘子时 lifted probability 很低；
- 但接近碗时曾短暂达到约 0.71，故必须使用 >0.8 且连续多帧确认。

## 8. 恢复控制实验及失败原因

主要在线脚本：

- `experiments/robot/libero/run_advantage_openvla_online.py`
- `experiments/robot/libero/run_advantage_openvla_online_stable.py`

### 8.1 向量恢复 v1

检测到未抓起后，使用 Target-State Estimator 的 target-EFF 向量直接生成 Cartesian 动作。

失败：碗已向后移动约 6.5cm，模型仍预测旧方向。恢复开始时方向误差约 10.6cm，随后扩大到 37.4cm，机械臂跑出画面。

输出：`/data2/lty/naive-e-slotssm-runs/target-recovery-online-v1/task9-episode7`

### 8.2 OpenVLA 重新提示恢复 v2

检测失败后重新调用 OpenVLA，提示其重新寻找柜子上的碗，并在确认抓起前不要去盘子。

失败：恢复从第 94 步开始，真实 EEF-target 距离从 17.9cm 增加到 30.2cm。80 步后进入 hold，最后 46 步原地保持。

输出：`/data2/lty/naive-e-slotssm-runs/target-recovery-online-v2-replan/task9-episode7`

结论：Failure detector 有一定效果，但恢复策略没有可靠目标 grounding。不要继续调整提示词、XYZ 增益或门槛。

## 9. 已采集的 on-policy 失败/恢复分布

原始 OpenVLA 失败 Slot（每个 220 帧）：

```text
/data2/lty/naive-e-slotssm-runs/onpolicy-failure-slots-v2/task9-episode5
/data2/lty/naive-e-slotssm-runs/onpolicy-failure-slots-v2/task9-episode6
/data2/lty/naive-e-slotssm-runs/onpolicy-failure-slots-v2/task9-episode7
```

错误 replan 恢复分布（每个 220 帧）：

```text
/data2/lty/naive-e-slotssm-runs/recovery-distribution-v1/task9-episode5
/data2/lty/naive-e-slotssm-runs/recovery-distribution-v1/task9-episode6
/data2/lty/naive-e-slotssm-runs/target-recovery-online-v2-replan/task9-episode7
```

每个目录包含 `steps.jsonl`、`slots.safetensors`、`metadata.json` 和视频。错误恢复轨迹只能用于覆盖状态分布，不能直接当专家动作监督。

## 10. 为什么决定重构

根因不是简单的恢复控制器参数：

1. Stage-1 是类别无关 Slot，没有显式 target/distractor 语义；
2. 在线每帧调用 tokenizer 时使用 `past_slots=None`，没有跨帧 identity；
3. Stage-2 Mamba2 没有进入在线 Failure detector；
4. 小模型使用哈希词袋语言，空间关系表达弱；
5. 数据状态多但独立轨迹少，极易过拟合；
6. lifted 验证类别严重不平衡；
7. counterfactual 只覆盖正常轨迹的一步/短期分叉；
8. 恢复轨迹不是 oracle 专家示范。

## 11. 当前正式工作流：Stage A–F

完整门槛见 `Failure_Aware_Embodied_SlotSSM_root_cause_and_rebuild_plan.md`。

### Stage A：Slot 表示审计（当前阶段）

必须回答：

- Slot mask 与实例 segmentation 的 IoU；
- 两个黑碗是否被不同 Slot 分离；
- target Slot 是否跨帧稳定；
- 碗移动后 target Slot 是否随碗移动；
- `past_slots=None` 与真正传递 past slots 的差异；
- Slot probe 是否显著优于 joints-only；
- shuffled Slot/language 和遮挡目标后的性能。

### Stage B：目标语义 Slot

若 Stage A 不通过，用仿真 instance segmentation、对象身份、语言 target Slot、空间关系和跨帧一致性重新训练 Stage-1。

### Stage C：真正接入 SlotSSM

在线维护 past slots 和 Mamba2 hidden state；训练 approach、contact、grasp、lift、slip、target lost、placed 等事件，并与单帧/GRU/复制基线比较。

### Stage D：Failure Detector

用时序 target Slot、robot Slot、动作历史和任务阶段预测空抓、滑落、错目标、阶段许可与 OOD。

### Stage E：Recovery Policy

必须由仿真 oracle 生成正确恢复示范，再做 Behavior Cloning + DAgger。不要将错误恢复轨迹当专家动作。

### Stage F：严格评测

按 episode 隔离 train/validation/test，报告 OpenVLA 成功率、成功轨迹破坏率、failure precision/recall/F1、恢复成功率和最终成功率。

## 12. Stage A 已有初始证据

现有历史审计：

```text
/data2/lty/naive-e-slotssm-runs/stage1-multitask-acceptance/metrics.json
```

总体 10 个 LIBERO-Mem task：

- mean bbox IoU：0.714；
- mean mask IoU：0.546；
- objectness precision：0.999；
- objectness recall：0.970；
- identity consistency：0.966；
- total identity switches：27。

但是包含三个碗和多个盘子的 task 9 最差：

- bbox IoU：0.503；
- mask IoU：0.357；
- recall：0.888；
- identity consistency：0.879；
- 16 帧内 identity switches：9。

这与当前多个同类黑碗场景中的目标漂移一致，但该结果来自 LIBERO-Mem，不能替代当前 LIBERO Spatial task 9 的在线审计。

相关脚本：

- `scripts/evaluate_slot_stage1_multitask.py`
- `scripts/visualize_slot_stage1.py`

## 13. 下一步应立刻做什么

不要恢复旧的 Recovery Action Head 计划。下一步是实现并运行当前 task 9 的 Slot 审计器。

建议顺序：

1. 使用 `SegmentationRenderEnv` 而不是普通 `OffScreenRenderEnv`，确认 `instance_to_id` 中：
   - `akita_black_bowl_1`（目标）；
   - `akita_black_bowl_2`（干扰）；
   - `plate_1`；
   - robot。
2. 在 task 9 至少采集：
   - episode 0 成功轨迹；
   - episode 6 空抓轨迹；
   - episode 7 碗被碰动但未抓起轨迹。
3. 同时输出两套 Slot：
   - independent：每帧 `past_slots=None`；
   - tracked：上一帧 Slot 传给下一帧。
4. 用 mask head 输出 16 个预测 mask，与实例 segmentation 计算每帧 IoU 矩阵。
5. 通过 Hungarian matching 或目标 mask 最大 IoU 得到 target Slot，报告：
   - target mask IoU；
   - distractor mask IoU；
   - target/distractor collision rate；
   - target Slot switch rate；
   - 目标移动前后 Slot 是否随动。
6. 保存 JSON、CSV 和叠加视频。
7. 再实现 joints-only、Slot-only、Slot+joints 和 shuffled probes。

LIBERO 已提供 `SegmentationRenderEnv`：

```text
experiments/robot/libero/libero/libero/envs/env_wrapper.py
```

其中有 `segmentation_id_mapping`、`instance_to_id` 和 `get_segmentation_instances`。普通 `OffScreenRenderEnv` 虽然 observation 中可能含 segmentation key，但严格实例审计应使用 `SegmentationRenderEnv`。

## 14. 必须避免的坑

1. **不要相信 smoke success 等于论文复现。** Smoke 只证明形状、梯度和保存路径可用。
2. **不要用 `missing=982` 判断 Stage-1 没加载。** 53MB adapter 加载到含冻结 base 的 wrapper 时，大量 base key missing 是预期的；要检查 Slot key 是否实际加载。
3. **不要把离线一步 advantage 当在线闭环效果。** 已经造成第 0 步破坏成功轨迹。
4. **不要连续使用 `translation_delta=0.35`。** 它是归一化动作量，不是 0.35 米，足以取消/反转 OpenVLA 意图。
5. **不要只报告 accuracy。** episode 7 lifted 全是负样本，必须报告正负分离指标。
6. **不要把连续 220 帧当 220 条独立轨迹。** 有效样本量按 episode 计算。
7. **不要在 episode 7 上继续反复调参。** 它已经被用作开发观察，应另留真正测试 episode。
8. **不要将错误 recovery rollout 当专家动作。** 必须由 oracle 标注正确恢复动作。
9. **不要忽略在线 `past_slots=None`。** 这是 identity 不稳定的直接原因之一。
10. **不要在多卡模型并行下把 Mamba2 输入留在 CPU 或不同 GPU。** Triton 会报 `Pointer argument ... cannot be accessed`。
11. **不要错误使用 `FlashMHA` 占位符做 `isinstance`。** 它可能不是 type，相关兼容修复在 `modeling_prismatic.py`。
12. **不要复用环境后假设 episode 与新建环境严格相同。** 批量环境复用曾导致首帧与独立运行不同；严格成对实验应使用相同环境生命周期。
13. **不要用系统 Python。** 使用 `.conda/envs/openvla-a/bin/python`。
14. **不要删除或重置工作树。** 当前有大量用户/本会话修改和未跟踪文件；禁止 `git reset --hard`、`git checkout --` 或批量清理。
15. **不要误杀 GPU 上其他用户进程。** 当前机器为共享服务器。

## 15. 代码状态与重要新增文件

工作树很脏，大量文件未跟踪或已修改。现有改动属于用户项目，不要清理。

重点新增/修改：

- `experiments/robot/libero/run_openvla_baseline.py`
- `experiments/robot/libero/screen_openvla_episodes.py`
- `experiments/robot/libero/run_advantage_openvla_online.py`
- `experiments/robot/libero/run_advantage_openvla_online_stable.py`
- `experiments/robot/libero/collect_counterfactual_transitions.py`
- `scripts/train_target_state_estimator.py`
- `scripts/precompute_counterfactual_slots.py`
- `scripts/precompute_task_progress_labels.py`
- `scripts/train_task_aware_advantage_scorer.py`
- `scripts/evaluate_slot_stage1_multitask.py`
- `scripts/collect_onpolicy_failure_slots.sh`
- `scripts/collect_recovery_distribution.sh`
- `prismatic/extern/hf/modeling_prismatic.py`
- `prismatic/losses/training_losses.py`
- `prismatic/vla/datasets/datasets.py`

## 16. 快速验证命令

查看环境：

```bash
cd /data/lty/code/naive-e-slotssm
.conda/envs/openvla-a/bin/python -V
nvidia-smi
```

查看 Stage A 历史指标：

```bash
cat /data2/lty/naive-e-slotssm-runs/stage1-multitask-acceptance/metrics.json
```

查看 v3 指标：

```bash
cat /data2/lty/naive-e-slotssm-runs/target-state-estimator-failure-aware-v3/best-metrics.json
```

查看严格 OpenVLA episode 0/6：

```bash
cat /data2/lty/naive-e-slotssm-runs/openvla-online-control/task9-episode0/metadata.json
cat /data2/lty/naive-e-slotssm-runs/openvla-standalone-repeat-v1/task9-episode6/metadata.json
```

## 17. 给下一会话的最终指令

当前不是继续训练 Recovery Action Head，也不是继续跑 episode 7 恢复。

请从 Stage A 开始：

1. 先实现 task 9 的 `SegmentationRenderEnv` + Slot mask/identity 审计脚本；
2. 先跑 1–2 帧预检，确认 segmentation ID 与图像方向完全一致；
3. 再跑成功/空抓/移动目标三类轨迹；
4. 输出定量报告；
5. 根据审计结果决定 Stage-1 是否必须重训；
6. 没有通过 Stage A 门槛前，不要再上线任何恢复控制器。

## 18. 2026-07-22 更新：Stage A task 9 已完成并判定失败

已新增并运行：

- `scripts/preflight_slot_segmentation_task9.py`
- `scripts/audit_slot_stage1_task9.py`

三条轨迹共 549 帧完成严格动作回放，EEF 和对象位置最大误差低于 `6e-8 m`。实例 ID、RGB/segmentation 方向及 mask-head checkpoint 加载均已核验。

正式聚合指标：

| 指标 | independent | tracked |
|---|---:|---:|
| 目标碗 soft IoU | 0.0036 | 0.0060 |
| 干扰碗 soft IoU | 0.1457 | 0.0775 |
| target switch rate | 13.2% | 17.2% |
| 两个黑碗 collision rate | 20.6% | 51.4% |

因此 Stage A 未通过。传递 `past_slots` 不但没有解决 identity，还放大了错误对象绑定。下一步不再继续旧 Stage-1 的 probe 调参，也不进入 Stage C/Recovery；应进入 Stage B，设计并训练带以下监督的新 Stage-1：

1. 仿真实例 mask 和对象身份；
2. `target_slot / distractor_slot / goal_slot / robot_slot` 显式角色；
3. 语言到 target Slot 的匹配，尤其是空间关系；
4. 跨帧同一实例一致性；
5. 目标移动、遮挡、空抓和恢复分布。

完整结果：

```text
/data2/lty/naive-e-slotssm-runs/stage-a-task9-slot-audit/metrics.json
/data2/lty/naive-e-slotssm-runs/stage-a-task9-slot-audit/task9-episode{0,6,7}/
```

## 19. 2026-07-22 更新：Stage B 数据准备与视觉预计算

Stage B 已开始，当前先验证 task 9 的固定角色 Slot 机制，不能将此单任务结果冒充语言泛化。

已采集 task 9 episode 0–9 的实例级训练数据，共 1,520 帧：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-task9-instance-data
```

固定角色协议：

```text
slot 0 = target      = akita_black_bowl_1
slot 1 = distractor  = akita_black_bowl_2
slot 2 = goal        = plate_1
slot 3 = robot
```

按 episode 隔离：

- train：0、1、2、5，共 594 帧 / 72 个 16 帧窗口；
- validation：3、6、8，共 459 帧 / 56 个窗口；
- test：4、7、9，共 467 帧 / 56 个窗口。

manifest：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-task9-instance-data/manifest.json
```

已完成冻结 OpenVLA vision patch 预计算，不再需要在 Stage-B Slot 训练中加载 7B backbone：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-task9-semantic-patches
```

- 10 个 safetensors shard；
- 总大小约 1.7 GB；
- patch shape：每帧 `[256, 2176]`，bf16；
- role mask shape：每帧 `[4, 64, 64]`，uint8；
- 所有 1,520 帧 patch 均为有限值；
- 四类角色在所有帧都有有效像素。

新增脚本：

- `scripts/build_stage1_semantic_manifest.py`
- `scripts/precompute_stage1_semantic_patches.py`

下一项工作：实现固定角色 mask BCE/Dice、target-distractor separation 和跨帧 identity loss，先跑 1–2 step 前向/反向/checkpoint 预检。之后再加入语言条件化并扩展到 Spatial 全任务。

### 19.1 固定角色训练链路预检已通过

新增：

- `scripts/train_stage1_semantic_slots.py`

已完成 `1 step × 2 frames` 预检：

- 旧 Stage-1 tokenizer/mask head 权重 strict load 成功；
- fixed role mask BCE + Dice 正常；
- target/distractor cosine separation 正常；
- detached `past_slots` 时序一致性 loss 正常；
- 所有梯度有限，optimizer 更新成功；
- validation 前向成功；
- 51 MB checkpoint 保存并 strict reload 成功。

输出：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-task9-semantic-preflight
```

预检原始梯度范数约 499.6，已经裁剪到 1.0。这说明链路可用，但正式训练前必须先用更低学习率和短稳定性运行观察 loss/grad norm，不能直接启动长训练。当前 checkpoint 只证明训练协议可运行，不是可评测模型。

### 19.2 固定角色 Stage-B 原型的严格结果：未通过

新增全 episode 评测器：

- `scripts/evaluate_stage1_semantic_slots.py`

对 validation 459 帧和 test 467 帧进行了固定 slot 角色的完整评测，不再依赖少量随机窗口。

关键结果：

| checkpoint | validation target soft IoU | test target soft IoU |
|---|---:|---:|
| 旧 Stage-1 固定 slot 0 | 0.00436 | 0.00387 |
| 标准 loss 20-step | 0.02382 | 0.00211 |
| 标准 loss 100-step | 0.00116 | 未继续采用 |
| 小对象加权 100-step | 0.00165 | 未继续采用 |
| attention 对齐 20-step | 0.00279 | 未继续采用 |

标准 20-step 在 validation 上短暂改善，但在严格 test 上 target 反而下降。100-step 原型主要学会了容易的 goal/robot：标准 100-step validation goal IoU 为 0.0864，但 target 塌到 0.00116。直接监督 16×16 SlotAttention map 也没有改善小碗目标。

结论：固定 slot 索引 + mask/Dice + detached past slots 不能提供可泛化的目标绑定。不要继续增加这条训练的步数或调 loss 权重。下一版 Stage B 必须加入：

1. 任务语言和空间关系条件化的 target query；
2. reference-object query（例如 cabinet）与 target query 的显式关系；
3. 比 16×16 更高分辨率的小目标视觉特征或多尺度 mask head；
4. 全 Spatial tasks 的语言/关系变化，而非只训练 task 9；
5. 继续使用 episode-isolated validation/test 和当前完整评测器。

评测结果：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-task9-semantic-evaluation
```

### 19.3 Stage-B v2：关系 schema 已建立

固定角色原型在严格 test 上未通过后，已停止继续堆训练步数。新增机器可读的 Spatial 关系标签：

- `experiments/robot/libero/stage_b_spatial_role_schema.json`
- `scripts/validate_stage_b_spatial_schema.py`

schema 已与全部 10 个 LIBERO Spatial BDDL 文件逐项校验通过。共同角色为 target bowl 1、distractor bowl 2、goal plate 和 robot；变化的关系包括：

- `between(plate, ramekin)`；
- `next_to(ramekin/cookies/plate)`；
- `on(cookies/ramekin/stove/cabinet)`；
- `in(cabinet)`；
- `table_center`。

下一步应以该 schema 为唯一标签源，采集 target/reference/goal/distractor/robot 实例 mask，并训练语言/关系条件化 target query。不要再回到固定 slot 编号假设。

### 19.4 Stage-B v2：关系 mask collector 已打通

新增：

- `scripts/collect_stage_b_spatial_relation_data.py`

该 collector 直接读取 `stage_b_spatial_role_schema.json`，不在脚本内重复定义任务关系。每个样本保存 RGB、人工检查 overlay，以及命名的压缩二值 mask：`target`、`distractor`、`goal`、`robot`、可选 `reference_0/reference_1`。mask 不再编码为互斥颜色，因此 task 0 的双 reference 和 plate 同时作为 goal/reference 的情况不会丢失标签。

已完成全部 10 个 Spatial task 各 1 个初始状态的 smoke test：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-spatial-relation-smoke-all
```

- 共 10 个样本，覆盖 `between/in/next_to/on/table_center`；
- task 0 正确生成两个 reference；task 2 正确不生成 reference；其余任务生成一个；
- 所有 target、distractor、goal、reference 和 robot mask 均超过 16 像素；
- target 可见面积范围为 537–1,243 像素，未发现空 mask；
- 输出 NPZ 的数组均为 `[256, 256] uint8`，overlay 已人工检查。

下一步不是马上长训练。先采集全部 initial states，按 `(task_id, state_id)` 做隔离 split；然后在冻结视觉 patch 上实现 relation embedding + reference-query 与 target-query 的交互，并用 held-out task/relation 评测目标绑定。

### 19.5 Stage-B v2：全量 initial-state 数据与双评测协议已完成

已采集全部 10 个 Spatial task × 50 个官方 initial states，共 500 个关系监督样本：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-spatial-relation-data
```

数据约 62 MB，共 1,502 个文件（每样本 RGB、overlay、NPZ，加两份 manifest）。全量质量校验已通过：

- target：500/500 可见，414–1,302 像素；
- distractor：500/500 可见，455–1,233 像素；
- goal：500/500 可见，1,299–1,450 像素；
- `reference_0`：450 个需要 reference 的样本全部可见，179–9,762 像素；
- `reference_1`：50 个 `between` 样本全部可见，300–480 像素；
- 所有 NPZ role key、`[256,256] uint8` shape/dtype、二值性和最低 16 像素约束通过。

关系分布：`between=50`、`in=50`、`next_to=150`、`on=200`、`table_center=50`。

新增严格 split 生成与校验器：

- `scripts/build_stage_b_spatial_relation_manifest.py`
- `/data2/lty/naive-e-slotssm-runs/stage-b-spatial-relation-data/split_manifest.json`

两套协议：

1. state generalization：train 350（state 0–34）、validation 70（35–41）、test 80（42–49），state ID 不交叉；
2. compositional task generalization：train tasks 0/1/2/4/7/9 共 300，validation tasks 5/6 共 100，test tasks 3/8 共 100，task 不交叉；训练集包含全部关系词，验证/测试保留未见过的 relation-reference-layout 组合。

采集器还完成两项等价加速：沉降的 20 个 no-op 步关闭中间相机 observable，只在最终状态渲染；同一 task 使用 soft reset。快速版本与原始版本的实例 mask 逐像素完全一致。

下一步：为这 500 张图预计算冻结视觉特征，并实现最小的 relation-conditioned query 原型。先做 state-generalization 与 compositional-task 两个协议的短训练/完整评测，不进入 Stage C。

### 19.6 Stage-B v2：关系视觉 patch 已完成

新增：

- `scripts/precompute_stage_b_spatial_relation_patches.py`

已在隔离的物理 GPU 0 上完成全部 500 张图的冻结 OpenVLA vision patch 预计算：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-spatial-relation-patches
```

- 10 个 task shard，总大小约 544 MB；
- 每 task 50 个状态；
- patch shape：`[50,256,2176]`，bf16；
- role mask shape：`[50,6,64,64]`，通道顺序为 target、distractor、goal、reference_0、reference_1、robot；
- 额外保存 `role_presence [50,6]`、`state_ids [50]` 和 `relation_ids [50]`；
- relation 顺序固定为 between、in、next_to、on、table_center。

全量 safetensors 回读校验通过：500/500 patch 均为有限值；mask 全部为二值；presence 与 mask 非空性完全一致；不存在的 reference mask 严格为零；500 个 `(task_id,state_id)` 无重复。task 0 双 reference、task 2 无 reference，其余 task 单 reference 的 presence 数量均正确。

下一步可直接在 544 MB patch 数据上实现 relation-conditioned query，不再加载 7B backbone。最小模型必须显式消费 `relation_id` 和 `role_presence`，并同时输出 target/reference mask；禁止退化回仅靠 task ID 或固定 slot 0 的分类器。

### 19.7 Relation-query 原型：state 泛化通过，组合泛化未通过

新增：

- `scripts/train_stage_b_relation_queries.py`

模型不接收 task ID。它先用 reference queries 对 patch 做 cross-attention，再用 `relation embedding + reference context` 构造 target query；空间相似度通过 16×16→64×64 残差上采样头同时预测六类 mask。训练链路的有限梯度、optimizer、checkpoint 保存和 strict reload 均通过。

200-step state-generalization 结果：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-relation-query-state-200step
```

- validation target soft IoU：0.5534；test：0.5600；
- validation reference_0：0.4376；test：0.4329；
- validation distractor：0.5090；goal：0.7655；robot：0.7921；
- 训练末 target/distractor 概率重叠约 0.01。

这证明冻结 patch 上的空间 query/mask head 有能力学习小碗和 reference 分割，修复了旧固定-slot 头几乎为零的结果。但严格 compositional-task 协议未通过：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-relation-query-compositional-200step
```

- held-out validation tasks 5/6：target 0.0822，reference_0 0.0177；
- held-out test tasks 3/8：target 0.3622，reference_0 0.0583；
- task 5：target 0.1529 / reference 0.0354；
- task 6：target 0.0115 / reference 0.000015；
- task 3：target 0.3874 / reference 0.00213；
- task 8：target 0.3371 / reference 0.1145。

根因进一步收窄：当前 query 只有 relation ID，没有 reference object 的语言/类别语义。compositional train tasks 的 reference 包括 plate/ramekin/cabinet/stove，但 held-out tasks 3/6 使用从未作为训练 reference 出现的 cookies，因此 reference query 几乎完全失败；target 在部分布局上的中等 IoU 不能证明关系绑定成功。

下一步必须加入 instruction/reference phrase 的冻结语言表示，并做消融：完整语言、打乱 relation、打乱 reference phrase、移除 reference context。只有完整模型在 held-out task 上显著优于这些 counterfactual baselines，才算真正学到空间语言绑定。当前 200-step checkpoint 不进入 Stage C。

### 19.8 冻结 Llama phrase 特征：改善已见 reference，未解决开放词汇 grounding

新增：

- `scripts/precompute_stage_b_language_features.py`
- `/data2/lty/naive-e-slotssm-runs/stage-b-spatial-relation-patches/language_features.safetensors`

该脚本只从 OpenVLA checkpoint 的 embedding shard 读取 `language_model.model.embed_tokens.weight`，不加载完整 7B。已为 10 条 instruction、relation phrase 和最多两个 reference phrase 生成 `[4096]` mean-token frozen features。训练模型使用这些特征构造 reference/target query，数字 task ID 不进入模型。

200-step compositional 结果：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-language-query-compositional-200step
```

相对无语言 relation-query：

- aggregate validation reference_0：0.0177 → 0.0944；
- aggregate test reference_0：0.0583 → 0.3764；
- aggregate test target：0.3622 → 0.3894；
- task 8 plate reference：0.1145 → 0.7462。

但未见类别 `cookie box` 仍失败：

- task 6 reference：0.000015 → 0.000143，target 0.0215；
- task 3 reference：0.00213 → 0.00665，target 0.4189。

因此 raw Llama token embedding 只能帮助已见/相似 phrase，不能把未见文本类别可靠对齐到 DINO+SigLIP image patches。task 3 的 target 中等 IoU 在 reference 几乎为零时仍不能算关系绑定成功。

已检查本机 Hugging Face 缓存：`google/siglip-so400m-patch14-384` 只有空的 repo ref，没有 config 或 text-tower 权重；当前环境在线加载失败。下一步需要取得与 vision backbone 配套的 SigLIP text tower，或从 OpenVLA 联合多模态层提取 contextualized language/patch features。没有对齐 text tower 前，不继续堆 relation-query 步数，也不进入 Stage C。

### 19.9 SigLIP text tower 已验证；问题进一步定位为 dense/关系泛化

用户提供的模型已验证可离线完整加载：

```text
/data2/lty/models/siglip-so400m-patch14-384
```

- `SiglipModel` 共约 8.78 亿参数；
- text/vision hidden dim 均为 1,152；
- 新增 `scripts/precompute_stage_b_siglip_language_features.py`；
- 已生成 20 个唯一 instruction/relation/reference phrase 的归一化 SigLIP text features：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-spatial-relation-patches/siglip_language_features.safetensors
```

200-step 原 open-vocabulary compositional 协议结果：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-siglip-query-compositional-200step
```

- aggregate test target 0.4068，reference_0 0.3524；
- task 8 plate reference 0.7018；
- 但 task 3 cookie reference 0.00264、task 6 cookie reference 0.00020，仍未通过。

进一步检查发现 OpenVLA patch 是 DINO 1,024 + SigLIP 1,152 的拼接，但取自两个视觉塔的倒数第二层。原生 SigLIP text 与旧 SigLIP patch 的 top-k reference IoU 几乎为零。使用下载模型的 384px final-layer 27×27 patch 做少量零样本检查，cookie top-k IoU 仍只有 0–0.018；SigLIP 全局对比特征本身不提供可靠的小物体 dense grounding。

为避免把新类别 grounding 与关系组合泛化混为一谈，新增两个无 task 泄漏且 reference-category-seen 的协议：

- fold A：task 6 `next_to(cookie)` 在训练，task 3 `on(cookie)` 在测试；
- fold B：task 3 `on(cookie)` 在训练，task 6 `next_to(cookie)` 在测试。

结果：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-siglip-reference-seen-fold-a-200step
/data2/lty/naive-e-slotssm-runs/stage-b-siglip-reference-seen-fold-b-200step
```

- fold A test task 3：target 0.6188，但 reference 仅 0.0480；
- fold B test task 6：target 0.0200，reference 0.000065；
- 两个 fold 都未满足“target 与 reference 同时正确”的绑定标准。

结论：下载模型有效，但并不能直接解决 dense grounding。当前瓶颈已不再是缺少 text tower，而是：全局 SigLIP 特征无法稳定定位小 reference，且 `on ↔ next_to` 的关系组合泛化高度不对称。下一步应引入显式 dense grounding 模型/更高分辨率实例特征，或在训练数据中加入同一场景的关系反事实监督；不能把 fold A 的 target 0.62 单独当作通过，也不进入 Stage C。

### 19.10 下一阶段启动：relation-invariant grounding 与原生反事实 schema

首先修正了结构错误：reference query 不再加入 relation embedding；物体身份 grounding 与 `on/next_to/in` 解耦，relation 只进入 target query。相同 200-step reference-seen folds 结果：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-invariant-reference-fold-a-200step
/data2/lty/naive-e-slotssm-runs/stage-b-invariant-reference-fold-b-200step
```

- fold A `next_to(cookie)→on(cookie)`：test reference 0.0480 → 0.1827，target 0.5875；
- fold B `on(cookie)→next_to(cookie)`：test reference 仍为 0.000059，target 0.0292。

这证明 reference/query 解耦正确，但仍不足。fold B 失败的原因是 task 3 中 cookie 被 bowl 覆盖、只提供很小的可见监督，无法迁移到 task 6 的完整外观。

随后检查全部 Spatial BDDL，发现每个场景天然给 bowl 2 指定了另一条空间关系。例如：

- task 1：bowl 1 next_to ramekin，bowl 2 next_to cookie；
- task 5：bowl 1 on ramekin，bowl 2 on cookie；
- task 6：bowl 1 next_to cookie，bowl 2 on stove；
- task 9：bowl 1 on cabinet，bowl 2 on stove。

因此无需人工移动物体即可构造同场景双候选反事实监督。`stage_b_spatial_role_schema.json` 已为全部 10 task 新增 `counterfactual`：bowl 2 target、relation、references 和原始 BDDL predicate；`validate_stage_b_spatial_schema.py` 已逐项验证全部谓词与实例通过。

下一步应扩展 mask collector/precompute 格式，保存 counterfactual target/reference masks；训练时同一 patch 同时形成 bowl1-query 和 bowl2-query 两个样本。重点消融是：交换 relation/reference phrase 后 target mask 是否随之从 bowl1 切到 bowl2。只有这种同图 counterfactual switch 成功，才证明模型没有靠布局或固定 bowl identity 投机。

### 19.11 同图双候选反事实数据已完成

`scripts/collect_stage_b_spatial_relation_data.py` 已扩展为读取 schema v2 的 `counterfactual`，每帧额外保存：

- `counterfactual_target`：bowl 2 mask；
- `counterfactual_reference_0/1`；
- counterfactual relation、reference instances 和 BDDL predicate。

全量 500 状态已重新采集到独立目录，不覆盖原数据：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-spatial-counterfactual-data
```

严格校验：500/500 counterfactual target 与原 distractor mask 逐像素完全一致；target 可见 455–1,233 像素；reference 可见 178–9,762 像素；所有 mask 均为 `[256,256] uint8` 二值数组。

新增：

- `scripts/build_stage_b_counterfactual_patches.py`

该脚本复用已有冻结视觉 patch，不重新运行 7B backbone，附加反事实 mask/relation：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-spatial-counterfactual-patches
```

- 10 个 task shard，500 样本，约 549 MB；
- `counterfactual_target_masks [50,64,64]`；
- `counterfactual_reference_masks [50,2,64,64]`；
- `counterfactual_reference_presence [50,2]`；
- `counterfactual_relation_ids [50]`；
- patch 全部有限，反事实 target 与 `role_masks[:,distractor]` 完全相等，presence 与非空 reference mask 完全一致。

SigLIP language feature 文件也已加入 10 个 counterfactual relation/reference phrase features。下一步训练器应把每个物理样本展开为 primary/counterfactual 两条 query，共 1,000 个查询；评测必须报告同图 switch accuracy、primary target IoU、counterfactual target IoU，以及错误 bowl IoU。不要只报告平均 mask IoU。

### 19.12 双查询训练闭环完成；普通 IoU 通过但严格 switch 失败

`scripts/train_stage_b_relation_queries.py` 现支持 `--counterfactual-training`：每次抽取物理图像后展开成 primary/counterfactual 成对 query；counterfactual role 顺序重新组装为 bowl 2 target、bowl 1 distractor，并使用对应 relation/reference 和语言特征。训练器新增同图严格指标：

- primary prediction 对正确/错误 bowl 的 soft IoU；
- counterfactual prediction 对正确/错误 bowl 的 soft IoU；
- 两个方向同时选对才计 1 的 `paired_switch_accuracy`；
- 两条预测的 `prediction_change_l1`。

SigLIP 文本文件已重新生成（22 个唯一文本，feature dim 1,152）：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-spatial-counterfactual-patches/siglip_language_features.safetensors
```

1-step smoke 的梯度、checkpoint 保存及严格重载均通过。随后完成 fold B 200-step：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-counterfactual-query-fold-b-200
```

训练 target soft IoU 已到 0.7426；普通 test target 0.5703、reference_0 0.4217，看平均指标会误判为成功。但严格 test switch 显示：

- primary：正确 bowl 0.5703，错误 bowl 0.2239；
- counterfactual：正确 bowl 0.2651，错误 bowl 0.5452；
- `paired_switch_accuracy = 0.0`；
- `prediction_change_l1 = 0.0060`。

validation switch accuracy 为 0.48，但 held-out task 6 完全不切换。schema 核查确认 task 6 两条 query 并不相同：primary 是 `next_to(cookie)`，counterfactual 是 `on(stove)`，因此不是重复文本或不可辨识标签造成的。当前模型仍主要依赖 scene/task 布局选择固定 bowl；像素 BCE/Dice 的双查询监督本身不足以让 attention 学会关系组合。Stage B 仍未通过，不能进入 Stage C。

下一步应加入显式成对排序/交换约束：primary logits 在 bowl 1 区域必须高于 bowl 2，counterfactual logits 反之，并直接惩罚两条 target prediction 过于相同；之后先在 fold B 复跑，严格门槛以 held-out `paired_switch_accuracy` 和两个方向 correct-vs-wrong margin 为准。

显式排序已实现并完成同配置 200-step 对照：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-counterfactual-ranking-fold-b-200
```

排序约束直接比较每条 target logit 在正确 bowl 与错误 bowl mask 内的均值，margin 设 1.0、权重 0.5。训练末 `switch_ranking_loss=0`、平均 margin 11.375，说明约束已充分拟合；普通 test target 也由 0.5703 微升到 0.5930。然而 held-out 严格结果仍失败：

- primary：正确 0.5930，错误 0.2020；
- counterfactual：正确 0.2736，错误 0.5210；
- `paired_switch_accuracy=0.0`；
- `prediction_change_l1=0.00726`。

因此继续调大 ranking 权重没有依据：它已经在训练集零损失，却不能跨 task 切换。下一步需要改变表征/推理分解，而非继续修改同一个 loss。优先方案是把 target 选择显式写成“先独立分割两个 bowl 候选，再分别计算候选与已 grounding reference 的空间谓词分数，最后由 relation query 在候选间选择”；关系分数应使用 mask 几何（on/in/next_to/between/table_center）监督并跨 task 共享，不再让一个自由 attention query 同时学习物体、关系和 task 布局。

### 19.13 Oracle mask 几何 probe：分解路线成立，但暴露 cookie 遮挡上限

新增：

- `scripts/probe_stage_b_oracle_geometry.py`

probe 不使用 image patch、task id 或语言 embedding，只从两个真值 bowl 候选及 query-specific reference masks 提取 40 维共享几何特征：mask 面积、质心、bbox、候选/reference 相对位移和距离、交并比、双 reference 中点距离、table-center 距离，再用同一个小 MLP 对两个候选打分。

fold B 200-step：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-oracle-geometry-fold-b.json
```

- train 1.000；validation 0.995；held-out task 6 test 1.000；
- test mean correct margin 17.25。

这证明 task 6 的 `next_to(cookie)` / `on(stove)` 切换在二维 mask 几何中完全可分，端到端 attention 的失败不是数据本身不可辨识。

fold A：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-oracle-geometry-fold-a.json
```

- train 1.000；validation 1.000；held-out task 3 test 0.700；
- task 3 counterfactual `on(cabinet)` 为 1.000；
- task 3 primary `on(cookie)` 仅 0.400。

task 3 的 cookie 被 bowl 遮挡，真值可见 reference mask 本身就不足以稳定表达 `on(cookie)`；这与此前 fold A reference grounding 明显优于 fold B、但仍不完整的观察一致。结论是显式几何分解应优先在 fold B 落地，它有 100% oracle 上限；fold A 不能要求纯 RGB 可见 mask 几何达到 100%，后续需要 amodal reference/object-state 信息或 simulator 3D 几何处理遮挡。

### 19.14 预测 mask 三级级联：主瓶颈是两个 bowl candidate 塌缩

oracle geometry probe 已支持加载 relation-query checkpoint，并保持 geometry scorer 只在 oracle train masks 上训练，再进行四种替换评测。fold B 结果：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-geometry-cascade-fold-b.json
```

held-out task 6：

- oracle bowls + oracle references：1.00；
- predicted bowls + oracle references：0.71；
- oracle bowls + predicted references：1.00；
- predicted bowls + predicted references：0.65。

因此 reference prediction 虽然 mask IoU 不完美，但保留的几何信息足以在 oracle bowl 候选间 100% 选择；主要损失来自 bowl candidates。test mask 诊断进一步确认：

- candidate 0 IoU：0.630；
- candidate 1 IoU：0.243；
- 两个 predicted candidate 的相互 IoU：0.906；
- empty candidate rate：0；
- primary/counterfactual reference_0 IoU：0.456/0.495。

也就是说不是 bowl 漏检为空，而是 target/distractor 两个 head 几乎输出同一个 bowl。下一步必须增加独立于 relation/language 的 class-agnostic bowl instance proposal head，固定输出两个互斥候选；关系 query 只能在候选生成之后选择，不能反向定义候选身份。候选阶段的严格门槛应为：两个候选分别匹配 bowl1/bowl2 的 permutation IoU，同时 candidate-pair overlap 显著低于当前 0.906。

### 19.15 独立 bowl proposals + 显式几何选择：fold B 严格 switch 首次通过

新增：

- `scripts/train_stage_b_bowl_proposals.py`

该模型只输入冻结的 2,176-d image patches，完全不接 instruction、relation、reference、task id。两个 proposal queries 用 permutation-matched BCE/Dice 监督，并显式惩罚 proposal overlap。

fold B 300-step：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-bowl-proposals-fold-b
```

- train permutation soft IoU 0.838；pair overlap 0.00005；
- held-out task 6 permutation soft IoU 0.811；pair overlap 0.0010；
- validation tasks 5/8 permutation soft IoU 0.329，说明跨场景 proposal 泛化仍不均衡，后续必须单独处理。

随后将该 proposal checkpoint 接入 oracle-trained geometry scorer，并继续使用旧 relation model 的 predicted reference masks：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-geometry-cascade-proposals-fold-b.json
```

held-out task 6 的 100 个 primary/counterfactual queries：

- oracle bowls + oracle references：1.00；
- predicted bowls + oracle references：1.00；
- oracle bowls + predicted references：1.00；
- predicted bowls + predicted references：1.00；
- 两个 query 方向及 `next_to/on` relation 均为 1.00；
- full-predicted mean correct margin 10.04；
- predicted bowls permutation mean binary IoU 0.866，pair overlap 0.0010；
- predicted primary/counterfactual reference IoU 0.456/0.495。

这是 Stage B 首次在 held-out task 上通过同图严格 switch，验证了正确结构是：language-independent instance proposals → reference grounding → shared mask geometry → relation-conditioned candidate selection。proposal slot 的编号发生全局交换是 permutation 模型的正常现象；选择器输出 proposal index 后直接取对应 mask，不需要恢复 bowl1/bowl2 名称。

但目前只能宣布 fold B 通过，不能宣布整个 Stage B 完成：fold A 存在 cookie 遮挡的 oracle 上限，且 validation task 5/8 proposal IoU 仅 0.329。下一步先评测独立 proposals 在 fold A 及全部逐 task 的 permutation IoU，再决定增加训练增强/更强 instance decoder，还是为遮挡任务引入 amodal/3D state。

相同配置的 fold A proposal 已完成：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-bowl-proposals-fold-a
```

- train permutation soft IoU 0.836，overlap 0.00003；
- validation 0.492，overlap 0.137；
- held-out task 3 test 0.246，overlap 0.0067。

fold A 的主要问题在进入关系几何之前就已出现：language-independent proposal 无法从其他 task 的冻结 16×16 patches 泛化到 task 3 的两个 bowl。即使 proposal 修复，task 3 `on(cookie)` 仍受前述 oracle-visible-geometry 仅 40% 的第二重上限。因此近期工程主线应以 fold B 的已通过分解为可行基线；fold A 作为遮挡/跨场景实例泛化专项，不能通过继续调 relation loss 解决。

### 19.16 Proposal 泛化增强与 fold A 完整级联

state-generalization proposal 300-step：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-bowl-proposals-state-generalization
```

所有 10 task 在严格 state-held-out test 上 permutation soft IoU 均为 0.767–0.874，整体 0.816、pair overlap 0.004。这证明 decoder 对每个已见场景都能稳定分开两个 bowl；fold A 的低值是 held-out scene/layout 泛化，而非模型容量不足。

训练器新增同步水平翻转（16×16 patch grid 与 64×64 masks）和 patch dropout。fold A 225-step 消融：

- 无增强：test IoU 0.246，overlap 0.0067；
- flip only：0.446，overlap 0.218；
- flip + 5% dropout：0.547，overlap 0.136；
- flip + 5% dropout + overlap weight 2.0：0.566，overlap 0.118。

当前最佳 fold A proposal：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-bowl-proposals-fold-a-augmented-overlap2-225
```

增强显著削弱布局 shortcut，但仍低于 fold B。对应 fold A relation/reference 模型使用无泄漏 split 训练 75 steps（ranking 从 step 50 起为零）：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-counterfactual-ranking-fold-a-75
```

最终结构化级联：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-geometry-cascade-proposals-fold-a.json
```

held-out task 3：

- oracle bowls + oracle visible references：0.70（primary `on(cookie)` 0.40，counterfactual 1.00）；
- predicted bowls + oracle references：0.83；
- oracle bowls + predicted references：1.00；
- predicted bowls + predicted references：0.93；
- full-predicted primary 0.86，counterfactual 1.00；
- proposal permutation binary IoU 0.595，pair overlap 0.118；
- predicted primary cookie reference visible-mask IoU 0.298，counterfactual cabinet reference 0.833。

重要现象：predicted reference 超过 oracle visible reference，并非评测泄漏——geometry scorer 仍只用训练任务 oracle masks 训练。更合理的解释是 reference head 利用上下文生成了超出可见 cookie 像素的近似 amodal/关系区域，使被 bowl 遮挡的 `on(cookie)` 从 40% 提升到 oracle-bowl 条件下 100%。这说明不能把可见 instance-mask IoU 当作遮挡关系 grounding 的唯一指标。

当前严格状态：fold B full-predicted switch 1.00；fold A 0.93。Stage B 的结构性根因已修复，但若门槛要求两个 folds 都 100%，仍差 task 3 primary 的 7/50 状态。下一步应逐帧导出这 7 个错误状态及 proposal/reference/geometry 可视化，判断剩余错误来自 proposal 缺失、候选重叠还是几何 scorer margin，再做有针对性的最后修复。

### 19.17 错误审计、全局 matching 修正与 Stage B 双 fold 严格通过

级联报告现保存逐 query errors（task/state/variant/relation/scores/margin），并新增：

- `scripts/render_stage_b_cascade_errors.py`

最初的 7 个 fold A “错误”可视化后发现评测标签存在问题：permutation proposal 中，一个 slot 有时覆盖两只 bowl，另一个 slot 是更干净的单 bowl；旧代码分别为每个 target 选择最高 IoU proposal，会产生冲突匹配和假阴性。现已修正为对完整 2×2 proposal-target IoU matrix 做全局二选一 permutation matching，与 proposal 训练/评测定义一致。

修正后 75-step reference + 原 proposals 的 fold A full-predicted 从 0.93 更新为 0.95，真实错误为 states 14/17/24/29/32。错误面板与数值诊断：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-fold-a-true-error-panels.png
/data2/lty/naive-e-slotssm-runs/stage-b-fold-a-true-error-panels.json
```

5 个错误共同表现为 proposal0 混合两只 bowl、proposal1 只覆盖部分 bowl1；state32 proposal1 为空。新增纯后处理 `--component-proposals`：对两个 proposal union 做 connected components，取最大两个互斥实例。结果：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-geometry-cascade-components-fold-a.json
```

- proposal permutation binary IoU：0.595 → 0.717；
- pair overlap：0.118 → 0；
- state32 修复；
- full-predicted switch：0.95 → 0.96；
- 剩余 states 14/17/24/29 与 oracle bowls + predicted references 的错误完全一致，说明最后瓶颈已转为 reference contextual geometry。

此前 fold A reference 模型仅因 GPU0 拥堵训练 75 steps。切换到空闲 GPU2 后完成原计划 200 steps：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-counterfactual-ranking-fold-a-200
/data2/lty/naive-e-slotssm-runs/stage-b-geometry-cascade-components-fold-a-reference200.json
```

最终 fold A held-out task 3：

- oracle bowls + predicted references：1.00；
- component predicted bowls + predicted references：1.00；
- primary `on(cookie)`：50/50；
- counterfactual `on(cabinet)`：50/50；
- full-predicted strict switch：100/100。

值得注意的是 primary reference 对 visible cookie mask 的 IoU 从 75-step 的 0.375 降到 200-step 的 0.289，但关系选择反而从 96% 到 100%。这再次确认遮挡 reference 的正确表征是 contextual/amodal relation region，不应以 visible instance IoU 单独选 checkpoint。

至此，两个 reference-seen compositional folds 的结构化 full-predicted strict switch 均为 1.00：

- fold A：held-out `on(cookie)` task 3，1.00；
- fold B：held-out `next_to(cookie)` task 6，1.00。

Stage B 的关系绑定/同图反事实门槛已通过。有效结构为：language-independent bowl proposals（必要时 union connected-components 实例化）→ contextual reference grounding → shared mask geometry → relation-conditioned candidate selection。下一步可以整理成单一可加载 inference module、增加严格回归测试，然后再讨论进入 Stage C；不要重新退回自由 target-attention 作为 selector。

### 19.18 Stage B 已固化为统一 inference module 与双 fold 回归

新增统一可加载模块：

- `scripts/stage_b_spatial_selector.py`
- `StageBSpatialSelector.from_config(...)`

固定推理路径：

```text
frozen patches
  → language-independent BowlProposalSegmenter
  → proposal-union connected components
  → RelationQuerySegmenter 的 contextual reference masks（不使用其 target 输出）
  → 40-d candidate/reference mask geometry + relation one-hot
  → FP32 GeometryScorer
  → selected target mask / candidate scores / selection margin
```

统一输出包含：`target_masks`、两个互斥 `candidate_masks`、原始 proposal masks、reference masks、candidate scores、selected index、selection margin 和 geometry features。forward 接口不接受 task id。

geometry probe 已支持 `--geometry-checkpoint`，保存 safetensors 后立即 strict reload。确定性 fold checkpoints：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-selector-fold-a/geometry-scorer.safetensors
/data2/lty/naive-e-slotssm-runs/stage-b-selector-fold-b/geometry-scorer.safetensors
```

完整配置：

- `experiments/robot/libero/stage_b_selector_fold_a.json`
- `experiments/robot/libero/stage_b_selector_fold_b.json`

新增严格回归：

- `scripts/regress_stage_b_spatial_selector.py`

它不再拼接 probe 的中间输出，而是从配置严格加载 proposal/reference/geometry 三个 checkpoint，只调用统一 selector 重放两个 held-out test folds。最终报告：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-selector-regression.json
```

结果 `passed=true`：

- fold A：primary 1.00、counterfactual 1.00、paired switch 1.00；minimum margin 0.114；
- fold B：primary 1.00、counterfactual 1.00、paired switch 1.00；minimum margin 0.967；
- 两 fold maximum candidate overlap IoU 均为 0；
- checkpoint strict reload、无 task-id input、split leakage checks 全部通过。

统一重放首次曾在 fold A state39 得到 BF16 geometry scores 完全相等（3.796875/3.796875），使准确率短暂为 0.98。原因是旧 probe 的 geometry scorer 使用 FP32，而统一模块被外层 autocast 到 BF16。现已强制这个仅 20 KB 的控制头使用 FP32；修正后 state39 margin 为正且双 fold 100%。后续不得将 geometry scorer/margin 降为 BF16。

Stage B 现在不仅实验上通过，也完成了可加载、可复现、可回归的工程闭环。下一步可以正式设计 Stage C 输入契约：优先消费 `target_masks/candidate_masks/reference_masks/relation_ids/selection_margin`，并将低 margin 作为 failure-aware uncertainty；不要让 Stage C 重新学习自由 target selection。

### 19.19 单场景无头恢复演示已成功（近期目标达成）

用户将近期目标收敛为：允许单场景过拟合，至少得到一个“发生失误后恢复并最终成功”的无头仿真 MP4。新增：

- `experiments/robot/libero/run_single_scene_recovery.py`

选择已知 nominal 成功的 LIBERO Spatial task 9 episode 0：`pick up the black bowl on the wooden cabinet and place it on the plate`。rollout 结构：

- steps 0–24：nominal OpenVLA；
- steps 25–32：强制执行 8 步错误动作 `[-0.8,+0.8,+0.4,0,0,0,-1]`，把机械臂从原接近轨迹移开；
- step 33：标记 failure detected；
- steps 33–67：从受扰动后的真实 observation 逐帧重新调用 OpenVLA 规划；
- 之后恢复原任务闭环，step 137 `env.check_success()=true`。

产物：

```text
/data2/lty/naive-e-slotssm-runs/single-scene-recovery-v1/task9-episode0.mp4
/data2/lty/naive-e-slotssm-runs/single-scene-recovery-v1/task9-episode0/metadata.json
/data2/lty/naive-e-slotssm-runs/single-scene-recovery-v1/task9-episode0/steps.jsonl
```

视频验收：H.264，256×256，20 FPS，148 frames，7.4 秒；画面叠加 `NOMINAL / FORCED ERROR / RECOVERY / RESUMED TASK / SUCCESS`。metadata 中 `completed=true, success=true`；steps 分布为 nominal 25、forced error 8、recovery 35、resumed task 70，最后一条仿真 success 为 true。

必须准确描述其局限：这是单场景 overfit demonstration；错误注入时间和 failure detection 时间由脚本预设，不是学习到的 detector。恢复动作不是录制轨迹或 simulator state reset，而是 OpenVLA 从受扰动状态的 RGB observation 重新闭环规划并最终完成任务。因此它满足“至少一个可见恢复成功场景”的近期交付目标，但不证明跨场景 failure-aware 泛化。

### 19.20 更晚且可辨认的 LOST GRASP 恢复成功

用户指出 19.19 的错误发生在抓取前，画面上只是机械臂路径偏移，不能直观看成“操作失败”。因此重新设计为语义明确的 late failure：

1. nominal OpenVLA 先正确接近、闭合夹爪并抬起目标碗；
2. steps 58–64 强制张开夹爪并向上移动，使碗从夹爪掉落；
3. detector 要求目标曾被抬升超过 6 cm，随后高度跌回 3 cm 以下，step 67 才触发 `FAILURE DETECTED`；
4. 单场景 oracle recovery state machine 执行 `RETURN TO BOWL → DESCEND → CLOSE GRIPPER → VERIFY LIFT`；
5. 只有碗重新抬升超过 6 cm 才将控制交回 OpenVLA完成放置。

最终成功产物：

```text
/data2/lty/naive-e-slotssm-runs/single-scene-recovery-lost-grasp-v3/task9-episode0.mp4
/data2/lty/naive-e-slotssm-runs/single-scene-recovery-lost-grasp-v3/task9-episode0/metadata.json
/data2/lty/naive-e-slotssm-runs/single-scene-recovery-lost-grasp-v3/task9-episode0/steps.jsonl
```

严格日志证据：

- step 57：nominal 已抓起，target lift 3.6 cm 且继续上升；
- step 58–64：`LOST GRASP`，开爪并上移，peak lift 约 9.2 cm；
- step 67：碗已落回约 1.1 cm，触发 failure detector；
- recovery：return 18 steps、descend 13、close 8、verify lift 11；
- step 130：重抓后 target lift 12.5 cm，`recovery_completed=true`；
- step 173：`env.check_success()=true`；
- MP4：H.264，256×256，20 FPS，184 frames，9.2 秒。

中间尝试必须保留为负结果：

- `missed-grasp-v1` 确实重抓并抬起，但最终把碗掉在 plate 前方，success=false；
- `lost-grasp-v1` 强制开爪后碗仍卡在夹爪中，detector 到放置时才误触发，虽然最终任务成功但不能算有效恢复；
- `lost-grasp-v2` 成功制造掉碗并正确检测，但纯 OpenVLA 再次空抓后仍前往 plate，success=false。

局限必须明确：v3 的失败检测使用 simulator target height，重抓使用 task9 episode0 的 privileged bowl pose 和手工 Panda grasp offset，属于允许过拟合的 oracle recovery；只有掉碗后的最终放置由 OpenVLA继续闭环完成。它满足“错误肉眼可见、检测发生在错误之后、重新抓取经抬升验证、最终任务成功”的单场景视频目标，但尚不是 learned/general recovery。下一步若继续，应以 StageB target mask + robot mask/深度替代 privileged pose 和高度 detector，而不是再改变演示定义。

### 19.21 已用像素运动替换 privileged height failure detector

`experiments/robot/libero/run_single_scene_recovery.py` 新增
`--failure-detector segmentation-motion`。该模式使用
`SegmentationRenderEnv` 的 agent-view target instance mask，但 detector 本身只消费
mask 的像素中心：

1. 记录初始 target mask 中心行（109.98 px）；
2. target 历史上升至少 30 px，才确认发生过 lift；
3. 随后 target 回落到距初始行 16 px 以内，判定 lost grasp。

在同一 task9 episode0 中，step 64 的视觉上升为 39.24 px；step 67 回落到
13.66 px，因此恰好在真实掉落之后触发，而不是依赖
`akita_black_bowl_1_pos[2]` 做 detector 决策。最终验证产物：

```text
/data2/lty/naive-e-slotssm-runs/single-scene-recovery-visual-detector-v2/task9-episode0.mp4
/data2/lty/naive-e-slotssm-runs/single-scene-recovery-visual-detector-v2/task9-episode0/metadata.json
/data2/lty/naive-e-slotssm-runs/single-scene-recovery-visual-detector-v2/task9-episode0/steps.jsonl
```

结果仍为 `failure_detected_step=67, recovery_completed=true, success=true`；
step 173 `env.check_success()=true`。视频为 H.264、256×256、20 FPS、184
frames、9.2 秒，并修正了 failure overlay 为
`target fell after being lifted`。

边界必须继续说明：这一步移除了 detector 对 simulator 三维 object height 的依赖，
但 mask 仍是 simulator instance segmentation ground truth，不是 Stage B 的预测 mask；
重抓控制器仍读取 privileged bowl xyz 和手工 grasp offset，重抓后的 lift verification
也仍读取 object height。下一步应先把同一 detector 的输入切换到 Stage B 在线
target mask，并做 recorded-trajectory parity regression；之后再用 depth + camera
calibration 从 mask 恢复 3D 目标位置，替换 oracle regrasp pose。

### 19.22 Stage B predicted-mask detector 已在线恢复成功

新增离线审计：

```text
scripts/audit_stage_b_recovery_masks.py
/data2/lty/naive-e-slotssm-runs/stage-b-recovery-mask-audit-v1/report.json
```

原 production Stage B 直接用于 recovery trajectory 时没有通过：

- ground-truth mask detector：step 67；
- stateless Stage B detector：step 65，提前误触发；
- 简单 temporal tracking：step 127，严重滞后；
- step 57 抓取遮挡时 proposal 已绑定 distractor，target IoU=0。

这证明原 proposal 只覆盖静态空间关系状态，不覆盖机械臂遮挡、目标抬起和掉落；
也证明不能把每帧 relation selector 当成 object identity tracker。

为满足当前允许单场景过拟合的目标，使用
`scripts/finetune_stage_b_recovery_proposals.py` 从 production proposal 初始化，
在 task9 episode0 recovery patches 上做 400-step adaptation。固定
`timestep % 5 == 2` 的 35 帧为 validation，包含 step 57 和 step 67；结果：

- train permutation soft-IoU：0.8372；
- validation permutation soft-IoU：0.8303；
- candidate binary overlap IoU：0；
- checkpoint：
  `/data2/lty/naive-e-slotssm-runs/stage-b-recovery-proposals-v1/bowl-proposals-step-000400.safetensors`。

适配后回放报告：

```text
/data2/lty/naive-e-slotssm-runs/stage-b-recovery-mask-audit-v2/report.json
```

parity 结果：GT、Stage B stateless、Stage B temporal 三者均在 step 67 触发；
temporal target mean IoU=0.8466，distractor IoU=0。step 57/64/67 的 target
IoU 分别为 0.7547/0.8980/0.7679。任务末尾 stateless selector 再次按已失效的
`on(cabinet)` 关系切到 distractor，但 temporal tracker 在 step 173 仍保持 target
IoU=0.8478，因此在线 detector 使用“首帧关系 grounding + proposal mask IoU/centroid
时序跟踪”，而不是每帧重新选择语义目标。

最终在线运行使用普通 `OffScreenRenderEnv`，没有请求 simulator segmentation；
`--failure-detector stage-b-tracked` 直接复用 OpenVLA vision backbone，逐帧产生
Stage B bowl proposals 和 tracked target mask。最终产物：

```text
/data2/lty/naive-e-slotssm-runs/single-scene-recovery-stage-b-detector-v1/task9-episode0.mp4
/data2/lty/naive-e-slotssm-runs/single-scene-recovery-stage-b-detector-v1/task9-episode0/metadata.json
/data2/lty/naive-e-slotssm-runs/single-scene-recovery-stage-b-detector-v1/task9-episode0/steps.jsonl
```

在线证据：

- initial predicted target row=110.17 px；
- step 64 predicted row=70.92，历史 rise=39.25 px；
- step 67 predicted row=94.54，rise 回落至 15.63 px，触发 failure；
- `failure_detected_step=67, recovery_completed=true`；
- step 173 `env.check_success()=true`；
- MP4 为 H.264、256×256、20 FPS、184 frames、9.2 秒。

当前 detector 已不读取 object xyz/height，也不使用 simulator segmentation；
但 recovery proposal 是单场景轨迹适配，不能宣称泛化。重抓 controller 以及
regrasp lift verification 仍读取 privileged bowl xyz/height。下一步应以 predicted
mask + agent-view depth + camera calibration 估计 bowl 3D centroid，先离线对齐
oracle pose，再替换 `RETURN TO BOWL / DESCEND` 中的 privileged bowl position。

### 19.23 Recovery expert 数据采集闭环已建立

恢复策略需要“失败状态 → 正确恢复动作”，普通成功轨迹和旧失败 rollout 都不能
提供该监督。因此当前 oracle state machine 不再视为最终 controller，而是作为
simulator expert teacher 自动生成行为克隆/DAgger 数据；不需要人工遥操作采集。

`run_single_scene_recovery.py` 新增 `--expert-dataset-dir`。开启后环境同时输出
agent-view depth，并逐步保存：

- 256×256 RGB；
- 256×256 float32 MuJoCo depth buffer；
- 64×64 Stage-B predicted + temporally tracked target mask；
- robot proprio；
- nominal OpenVLA action；
- expert/executed action；
- recovery phase 和 failure/success 标志；
- privileged object state 仅作为 teacher/debug 字段。

schema 文档：

```text
recovery_expert_dataset_schema.md
```

首个 smoke episode：

```text
/data2/lty/naive-e-slotssm-runs/recovery-expert-dataset-v1/task9-episode0/
/data2/lty/naive-e-slotssm-runs/recovery-expert-smoke-v1/task9-episode0.mp4
```

验证结果：

- 174 条严格对齐的 observation/action transitions；
- 50 条 `is_recovery_expert_action=true`：
  return 18、descend 13、close 8、verify lift 11；
- RGB/depth/mask 文件无缺失；
- dataset 大小约 21 MB；
- Stage-B detector 仍在 step 67 触发；
- recovery 完成，step 173 task success；
- MP4 仍为 H.264、256×256、184 frames、9.2 秒。

该单 episode 只能验证 schema 和采集闭环，不能直接训练并宣称 learned recovery。
正式数据应覆盖多个 episode、多个 failure injection 时间、横向掉落偏移、空抓、
目标被推移和二次抓取失败；必须按 episode 划分 train/validation/test，不能随机
拆相邻帧。第一版动作模型建议预测 bounded residual：
`expert_action - nominal_openvla_action`，输入只能使用 RGB/depth/predicted
slots/masks/proprio/history，禁止使用 `teacher_privileged_object_state`。

### 19.24 动态失败注入、跨 episode 审计与 dataset manifest

固定 step 58 只适用于 episode0。`run_single_scene_recovery.py` 已新增：

```text
--mistake-trigger visual-lift
--injection-lift-threshold-px
```

Stage-B tracked target mask 相对初始行上升到阈值后才开始强制开爪，因此不同
episode 的抓取速度不需要共享固定 timestep。

episode1 的第一次运行暴露了固定 detector pixel threshold 的视角依赖：

- step 75 开始注入；
- bowl 实际 lift 约 5.8 cm，但 predicted-mask centroid 只上升 22.8 px；
- episode0 标定的 30 px 条件未满足，第一次掉落被漏检；
- step 142 才在后续运动中错误触发。

该 rollout 为 detector hard negative，不应作为成功专家 episode。将 lift
confirmation 降为 20 px 后重新运行：

- step 75 注入；
- step 82 在第一次掉落后正确检测；
- step 133 oracle regrasp 完成并重新抬升；
- `recovery_completed=true`；
- 交还 OpenVLA 后最终放置失败，`final task success=false`。

因此必须区分：

- `recovery_completed`：掉落后是否正确重新抓住；
- `final_task_success`：恢复后基础策略是否最终完成整个任务。

episode1 对 recovery behavior cloning 是有效专家数据，但不能进入
end-to-end-success-only 数据子集：

```text
/data2/lty/naive-e-slotssm-runs/recovery-expert-dataset-v2/task9-episode1/
/data2/lty/naive-e-slotssm-runs/recovery-expert-smoke-v3/task9-episode1.mp4
```

新增数据集工具：

```text
scripts/collect_recovery_expert_batch.sh
scripts/build_recovery_expert_manifest.py
```

首个 episode-level manifest：

```text
/data2/lty/naive-e-slotssm-runs/recovery-expert-manifest-smoke-v1.json
```

其中 episode0=train、episode1=validation；两者均 recovery-BC eligible，
共 100 条 recovery expert actions，只有 episode0 end-to-end success eligible。
manifest 明确显示当前没有 test split，因此仍不允许开始正式训练并报告泛化结果。
批采集默认候选为 OpenVLA nominal 已知成功的 task9 episodes
`0,1,2,3,4,8,9`；下一轮需继续运行并根据 `recovery_completed` 筛选，而不能仅按
脚本退出码或最终任务 success 丢弃有效重抓样本。

### 19.25 多 episode 采集与首个 learned recovery residual 离线通过

并行采集了 task9 episodes 2、3、4、8、9：

- episode2：step67 检测，recovery completed，最终放置失败；BC eligible；
- episode3：基础任务成功，但未达到 visual injection threshold，未产生 recovery；
- episode4：step74 检测，recovery completed，最终任务成功；BC eligible；
- episode8：step123 延迟检测，oracle 卡在 descend，recovery incomplete；排除；
- episode9：step73 检测，recovery completed，最终任务成功；BC eligible。

合并 episode0/1 后，正式 smoke manifest：

```text
/data2/lty/naive-e-slotssm-runs/recovery-expert-manifest-v2.json
```

episode split：

- train：0、2（102 条 eligible expert actions）；
- validation：1（50 条）；
- test：4、9（95 条）；
- 3、8 保留为正常负样本/采集失败记录，不进入正确恢复动作训练。

新增 `scripts/train_recovery_residual_smoke.py`。第一版直接训练 3000 steps 严重
过拟合：train action MAE=0.0023，但 validation=0.201、test=0.154，均差于
nominal/phase-mean baseline，因此不能上线。

第二版缩小 MLP、对低方差输入使用最小 normalization scale、增加 weight decay，
并只按 validation action MAE 选择 checkpoint（best step=550）。输入为：

```text
Stage-B mask centroid/area/margin
+ robot proprio
+ nominal OpenVLA action
+ oracle recovery phase one-hot
```

目标为 `expert_action - nominal_action`，没有读取 simulator object state。结果：

```text
/data2/lty/naive-e-slotssm-runs/recovery-residual-smoke-v2/metrics.json
/data2/lty/naive-e-slotssm-runs/recovery-residual-smoke-v2/recovery-residual-mlp.safetensors
```

- train action MAE：0.0446；
- validation action MAE：0.0808，nominal=0.1208，改善33.2%；
- test action MAE：0.0832，nominal=0.1381，改善39.7%；
- test phase-mean baseline=0.1212，learned model 也优于该基线；
- test translation MAE=0.1243；
- test gripper sign accuracy=93.7%。

这是第一个在 episode-held-out 离线动作误差上通过 baseline 的 learned recovery
controller smoke，但不能直接声称方案一完成：

1. recovery phase 仍由 oracle state machine 提供；
2. 输入是 mask measurement，不是真正 SlotSSM history；
3. 训练只有两个 episode，test 仍是同一 task9；
4. 12.4% normalized translation MAE 和 93.7% gripper sign accuracy 对在线控制仍有风险；
5. 尚未闭环运行 learned residual。

下一步应先做 teacher-forced sequential replay / action safety audit，增加 residual
clip、phase-dependent gripper override 和 uncertainty gate；只有留出 episode 的
预测动作没有无界/反向运动，才进行 episode4/9 在线 learned-controller smoke。

### 19.26 Teacher-forced sequential audit 阻止不安全在线执行

新增：

```text
scripts/audit_recovery_residual_sequential.py
```

它按 oracle recovery trajectory 的时间顺序逐步预测，但不执行模型动作。只用
validation 选择 residual clip，并在冻结 test 上检查：

- translation cosine 与反向动作比例；
- z direction sign accuracy；
- `DESCEND` 阶段单独的 z sign accuracy；
- 连续 translation jump；
- clip saturation；
- phase-dependent gripper safety override。

v2 checkpoint 在 validation 上选择 clip=0.5 并通过，但 episodes4/9 test 未通过：

```text
/data2/lty/naive-e-slotssm-runs/recovery-residual-smoke-v2/
sequential-safety-audit.json
```

- aggregate negative translation cosine rate=6.33%；
- aggregate z sign accuracy=93.67%；
- `DESCEND` z sign accuracy=80%；
- episode9 negative cosine rate=12.20%，z sign accuracy=87.80%；
- `online_smoke_allowed=false`。

最初 `max_action_jump=2` 来自合法的 gripper -1→+1 切换，不是 arm jump；审计已
修正为只测 xyz translation jump。修正后阻止上线的仍是 episode9 方向错误，而非
指标实现问题。

为了做开发迭代，将已经审计过的 episode4 转入 train，episode9 保留为 hard
holdout，得到 manifest v3。必须注明：因为此前已查看 episode9 指标，它不再是
严格意义上从未用于决策的最终 test，后续仍需新采 final test。

```text
/data2/lty/naive-e-slotssm-runs/recovery-expert-manifest-v3.json
/data2/lty/naive-e-slotssm-runs/recovery-residual-smoke-v3/
```

v3 的 frame-level 离线误差明显改善：

- episode9 action MAE=0.0599，nominal=0.1525，改善60.7%；
- translation MAE 从 v2 的0.1303 降至0.0691。

但 sequential direction safety 仍失败：

- episode9 negative translation cosine rate=12.20%；
- z sign accuracy=87.80%；
- `DESCEND` z sign accuracy进一步只有61.54%；
- `online_smoke_allowed=false`。

因此不能用更低 MAE 掩盖连续控制中的方向错误，也不能放宽阈值强行上线。当前
feature 只有 2D mask measurement + proprio + nominal action + oracle phase，
缺少 target/eef 的可靠 3D 相对几何和时序 hidden state。下一步优先：

1. 将 agent-view depth 转成 target-mask 内的 metric 3D centroid；
2. 显式构造 `target_xyz - eef_xyz`（输入来自视觉估计，不用 simulator pose）；
3. 收集更多横向位置/高度扰动的成功 oracle episodes；
4. 再训练 residual，并要求新 final-test 的 sequential audit 通过；
5. 在此之前禁止 learned controller 在线控制。

### 19.27 视觉3D低维 residual 首次在线闭环恢复成功

新增 `scripts/audit_recovery_visual_3d.py`，使用 Stage-B predicted target mask +
agent-view MuJoCo depth + robosuite agentview camera calibration，将 mask 点云反投影
到 world frame。orientation 审计确认保存图像到 camera pixel 只需恢复水平轴：
`camera_y=saved_y, camera_x=255-saved_x`。

在 manifest v3 上按 train+validation 选择 mask 点云 estimator，`all_median`
最稳定：

```text
/data2/lty/naive-e-slotssm-runs/recovery-visual-3d-audit-v2/report.json
```

未校准的 surface-point median 对 simulator object center：

- train mean error=3.85 cm；
- validation=3.58 cm；
- hard holdout episode9=3.81 cm，P90=4.44 cm。

train+validation teacher pose 仅用于估计固定表面偏差：

```text
estimate - object_center = [-0.0041, -0.0097, +0.0313] m
```

校准后：

- train mean=1.79 cm；
- validation=1.49 cm；
- episode9 mean=1.87 cm，P90=2.78 cm，max=3.32 cm。

直接把视觉 xyz 加入原高维 MLP（v4/v5）仍失败，且常量校准在 feature mean
normalization 后被抵消。根因是小数据下 39-D proprio + 2D + 3D 重复特征过拟合。
随后改为低维 geometry feature：

```text
visual_target_xyz - eef_xyz
+ nominal OpenVLA action
+ oracle recovery phase one-hot
```

模型：

```text
/data2/lty/naive-e-slotssm-runs/recovery-residual-smoke-v6-geometry/
```

episode9 离线：

- action MAE=0.0163；
- translation MAE=0.0267；
- nominal MAE=0.1525，改善89.3%；
- gripper sign accuracy=100%。

sequential safety audit：

- negative translation cosine rate=0；
- z sign accuracy=100%；
- `DESCEND` z sign accuracy=100%；
- `online_smoke_allowed=true`。

第一次在线 learned-controller 运行：

```text
/data2/lty/naive-e-slotssm-runs/recovery-learned-residual-online-v1/
task9-episode9.mp4
/data2/lty/naive-e-slotssm-runs/recovery-learned-residual-online-v1/
task9-episode9/metadata.json
/data2/lty/naive-e-slotssm-runs/recovery-learned-residual-online-v1/
task9-episode9/steps.jsonl
```

在线流程与证据：

- step64–70：动态 `LOST GRASP`；
- step73：Stage-B predicted-mask detector 触发；
- 之后 89 个 recovery actions 全部由 learned geometry residual 执行：
  return 26、descend 44、close 8、verify 11；
- step163：regrasp verified，`recovery_completed=true`；
- step225：`env.check_success()=true`；
- learned action xyz 绝对最大值 `[0.200, 0.171, 0.677]`，无无界动作；
- 视觉 target xyz 全程 finite；
- MP4：H.264，256×256，20 FPS，236 frames，11.8 秒。

准确边界：

1. 这是首个 learned recovery **action** closed loop 成功，不再由手工 return/
   descend/close/lift 动作控制；
2. recovery phase 和 phase transition/exit 仍由 oracle state machine 及 privileged
   bowl pose/height提供；
3. gripper sign 使用 phase safety override；
4. episode9 曾用于前一轮 safety diagnosis，不是从未查看的最终 generalization test；
5. Stage-B proposal 和 recovery model 都仍是 task9 recovery adaptation。

因此方案一已从“手工恢复动作”推进到“手工 phase manager + learned visual-geometry
actions”。下一步应采集一个全新 final-test failure variant/episode，复现在线成功；
随后训练 phase/mode predictor 和视觉 regrasp verification，逐步删除剩余 oracle
state machine。

### 19.28 冻结配置的真正未见 episode10 final test 成功

在 episode10 运行前冻结以下内容，不再调参：

- Stage-B recovery proposal/temporal tracker；
- predicted-mask failure detector thresholds；
- visual depth 3D estimator 和 train+validation surface bias；
- v6 geometry residual checkpoint；
- residual clip=1.0；
- oracle phase manager 和 gripper safety rule。

episode10 此前未参与数据采集、训练、validation、test diagnosis 或阈值选择。产物：

```text
/data2/lty/naive-e-slotssm-runs/recovery-learned-residual-final-test-v1/
task9-episode10.mp4
/data2/lty/naive-e-slotssm-runs/recovery-learned-residual-final-test-v1/
task9-episode10/metadata.json
/data2/lty/naive-e-slotssm-runs/recovery-learned-residual-final-test-v1/
task9-episode10/steps.jsonl
```

final-test evidence：

- step57–63：visual-lift trigger 后注入 `LOST GRASP`；
- step64：Stage-B predicted-mask detector 在第一次掉落后触发；
- 78 个 recovery actions 全部来自 frozen learned geometry residual：
  return 18、descend 40、close 8、verify 12；
- step143：regrasp verified，`recovery_completed=true`；
- step190：`env.check_success()=true`；
- learned executed xyz absolute max=`[0.225, 0.174, 0.732]`；
- visual target xyz 全程 finite；
- MP4：H.264，256×256，20 FPS、201 frames、10.05 秒。

这是当前最强证据：同一 frozen learned recovery action model 在 episode9 开发在线
smoke 和从未见过的 episode10 final test 都完成“明显掉落 → 检测 → 重抓 → 最终
放置成功”。因此可以宣布：

> task9 内跨初始状态的 learned visual-geometry recovery action feasibility 已通过。

仍不能宣布完整 Failure-Aware SlotSSM 或跨任务泛化，因为：

1. phase manager/transition/exit 仍用 oracle bowl pose/height；
2. gripper sign 仍由 phase safety override；
3. recovery model 是低维 geometry MLP，不是 SlotSSM temporal policy；
4. Stage-B 和 recovery 数据均围绕 task9；
5. failure 类型只有 lost grasp。

下一阶段可以正式进入 phase/mode predictor：先用现有 expert phase labels 训练
teacher-forced classifier，输入只允许 visual target-eef geometry、gripper/proprio
和历史 phase；必须在新 episode 上证明 phase order、切换延迟和无非法跳转，再替换
oracle phase transition。视觉 regrasp verification 应独立替换 privileged height。

### 19.29 learned phase manager 在线成功，并通过未见 episode11

新增 `scripts/train_recovery_phase_predictor.py`。phase predictor 输入限制为：

```text
visual_target_xyz - eef_xyz
+ gripper qpos/qvel
+ previous phase
```

预测 `RETURN/DESCEND/CLOSE/VERIFY`，并在自回归推理中只允许 stay 或前进一个
phase，禁止跳 phase 和倒退。v1 离线 gate 虽通过，但第一次 episode10 在线闭环
失败：模型只执行9步 `DESCEND` 就过早进入 `CLOSE`，随后长期停在 `VERIFY`，没有
重新抓住碗。这说明 teacher-forced/离线准确率不能替代在线时序验证。

随后把 episode10 的成功 learned-action + oracle-phase 在线 trace 作为
DAgger-style teacher trace 加入训练：

```text
/data2/lty/naive-e-slotssm-runs/recovery-phase-predictor-v2-dagger/
recovery-phase-predictor.safetensors
```

v2 离线 gate：

- extra episode10 trace autoregressive accuracy=97.4%；
- phase transition delay：return→descend +1、descend→close +1、close→verify 0；
- validation accuracy=90.0%，episode9 test accuracy=91.8%；
- 全部保持 monotonic phase order 并能到达 VERIFY。

episode10 在线回归验证成功：

```text
/data2/lty/naive-e-slotssm-runs/recovery-learned-phase-online-v2-dagger/
task9-episode10.mp4
```

- step64 failure detected；
- learned phase 时序：return 20、descend 74、close 8、verify/lift 6；
- phase transitions：64 return、84 descend、158 close、166 verify；
- step172 recovery complete，step219 env success；
- 共108个 learned recovery actions，视觉 xyz 有效帧全部 finite。

然后冻结 Stage-B、visual 3D estimator、v6 geometry residual、v2 phase predictor
和所有阈值，在未参与 phase DAgger、训练或诊断的 episode11 上做 final test：

```text
/data2/lty/naive-e-slotssm-runs/recovery-learned-phase-final-test-v1/
task9-episode11.mp4
/data2/lty/naive-e-slotssm-runs/recovery-learned-phase-final-test-v1/
task9-episode11/metadata.json
/data2/lty/naive-e-slotssm-runs/recovery-learned-phase-final-test-v1/
task9-episode11/steps.jsonl
```

episode11 evidence：

- step68–74 注入明显 lost grasp，step75 Stage-B detector 触发；
- learned phase 时序：return 26、descend 43、close 8、verify/lift 15；
- phase transitions：75 return、101 descend、144 close、152 verify；
- 92个 recovery actions 全部由 frozen learned geometry residual 执行；
- step167 recovery complete，step211 `env.check_success()=true`；
- learned executed xyz absolute max=`[0.796, 0.206, 0.912]`，仍在
  residual clip=1.0 内；
- 视觉 target xyz 有效帧全部 finite；
- MP4：H.264，256×256，20 FPS，222 frames，11.1秒。

因此当前可以宣布：task9 内 frozen learned visual recovery actions +
learned monotonic phase transitions 已跨未见初始状态在线闭环成功。相比19.28，
手工 phase transition 已被学习模型替换。

仍保留的关键 oracle/规则边界：

1. `VERIFY` 后的 recovery-complete/退出仍用 privileged bowl height；
2. gripper open/close sign 仍按 phase 做 safety override；
3. phase predictor 是低维 MLP，不是最终两阶段 SlotSSM temporal model；
4. 仅验证 task9 的 lost-grasp failure，尚未证明跨任务或放置错误恢复。

下一步应优先训练视觉 regrasp verifier，替换 privileged-height recovery exit；
随后去掉 gripper sign override。完成这两步后，整段恢复闭环才不再依赖 object-state
oracle 和手工 phase 动作规则。

### 19.30 视觉时序 regrasp verifier 替换 privileged-height exit

新增：

```text
scripts/train_visual_regrasp_verifier.py
/data2/lty/naive-e-slotssm-runs/visual-regrasp-verifier-v10-temporal/
visual-regrasp-verifier.safetensors
```

输入仅包含 VERIFY 起点后的可观测时序量：

- visual target z displacement；
- eef z displacement；
- target-eef relative drift / xy drift；
- target/eef z motion mismatch；
- gripper qpos/qvel；
- elapsed VERIFY steps。

privileged bowl height 只用于离线 teacher label，不进入模型输入。审计发现失败的
learned-phase-v1 轨迹并非从未抓起：前7步曾短暂带起碗约5.7 cm，随后滑落。因此
单帧 lift classifier 会产生危险的提前成功。v10 改为 threshold=0.75，并要求连续
3帧超过阈值。

训练数据中的硬负轨迹共180个 VERIFY frames，没有连续确认事件；extended
validation teacher trace：

```text
/data2/lty/naive-e-slotssm-runs/recovery-verifier-teacher-extended-v1/
task9-episode11/steps.jsonl
```

通过把 teacher oracle exit 临时提高到10 cm，得到20个 VERIFY frames，其中7个
稳定正帧。v10 离线：

- hard-negative false confirmation=0；
- extended validation false positive=0、false negative=0；
- held-out episode10 trace false positive=0，首次确认比 teacher 晚1帧；
- `offline_gate_passed=true`。

`run_single_scene_recovery.py` 新增：

```text
--visual-regrasp-checkpoint
--visual-regrasp-confirm-steps
--visual-regrasp-settle-height-m
```

当提供 visual verifier 时，VERIFY exit 不再读取 bowl height。第一次在线 smoke
确认了 verifier 能触发，但若在约7.7–9.4 cm 直接交还 OpenVLA，后续放置会因恢复
出口分布偏移失败。最终加入一个明确标记为手工安全过渡的 `VISUAL SETTLE`：
视觉确认后只用 eef encoder 将末端降回 VERIFY-start +4 cm，再交还 OpenVLA；
不读取 bowl pose/height。

成功产物：

```text
/data2/lty/naive-e-slotssm-runs/recovery-visual-verifier-online-smoke-v4-settle/
task9-episode10.mp4
/data2/lty/naive-e-slotssm-runs/recovery-visual-verifier-online-smoke-v4-settle/
task9-episode10/metadata.json
/data2/lty/naive-e-slotssm-runs/recovery-visual-verifier-online-smoke-v4-settle/
task9-episode10/steps.jsonl
```

在线证据：

- step64 failure detected；
- learned phase/action 执行 return/descend/close；
- step166–172 verifier probability 从0.629增至0.795；
- step173 probability=0.812，满足连续3帧确认，进入 `VISUAL SETTLE`；
- step173–189 共17步，仅用 eef encoder 回到交还高度；
- step190 `recovery_completed=true`，过程中不以 privileged bowl height 决策；
- step238 `env.check_success()=true`；
- MP4：H.264，256×256，20 FPS，249 frames，12.45秒。

重要边界：

1. 这是 episode10 online smoke；episode10 参与过 phase DAgger，不能当全新 final
   test；
2. episode12/14 因 OpenVLA 名义抓取未抬起而无法触发错误注入；
3. episode13 触发失败后 learned phase 卡在 DESCEND，未进入 VERIFY；
4. 因此 visual verifier 已证明能替换在线 privileged-height exit，但尚未取得
   未见初始状态的端到端 verifier final test；
5. `VISUAL SETTLE` 是新的手工 eef-only safety transition，不是 learned action；
6. gripper sign phase override 仍存在。

下一步不应继续微调 verifier 阈值。应收集/合并 episode13 这类 phase
distribution-shift trace，增强 phase predictor，使更多未见初始状态能进入 VERIFY；
同时把 `VISUAL SETTLE` action 加入 recovery action dataset，训练模型覆盖
verify-confirmed → nominal handoff，最终删除手工 settle 和 gripper override。

### 19.31 当前代码结构与统一 Recovery SlotSSM 决策

2026-07-28 重新审视代码后确认，当前成功恢复链路虽然证明了各个子问题可解，但
结构已经不适合作为最终系统。在线恢复路径目前由以下组件串联：

```text
OpenVLA nominal policy
→ Stage-B target selector
→ temporal mask tracker
→ mask-motion failure detector
→ mask + depth visual 3D estimator
→ learned monotonic phase predictor
→ learned geometry residual controller
→ visual temporal regrasp verifier
→ hand-written VISUAL SETTLE
→ phase-based gripper sign override
→ resume OpenVLA
```

主要代码位置：

```text
prismatic/                         OpenVLA/Prismatic 基础框架
vla-scripts/                       原始 Stage-1/2/3 SlotSSM 训练入口
scripts/                           数据、训练、评测和审计脚本
experiments/robot/libero/          LIBERO 在线仿真
experiments/robot/libero/
  run_single_scene_recovery.py     当前模块化恢复主入口
```

`run_single_scene_recovery.py` 当前同时负责环境、OpenVLA、Stage-B、深度反投影、
phase、动作、verification、settle、日志和视频，职责过多。不要继续把新的模型或
phase 直接堆进该文件。

#### 不采用传统 logits 蒸馏作为主路线

下一阶段不是让统一模型机械模仿旧 phase/verifier MLP 的概率。旧模块存在 episode13
卡在 DESCEND、分布外提前/延迟切换等错误，直接蒸馏会把错误一起复制。

采用的路线是：

> 直接在恢复序列数据集上训练一个统一 Recovery SlotSSM；现有模块和 simulator
> privileged state 只用于采集成功轨迹、生成 teacher labels 和离线审计。

privileged bowl pose/contact/height 可以用于训练标签，但禁止进入在线输入。

#### 第一版统一范围

第一版不立即从原始 RGB 完全端到端，也不立即删除 Stage-B。先保留可靠的感知前端：

```text
Stage-B target feature/mask
+ visual target xyz
+ proprio
+ previous action
+ OpenVLA nominal action
+ continuous history
        │
        ▼
Unified Recovery SlotSSM
        │
        ├── failure/type head
        ├── phase/mode head
        ├── grasp/verification/handoff head
        └── complete 7-D residual action head
```

目标是用一个共享 temporal backbone、一个 checkpoint 和一次前向传播，优先替换：

1. 独立 phase predictor；
2. 独立 residual controller；
3. 独立 visual verifier；
4. hand-written `VISUAL SETTLE`；
5. phase-based gripper override；
6. mask-motion failure rule（可稍晚切换）。

Stage-B 和 visual depth xyz 暂时作为统一模型输入及 teacher。第一版稳定后，再将
它们蒸馏/合并进 target-conditioned visual slot encoder。

#### 直接训练需要的数据

现有约5条有效 recovery expert episodes 只够代码 smoke 和单任务过拟合，不足以
训练可靠统一时序模型。第一版 task9 lost-grasp overfit/feasibility 建议至少整理或
采集20–30条完整序列；task9 稳定版本需要约100–300条恢复尝试。

每个 timestep 在线允许输入：

```text
RGB/depth or frozen visual features
language
proprio
previous executed action
OpenVLA nominal action
```

训练监督：

```text
target mask / target xyz teacher
failure type
recovery phase/mode
grasped / attached
verified
handoff ready
expert complete 7-D action or residual
episode recovery/final-task result
```

数据不能只有成功轨迹，必须包含：

- 正常成功轨迹，学习不干扰 OpenVLA；
- lost grasp 后成功恢复；
- 短暂带起后再次滑落的 hard negative；
- phase 卡住、提前 CLOSE、错误 VERIFY；
- recovery 失败后第二次规划；
- 后续扩展 missed grasp、off-center grasp、wrong placement。

#### 推荐训练顺序

1. 定义统一 sequence schema，禁止 privileged input 泄漏；
2. 转换已有 episode0/1/2/4/9/10/11 和失败 episode13；
3. 从关键事件附近构造连续32或64步窗口，而不是随机单帧；
4. 先训练共享状态和 failure/phase/grasp/verification/handoff 辅助头；
5. 再训练完整7-D residual action，并审计方向、gripper sign、平滑性；
6. 联合多任务训练同一个 checkpoint；
7. 做 teacher-forced 之外的 autoregressive sequential safety audit；
8. 在线 rollout 暴露分布外状态后做统一模型 DAgger；
9. 用统一 checkpoint 替换旧 phase/residual/verifier；
10. 最后再删除 Stage-B/显式 visual xyz 前端。

#### 建议新增代码目录

不要继续将统一模型放进散乱的 `scripts/` 或旧主入口内部。建议新增：

```text
recovery/
├── data/
│   ├── schema.py
│   ├── dataset.py
│   └── sequence_builder.py
├── models/
│   ├── slot_encoder.py
│   ├── recovery_slotssm.py
│   └── heads.py
├── training/
│   ├── losses.py
│   └── trainer.py
├── evaluation/
│   ├── offline_audit.py
│   └── online_rollout.py
└── teachers/
    ├── stage_b_teacher.py
    └── recovery_labeler.py
```

旧 `run_single_scene_recovery.py` 应保留为模块化 teacher/baseline，不做破坏性重写；
统一模型使用新的在线入口，便于 A/B 对比和回归。

#### 当前最准确的项目定位

现阶段已经完成：

> task9 lost-grasp 的“视觉感知 → 失败检测 → 恢复阶段 → 恢复动作 → 抓稳验证 →
> 继续任务”模块化 feasibility prototype。

尚未完成：

> 共享 SlotSSM 时序状态、单 checkpoint、自主 handoff/gripper、多轮重规划、多失败
> 类型和 benchmark 级泛化。

下一位执行者应先建设统一恢复序列数据集和 `recovery/` 代码骨架，不要先为
episode13 再增加一个独立 checkpoint，也不要继续微调 visual verifier 阈值。

## 20. 2026-07-28 更新：统一 recovery schema 与首批转换已完成

已正式开始 19.31 决定的 Unified Recovery SlotSSM 路线。新增：

```text
recovery/
├── README.md
├── data/
│   ├── schema.py
│   ├── dataset.py
│   ├── conversion.py
│   ├── sequence_builder.py
│   └── validation.py
└── tests/test_data_contract.py

scripts/convert_recovery_sequences.py
scripts/convert_recovery_online_traces.py
scripts/audit_recovery_sequences.py
```

`schema_version=2` 将每步数据强制分为：

```text
online_inputs
labels
teacher_only
provenance
```

在线模型只能通过显式 `ONLINE_INPUT_ALLOWLIST` 读取字段。object pose、height、
contact、instance segmentation 和 raw privileged state 不允许出现在
`online_inputs`。当前5项 contract tests 全部通过，包括 privilege leakage 拒绝、
JSONL round-trip、episode/split 隔离和非连续窗口过滤。

首批完整 expert 数据：

```text
/data2/lty/naive-e-slotssm-runs/recovery-unified-sequences-v1
```

包含 task9 episode0/1/2/3/4/8/9，共1,368 rows，其中363个 recovery expert
actions。7个 episode 全部具有对齐 RGB/depth/target mask。split 为 train 5、
validation 1、development-test 1；episode9 已明确降为 development test。

首批在线时序数据：

```text
/data2/lty/naive-e-slotssm-runs/recovery-unified-online-traces-v1
```

包含 episode10、11、13，共751 rows。episode10/11 是成功 trace，episode13 是
长期卡在 DESCEND 的 hard negative。这些旧 `steps.jsonl` 没有逐帧媒体路径，转换
结果明确标为 temporal-label-only，不可冒充完整视觉训练数据。

合并审计：

```text
/data2/lty/naive-e-slotssm-runs/recovery-unified-audit-v1/report.json
```

- 10 episodes / 2,119 rows；
- 32-step windows=1,809，64-step windows=1,489；
- event 32-step windows=1,076，event 64-step windows=1,163；
- 最终成功 episode=6；
- missing files=0，schema errors=0；
- 对齐 RGB/depth/mask rows=1,368；
- visual target xyz available rows=419。

下一步：

1. 建立可训练 tensor sequence dataset/collator；
2. 用现有 Stage-B mask + depth 离线补算 expert rows 的 visual target xyz；
3. 用 simulator teacher 离线补充 failure/grasp/slip/verified/handoff labels；
4. 补采至少20–30条带完整逐帧媒体的 task9 recovery 序列；
5. 然后实现首个 shared Recovery SlotSSM auxiliary-head smoke model。

### 20.1 Tensor dataset、视觉3D质量 gate 与共享模型 smoke

已新增：

```text
recovery/data/tensor_dataset.py
recovery/data/visual_geometry.py
recovery/models/recovery_slotssm.py
scripts/audit_recovery_tensor_dataset.py
scripts/enrich_recovery_visual_xyz.py
scripts/train_recovery_slotssm_smoke.py
```

真实 schema-v2 数据已成功构造成 `[B,T,D]` tensor batch。第一版 observable input
为76维，包含 proprio及mask、nominal/previous action、target-eef geometry及valid
mask、target centroid/measurement及mask。训练标签不进入该 input。

visual xyz enrichment 产物：

```text
/data2/lty/naive-e-slotssm-runs/recovery-unified-sequences-v2-visual-xyz
```

重要新发现：Stage-B mask 在部分 episode 的正常后段、resumed task 或 phase-stuck
后会跳到约40–50cm外的错误 proposal。不能把“xyz finite”当作“xyz valid”。新增只
使用可观测量的 gate：

```text
visible_pixels >= 32 at 64×64
and jump_from_last_reliable_xyz <= 0.12m
```

teacher-only 审计：

- 全部不加 gate：mean=11.34cm、P90=46.25cm、max=49.62cm；
- gate 后951帧：mean=2.19cm、P90=3.10cm、max=4.35cm；
- 被拒417帧：mean=32.19cm；
- gate 不读取 simulator state。

tensor audit：

```text
/data2/lty/naive-e-slotssm-runs/recovery-tensor-audit-v1/report.json
```

32-step/stride8：train 101 windows、validation 27、development-test 102；所有 tensor
finite。

共享多头 temporal model 已分别通过：

```text
/data2/lty/naive-e-slotssm-runs/recovery-slotssm-smoke-v1-gru
/data2/lty/naive-e-slotssm-runs/recovery-slotssm-smoke-v1-mamba2
```

GRU reference 完成50 steps；Mamba2 完成2-step CUDA forward/backward/validation/
checkpoint。Mamba2 `model_dim=64` 必须显式用 `headdim=16`，避免 causal-conv
channel-last stride 错误。

准确边界：当前只是 shared backbone + heads 的训练路径 smoke。legacy phase label
来自旧 controller，failure/grasp/verification/handoff 标签还不完整，类别严重不
平衡，未做 autoregressive safety audit，禁止在线运行该 checkpoint。

### 20.2 Teacher event labels 与 causal sequential audit

新增：

```text
recovery/teachers/recovery_labeler.py
scripts/enrich_recovery_teacher_labels.py
scripts/evaluate_recovery_slotssm_sequential.py
```

teacher-label 数据：

```text
/data2/lty/naive-e-slotssm-runs/recovery-unified-sequences-v3-teacher-labels
```

7 episodes / 1,368 rows 中：failure-active=369、lifted=362、attached=682、
slip-event=39、verified=273、handoff-ready=388。旧日志没有 contact，所以
attached/grasped 是 target-EFF distance + 连续共运动的保守 proxy，不应表述为
contact oracle。privileged state 只写 label，不进入76-D policy input。

模型已增加 grasped/lifted/slip heads，训练 loss 加入 phase weights 和 binary
positive weights。GRU 200-step balanced preflight：

```text
/data2/lty/naive-e-slotssm-runs/recovery-slotssm-preflight-v3-gru-balanced
/data2/lty/naive-e-slotssm-runs/recovery-slotssm-sequential-audit-v3-gru-balanced
```

validation+development causal audit：

- phase accuracy=76.1%；
- current-failure precision=0.556、recall=0.891、F1=0.684；
- failure onset 后检测延迟=0；
- onset 前误报率 episode1=7.3%、episode9=17.8%；
- verification F1=0.876；
- handoff F1=0.857；
- grasped F1=0.923；
- slip precision=0.344、F1=0.478；
- `online_smoke_allowed=false`。

Mamba2 25-step preflight 也已完成：

```text
/data2/lty/naive-e-slotssm-runs/recovery-slotssm-preflight-v2-mamba2-labels
/data2/lty/naive-e-slotssm-runs/recovery-slotssm-sequential-audit-v2-mamba2
```

但训练太短，phase 塌缩到 NOMINAL/RESUMED，phase accuracy=43.97%，failure
F1=0.451，slip F1=0.296。禁止上线。

下一步必须做 validation threshold calibration、rare/event window sampling、更多
正常负轨迹和带contact/完整媒体的新恢复数据。不要因为 GRU 某些辅助头 F1 较高就
跳过 Mamba2 正式训练或直接在线控制。

### 20.3 Validation-only calibration、采样消融与失败门控解码

已完成 validation-only threshold calibration：阈值只在 validation episode 上
选择，再原样用于 development-test，避免直接用测试集调阈值。也加入 causal legal
phase graph 和 failure-gated constrained decode；后者要求 current-failure 连续两帧
超过 validation 阈值后才离开 NOMINAL。

两组较长 preflight：

```text
/data2/lty/naive-e-slotssm-runs/recovery-slotssm-preflight-v5-gru-mixed-sampling
/data2/lty/naive-e-slotssm-runs/recovery-slotssm-sequential-audit-v5-gru-mixed-sampling
/data2/lty/naive-e-slotssm-runs/recovery-slotssm-preflight-v6-mamba2-mixed-sampling
/data2/lty/naive-e-slotssm-runs/recovery-slotssm-sequential-audit-v6-mamba2-mixed-sampling
```

GRU mild mixed sampling：raw phase accuracy=63.36%；validation 校准后的
development-test current-failure F1=0.876、verification F1=0.959、slip
F1=0.714。Mamba2 100-step mild mixed sampling：raw phase accuracy=52.72%；
development-test current-failure F1=0.916、verification F1=0.936，但 slip
F1=0.222。

这些数字仍不能作为上线依据：

- validation 目前只有 episode 1，development-test 只有 episode 9，阈值方差无法估计；
- GRU failure-gated phase accuracy 在 validation 为93.3%，到 development-test
  降为34.4%；
- Mamba2 failure-gated phase accuracy 在 validation 为45.0%，development-test
  为33.3%；早期连续两帧假阳性会使单向状态机永久离开 NOMINAL；
- 最大20倍的 aggressive rare-window sampling 已导致 phase collapse 和超过57%的
  onset前误报，应明确弃用；最大4倍 mild sampling 仅保留为消融基线；
- 两个 report 都保持 `online_smoke_allowed=false`。

因此下一项主任务不是继续在这两个 episode 上调阈值，而是修改 collector，使每帧
保存对齐 RGB/depth/mask、真实 contact/attached teacher fields、nominal action 和
executed action，并新增独立正常负轨迹、失败和恢复轨迹。最低先收集20–30个 episode，
按 episode 划分 train/validation/development-test；在 validation 至少覆盖多个随机种子
和失败类型前，不冻结阈值，不接入在线控制。

### 20.4 Collector v2 字段改造已完成

`experiments/robot/libero/run_single_scene_recovery.py` 的 expert collector 已升级：

- expert collection 自动启用实例分割；RGB、depth 和64×64 target mask 与动作前
  observation 对齐；
- 每行继续同时保存同一 observation 的 OpenVLA `nominal_action` 和实际
  `expert_or_executed_action`；
- 新增 `teacher_contact_state`，保存 MuJoCo 中目标物与夹爪的具体 contact geom pairs；
- 新增 `teacher_attached_truth`，仅在可识别的左右两指同时接触目标物时为真。这是
  保守 pinch/contact proxy，不是 weld truth；
- 新增 `--disable-failure-injection`，用于采集真正的 nominal negative episodes；
- schema-v2 转换会把 attached truth 写入 labels，同时只把原始 contact state 放在
  `teacher_only`，不会泄漏到 policy inputs；
- dataset audit 现在统计 contact/attached teacher coverage。

正式采集入口：

```text
scripts/collect_recovery_dataset_v3.sh
```

默认模板包含10个 failure episodes 和10个 nominal episodes，可通过
`FAILURE_EPISODES`、`NORMAL_EPISODES`、`GPU`、`DATA_ROOT` 和 `ROLLOUT_ROOT`
覆盖。脚本故意不预先决定数据 split；必须等采集完成后根据 episode、seed 和 failure
type 做分组，再固定 train/validation/development-test。

新增 contact 单元测试和 collector-v2 conversion round-trip 测试；当前 recovery
tests 共12项全部通过。正式20–30条批量采集前，先完成下面两条端到端 preflight。

### 20.5 Collector v2 两类端到端 preflight 已通过

实际采集路径：

```text
/data2/lty/naive-e-slotssm-runs/recovery-expert-dataset-v3-preflight
/data2/lty/naive-e-slotssm-runs/recovery-expert-rollouts-v3-preflight
```

- episode0-failure：167 transitions；第56步开始注入，第65步检测失败，oracle
  recovery完成，最终任务成功；
- episode10-normal：108 transitions；`failure_injection_enabled=false`，没有错误
  注入、没有失败检测，OpenVLA nominal rollout最终成功。

统一转换、observable xyz 和 teacher enrichment：

```text
/data2/lty/naive-e-slotssm-runs/recovery-unified-sequences-v4-collector-preflight
/data2/lty/naive-e-slotssm-runs/recovery-unified-sequences-v5-collector-preflight-visual
/data2/lty/naive-e-slotssm-runs/recovery-unified-sequences-v6-collector-preflight-teacher
```

最终审计 `v6.../audit-report.json`：

- 2 episodes / 275 rows，schema valid；
- 275/275 有 RGB、depth、target mask 和 observable visual xyz；
- 275/275 有 simulator contact state 和 attached truth；
- 122 rows 为 bilateral-finger attached；
- teacher enrichment 的275行全部使用 simulator contact，co-motion fallback=0；
- observable xyz gate 接受226行，接受行 teacher mean error=2.11cm、
  P90=3.33cm、max=3.95cm；
- missing files=0，errors=0；
- 12项 recovery tests、compileall、shell syntax 和 `git diff --check` 全通过。

preflight 已消除 collector/schema 风险。下一步可扩大采集，但不要机械重复同一失败：
至少应把20–30条配额分为 nominal negatives、lost grasp、missed grasp、early/late
failure，以及多个 init-state/seed；并保证 validation 和 development-test 各自包含
多个独立 episode。

### 20.6 24-episode diverse dataset 与正式离线重训

正式采集入口：

```text
scripts/collect_recovery_diverse_v4.sh
```

collector 新增 `--dataset-episode-id` 和 `--seed`，将 LIBERO init-state index 与数据
episode identity 分开，避免同一初始状态的不同干预变体被错误合并。8卡分片实际采集：

- 8 nominal negative episodes；
- 6 early lost-grasp；
- 6 late lost-grasp；
- 4 missed-grasp；
- 合计24 episodes / 4,791 transitions。

原始与最终数据：

```text
/data2/lty/naive-e-slotssm-runs/recovery-expert-dataset-v4-diverse
/data2/lty/naive-e-slotssm-runs/recovery-expert-rollouts-v4-diverse
/data2/lty/naive-e-slotssm-runs/recovery-unified-sequences-v7-diverse
/data2/lty/naive-e-slotssm-runs/recovery-unified-sequences-v8-diverse-visual
/data2/lty/naive-e-slotssm-runs/recovery-unified-sequences-v9-diverse-teacher
```

split按 failure type 分层：train=10 episodes、validation=7、development-test=7。
最终审计 valid；4,791/4,791有RGB/depth/mask/contact/attached truth，4,782有可计算
visual xyz；目标完全不可见的9帧被合法标为 observation unavailable。observable gate
接受3,718帧，接受帧 teacher mean=1.90cm、P90=2.53cm。teacher labels：
attached=1,035、slip=75、verified=488、failure-active=825、handoff=368，全部使用
sim contact，co-motion fallback=0。

采集中修复一个真实 tracker bug：当上一帧所有 proposal 都为空时，
`previous_centroid=None` 曾导致距离计算崩溃。现在改为 relation re-grounding，并成功
重采 dataset episode1201。注意 early lost-grasp 只有1/6触发现有“先抬高再落下”
检测规则；其余保留为 injected-but-undetected failure，不能伪装成已恢复样本。

tensor audit：

```text
/data2/lty/naive-e-slotssm-runs/recovery-tensor-audit-v2-diverse/report.json
```

32-step/stride8：train=188 windows、validation=177、development-test=155。

500-step mild-balanced 重训：

```text
/data2/lty/naive-e-slotssm-runs/recovery-slotssm-v7-gru-diverse
/data2/lty/naive-e-slotssm-runs/recovery-slotssm-v8-mamba2-diverse
/data2/lty/naive-e-slotssm-runs/recovery-slotssm-sequential-audit-v7-gru-diverse
/data2/lty/naive-e-slotssm-runs/recovery-slotssm-sequential-audit-v8-mamba2-diverse
```

Mamba2 当前明显优于 GRU：

- raw phase accuracy：Mamba2=90.72%，GRU=88.96%；
- constrained phase：Mamba2=76.01%，GRU=67.56%；
- failure-gated phase development-test：Mamba2=85.32%，GRU=73.82%；
- validation校准后 development-test current-failure F1：
  Mamba2=0.843（precision=0.919，recall=0.779），GRU=0.760；
- verification F1：Mamba2=0.728，GRU=0.748；
- slip F1：Mamba2=0.710，GRU=0.397。

这是首次在7个validation和7个development-test episodes上得到较稳定的结果，但仍
`online_smoke_allowed=false`：没有冻结final-test，early failure detector coverage
不足，当前恢复动作仍主要是oracle controller label，尚未证明learned residual在线
安全。下一步应先修正/学习early failure onset supervision，冻结final-test，再做
candidate recovery action scoring和只读 shadow evaluation。

### 20.7 Causal failure onset 与冻结 final-test

旧 teacher label 只在 legacy detector 报警后设置 `failure_active=true`，因此大量
early injected failures 被错误当作 normal。现在 onset teacher 按以下因果规则生成：

1. transition row 对齐动作执行前 observation；
2. 人工错误动作执行后的下一帧标记
   `failure_onset_reason=post_intervention_observation`；
3. failure-active 持续到 `recovery_completed`；
4. 非人工干预轨迹中，已抬起目标失去 bilateral finger contact 也可触发
   `lifted_contact_loss` onset；
5. legacy detector 只作为第三种 fallback onset，不再是唯一来源。

24条开发数据中产生18个onset，failure-active rows由旧标签的825增加到因果标签的
1,786。这个变化使指标下降，但旧指标是由漏标造成的虚高，不能恢复旧标签。

另外新采6条从未参与开发的 holdout：

```text
scripts/collect_recovery_final_holdout_v1.sh
/data2/lty/naive-e-slotssm-runs/recovery-expert-final-holdout-v1
/data2/lty/naive-e-slotssm-runs/recovery-rollouts-final-holdout-v1
/data2/lty/naive-e-slotssm-runs/recovery-unified-sequences-v13-frozen-final-onset
```

合并数据共30 episodes / 5,895 rows，固定为train=10、validation=7、
development-test=7、final-test=6。final-test只做schema/media completeness audit；
未输入训练、阈值校准、architecture selection或本轮模型性能评估。最终audit valid，
5,895/5,895有媒体/contact truth，5,886有可计算visual xyz，missing files=0。

因果标签Mamba2开发实验：

```text
/data2/lty/naive-e-slotssm-runs/recovery-slotssm-v11-mamba2-causal-onset
/data2/lty/naive-e-slotssm-runs/recovery-slotssm-sequential-audit-v11-mamba2-causal-onset
```

500-step结果：raw phase=86.33%，constrained phase=69.93%；validation校准后
development current-failure precision=0.686、recall=0.600、F1=0.640；
verification F1=0.724，slip F1=0.712；failure-gated phase=46.37%。1500-step版本
current-failure F1进一步降到0.586，判定为过拟合，不采用。

这说明“旧检测器报警后识别恢复阶段”已经较容易，但“从可观测历史自主判断early
failure是否持续存在”仍未解决。下一步不能打开final-test；应在development上改为
onset/duration分离建模，加入contact-loss可观测代理、target motion与action-conditioned
effect，并先做candidate scorer/shadow ranking。方法冻结后，final-test只评一次。

### 20.8 Onset / persistence / resolution 分离模型

已完成最小无重复的三部分设计：

- 新增 `failure_onset_head`，监督单帧因果 onset；
- 原 `current_failure_head` 明确作为 persistence 状态头；
- 原 `handoff_head` 复用为 resolution / safe-to-handoff 头，避免再造同义输出；
- phase gate 改由 validation-calibrated onset probability 单帧触发，不再用持续状态
  probability冒充瞬时事件。

相关代码：

```text
recovery/data/tensor_dataset.py
recovery/models/recovery_slotssm.py
scripts/train_recovery_slotssm_smoke.py
scripts/evaluate_recovery_slotssm_sequential.py
```

500-step Mamba2：

```text
/data2/lty/naive-e-slotssm-runs/recovery-slotssm-v13-mamba2-split-head-500
/data2/lty/naive-e-slotssm-runs/recovery-slotssm-sequential-audit-v13-mamba2-split-head-500
```

validation选择onset threshold=0.95。development-test：

- onset：precision=1.000、recall=0.833、F1=0.909（5/6 onset，0 FP）；
- persistence：precision=0.676、recall=0.478、F1=0.560；
- resolution/handoff：precision=0.932、recall=0.971、F1=0.951；
- verification F1=0.700；
- slip F1=0.706；
- raw phase=90.09%，constrained phase=68.63%；
- onset-gated phase=63.37%，相比单头因果基线46.37%明显改善；
- `online_smoke_allowed=false`，final-test未打开。

结论：early failure“发生时刻”已能较可靠识别，resolution也较强；当前主要问题收敛为
persistence，即在没有privileged contact的线上输入中判断失败是否仍持续。下一步应
加入policy-observable target-motion/contact-loss proxy，并开始action-conditioned
candidate effect scorer；不能用打开final-test来指导这些修改。

### 20.9 Recovery candidate effect/risk 最小训练闭环

Phase 6 的代码骨架已建立，目的是先固定因果正确的数据和张量接口，再采集昂贵的
MuJoCo recovery-state branches。新增：

```text
recovery/data/candidate_schema.py
recovery/data/candidate_tensor_dataset.py
recovery/models/candidate_effect_scorer.py
scripts/train_recovery_candidate_scorer.py
recovery/tests/test_candidate_effect.py
```

当前候选由 nominal、Recovery SlotSSM residual center、有界 xyz / z-rotation 扰动、
flip-gripper 和 hold 组成，全部裁剪到合法7-D动作范围并数值去重。同一 observation
下的所有候选必须作为一个 matched group 保存，不能拆散；每个 branch 的 teacher
outcome包含 EEF/target位移、attach、lift、slip、failure persistence、handoff-ready
和 task-aware `teacher_value`。teacher outcome只能作为监督，评分器输入仅使用
schema-v2 policy-observable 76-D context、nominal action和candidate action。

评分器对候选轴 permutation-equivariant，联合学习物理效果、二分类未来事件、候选
价值及成对排序。可变候选数由显式mask处理，padding和无效仿真分支不进入loss。
17项 recovery tests、compileall和`git diff --check`已通过。

注意这仍是“可训练闭环”，不是有效性结论：尚未采集recovery-state matched branches，
未训练checkpoint，也未证明action shuffle degradation或held-out ranking gain。
当前76-D context是统一模型的可观测输入接口，下一版应允许直接读取
Recovery SlotSSM连续hidden/slot memory，以实现真正共享时序表示。

下一步严格顺序：

1. 在独立 fork env 中，从 failure/recovery/verify 状态恢复完全相同的MuJoCo state；
2. 对每组候选短 rollout 4步，写入`candidates.jsonl`及可追溯teacher outcome；
3. 先采1个failure episode做restore determinism、动作差异和标签审计；
4. 扩展train/validation candidate groups，但继续封存final-test；
5. 训练评分器，检查action shuffle、best-candidate accuracy、regret和校准；
6. 通过离线gate后才做只读shadow ranking，仍不立即改变在线动作。

### 20.10 第一份 recovery-state matched candidate 仿真数据

`run_single_scene_recovery.py` 已接入可选的独立MuJoCo fork collector。主环境继续
生成视频和闭环轨迹；第二个环境在候选采集点恢复主环境的flattened state，每个候选
都从同一个state重新开始，因此候选branch不会修改主策略observation或tracker状态。
新增参数包括：

```text
--candidate-dataset-dir
--candidate-collect-every
--candidate-max-forks
--candidate-branch-steps
--candidate-split
--candidate-translation-delta
--candidate-rotation-delta
```

首次smoke使用task9 init-state10、dataset episode9000、2个RETURN TO BOWL状态、
4-step horizon：

```text
/data2/lty/naive-e-slotssm-runs/recovery-candidate-smoke-v1
/data2/lty/naive-e-slotssm-runs/recovery-candidate-expert-smoke-v1
/data2/lty/naive-e-slotssm-runs/recovery-candidates-smoke-v1
/data2/lty/naive-e-slotssm-runs/recovery-candidate-unified-smoke-v1
```

结果：

- 2 matched groups，分别11和12个去重候选，23/23 branch有效；
- 第一组teacher value跨度0.454，EEF 4-step位移范数跨度1.19–7.23cm；
- 第二组teacher value跨度0.110，EEF位移范数跨度0.67–2.08cm；
- 两组的recovery center都不如nominal，证明候选筛选不是无意义恒等操作，也说明
  当前learned mean recovery action并不总是短视野最优；
- 170行主轨迹schema-v2 audit valid，媒体/contact完整且missing files=0；
- 真实候选batch形状为context `[2,76]`、actions `[2,12,7]`，23个有效branch；
- effect/risk/ranking联合loss成功反向传播，所有梯度finite。

这次主episode在170步限制内未完成恢复，不影响candidate data smoke的成立。当前还
不能报告学习排序精度：只有一个episode和两个group，且全部是RETURN状态。下一步
必须扩展到多个episode并覆盖RETURN、DESCEND、CLOSE、VERIFY，按episode划分
train/validation；随后才能训练和做action-shuffle / regret评估。

### 20.11 第一轮跨 episode candidate scorer

新增可分片复现脚本：

```text
scripts/collect_recovery_candidates_v1.sh
```

候选配额尝试4个开发初始状态；init9和init0未触发失败检测，因此零候选且未伪造
样本。有效数据：

```text
/data2/lty/naive-e-slotssm-runs/recovery-candidates-v1
/data2/lty/naive-e-slotssm-runs/recovery-candidate-expert-v1
/data2/lty/naive-e-slotssm-runs/recovery-candidate-unified-v1
```

- train episode9101/init10：12 groups、143 valid branches，覆盖RETURN、DESCEND、
  CLOSE、VERIFY；
- validation episode9102/init11：12 groups、142 valid branches，覆盖RETURN和
  DESCEND；
- 统一主轨迹4 episodes / 920 rows，schema/media/contact audit valid；
- split严格按episode，未使用冻结的6条final-test。

500-step scorer：

```text
/data2/lty/naive-e-slotssm-runs/recovery-candidate-scorer-v2
```

step400按validation regret优先选为best：

- held-out best-candidate accuracy=16.7%，约12候选的uniform random为约8.3%；
- pairwise accuracy=67.0%；
- mean regret=0.01057，优于nominal=0.01672和uniform random expected=0.03661；
- 正常validation total loss=0.06556，打乱candidate action后=0.09908，
  增加0.03352（约51.1%），说明value/ranking输出确实读取动作；
- step500 action-shuffle loss仍增加约44.6%。

这是第一次得到held-out episode上的正向candidate ranking信号，但样本仍极小：
只有1个有效train episode和1个validation episode，validation也没有CLOSE/VERIFY。
不能据此开启在线候选控制。下一步需改进failure-trigger coverage并再采至少3–5个
有效train、2–3个validation episodes；同时应把当前手工teacher value分解为
phase-aware progress与安全分数做消融，避免utility定义主导结论。

### 20.12 首次只读 online shadow ranking 与MP4

`run_single_scene_recovery.py` 新增：

```text
--candidate-scorer-checkpoint
```

提供checkpoint后，每个恢复帧都从当时的observable 76-D context生成并评分候选，
但只记录结果，`execution_changed=false`，主环境仍执行原recovery action。视频用
第二行蓝色字幕明确显示：

```text
SHADOW only: <candidate name> value=<predicted value>
```

首次shadow产物：

```text
/data2/lty/naive-e-slotssm-runs/recovery-candidate-shadow-v1/
task9-dataset9200.mp4
/data2/lty/naive-e-slotssm-runs/recovery-candidate-shadow-v1/
task9-dataset9200/steps.jsonl
/data2/lty/naive-e-slotssm-runs/recovery-candidate-shadow-branches-v1/
task9-dataset9200/candidates.jsonl
```

init11、190-step运行：

- 81个恢复帧执行shadow评分，0帧修改实际动作；
- 4个时刻同时执行真实MuJoCo branches；
- shadow best命中2/4；
- shadow mean teacher regret=0.01846；
- 同4状态始终选nominal的mean regret=0.04518；
- 具体包含一次正确选择`translate_x_plus`、一次正确选择`nominal`，以及两次错误；
- MP4为H.264、256×256、20 FPS、190帧、9.5秒，字幕已人工检查；
- 主任务在190步内未恢复成功，因此本结果只说明候选排序shadow收益，不说明闭环
  成功率提升。

注意init11曾是v1 scorer的validation episode，这次是部署链路回放/可视化，不是新的
泛化测试。下一步应在未参与scorer训练和checkpoint选择、但仍属于development范围的
有效episode上运行同一冻结checkpoint；在增加failure-trigger coverage前继续禁止
让shadow选择改变执行动作。

### 20.13 Level-1 unified Recovery SlotSSM 首次实现

新增：

```text
recovery/models/unified_recovery.py
recovery/data/candidate_tensor_dataset.py
scripts/train_unified_recovery_candidate_head.py
```

统一结构不再让candidate scorer独立编码单帧76-D context，而是：

```text
32-step observable history
        ↓
frozen Recovery Mamba2
        ↓
shared temporal_state [B,T,128]
   ├── 原 phase/onset/persistence/verification/handoff/action heads
   └── candidate action-conditioned future/value head
```

`UnifiedRecoveryModel`保留原`RecoverySlotSSM`对象，candidate head读取指定时刻或窗口
最后时刻的temporal state。旧backbone和current heads从v13 checkpoint加载并全部
冻结，只有candidate head进入optimizer。18项tests证明旧phase logits逐元素不变、
候选轴permutation equivariant、共享memory候选loss可反向传播。

首次500-step实验：

```text
/data2/lty/naive-e-slotssm-runs/unified-recovery-candidate-head-v1
```

按validation regret选择step300：

- unified mean regret=0.01270；
- nominal=0.01672，uniform random expected=0.03661；
- best-candidate accuracy=8.3%；
- validation total loss=0.04450；
- reverse-shuffle candidate actions后loss=0.08644，增加0.04194（约94.3%）；
- 独立单帧scorer的best regret=0.01057，因此统一版已优于基线但尚未超过独立scorer；
- unified motion loss约0.00304，高于独立scorer约0.00055，说明当前temporal state对
  短视野精确几何位移的保留仍不足。

单checkpoint：

```text
/data2/lty/naive-e-slotssm-runs/unified-recovery-candidate-head-v1/
unified-recovery-best.safetensors
```

包含68个tensors：48个原Recovery SlotSSM tensors和20个candidate-head tensors。
逐tensor审计确认48个recovery tensors与v13完全一致，因此这一阶段不会造成旧任务
遗忘。

结论：Level-1统一架构已经证明可实现，candidate future head能够从共享Mamba2 memory
学习动作敏感排序；但极小数据下尚无统一优于独立scorer的证据。下一步不应立即解冻
Mamba2，而应先扩大candidate episodes，并测试在temporal state之外拼接最后时刻
observable geometry的轻量skip connection，判断motion精度差距来自信息压缩还是数据
不足。

### 20.14 Observable geometry skip 消融：暂不采用

为检验统一模型motion loss偏高是否来自Mamba state压缩，新增可选11-D旁路：

```text
target_to_eef (3) + validity
target centroid (2) + validity
mask measurement (2) + validity mask (2)
```

旁路只进入candidate head；Recovery Mamba2和旧heads继续冻结。首先发现candidate
schema-v2 v1虽有depth/mask但未运行visual xyz enrichment，导致`target_to_eef`为零。
因此补做纯可观测mask+depth enrichment：

```text
/data2/lty/naive-e-slotssm-runs/recovery-candidate-unified-v2-visual
```

920/920行成功计算xyz，validity gate接受613行；接受行teacher audit mean error
1.81cm、P90 2.86cm。teacher pose只用于audit。

同seed、同500-step设置的严格消融：

```text
v1  no visual enrichment, no skip:
    unified-recovery-candidate-head-v1
v3  visual enrichment + 11-D geometry skip:
    unified-recovery-candidate-head-v3-visual-geometry
v4  visual enrichment, no skip:
    unified-recovery-candidate-head-v4-visual-no-skip
```

按validation regret选择：

- v1 no-skip：regret=0.01270，motion=0.00304；
- v3 visual+skip：regret=0.01932，motion=0.00283；
- v4 visual/no-skip：regret=0.01684，motion=0.00325；
- nominal regret=0.01672。

geometry skip使对应最佳点motion约改善7%，但regret明显变差；仅加入visual enrichment
也没有稳定改善。极小数据下无法区分是visual-validity分布、单episode过拟合还是
feature融合方式导致。当前模型选择仍为v1 `unified-recovery-best.safetensors`，
保留geometry-skip代码作为可选消融，但不进入shadow/在线控制。下一步回到扩大有效
candidate episode，而不是继续在同一validation episode上调结构。

### 20.15 Failure-trigger coverage修复与3-episode训练

零候选init9/init0的根因是collector阈值自相矛盾：

```text
visual rise >= 15px  → 触发LOST GRASP注入
visual rise >= 20px  → 才承认曾经lifted
```

两条轨迹峰值分别19.13px和17.12px，因此已由视觉抬升触发注入并真实掉落，却永远
不能进入failure detected。新增`recovery/detection/failure_trigger.py`：当干预确实
由`visual-lift`触发时，该触发事件本身构成因果had-lifted证据；仍必须观察到注入后
visual rise回落才确认drop。fixed-step注入不会凭空获得lift证据，当前测量缺失也不会
确认drop。

旧日志回放预测init9在step71、init0在step63检测；实际重采完全一致：

```text
/data2/lty/naive-e-slotssm-runs/recovery-candidates-v2-trigger-fix
/data2/lty/naive-e-slotssm-runs/recovery-candidate-unified-v3-trigger-fix
```

- init9：0→12 groups，144/144 valid branches；
- init0：0→12 groups，144/144 valid branches；
- 两条均从0%提升为100% failure-detection/candidate coverage；
- 新增24 groups全部为DESCEND，反映learned phase在这两个状态迅速进入descend；
- 21项tests通过。

统一训练因此从1个有效train episode / 12 groups扩到3 episodes / 36 groups，固定原
episode9102为validation：

```text
/data2/lty/naive-e-slotssm-runs/
unified-recovery-candidate-head-v5-trigger-coverage
```

结果没有改善模型选择：

- v5 validation regret=0.01801；
- nominal=0.01672，旧v1统一模型=0.01270；
- action shuffle loss delta仍为+0.04954，动作敏感性存在；
- RETURN 3 groups：模型全部选nominal，regret=0.05244，与nominal相同；
- DESCEND 9 groups：模型全部选`translate_y_minus`，regret=0.00653，略差于nominal
  0.00482。

说明只增加同一phase的高度相关数据会造成phase-conditioned choice collapse，并不等于
有效泛化。正式模型继续保留v1。下一次采集必须采用phase quota而非固定时间间隔，
并增加独立validation episodes；在此之前不再用episode9102反复选结构。

### 20.16 新场景task4首条受扰成功轨迹

阶段目标从扩充task9改为先跑通不同任务。选用LIBERO Spatial task4：
“从木柜顶层抽屉取出黑碗并放到盘子上”，init0正常OpenVLA基线111步成功。

`run_single_scene_recovery.py`新增可配置的`--regrasp-return-offset`和
`--regrasp-descend-offset`，解除task9抓取偏移硬编码。掉碗实验可在step70正确检测，
但碗落回抽屉后受柜体碰撞约束，原姿态无法二次抓取；该故障需要新增退出抽屉和重调
姿态阶段，不能仅调xyz偏移。

可逆的抓取前轨迹偏移已跑通：step20起执行3步
`[-0.4, 0.4, 0.2, 0, 0, 0, -1]`，step23交还OpenVLA，最终step124成功。
视频：

```text
/data2/lty/naive-e-slotssm-runs/task4-new-scene-v1/
recoverable-deviation-task4-episode0-v2.mp4
```

下一步应先把task4的纯视觉失败检测跑通，再为“掉回抽屉”增加
RETREAT/REORIENT阶段并采集该任务专用训练数据。

### 20.17 LIBERO-10自然失败基线与路线调整

主实验路线已从“人为破坏本来成功的轨迹”调整为标准benchmark自然失败恢复：

```text
冻结OpenVLA
vs.
冻结OpenVLA + Failure-Aware SlotSSM
```

新增本地模型：

```text
/data2/lty/models/openvla-7b-finetuned-libero-object
/data2/lty/models/openvla-7b-finetuned-libero-10
```

LIBERO-10完成10 tasks × 10 initial states，共100条无故障注入基线；100个MP4均可读。
总体49成功、51失败，成功率49.0%。重点审计task4、task6、task9的17条失败后，
建议以task6为第一主任务：它同时包含第一抓取停滞、第一子目标完成后未切换到布丁、
不稳定放置和近目标谓词失败。

完整统计和逐episode行为分类见`LIBERO10_NATURAL_FAILURE_AUDIT.md`。下一步不是继续
盲目扩充rollout，而是标注task6子目标完成/停滞时刻，先验证自然失败的oracle可恢复
上限，再训练共享时序状态的SlotSSM进度、失败和候选未来头。

### 20.18 Task6自然失败Oracle恢复上限

新增`run_task6_oracle_recovery.py`：连续5帧满足正式
`On(porcelain_mug_1, plate_1)`谓词后，仅将OpenVLA提示切换为布丁子目标，Oracle
不产生动作。完整episode0–9配对开发集由原始3/10成功提升为7/10，恢复episode1、4、
6、7，且保留原本成功的episode0、5、8，无回退。episode3虽正确切换仍失败，说明
后续动作能力也是独立瓶颈；episode2、9未完成第一子目标，说明还需处理第一阶段停滞。

成功视频与逐步日志位于：

```text
/data2/lty/naive-e-slotssm-runs/task6-oracle-stage-switch-v1
```

该结果首次在无人工故障注入的标准benchmark自然失败上证明：正确的持续子目标状态
和阶段切换本身可以显著提高冻结OpenVLA成功率。当前仍是使用仿真谓词且已经查看过的
开发集上限，不能作为学习SlotSSM结果；episode10–19保留作独立验证。

### 20.19 Task6视觉子目标标签集

新增`scripts/replay_task6_oracle_visual_dataset.py`，从精确初始状态重放10条Oracle
实验动作，导出原始RGB和每帧`On(porcelain_mug_1, plate_1)`真值：

```text
/data2/lty/naive-e-slotssm-runs/task6-oracle-visual-dataset-v1
```

数据集共3180帧、1023个正例，10条轨迹的最大物体位置重放误差均为0.0米。仿真状态只
写入监督标签，后续模型输入必须限定为RGB/Slot、proprio和历史状态。划分必须以episode
为单位，禁止随机拆分相邻帧造成泄漏。下一步依次训练RGB基线、冻结Slot特征分类头和
SlotSSM时序模型，用连续5帧预测替代Oracle触发器；最终只在未查看的episode10–19报告
独立结果。

### 20.20 RGB单帧开发基线

新增`scripts/train_task6_visual_state_baseline.py`，严格按episode划分：0–5训练、6/8
验证、7/9开发测试。模型输入只有原始agent-view RGB，连续5帧阳性才触发。

小型CNN在训练集达到99.4% balanced accuracy，但验证集为79.5%、开发测试为62.3%。
更重要的是，纯阴性episode8和9均发生误触发，阳性episode7的触发延迟25帧。因此
单帧RGB分类不满足在线切换要求；高帧准确率也不能代替轨迹级零误触发指标。下一步
接入已有冻结Slot编码器并训练时序头，模型选择优先最小化轨迹误触发，再考虑检测延迟。

产物：

```text
/data2/lty/naive-e-slotssm-runs/task6-rgb-state-baseline-v1
```

### 20.21 连续Slot时序触发器首次在线闭环

新增：

- `scripts/precompute_task6_visual_slots.py`：用冻结OpenVLA视觉骨干和已有Stage-1
  tokenizer，为10条轨迹提取连续因果Slot，`past_slots`只在episode边界清零；
- `scripts/train_task6_slot_temporal_state.py`：Slot注意力池化加两层因果GRU；
- `run_task6_oracle_recovery.py`的非Oracle学习触发模式。

3180帧全部完成Slot预计算。当前模型只读取RGB派生的`16×512` Slot和自身历史状态，
仿真谓词仅写日志用于事后核对。在线开发预检：

| Episode | 学习触发 | 最终结果 | 诊断 |
|---:|---:|---:|---|
| 6 | step 140 | 成功，236步 | 第一子目标真值首次成立后10步触发 |
| 7 | step 147 | 成功，260步 | 第一子目标真值首次成立后23步触发 |
| 9 | step 139 | 失败，520步 | 第一子目标从未成立，发生误触发 |

视频和日志：

```text
/data2/lty/naive-e-slotssm-runs/task6-learned-stage-v1
```

这是“连续Slot→时序状态→语言阶段切换→冻结OpenVLA动作”的首次非特权在线闭环，
但时序头目前是GRU而非最终Mamba2，且只有6条训练episode。episode9说明现有Slot更
容易识别总体任务进度，而不能可靠区分“杯子在盘附近”和正式`On`关系。下一步不动
episode10–19保留集，采集episode20–49及失败停滞hard negatives，再比较GRU/Mamba2，
达到轨迹级低误触发门槛后才运行保留集。

### 20.22 非保留数据扩充与GRU/Mamba2对照

保持episode10–19完全不查看，新采集episode20–29的10条标准初始状态Oracle轨迹：
4条成功（22/25/26/27），6条失败（20/21/23/24/28/29）。其中20/21/23/29从未完成
第一子目标，24完成第一子目标但后续动作失败，28短暂放好后又掉落。后两类说明后续
状态不能永远单向锁存。

新增数据共4323帧、1056个`mug_on_plate`正例，所有动作重放最大位置误差仍为0米；
全部完成连续Slot预计算。路径：

```text
/data2/lty/naive-e-slotssm-runs/task6-oracle-stage-switch-train20-29-v1
/data2/lty/naive-e-slotssm-runs/task6-oracle-visual-dataset-train20-29-v1
/data2/lty/naive-e-slotssm-runs/task6-continuous-slots-train20-29-v1
```

训练集扩为episode0–5和20–29，验证仍为6/8，开发测试仍为7/9。同一Slot输入下：

| 时序后端 | episode6 | episode8误触发 | episode7 | episode9误触发 |
|---|---:|---:|---:|---:|
| GRU | 准时检出 | 无 | 提前4帧检出 | 有 |
| Mamba2 | 准时检出 | 无 | 提前7帧检出 | 有 |

两者都修复了episode8，却都无法区分episode9。将episode9显式用于阈值校准也不能同时
保住正例和去掉误触发，所以问题不在GRU/Mamba2选择或阈值，而在当前Slot来自
LIBERO-Spatial训练，缺少精确杯—盘接触/支撑关系。下一步应针对task6微调Slot/关系头，
以杯、盘mask或仿真投影几何作训练监督；在此之前不运行episode10–19。

### 20.23 Task6专用Grounded Role Slots

仅在已有通用Slot上增加双角色查询仍未消除episode9误触发，说明上游表示缺少对象关系
信息。为此新增：

- `scripts/replay_task6_role_masks.py`：精确重放并导出杯子/盘子64×64 instance masks；
- `scripts/train_task6_grounded_role_slots.py`：RGB编码器同时预测杯/盘mask，按预测mask
  池化两个角色Slot，再用`cup, plate, cup-plate, cup*plate`和几何量判断`On`关系；
- `run_task6_oracle_recovery.py --grounded-role-checkpoint`在线模式。

mask和仿真谓词只作训练监督；推理输入仍只有RGB。20个开发/训练episode全部完成mask
导出，重放误差0米。最佳模型开发测试结果：

- episode7正确触发，episode9整条520帧不触发；
- frame balanced accuracy 99.15%；
- mug/plate mask IoU分别为0.559/0.758。

在线闭环：

| Episode | Oracle首次成立 | 学习切换 | 最终结果 |
|---:|---:|---:|---:|
| 6 | 130 | 198 | 成功，264步 |
| 7 | 124 | 119 | 成功，262步 |
| 9 | 从未成立 | 从未切换 | 失败，520步 |

视频：

```text
/data2/lty/naive-e-slotssm-runs/task6-grounded-role-online-v1
```

这首次同时保住episode6/7的在线恢复并消除episode9假阳性，证明对象角色监督比单纯更换
GRU/Mamba2更关键。当前关系头仍是单帧模型加连续5帧确认；下一步把两个grounded role
slots作为Mamba2输入，加入“完成后又掉落”的状态回退头，再决定是否达到保留集门槛。

### 20.24 Grounded Role Mamba2与可逆阶段状态

新增`precompute_task6_grounded_role_sequences.py`，从20条轨迹提取每帧两个96维
grounded role slots；新增`train_task6_grounded_role_mamba.py`，共享Mamba2主干输出：

1. 当前杯—盘关系是否完成；
2. 已完成后是否发生关系失效/掉落。

为防止时序模型再次仅凭任务进度误触发，采用安全门控：

```text
进入分数 = min(Mamba completion, grounded relation)
回退分数 = min(Mamba rollback, 1 - grounded relation)
```

进入与回退均需连续5帧，回退阈值提高到0.8。离线结果保持episode6/7可检出、
episode8/9不误触发；在训练hard case episode28上，真值回退step236，Mamba2预测
step237。`run_task6_oracle_recovery.py`已加入可逆状态机，可在
`FULL TASK ↔ PUDDING SUBGOAL`之间双向切换。

proper temporal在线episode28在step125切换，最终255步成功；该新策略没有重现原Oracle
轨迹中的掉落，因此没有实际触发回退。这是更好的任务结果，但只能证明回退链路已离线
对齐、在线可运行，不能声称已完成在线掉落恢复。视频：

```text
/data2/lty/naive-e-slotssm-runs/task6-grounded-temporal-online-v2/task6-episode28.mp4
```

下一步应构造不使用仿真状态作为输入的可复现掉落/滑落开发轨迹，在线验证回退后重新
抓杯；完成后再决定是否冻结开发并运行episode10–19。
