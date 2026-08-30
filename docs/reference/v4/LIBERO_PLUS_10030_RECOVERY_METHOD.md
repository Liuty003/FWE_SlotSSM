# LIBERO-Plus 10,030 条任务的通用失败恢复方法

最后更新：2026-08-26

本文整理当前正在推进的 LIBERO-Plus 大规模恢复实验。它描述研究目标、方法结构、数据契约、自动化流水线、训练方案、跨 backbone 对比协议，以及已经完成和仍待完成的工作。历史上的 Book、Mug + Pudding、Black Bowl 实验是方法原型；本文定义的是面向完整 10,030 条官方实例的新口径。

## 一、目标与核心主张

我们的目标不是重新训练或替换原始策略，而是在冻结的基础策略外增加一个短时失败恢复模块：

```text
基础策略正常执行
        ↓
gate 判断基础策略是否进入可恢复失败
        ↓ 是
failure router 选择恢复技能
        ↓
recovery policy 短暂接管并纠正状态
        ↓
handoff gate 确认基础策略重新具备完成能力
        ↓
归还基础策略继续执行
```

目标实验结论应当是：

> 在不增加推理传感器或 simulator 特权信息的条件下，只在训练阶段利用额外监督，为 OpenVLA-OFT、Diffusion Policy、ACT 等基础策略加入统一的恢复模块，并提高它们在相同 LIBERO-Plus 评测实例上的成功率。

恢复模块包含四个逻辑部分：

1. `ownership gate`：判断保持基础策略、恢复接管或归还基础策略。
2. `failure router`：把失败映射到少量通用恢复技能，而不是映射到 task ID。
3. `recovery action policy`：执行局部恢复动作。
4. `handoff verifier`：确认恢复后的状态确实适合交还基础策略。

## 二、官方实验范围

实验对象是 LIBERO-Plus 官方定义的 10,030 条实例，而不是我们自行生成的扰动：

| Suite | 实例数 |
|---|---:|
| `libero_spatial` | 2,402 |
| `libero_object` | 2,518 |
| `libero_goal` | 2,591 |
| `libero_10` | 2,519 |
| 合计 | 10,030 |

官方扰动共有七类：

- Background Textures
- Camera Viewpoints
- Language Instructions
- Light Conditions
- Objects Layout
- Robot Initial States
- Sensor Noise

suite、扰动类别和 difficulty 用于评测分层与统计，不直接决定恢复模型。在线恢复路由依据自动推断的失败类型，不能依赖 task ID selector。

冻结的官方索引位于：

```text
/data2/lty/naive-e-slotssm-runs/libero-plus-global-v1/official-manifest.json
```

## 三、公平性与输入契约

### 3.1 主实验的硬约束

恢复 student 在推理和正式评测时，只能使用对应基础策略本来就能获得的输入，以及该基础策略自己输出的 nominal action。

以 OpenVLA-OFT 为例，允许的输入是：

- 基础策略实际使用的主视角 RGB；
- 基础策略实际使用的腕部 RGB；
- 语言指令；
- proprio；
- OpenVLA-OFT 自己产生的 nominal action。

禁止作为主实验在线输入的信息包括：

- depth；
- instance segmentation；
- 物体或 fixture 真值位姿；
- MuJoCo state；
- simulator contact；
- BDDL predicate truth；
- oracle phase、目标名称或失败类型。

nominal action 是基础策略自身的输出，不是额外传感器，因此可以作为 gate 和 residual recovery 的条件。

### 3.2 训练阶段允许 privileged supervision

训练时可以使用 RGB-D 和 simulator 信息，但只能作为标签、teacher 输入或辅助监督：

| 信息 | 离线训练用途 | 主评测在线输入 |
|---|---|---:|
| backbone 原生 RGB/proprio/language | student 输入 | 允许 |
| backbone nominal action | student 条件/残差基准 | 允许 |
| depth | 几何辅助标签、teacher 规划 | 禁止，除非原 backbone 本身使用 |
| instance segmentation | Slot/目标表征辅助标签 | 禁止 |
| object/fixture pose | 相对几何、目标方向标签 | 禁止 |
| contact | 抓取质量与接触辅助标签 | 禁止 |
| BDDL predicates | phase、进度、handoff 标签 | 禁止 |
| MuJoCo state | 失败点复现、反事实 teacher | 禁止 |

最干净的实现是 privileged teacher 到 native-input student 的蒸馏：teacher 可以看 RGB-D 和 simulator truth，student 从训练开始就只能看 backbone 原生输入。

### 3.3 短时历史的口径

历史观测和历史 executed action 不是新传感器，但会给恢复模块增加记忆能力。为避免表述含混，最终至少区分两组：

1. `strict matched-input`：当前原生观测 + nominal action，作为主结果。
2. `temporal recovery`：额外使用原生观测和已执行动作的短时历史，作为明确标注的增强实验。

### 3.4 跨 backbone 的输入匹配

不能给所有恢复模型强制使用同一套 RGB-D 输入。每个 backbone 使用自己的 native observation adapter：

```text
OpenVLA-OFT native observation ─┐
DP native observation          ├─> modality adapter ─> shared recovery representation
ACT native observation         ┘
```

- 如果 DP 不使用腕部相机，对应 DP 主实验中的恢复模块也不能使用腕部相机。
- 如果 ACT 只使用固定相机 RGB 和关节状态，恢复模块也只能使用这些输入。
- 如果某个基础策略本身使用 depth，该 backbone 的对比才可以在线使用 depth。

## 四、当前正在做什么

当前阶段不是训练恢复 checkpoint，而是在完整 10,030 条实例上运行冻结的 OpenVLA-OFT+ 基线，并构造一次采集、可长期复用的数据底座。

正式 schema 是 `oft_archival_rollout_v4`。每条 rollout 保存：

- 是否完成任务、轨迹长度、语言和任务 provenance；
- 无损主视角和腕部 RGB；
- 双视角 float32 depth；
- 双视角 instance segmentation；
- 完整 proprio；
- planned action、executed action、真实 action chunk id 和 offset；
- 每一步 BDDL 子目标真假；
- 对象/fixture 位姿、关节和 contact pairs；
- 每一步及终态 MuJoCo state、ctrl、camera pose；
- 官方初始状态和每一步 NumPy RNG 状态；
- 主视角和腕部 MP4。

这里保存 RGB-D 不代表部署时使用 RGB-D。它们的主要用途是离线监督、teacher 构造、失败分析和训练诊断。

全量数据根目录：

```text
/data2/lty/naive-e-slotssm-runs/libero-plus-global-v1/full-audit-v1
```

数据完整性已经通过 Camera Viewpoint 和 Sensor Noise smoke 验证。Sensor Noise task1374 从官方初态重放到 step20 时，逐步 MuJoCo state 误差为 0，下一帧受噪 RGB 逐像素一致。这说明保存的初态、executed actions 和 RNG 足以重建准确失败前缀。

## 五、完整自动化流水线

### 5.1 冻结全局索引

从 LIBERO-Plus 官方 `task_classification.json` 生成 10,030 条 manifest，验证 suite 数量、扰动类别、difficulty 和有效 BDDL 路径。所有后续数据都使用稳定的 `suite:task_id:episode_id` 标识。

代码：

```text
scripts/build_libero_plus_global_manifest.py
```

### 5.2 运行基础策略并建立基线

对每条实例运行冻结基础策略，记录成功和失败。成功轨迹用于：

- 统计原始成功率；
- gate 的 `NORMAL / KEEP_BASE` 监督；
- 检查恢复模块是否破坏原本成功的任务；
- 学习什么状态不应该接管。

失败轨迹用于后续失败挖掘和 recovery teacher 构造。

代码：

```text
scripts/collect_oft_nominal_trajectories.py
scripts/run_libero_plus_audit_shard.py
scripts/validate_libero_plus_archival_rollouts.py
```

### 5.3 自动失败分类

利用仅离线可见的 BDDL 进度、夹爪变化、末端运动、物体运动、fixture joint 和 contact 信息，为失败轨迹生成粗粒度类型、置信度和候选恢复技能。

当前分类包括：

- `progress_stall_before_manipulation`
- `empty_grasp_or_failed_contact`
- `transfer_or_placement_failure`
- `partial_goal_completion`
- `goal_regression`
- `articulation_contact_failure`
- `articulation_incomplete`
- `unresolved_physical_failure`

分类器宁可输出低置信，也不能伪造精确原因。只有低置信或未知簇需要少量人工视频抽检，不能逐条人工审查 10,030 个任务。

代码：

```text
scripts/classify_libero_plus_failures.py
```

### 5.4 自动确定失败边界与接管边界

对失败轨迹自动寻找：

- 最后一个仍然正常的状态；
- 基础策略开始失去能力的状态；
- 第一次空抓、掉落、错误接触、目标回退或长时间停滞；
- 可作为 recovery fork 的候选时刻。

由此构造 gate 标签：

```text
失败边界前                         -> KEEP_BASE / NORMAL
失败边界附近且仍可恢复             -> TAKEOVER
teacher 恢复过程中                 -> KEEP_RECOVERY
经基础策略续跑认证的 handoff 边界   -> RETURN_BASE
```

边界需要连续确认和 hysteresis，避免在相邻状态反复切换控制权。

### 5.5 精确恢复失败状态

MuJoCo state 本身不包含 robosuite OSC controller 的全部 Python 缓存，所以不能把“直接加载单步快照”误称为完全等价续跑。

正式方法是：

1. 加载保存的官方初始状态；
2. 执行原轨迹保存的全部 executed actions；
3. 对 Sensor Noise 在每一步恢复保存的 NumPy RNG；
4. 在目标失败时刻校验当前 state 与保存 state 一致；
5. 从该状态启动反事实 recovery teacher。

这样 controller 内部状态会由前缀重放自然重建。已经验证 prefix replay 的最大 state error 为 0。

代码：

```text
scripts/save_oft_recovery_fork_snapshot.py
```

### 5.6 运行 privileged recovery teacher

teacher 只在离线数据构造阶段使用 simulator truth。它根据失败类型运行少量通用技能，而不是为每个 task 写一个完整脚本：

- `relocalize_and_approach`
- `regrasp`
- `stabilize_and_lift`
- `reposition_or_replace`
- `continue_remaining_subgoal`
- `relocalize_and_recontact_fixture`
- `continue_articulation`

teacher 分支必须通过 simulator 物理验证，不能仅凭动作看起来合理：

- 抓取技能检查有效接触、夹持和稳定抬升；
- 放置技能检查对应 `on` 或 `in` predicate；
- articulation 检查 `open`、`close` 或 `turnon`；
- 多子目标任务检查目标是否真实增加且没有回退；
- 最终恢复完成需由任务 success 或经认证 handoff 支持。

未成功的 teacher 分支不进入 recovery action 正样本，但可以成为 verifier 的负样本。

### 5.7 认证归还基础策略的时刻

teacher 局部成功不等于应该立刻归还基础策略。对 teacher 轨迹的多个候选状态运行冻结基础策略续跑：

- 如果基础策略可以从该状态完成任务，则该状态是认证 handoff；
- handoff 前仍由 recovery 控制；
- handoff 后完全 bypass recovery action，继续使用基础策略；
- 如果没有认证 handoff，恢复技能必须继续或判定失败。

这一步为 `RETURN_BASE` 提供反事实监督，避免 gate 只学会接管、不会归还。

### 5.8 构造训练数据

最终至少构造三类数据：

1. `normal/base-preservation`：原策略成功轨迹和失败边界前状态。
2. `ownership/router`：接管、保持恢复、技能类型和归还标签。
3. `recovery action`：经过 simulator 验证的 teacher action 和 DAgger action。

辅助训练目标可包括：

- depth/relative 3D geometry prediction；
- target mask 或 Slot consistency；
- contact 和 grasp-quality prediction；
- phase/subgoal progress；
- target direction/distance；
- recovery completion 和 handoff readiness。

这些辅助头用于塑造表示，不能在推理时直接读取 oracle 标签，也不能不经验证地直接支配夹爪和动作。

### 5.9 训练 gate、router 和 recovery action

建议模型结构：

```text
backbone-native observation encoder
               +
       nominal action encoding
               ↓
     shared recovery representation
       ├── ownership gate
       ├── failure skill router
       ├── auxiliary heads（训练期）
       └── recovery action head
```

基础策略默认冻结。恢复 action 可以采用：

- 对 nominal action 的 residual；
- 直接预测短 action chunk；
- 技能条件化的 action head；
- 对确定性低层 controller 的学习式目标参数。

主版本优先采用少量通用 skill heads，而不是 10,030 个 task-specific checkpoint。若单个统一 action head 容量不足，可以采用共享 encoder/gate + 按恢复技能或 suite 划分的少量 head，并明确报告路由方式。

### 5.10 自动 DAgger

纯 behavior cloning 在 teacher 状态上 loss 很低，也可能在闭环中抽搐、空抓或浪费时间。历史 Mug/Pudding 和 Black Bowl 实验已经证明这一点。

因此在线恢复 rollout 需要自动保存偏离 teacher 分布的状态，从精确 fork 重新调用 teacher：

```text
student 闭环偏离
       ↓
保存候选状态
       ↓
privileged teacher 重新标注
       ↓
只保留物理验证有效的纠偏轨迹
       ↓
合并训练并再次闭环评测
```

DAgger 重点覆盖空中抽搐、过早闭合、接近误差、抓住后掉落、放置偏差和恢复超时。

## 六、跨模型实验方式

对每个基础策略分别执行相同框架：

```text
基础策略基线采集
        ↓
该策略自己的自然失败集合
        ↓
统一 failure miner / teacher skill library
        ↓
native-input recovery student
        ↓
基础策略 vs 基础策略 + recovery
```

OpenVLA-OFT、DP、ACT 的 action space、chunk 长度和 observation spec 不同，因此需要 backbone adapter，但 gate 语义、failure taxonomy、teacher 验证和评测指标应保持一致。

公平比较必须满足：

- 同一 backbone 的 baseline 和 hybrid 使用相同任务、初态和最大步数；
- hybrid 不增加 baseline 没有的在线传感器；
- teacher 和 privileged labels 只在训练期使用；
- 分别报告基础成功、修复成功和原成功回退；
- 不把某个 backbone 的失败轨迹直接当成另一个 backbone 的完全相同失败状态。

## 七、评测指标与结果口径

最终在完整 10,030 条实例上至少报告：

- baseline success rate；
- hybrid success rate；
- baseline failures repaired；
- baseline successes regressed；
- net success gain；
- gate false takeover rate；
- takeover precision/recall；
- handoff success rate；
- recovery 平均接管步数与超时率；
- 按 suite、扰动类别、difficulty 和 failure type 的分层结果。

用户当前允许训练和测试使用同一批固定实例，因此该主实验属于 closed-set / in-sample repair。结果可以证明“在这 10,030 条官方实例上为原模型增加恢复模块是否有效”，但不能表述成未见任务或未见初始化泛化。后续若增加 held-out 实验，需单独报告。

历史固定任务结果也要保持原有口径：

- Book34 的 task-ID selector 不能称为纯神经泛化恢复；
- Mug + Pudding 的 in-sample DAgger 不能称为未见任务泛化；
- Black Bowl 的学习 gate + 3D 几何 + 确定性抓取 FSM 是混合系统；
- 如果历史版本在线使用 depth，只能作为 RGB-D hybrid 结果，不能放入严格 matched-input 主表。

## 八、已经完成与尚未完成

### 已经完成

- 冻结并验证官方 10,030 条 manifest；
- 盘点已有 OFT+ baseline；
- 实现多 GPU、跨 suite、模型单次加载和断点续跑；
- 实现 `oft_archival_rollout_v4`；
- 实现 BDDL 目标进度和初版自动失败分类；
- 验证五类官方 BDDL predicate：`on`、`in`、`open`、`close`、`turnon`；
- 完成 141 条 suite × perturbation × difficulty 分层 pilot；
- 验证 Camera Viewpoint、Sensor Noise、checksum 和零误差 prefix replay；
- 启动完整 10,030 条 OpenVLA-OFT+ archival baseline 采集。

### 尚未完成

- 全量 v4 采集和完整性审计；
- 全量 success/failure 分层统计；
- failure boundary 自动定位；
- 低置信失败聚类及少量人工视频抽检；
- 通用 teacher skill library 的全量化；
- handoff 反事实认证；
- native-input gate/router/recovery student 数据集；
- 严格 matched-input OpenVLA-OFT 主模型；
- DP 和 ACT backbone adapter 与对比实验；
- 全量 hybrid 闭环评测和 DAgger 迭代。

## 九、近期执行顺序

1. 完成 10,030 条 OpenVLA-OFT+ archival baseline。
2. 运行全量 schema/checksum/唯一性校验。
3. 生成总体及 suite × perturbation × difficulty 基线报告。
4. 自动分类失败并输出置信度、剩余 BDDL 子目标和候选 skill。
5. 对低置信簇抽取少量代表 MP4，必要时人工修正规则。
6. 自动定位 takeover boundary 并生成精确 recovery forks。
7. 按通用 skill 并行运行 privileged teacher，只保留物理验证成功轨迹。
8. 运行 OFT 续跑，认证 handoff boundary。
9. 构造 native-input gate/router/recovery action/auxiliary 数据。
10. 先训练严格 matched-input OpenVLA-OFT recovery，再运行小规模闭环 smoke。
11. 自动 DAgger 修复闭环 covariate shift。
12. 全量评测 baseline 与 hybrid，并按官方维度报告。
13. 固化 backbone adapter 后迁移到 DP、ACT。

## 十、主要资产与代码入口

数据和报告：

```text
/data2/lty/naive-e-slotssm-runs/libero-plus-global-v1/
  official-manifest.json
  baseline-inventory.json
  stratified-pilot-v1/
  full-audit-v1/
```

当前主要代码：

```text
scripts/build_libero_plus_global_manifest.py
scripts/inventory_libero_plus_baselines.py
scripts/collect_oft_nominal_trajectories.py
scripts/run_libero_plus_audit_shard.py
scripts/validate_libero_plus_archival_rollouts.py
scripts/classify_libero_plus_failures.py
scripts/save_oft_recovery_fork_snapshot.py
```

历史 V4.3 参考实现：

```text
recovery/models/v43_local_recovery.py
recovery/models/v4_relational_recovery.py
recovery/data/v4_relation_features.py
recovery/data/v4_geometry.py
scripts/build_ownership_dataset.py
scripts/train_v43_local_recovery.py
scripts/run_v4_relational_online.py
scripts/run_v43_dagger_factory.py
```

长期资产索引见 [ARTIFACT_INDEX.md](./ARTIFACT_INDEX.md)，历史单任务流水线和经验见 [v4流程.md](./v4流程.md)。
