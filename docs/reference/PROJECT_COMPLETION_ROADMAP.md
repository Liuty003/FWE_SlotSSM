# Failure-Aware Embodied-SlotSSM 完整项目路线图

更新时间：2026-07-28（Asia/Shanghai）

本文定义从当前模块化 task9 lost-grasp feasibility prototype 出发，直到完成统一
Failure-Aware Embodied-SlotSSM、严格实验和最终交付所需的全部工作。后续执行以本文
和最新 `HANDOFF.md` 为准；较早文档中的阶段计划若与本文冲突，以本文为准。

---

## 1. 最终目标与完成定义

最终系统应在正常操作和失败恢复中共享持续对象表示与时序状态：

```text
RGB / depth / language / proprio / previous action
                         │
                         ▼
          target-conditioned persistent Slot encoder
                         │
                         ▼
               shared temporal SlotSSM
                         │
       ┌─────────────────┼──────────────────┐
       ▼                 ▼                  ▼
 normal action       current failure    recovery state
 conditioning          assessment        and actions
       │                 │                  │
       └─────────────────┴──────────────────┘
                         │
                         ▼
             closed-loop execution and verification
```

项目不能以“某个脚本能跑”或“单个 episode 成功”作为最终完成。完整完成需要同时满足：

1. 正常操作时维护持续、目标相关的对象 Slot，并让动作模型真实读取；
2. 识别空抓、滑落、错目标、错误放置和不合法阶段转换；
3. 对已经发生的失败进行 post-transition assessment；
4. 对多个未执行候选动作预测未来后果和风险；
5. 从候选中选择短视野恢复动作，并在执行后重新观测；
6. 视觉验证恢复结果，失败时允许再次规划；
7. 在线输入不包含 simulator object pose、height、contact 或 oracle segmentation；
8. 在 episode-isolated、未用于调参的测试集上提高最终成功率；
9. 不显著破坏原本成功的 nominal policy 轨迹；
10. 提供可复现代码、配置、checkpoint、数据 manifest、指标、视频和论文材料。

最终至少形成两套可比较的系统：

```text
OpenVLA nominal policy + Failure-Aware SlotSSM

可选增强：
π0.5 nominal policy + 同一 Failure-Aware SlotSSM 接口
```

π0.5 迁移不是统一模型主线的前置条件。先在 OpenVLA 上完成方法闭环，再决定是否将
π0.5 纳入最终主要实验。

---

## 2. 当前起点

### 2.1 已完成

- OpenVLA 7B 无头 LIBERO Spatial baseline；
- 原始 Embodied-SlotSSM Stage-1/2/3 训练和在线机械链路；
- Stage-A 严格审计，证明旧 Stage-1 无法可靠绑定 task9 目标碗；
- Stage-B target selector、proposal cascade 和 temporal mask tracker；
- predicted-mask motion failure detector；
- predicted mask + depth 的 visual target xyz；
- learned geometry residual recovery controller；
- learned monotonic phase predictor；
- visual temporal regrasp verifier；
- task9 lost-grasp 在若干 episode 中在线恢复成功；
- recovery expert 数据采集、manifest 和初版 schema；
- counterfactual state branching 和多个动作分支采集能力。

### 2.2 当前原型仍依赖

- 多个相互独立的小模型；
- 手写 `VISUAL SETTLE`；
- phase-based gripper sign override；
- mask-motion failure rule；
- Stage-B 和显式 visual xyz 前端；
- 少量 task9 recovery episode；
- 仅 lost-grasp 为主的失败分布。

### 2.3 当前最重要的架构决定

旧模块化链路保留为：

```text
teacher + data collector + baseline + offline labeler
```

不要继续为单个新失败状态增加独立 MLP、独立 checkpoint 或在线规则。新增学习能力
应进入共享 Recovery SlotSSM 或其正式的数据/评测组件。

---

## 3. 总体执行顺序

```text
Phase 0  冻结协议与基线
   ↓
Phase 1  统一 sequence schema 和数据转换
   ↓
Phase 2  扩充 task9 recovery / hard-negative / normal 数据
   ↓
Phase 3  持续目标 Slot 与正常动作 cross-attention
   ↓
Phase 4  共享 Recovery SlotSSM 辅助状态模型
   ↓
Phase 5  学习完整恢复动作、handoff 和 gripper
   ↓
Phase 6  action-conditioned future model 与候选风险选择
   ↓
Phase 7  task9 多失败类型闭环
   ↓
Phase 8  LIBERO 多任务扩展与 benchmark 严格评测
   ↓
Phase 9  π0.5 迁移与底座消融（条件允许时）
   ↓
Phase 10 最终代码、实验、论文和交付
```

每个阶段必须通过 gate 后才能进入依赖该能力的在线控制。离线 frame accuracy 或
teacher-forced accuracy 不能单独作为上线依据。

---

## 4. Phase 0：冻结协议、复现实验与基线

### 4.1 目标

在开始统一模型前，固化当前可复现事实，避免后续无法判断新系统是否改善。

### 4.2 工作

1. 固化当前 OpenVLA task9 baseline 配置和 episode 列表；
2. 固化当前模块化 recovery teacher 的全部 checkpoint、阈值和配置；
3. 固化成功、失败和未触发注入的代表性视频与 metadata；
4. 建立统一 experiment ID、config snapshot 和 git commit 记录；
5. 为所有运行统一保存：
   - `metadata.json`；
   - `steps.jsonl`；
   - MP4；
   - resolved config；
   - checkpoint hashes；
   - dataset manifest hash；
6. 定义 train/validation/development-test/final-test episode 隔离规则；
7. 禁止继续使用 episode7/9/10/11 作为“从未见过的 final test”；
8. 检查当前工作树，明确用户修改，不做重置或批量清理。

### 4.3 输出

```text
configs/baselines/
evaluation/protocols/
artifacts/baseline_registry.json
```

### 4.4 Gate

- 同一冻结配置可重跑代表性 baseline；
- 所有关键 checkpoint 和数据均能由 registry 定位；
- final-test episode 在运行前未用于训练、阈值选择或视觉诊断。

---

## 5. Phase 1：统一恢复序列数据协议

### 5.1 目标

建立最终模型唯一使用的数据 contract。privileged simulator state 只能作为 teacher
标签，不能泄漏到在线输入。

### 5.2 建议目录

```text
recovery/
├── data/
│   ├── schema.py
│   ├── dataset.py
│   ├── sequence_builder.py
│   ├── conversion.py
│   └── validation.py
├── models/
├── training/
├── evaluation/
└── teachers/
```

### 5.3 每步在线可用输入

```text
rgb
depth or frozen visual features
language instruction
proprio
previous executed action
OpenVLA nominal action / nominal action chunk
Stage-B target mask and target feature
visual target xyz and validity
previous Recovery SlotSSM state
```

第一版可保留 Stage-B 和 visual xyz。它们是可观测前端，不属于 privileged input。

### 5.4 Teacher-only 字段

```text
simulator target/object pose
instance segmentation
contact state
object height
grasped / attached truth
oracle phase
oracle recovery action
oracle failure onset and type
oracle placement relation
```

数据 loader 必须通过显式 allowlist 构建在线输入，不允许简单把整行 JSON 转 tensor。

### 5.5 训练标签

```text
normal / failure / recovery / handoff mode
failure onset
failure type
recovery phase
grasped / attached
lifted
slipped
verified
handoff ready
recovery completed
expert complete 7-D action
expert residual relative to nominal action
short-horizon task progress
episode recovery success
final task success
```

### 5.6 三种训练样本

#### Sequence window

连续 32 或 64 步，用于训练共享时序状态：

```text
normal → contact → grasp → lift → slip → recovery → verify → handoff
```

#### Observed transition

```text
(state_t, executed_action_t, observed_state_t+1)
```

用于 post-transition failure assessment 和 action-conditioned dynamics。

#### Matched candidate group

```text
same simulator state
+ K candidate actions
+ K resulting outcomes
+ K risk/progress labels
```

候选组必须作为一个整体保存和采样，不能在数据加载时拆散。

### 5.7 数据 split

- 按 episode 隔离；
- 多任务阶段还要按 task/relation/layout 隔离；
- 相邻帧不得随机分到不同 split；
- DAgger 新数据必须记录模型版本和采集轮次；
- final test 不得回流训练，除非正式关闭该轮 final test 并重新建立新 final split。

### 5.8 Gate

- schema round-trip、shape、dtype、时间对齐检查全部通过；
- teacher-only 字段泄漏测试通过；
- 同一 timestep 的 observation、nominal action、executed action 和 next observation
  对齐；
- episode split 无交叉；
- 可以从已有 episode 构造 32/64 步连续窗口。

---

## 6. Phase 2：扩充 task9 数据

### 6.1 近期数据规模

```text
代码和 loss smoke：现有数据即可
task9 unified-model feasibility：至少 20–30 条完整恢复序列
task9 相对稳定版本：约 100–300 条恢复尝试
```

数据有效规模按独立 episode/branch 计算，不能把连续帧数当独立样本数。

### 6.2 必须包含的数据

#### 正常轨迹

- 原始 OpenVLA 成功；
- 原始 OpenVLA 失败；
- 成功抓取和放置全过程；
- 不触发 recovery 的 hard normal states；
- handoff 后 OpenVLA 能继续成功的状态分布。

#### 成功恢复

- lost grasp 后重新抓取；
- 不同掉落位置；
- 不同 EEF 高度；
- 横向和纵向偏移；
- 不同失败检测延迟；
- visual verification 后不同 handoff 高度。

#### Hard negatives

- 短暂带起后再次滑落；
- 未抓住但 EEF 上升；
- 提前 `CLOSE`；
- 长期卡在 `DESCEND`；
- 错误进入 `VERIFY`；
- visual verifier 短暂高概率但未持续抓稳；
- recovery 后再次掉落；
- nominal policy 没有成功抓起，导致注入条件不成立。

#### 多轮恢复

- 第一次恢复失败；
- 重新观测；
- 第二次定位和候选选择；
- 最终成功或安全终止。

### 6.3 Teacher 采集

复用 `run_single_scene_recovery.py` 和 simulator oracle：

- 现有模块输出作为 baseline prediction；
- oracle state machine 生成正确 phase/action；
- simulator state 生成标签；
- 在线模型输入只保存可观测量；
- teacher privileged state 单独命名和隔离。

### 6.4 质量审计

每条 episode 自动检查：

- RGB/depth/mask 对齐；
- target mask 是否可见；
- visual xyz 是否 finite；
- action 是否有界；
- gripper convention 是否一致；
- failure 是否在视觉上真实发生；
- recovery expert 是否实际改善 target-eef geometry；
- `recovery_completed` 与最终 task success 不混淆；
- 视频帧数与 JSONL 对齐。

### 6.5 Gate

- 至少 20–30 条有效且独立的完整 task9 序列；
- normal、successful recovery、hard negative 均有覆盖；
- 至少一部分数据包含多轮恢复或恢复失败；
- privileged leakage audit 通过；
- 数据统计报告按 episode 而非 frame 输出。

---

## 7. Phase 3：持续目标 Slot 与正常动作读取

### 7.1 目标

正常操作期间保持持续对象表示，并让动作模型通过显式注意力读取 Slot，同时尽量不
破坏 OpenVLA 原行为。

### 7.2 感知前端

第一版使用：

```text
Stage-B target/reference/goal proposals
+ visual patch features
+ mask/depth geometry
→ target-conditioned object slots
```

需要至少表示：

```text
target slot
goal slot
robot/eef slot
reference slot(s)
distractor slot(s)
```

不再依赖“固定 slot 0 一定是目标”。角色由语言、关系和 proposal matching 决定。

### 7.3 持续状态

在线维护：

```text
previous slots
+ current observed slots
+ previous executed action
+ proprio
→ SlotSSM hidden state
```

必须处理：

- 目标遮挡；
- 相似物体；
- 目标移动；
- mask 暂时丢失；
- tracker identity uncertainty；
- episode reset 时 hidden state 清空。

### 7.4 OpenVLA 显式 Slot cross-attention adapter

推荐结构：

```text
OpenVLA action hidden/query
        │
        ▼
CrossAttention(
    Q=action hidden,
    K=persistent slots,
    V=persistent slots
)
        │
        ▼
small or zero-initialized gate
        │
        ▼
residual fusion → action head
```

第一版冻结 OpenVLA 主干，训练：

- Slot encoder/SlotSSM；
- Slot projector；
- gated cross-attention；
- continuous action/residual head。

如冻结主干无法利用 Slot，再尝试最后若干层 adapter 或 LoRA，不先全量微调 7B。

### 7.5 训练目标

- normal action imitation；
- target/goal/robot role supervision；
- temporal identity；
- mask/geometry辅助监督；
- action-to-slot attention stability；
- 保持 nominal action 的 residual regularization；
- success trajectory non-interference loss。

### 7.6 必须做的消融

```text
OpenVLA
OpenVLA + current-frame slots
OpenVLA + persistent slots
OpenVLA + persistent slots + explicit cross-attention
zero slots
shuffled slots
shuffled language
target mask occlusion
```

### 7.7 Gate

- target identity/geometry 在 held-out episode 上稳定；
- persistent model 显著优于单帧模型；
- action 输出对正确 Slot 敏感，对 shuffled Slot 有可解释退化；
- 正常成功轨迹破坏率处于预设容忍范围；
- online action 有界、平滑且不出现旧 Stage-3 均值策略塌缩。

---

## 8. Phase 4：共享 Recovery SlotSSM 辅助状态模型

### 8.1 目标

用一个共享时序 backbone 替换相互独立的 phase predictor 和 verifier 表示，但暂不
急于让它独立控制完整恢复动作。

### 8.2 输入

```text
persistent target/goal/robot/reference slots
visual target-eef geometry
proprio
previous executed action
OpenVLA nominal action
language/task embedding
observation validity and uncertainty
```

### 8.3 输出 heads

```text
current failure probability
failure type
recovery required
phase/mode
grasped/attached probability
slip probability
verification probability
handoff-ready probability
state/OOD uncertainty
```

### 8.4 当前失败的时间语义

Post-transition head 使用真实执行结果：

```text
previous state
+ previous executed action
+ current observed state
→ current/post-transition failure
```

不要把轨迹最终失败标签复制给所有帧。优先使用 simulator teacher 标注 failure onset
和事件窗口；轨迹级标签只作为弱监督补充。

### 8.5 训练与评测

- teacher-forced 分类；
- class-balanced metrics；
- 按 episode 报告 precision/recall/F1/AUPRC；
- failure onset 检测延迟；
- normal false trigger rate；
- autoregressive phase rollout；
- illegal transition rate；
- hard-negative false verification；
- uncertainty calibration。

### 8.6 Gate

- phase 自回归顺序和延迟通过；
- hard-negative verification 不提前确认；
- held-out failure episode 检测有足够 recall；
- normal episode 误触发受控；
- episode13 类 phase distribution shift 得到改善；
- 不读取 privileged online input。

---

## 9. Phase 5：学习完整恢复动作与自主 handoff

### 9.1 目标

统一模型输出完整 7-D recovery residual/action，依次删除：

1. 独立 geometry residual controller；
2. 手写 `VISUAL SETTLE`；
3. phase-based gripper sign override；
4. privileged 或规则式 recovery exit。

### 9.2 动作形式

第一版保持：

\[
a_t^{exec}
=
a_t^{nominal}
+
\Delta a_t^{recovery}
\]

并加入：

- per-dimension residual clip；
- jerk/smoothness penalty；
- phase-independent gripper supervision；
- uncertainty 下的 hold/observe action；
- normal mode residual 接近零的约束。

### 9.3 Handoff 学习

数据中加入明确状态：

```text
VERIFY_PENDING
VERIFY_CONFIRMED
HANDOFF_ALIGN
HANDOFF_READY
NOMINAL
```

模型学习从视觉确认高度回到 nominal policy 熟悉的状态分布，而不是用手写 EEF settle。

### 9.4 Gripper 学习

必须收集：

- 应该 close 的正样本；
- 应该保持 close 的样本；
- 不应提前 open 的 hard negative；
- handoff 后正常 OpenVLA gripper 行为；
- 滑落后的再次 close；
- 错误符号造成的失败样本。

先进行 gripper sign sequential audit，再删除 override。

### 9.5 DAgger

每轮：

```text
统一模型在线 rollout
→ 收集 OOD/失败状态
→ oracle teacher 重标 phase/action/outcome
→ 合并数据
→ 重训或继续训练
→ 新 held-out episode 评测
```

每轮必须保留模型版本和数据版本，避免测试集无意回流。

### 9.6 Gate

- 完整 action sequential direction audit 通过；
- gripper sign false action 低于安全阈值；
- 无手写 settle 仍能完成 handoff；
- 无 phase-based gripper override 仍能恢复；
- task9 多个未见 episode 在线成功；
- 原成功 episode 不被明显破坏。

---

## 10. Phase 6：未来状态预测与候选风险选择

### 10.1 目标

同时拥有：

```text
当前失败评估：判断已经发生什么
未来候选评估：判断执行某个候选后可能发生什么
```

二者共享 SlotSSM representation、action encoder 和 failure concepts，但不强行使用
完全相同的标量 head。

### 10.2 Action-conditioned future model

\[
\widehat M_{t+1}^{(k)}
=
F_\theta(M_t,a_t^{(k)},q_t)
\]

输出不应只预测 raw Slot embedding，还应预测可审计 outcome：

```text
target motion
eef-target geometry
grasp/attach probability
slip probability
phase progress
placement relation
observation uncertainty
```

### 10.3 数据

使用现有 MuJoCo state branching 扩展 matched counterfactual 数据：

```text
same state
├── nominal action
├── recovery policy samples
├── bounded xyz/rotation/gripper perturbations
├── hold/observe
└── teacher action
```

每个 branch 执行短 horizon，并保存真实 outcome。

### 10.4 必须通过的 action sensitivity

- shuffle candidate action 后 future prediction 明显退化；
- 同状态不同动作产生可区分 predicted outcomes；
- candidate ranking 优于随机和 nominal-only；
- world model 不能只靠当前状态猜平均后果。

旧 action-conditioned predictor 在 shuffle action 后几乎不退化，因此不能直接复用为
合格的 future model。

### 10.5 候选生成

OpenVLA 阶段可组合：

```text
nominal OpenVLA action
unified recovery mean action
recovery distribution samples
bounded spatial perturbations
hold/observe candidate
```

π0.5 阶段可使用不同 flow noise seed 生成更多自然 action chunks。

### 10.6 候选评分

\[
J_k =
\lambda_f p_{\mathrm{future-fail}}^{(k)}
-\lambda_p \widehat{\mathrm{progress}}^{(k)}
+\lambda_u U^{(k)}
+\lambda_s C_{\mathrm{safety}}^{(k)}
+\lambda_r C_{\mathrm{repeat}}^{(k)}
\]

只执行最优候选的前 1–2 步，随后重新观测并用 post-transition head 更新。

### 10.7 Gate

- matched candidate held-out ranking 显著优于随机；
- risk calibration 可用；
- candidate selection 在线优于单一 recovery mean action；
- 短闭环不会产生无界动作；
- future model 高不确定时选择 hold/observe 或安全回退；
- 当前失败 head 和 future-risk head 的时间语义在代码和日志中明确区分。

---

## 11. Phase 7：task9 多失败类型闭环

### 11.1 失败类型

至少覆盖：

```text
missed/empty grasp
off-center grasp
lost grasp/slip
wrong target
target pushed away
wrong placement
premature phase transition
false verification
```

### 11.2 多轮控制

统一状态机不应写死为单次线性恢复。模型应支持：

```text
detect
→ plan/select
→ execute
→ verify
→ success and handoff
或
→ verification failure
→ update memory
→ replan/select again
```

需要设置安全预算：

- 最大 recovery steps；
- 最大 recovery attempts；
- OOD/uncertainty 阈值；
- 无进展终止；
- 环境边界和动作安全检查。

### 11.3 Gate

- 每种失败都有独立 held-out episode；
- 报告 per-type detection 和 recovery 指标；
- 能完成至少一次二次规划；
- 不用 privileged online state；
- task9 最终成功率相对 nominal policy 和模块化 teacher 有可信提升。

---

## 12. Phase 8：LIBERO 多任务扩展与严格 benchmark

### 12.1 扩展顺序

1. task9 多初始状态；
2. 相同对象类别、不同关系的 Spatial tasks；
3. 全部 LIBERO Spatial；
4. LIBERO Object / Goal；
5. 条件允许时 LIBERO-10/90 或 LIBERO-Mem。

### 12.2 感知泛化

- target/reference/goal role schema；
- relation-conditioned proposal selection；
- 相同图像不同语言目标的 counterfactual query；
- held-out relation-reference-layout 组合；
- 新对象类别和新目标容器；
- proposal collapse 和 target/distractor collision regression。

### 12.3 数据量

扩展阶段不只按总帧数规划，应按：

```text
task 数
独立 initial state 数
failure type 数
成功/失败 recovery attempt 数
matched candidate group 数
```

逐步扩展，不在 task9 gate 未通过时盲目采集全 benchmark 大数据。

### 12.4 最终对比

```text
OpenVLA nominal
OpenVLA + current-frame Slot
OpenVLA + persistent Slot cross-attention
模块化 recovery teacher
Unified Recovery SlotSSM without candidate selection
Unified Recovery SlotSSM with candidate selection
```

### 12.5 指标

#### 正常操作

- task success rate；
- 原成功轨迹破坏率；
- action deviation；
- 推理延迟和显存；
- Slot/action attention 可解释性。

#### 失败检测

- precision；
- recall；
- F1；
- AUPRC；
- false trigger rate；
- detection delay；
- per-failure-type metrics。

#### 恢复

- conditional recovery success；
- final task success；
- recovery attempts；
- additional steps；
- repeated-failure rate；
- second-replan success；
- unsafe/OOD termination。

#### 候选选择

- pairwise/group ranking accuracy；
- regret relative to oracle branch；
- future outcome prediction error；
- calibration；
- online gain relative to single action。

### 12.6 统计协议

- 至少多个 random seeds/initial states；
- 报告 episode-level confidence interval；
- 所有阈值只在 validation 上选择；
- final test 只运行冻结配置；
- 失败视频与成功视频都保留；
- 禁止只挑选成功案例。

### 12.7 Gate

- benchmark 上最终成功率提高；
- 正常成功轨迹破坏率可接受；
- 结果在独立 episode 和多个 task 上成立；
- 候选选择有独立增益；
- 主要贡献均有消融支持。

---

## 13. Phase 9：π0.5 迁移与底座消融

此阶段在 OpenVLA 版本方法闭环后进行，不阻塞前面工作。

### 13.1 环境隔离

π0.5/openpi 与当前 OpenVLA 依赖不同，应使用独立环境，不直接破坏
`.conda/envs/openvla-a`。

### 13.2 第一阶段复现

- 下载官方 `pi05_libero`；
- 跑通纯 π0.5 LIBERO baseline；
- 核对 observation、action、normalization 和 camera convention；
- 固化成功率、速度和显存。

### 13.3 统一 nominal policy 接口

```python
encode_observation(observation, language)
sample_actions(policy_context, num_candidates)
extract_action_features(policy_context, actions)
execute_action_prefix(action_chunk, length)
```

OpenVLA 和 π0.5 各自实现 adapter，Failure-Aware 模块不依赖底座内部类。

### 13.4 Slot 注入

优先尝试：

- Slot condition token；
- action expert 的 gated cross-attention；
- conditioning adapter；
- 冻结主干 + 小模块训练。

不要假设 π0.5 自动理解我们新增的 Slot。

### 13.5 候选生成

使用不同 flow noise seed 生成候选 action chunks，复用同一个 future model 和
candidate-risk evaluation protocol。

### 13.6 最终判断

比较：

```text
OpenVLA + Failure-Aware SlotSSM
π0.5 + Failure-Aware SlotSSM
```

若 π0.5 显著改善正常动作或候选多样性，纳入最终主要结果；否则作为迁移可行性或
附录实验，不延迟核心项目交付。

---

## 14. Phase 10：最终代码、论文和交付

### 14.1 代码

- 清晰的 `recovery/` package；
- policy adapter；
- teacher/data collector；
- dataset conversion；
- trainer；
- offline audit；
- online rollout；
- benchmark runner；
- regression tests；
- 配置文件和环境说明。

### 14.2 可复现产物

- 数据 manifest；
- split manifest；
- checkpoint registry；
- config snapshots；
- 训练曲线；
- 离线指标；
- 在线逐步日志；
- 成功/失败视频；
- checkpoint hashes；
- 最终运行命令。

### 14.3 论文/报告

至少包含：

1. 问题定义；
2. 持续对象 Slot 表示；
3. 正常动作 Slot cross-attention；
4. shared Recovery SlotSSM；
5. post-transition failure assessment；
6. action-conditioned future model；
7. candidate risk selection；
8. 数据采集与 privileged teacher protocol；
9. 实验和消融；
10. 失败案例、局限和安全边界。

### 14.4 最终贡献表述

如果全部 gate 通过，可将贡献概括为：

1. 正常操作和失败恢复共享的持续目标对象表示；
2. 区分已发生失败与未执行候选风险、但共享时序表示和动力学模型的统一框架；
3. 基于预测对象后果的短视野候选恢复选择；
4. 使用 simulator privileged teacher、但在线仅依赖视觉/机器人可观测量的数据协议；
5. 在 LIBERO 上提高失败恢复与最终成功率，同时控制 nominal 成功轨迹破坏。

---

## 15. 必须持续执行的工程原则

1. 不把 smoke test 写成效果结论；
2. 不把 frame 数写成独立数据量；
3. 不把错误 recovery rollout 当 expert action；
4. 不把 privileged teacher field 放进在线输入；
5. 不在已经用于开发的 episode 上声称 final generalization；
6. 不用离线 MAE 掩盖 sequential direction/gripper 错误；
7. 不用 teacher-forced accuracy 替代自回归在线评测；
8. 不把当前 observed deviation 直接当作所有未来候选的风险；
9. future model 必须通过 shuffled-action 和 matched-counterfactual sensitivity；
10. 所有在线模型上线前必须通过 action bound、direction、gripper 和 OOD safety gate；
11. 每个新增模块必须有不使用该模块的消融；
12. 旧 teacher 主入口保留，不做破坏性重写；
13. 不继续在旧主入口中堆独立 phase、规则和 checkpoint；
14. 不清理或重置当前用户工作树；
15. 共享服务器启动前检查 GPU，不干扰其他用户进程。

---

## 16. 近期立即执行清单

按当前状态，下一轮工作从这里开始：

1. ~~建立 `recovery/` 目录骨架；~~（2026-07-28 已完成首版）
2. ~~实现严格的 sequence schema 和 online-input allowlist；~~（已完成并通过测试）
3. ~~转换首批已有 expert 与 episode10/11/13 trace；~~（已完成；媒体边界已显式记录）
4. ~~生成数据统计、split 和 privileged leakage audit；~~（首版审计已完成）
5. 将正常、成功恢复、滑落 hard negative、phase-stuck 和 handoff 数据构造成连续窗口；
6. 补采至至少 20–30 条 task9 完整序列；
7. 实现最小 shared Recovery SlotSSM；
8. 先训练 failure/phase/grasp/verification/handoff heads；
9. 完成 teacher-forced 与 autoregressive offline audit；
10. 再接完整 7-D residual action head；
11. 通过 sequential safety gate 后进行新 episode 在线 rollout；
12. 根据在线 OOD 状态启动统一模型 DAgger；
13. 并行设计 OpenVLA gated Slot cross-attention，但不阻塞 recovery data/model 主线；
14. 在 task9 统一模型通过后，再实现 candidate future model；
15. 最后扩展多失败类型、多任务和 π0.5。

### 16.1 2026-07-28 首轮实施结果

新增代码：

```text
recovery/data/schema.py
recovery/data/dataset.py
recovery/data/conversion.py
recovery/data/sequence_builder.py
recovery/data/validation.py
recovery/tests/test_data_contract.py
scripts/convert_recovery_sequences.py
scripts/convert_recovery_online_traces.py
scripts/audit_recovery_sequences.py
```

首批 expert 转换：

```text
/data2/lty/naive-e-slotssm-runs/recovery-unified-sequences-v1
```

- episode 0/1/2/3/4/8/9；
- episode1 自动选择较新的 v2 来源；
- 1,368 rows；
- 363 recovery expert action rows；
- 全部有对齐 RGB、depth 和 target mask；
- train 5 episodes、validation 1、development test 1；
- episode9 明确为 development test，不再作为 final test。

首批在线时序 trace：

```text
/data2/lty/naive-e-slotssm-runs/recovery-unified-online-traces-v1
```

- episode10 visual-verifier + settle 成功；
- episode11 learned-phase 成功；
- episode13 phase-stuck 失败硬负；
- 751 rows；
- 这些旧 trace 没有对齐逐帧 RGB/depth，因此只用于低维时序标签、状态覆盖和
  hard-negative 训练，不冒充完整视觉 episode。

合并审计：

```text
/data2/lty/naive-e-slotssm-runs/recovery-unified-audit-v1/report.json
```

- 10 episodes / 2,119 rows；
- 1,809 个连续 32-step windows；
- 1,489 个连续 64-step windows；
- 1,076 个含关键事件的 32-step windows；
- 1,163 个含关键事件的 64-step windows；
- 6 个最终成功 episode；
- missing files=0；
- schema errors=0；
- privileged online-input leakage=0（由 allowlist 和单元测试强制）。

下一项实际工作是将 sequence window 从索引结构推进为可训练 tensor dataset，并为
旧 expert 数据离线补算 visual target xyz、failure/grasp/verification/handoff
teacher labels；同时开始补采完整媒体的新 recovery episode。

### 16.2 2026-07-28 第二轮实施结果

已完成可训练 tensor sequence 层：

```text
recovery/data/tensor_dataset.py
scripts/audit_recovery_tensor_dataset.py
```

- 固定 proprio layout 为25维；
- nominal/previous/executed/expert action 均为7维；
- visual xyz、target-eef、centroid、mask measurement 均带独立 validity mask；
- 第一版 policy-observable temporal input 共76维；
- collator 输出统一 `[B,T,D]`；
- 不把缺失 xyz 的零填充值当真实坐标；
- 32-step、stride=8 的实际审计得到 train 101、validation 27、
  development-test 102个 windows；
- 所有训练张量 finite。

新增可观测 visual xyz enrichment：

```text
recovery/data/visual_geometry.py
scripts/enrich_recovery_visual_xyz.py
/data2/lty/naive-e-slotssm-runs/recovery-unified-sequences-v2-visual-xyz
```

直接把所有有限 xyz 标为有效会在 proposal/tracker 丢失目标后产生约40–50cm错误。
因此增加只使用可观测量的 gate：

```text
64×64 target mask visible_pixels >= 32
且相对上一可靠 xyz jump <= 0.12m
```

teacher pose 只用于离线审计，不进入 gate。结果：

- 1,368 rows 全部成功计算 xyz；
- 951 rows 通过可观测 gate；
- gated mean error=2.19cm；
- gated P90=3.10cm；
- gated max=4.35cm；
- 417 rows 被拒绝，其 teacher-audit mean error=32.19cm；
- 当前数据中所有 >10cm 错误绑定均被拒绝。

已建立首个共享时序多头模型：

```text
recovery/models/recovery_slotssm.py
scripts/train_recovery_slotssm_smoke.py
```

共享 backbone 输出 phase、current failure、recovery active、verification、handoff
和7-D residual action heads。GRU reference 后端完成50-step真实数据 smoke；Mamba2
后端完成2-step CUDA前向、反向、validation和checkpoint。Mamba2 小维度配置必须用
`headdim=16`，否则 causal-conv 的 channel-last projection stride 不是8的倍数。

产物：

```text
/data2/lty/naive-e-slotssm-runs/recovery-slotssm-smoke-v1-gru
/data2/lty/naive-e-slotssm-runs/recovery-slotssm-smoke-v1-mamba2
/data2/lty/naive-e-slotssm-runs/recovery-tensor-audit-v1/report.json
```

这些 checkpoint 只证明统一数据、多任务 loss、共享 temporal backbone 和保存链路
成立，不允许在线控制。下一步必须补全可靠 event labels、解决 class imbalance，
并建立 autoregressive sequential audit，再做正式 Mamba2 训练。

### 16.3 2026-07-28 第三轮实施结果

已新增 simulator-assisted temporal teacher：

```text
recovery/teachers/recovery_labeler.py
scripts/enrich_recovery_teacher_labels.py
/data2/lty/naive-e-slotssm-runs/recovery-unified-sequences-v3-teacher-labels
```

标签包括 failure-active、lifted、attached/grasped、slip event、verified 和
handoff-ready。privileged target pose 只生成 `labels`，不修改 `online_inputs`。
由于旧数据没有 contact state，attached 是“target 与 EEF 连续共运动、距离受限”的
保守代理，并带显式 validity/confidence。

7个 expert episodes / 1,368 rows 的标签统计：

```text
failure-active 369
lifted 362
attached 682
slip event 39
verified 273
handoff-ready 388
invalid privileged state 0
```

共享模型新增 grasped/lifted/slip heads，并对 phase 与二分类事件使用类别平衡 loss。
新增 causal whole-episode evaluator：

```text
scripts/evaluate_recovery_slotssm_sequential.py
```

它报告 phase accuracy、非法阶段跳转、failure precision/recall/F1、真实 onset 后检测
延迟、onset 前误报率，以及 grasp/lift/slip/verification/handoff 指标。

GRU balanced 200-step preflight：

```text
/data2/lty/naive-e-slotssm-runs/recovery-slotssm-preflight-v3-gru-balanced
/data2/lty/naive-e-slotssm-runs/recovery-slotssm-sequential-audit-v3-gru-balanced
```

- validation+development phase accuracy=76.1%；
- current-failure F1=0.684；
- onset 后检测延迟=0步；
- onset 前误报率分别为7.3%和17.8%；
- slip precision=0.344、F1=0.478；
- 能预测 CLOSE/VERIFY，但仍有非法阶段切换；
- `online_smoke_allowed=false`。

Mamba2 25-step preflight：

```text
/data2/lty/naive-e-slotssm-runs/recovery-slotssm-preflight-v2-mamba2-labels
/data2/lty/naive-e-slotssm-runs/recovery-slotssm-sequential-audit-v2-mamba2
```

训练链路稳定，但训练不足导致 phase 主要塌缩到 NOMINAL/RESUMED，phase accuracy
43.97%、current-failure F1=0.451、slip F1=0.296，因此同样禁止上线。

上述1、2、5已完成离线实现与消融，但由于 validation 只有单个 episode，门控解码
没有跨 episode 泛化，仍然禁止在线 rollout。当前下一步是：

1. 更新 collector，同时记录对齐媒体、contact/attached、nominal/executed action；
2. 收集至少20–30个独立 episode，增加正常无失败轨迹并覆盖多个 failure type；
3. 将 episode10/11/13 重新采集或补齐 contact/媒体后再纳入正式数据；
4. 按 episode/seed/failure type 重划 train/validation/development-test；
5. 在多 validation episodes 上重新训练、校准阈值并估计方差；
6. 达到 sequential gate 后才考虑在线 smoke。

2026-07-28 collector 更新：上述第1项代码已完成。expert collection 现在保存对齐
RGB/depth/64×64 mask、nominal/executed action、MuJoCo target-gripper contact pairs
和保守 bilateral-contact attached truth，并支持无故障 nominal negative episode。
正式批量采集模板为 `scripts/collect_recovery_dataset_v3.sh`。下一检查点是先采集1个
failure与1个normal preflight，完整审计后再扩到20–30个 episode。

该检查点已通过：episode0 failure 和 episode10 nominal 共275 transitions，最终
schema审计中 RGB/depth/mask/visual xyz/contact/attached coverage 均为275/275，
零缺失文件和零错误。observable xyz gate接受226行，接受行平均误差2.11cm。
下一工作项正式转为多seed、多failure-type的20–30 episode扩展采集。

扩展采集也已完成：24 episodes / 4,791 transitions，分层split为10/7/7。
Mamba2 500-step离线重训达到raw phase 90.72%、failure-gated development-test phase
85.32%、校准后current-failure F1=0.843、slip F1=0.710。当前主阻塞从“数据太少”
转为“early failure onset标签覆盖不足、尚无冻结final-test、learned recovery action
尚未通过shadow/online安全验证”。

随后确认上述current-failure结果受early-failure漏标影响。因果teacher现将错误动作
执行后的下一帧作为onset，并在物理恢复完成前保持failure-active；修正后开发集
current-failure F1=0.640、failure-gated phase=46.37%，这是更困难但可信的基线。
另新采6条并冻结为final-test，合并数据为30 episodes / 5,895 rows；final-test尚未
用于任何模型性能评估。当前下一研究点是onset/duration分离与action-conditioned
candidate scorer，而不是继续用旧漏标追求高分。

onset/duration分离第一版已完成：独立onset head在development达到F1=0.909且0 FP，
onset-gated phase提升到63.37%，resolution/handoff F1=0.951；persistence F1仍为
0.560。下一实现目标因此明确为observable persistence evidence和
action-conditioned candidate effect/risk scorer。final-test继续封存。

2026-07-28 Phase 6最小训练闭环已实现：matched candidate schema、可变候选数
dataset/collator、有界候选生成、action-conditioned effect/value scorer、物理/事件/
价值/排序联合loss和训练入口均已加入，17项tests通过。它当前只证明接口和梯度链路，
尚无recovery-state branch数据或排序结果。下一里程碑是复用独立MuJoCo fork env，
在failure/recovery/verify状态采集4-step matched outcomes，并依次通过restore
determinism、action sensitivity、held-out ranking和shadow evaluation；final-test
保持封存。

Phase 6第一份真实MuJoCo recovery candidate smoke也已完成：独立fork环境从2个
RETURN TO BOWL状态采得23/23有效4-step branches；同状态不同动作的teacher value
和EEF位移均有可见跨度。主轨迹成功转换为170行schema-v2，真实候选batch可直接进入
76-D context scorer且联合loss反向传播梯度finite。下一步从“实现collector”转为
多episode、多phase配额采集和严格按episode的train/validation ranking评测。

候选v1随后得到1个有效train episode和1个独立validation episode，各12 groups，
共285个有效branches。500-step scorer在validation的best-candidate accuracy为
16.7%（uniform随机约8.3%），pairwise accuracy约67.0%，mean regret 0.01057，
优于nominal 0.01672和uniform random expected 0.03661；打乱candidate action使
validation loss从0.06556升至0.09908。该结果证明了初步action sensitivity和排序
收益，但episode数远不足，且validation未覆盖CLOSE/VERIFY，仍禁止在线控制。

只读shadow部署链随后跑通：81个恢复帧实时显示候选选择，日志确认0帧修改实际动作；
其中4个状态有真实fork outcome，shadow命中2/4，mean regret 0.01846，优于同状态
nominal的0.04518。已生成带`SHADOW only`字幕的MP4。该轨迹复用了scorer validation
episode，因此只证明部署与可视化链路，不作为新泛化证据，仍禁止在线接管。

Level-1统一模型随后实现：冻结v13 Recovery Mamba2，以其32-step、128-D共享temporal
state同时服务原current heads和新的candidate future head。最佳validation regret
0.01270，优于nominal 0.01672和随机0.03661；action shuffle使loss增加约94.3%。
单一checkpoint已包含原48个Recovery tensors和20个candidate tensors，旧参数逐tensor
完全一致。统一版regret仍略差于独立scorer 0.01057，且motion loss更高，因此下一步
先扩数据和测试observable-geometry skip，不直接解冻Mamba2。

11-D observable geometry skip消融已完成，并补齐920行mask+depth visual xyz。
no-skip原版regret=0.01270；visual+skip=0.01932；visual/no-skip=0.01684。skip虽将对应
最佳点motion loss从约0.00304降至0.00283，但排序退化，故暂不采用，模型选择保持
Level-1 v1。此结果也表明不能继续针对单一validation episode调结构，下一优先级恢复
为扩大有效candidate episodes和failure-trigger coverage。

failure-trigger coverage根因随后修复：visual-lift在15px已触发注入却要求20px才承认
lift的矛盾被消除。init9/init0实际检测step与离线回放一致，均从0增至12 candidate
groups、144/144有效branches。训练扩至3个有效episodes/36 groups后，validation
regret反而为0.01801，差于nominal 0.01672和旧统一v1 0.01270；新增数据全部集中在
DESCEND并导致候选选择塌缩。下一采集必须按phase quota平衡，不再只按固定时间间隔。

---

## 17. 项目结束检查表

只有以下项目全部完成，才能宣布整个工作完成：

- [ ] 持续目标 Slot 在严格审计中通过；
- [ ] 正常动作模型真实读取 Slot；
- [ ] OpenVLA 正常轨迹不被明显破坏；
- [ ] shared Recovery SlotSSM 单 checkpoint 运行；
- [ ] 当前失败检测不依赖 privileged input；
- [ ] phase、grasp、verification 和 handoff 自主完成；
- [ ] 手写 `VISUAL SETTLE` 已删除；
- [ ] gripper sign override 已删除；
- [ ] 完整 7-D recovery action 由统一模型产生；
- [ ] action-conditioned future model 对 action 真正敏感；
- [ ] 多候选风险排序离线和在线均有增益；
- [ ] 支持 verification failure 后再次规划；
- [ ] 覆盖多种 failure type；
- [ ] 跨 episode 和多 task 严格测试通过；
- [ ] 最终成功率提高且 nominal 破坏率受控；
- [ ] 所有主要模块有消融；
- [ ] 数据、代码、配置、checkpoint 和视频可复现；
- [ ] 最终报告/论文完整记录方法、结果和局限；
- [ ] π0.5 迁移已完成，或有明确、经实验支持的不采用理由。
