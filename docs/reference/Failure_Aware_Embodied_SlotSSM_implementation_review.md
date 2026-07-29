# Failure-Aware Embodied-SlotSSM 实现评审与阶段规划

> 本文是对 `Failure_Aware_Embodied_SlotSSM.md` 的实现可行性评审。内容结合当前 `naive-e-slotssm` 仓库的实际完成状态，区分可以直接复用的模块、需要新增的模块，以及尚不成立或需要进一步验证的假设。

## 1. 总体判断

整体想法是可实现的，而且核心方向有价值：利用“动作执行后对象状态是否符合预期”作为失败信号，比单纯使用 VLA hidden 做二分类更有解释性。

但当前设计有几个关键问题必须调整，否则 Failure Scorer 可能离线指标不错，却不能真正检测失败或选择恢复动作。最大的问题是：文档把“事后失败检测”和“执行前候选风险预测”混成了同一个问题。

建议将第一版核心贡献收敛为：

> 基于动作条件化对象状态预测偏差的事后失败检测，以及由对象偏差条件化的短视野闭环恢复。

候选动作风险选择可以作为增强模块，但不应成为第一版成功与否的前提。

---

## 2. 当前设计中最需要修改的地方

### 2.1 `p_fail` 的时间语义不明确

当前文档同时包含两类风险：

```text
事后检测：
执行 a_t 后观察到 S_{t+1}，利用预测偏差判断刚才是否失败。

执行前预测：
候选动作 a_t^(k) 尚未执行，就给候选动作计算失败风险。
```

两者的信息条件不同。

事后检测可以使用：

\[
D_{t+1}
=
\operatorname{dist}
\left(
\widehat S_{t+1}(S_t,a_t),
S_{t+1}
\right)
\]

因为真实的 \(S_{t+1}\) 已经出现。

但候选动作还没有执行，因此没有真实的 \(S_{t+1}\)，无法计算候选对应的真实 deviation。当前方案二实际使用的是“上一动作造成的 deviation”来评价“下一候选动作”，这不是严格的候选风险。

建议明确拆成：

```text
Post-transition Failure Detector
输入：(状态_t, 已执行动作_t, 实际状态_t+1, deviation_t+1)
输出：刚才的执行是否异常

Pre-action Candidate Risk Scorer
输入：(当前状态_t, 候选动作_t)
输出：候选动作未来失败的风险
```

二者可以共享编码器，但第一版最好保留两个输出接口：

\[
p_{\mathrm{detect},t+1}
\]

和：

\[
p_{\mathrm{risk},t}^{(k)}
\]

它们都可以是单标量，不需要失败类型分类。

### 2.2 轨迹级标签复制到所有时间步不够合理

当前定义：

\[
y_t=y_\tau
\]

意味着只要 episode 最终失败，从第一帧开始所有时间步都被标成失败。

这容易使模型学习：

- 哪个任务更困难；
- 哪种场景更容易失败；
- 哪些轨迹最后会超时；

而不是学习失败发生的时刻。

例如第 300 帧发生抓取失败，前 299 帧也被标为失败，模型会被迫从一开始就输出高风险。

建议：

1. 仿真注入扰动时记录准确的扰动时间和失败发生窗口；
2. 将失败标签定义成有限预测窗口：

   \[
   y_t=1
   \quad\text{if failure occurs in }[t,t+H_f]
   \]

3. 对只有 episode 标签的数据使用 Multiple Instance Learning，而不是把全部时间步直接标记为 1：

   \[
   p_{\mathrm{episode}}
   =
   \max_t p_{\mathrm{fail},t}
   \]

   也可以使用 softmax 或 log-sum-exp 聚合。

如果要尽快得到稳定结果，第一版应主要使用我们自己在 LIBERO 中生成、具有已知扰动时间的失败 rollout。

### 2.3 当前数据全是成功示范，不能直接训练 Failure Scorer

目前的 961 条 LIBERO-Mem 数据本质上是成功 demonstration，可以训练：

- Stage-1 Slot；
- Stage-2 SlotSSM；
- action-conditioned Slot Predictor；
- 成功状态下的正常对象动态模型。

但这些数据不能单独训练可靠的 Failure Scorer。

必须额外生成失败数据。工程上需要建立一个 derived failure rollout corpus，即使不将它称作新的数据集。

建议在仿真中自动注入：

- 抓取前目标物体小范围平移；
- gripper 动作延迟或反转；
- 执行动作加入位置或旋转噪声；
- 抓起后让对象滑落；
- 放置时移动目标容器；
- 截断一段 action chunk；
- 暂时冻结机械臂动作；
- 将对象移动到错误容器附近。

每条轨迹保存：

```text
扰动类型（仅用于评测）
扰动时间
是否最终失败
失败确认时间
是否恢复成功
```

模型本身仍只输出一个标量，不要求它预测扰动类型。

### 2.4 Slot Predictor 必须使用实际执行动作

预测器应该使用：

```text
executed_action
```

而不是 VLA 原计划生成的完整 action chunk。

因为 action chunk 可能：

- 只执行前两步；
- 被 Failure Detector 中断；
- 被仿真环境裁剪；
- 受安全控制器约束；
- 在恢复阶段被候选选择器替换。

正确的训练样本是：

\[
(S_t,q_t,a_t^{\mathrm{executed}},S_{t+\Delta})
\]

而不是：

\[
(S_t,q_t,A_t^{\mathrm{planned}},S_{t+1})
\]

第一版建议只预测单步或固定小间隔 \(\Delta\)，不要直接 flatten 很长的 action chunk。

### 2.5 预测 raw Slot embedding 容易受到 Slot 漂移影响

即使图像中的对象没有异常，Slot embedding 也可能因为图像增强、视角变化、遮挡、Slot 索引交换或 Slot Attention 随机性发生较大变化。

直接使用：

\[
\left\|
\widehat s_{t+1}^i-s_{t+1}^i
\right\|
\]

容易产生误报。

需要增加：

- objectness mask，只对有效对象计算 deviation；
- Slot matching，必要时使用 Hungarian matching；
- 对 embedding 做 LayerNorm 或单位归一化；
- 使用训练集对象级均值和方差归一化 deviation；
- 对 deviation 做时间 EMA；
- 同时比较 bbox、mask centroid 和 objectness 等可解释状态。

建议对象偏差不是单一 embedding distance，而是：

\[
d_i
=
w_e d_{\mathrm{embed}}
+w_b d_{\mathrm{bbox}}
+w_m d_{\mathrm{mask\ centroid}}
+w_o d_{\mathrm{objectness}}
\]

第一版至少加入 embedding、bbox centroid 和 objectness。

### 2.6 不能直接假设“上一帧 Slot 初始化”已经实现

设计文档写了：

\[
S_t
=
\operatorname{SlotAttention}(F_t,S_{t-1})
\]

但当前正在训练的 Stage-1 主要是：

- 连续帧输入；
- bbox、mask、objectness 监督；
- 对象 track 相关约束。

这不等于已经实现严格的 online recurrent Slot Attention。

当前代码还需要核实或改造：

- 推理时能否传入上一帧 Slot；
- episode reset 时能否清空 Slot；
- Slot index 是否稳定；
- 遮挡后能否恢复原对象 identity。

因此，这部分应该作为待实现接口，而不是当作已经具备的功能。

### 2.7 当前 OpenVLA 注入方式没有可直接读取的任务相关性 `α`

设计希望从交叉注意力中获取：

\[
\alpha_t^i
\]

但我们目前跑通的 Stage-3 是把 16 个 SlotSSM token 投影成 4096 维，然后直接插入 Llama input embeddings。它不是独立的 gated cross-attention，因此没有天然、干净的：

```text
action token → object Slot attention
```

权重可以直接读取。

可选实现：

1. 在 OpenVLA 中加入真正的 gated cross-attention adapter；
2. 使用 LLM self-attention 中 action token 对 Slot token 的注意力；
3. 训练独立 task relevance head：

   \[
   \alpha_i
   =
   \operatorname{softmax}
   \left(
   f(m_i,h_{\mathrm{language}},h_{\mathrm{action}})
   \right)
   \]

第一版建议第三种，改动小、解释明确，也不依赖 Flash/SDPA 是否返回 attention。

### 2.8 直接缩放 Slot embedding 可能造成分布漂移

当前恢复表示：

\[
M_t^{\mathrm{rec}}
=
\{\beta_i m_i\}
\]

可能把 Slot token 的模长改得很大或很小。经过 LayerNorm 后，这种权重效果还可能被部分抵消。

更稳的方式是：

```text
原 Slot token
+ 异常程度 embedding
+ 恢复模式 embedding
```

例如：

\[
m_i^{\mathrm{rec}}
=
m_i
+
\lambda
\operatorname{MLP}
\left(
[m_i,\alpha_i,d_i,e_{\mathrm{recovery}}]
\right)
\]

或者将 \(\beta_i\) 作为 cross-attention logit bias，而不是直接乘 token。

原文提到的门控残差版本比直接相乘更合理，建议将其作为正式方案。

### 2.9 短 action chunk 与当前实现不完全一致

我们当前真实 OpenVLA smoke 输出的是：

```text
(1, 7)
```

即单步连续动作，不是已经实现的 8～16 步 action chunk。

因此：

```text
H_normal = 8～16
H_recovery = 1～2
```

目前还不能直接使用。需要选择：

- 实现真正的 action chunk head；
- 正常模式连续执行单步动作，但降低 VLA 调用频率；
- 一次预测多步，恢复模式只执行首步。

如果目标是尽快完成，建议正式 Stage-3 直接训练：

```text
[B, H_action, 7]
```

例如 `H_action=8`，部署时：

```text
正常模式执行前 4～8 步
恢复模式只执行第 1 步
```

### 2.10 候选动作风险排序是当前方案风险最高的部分

Failure Scorer 主要通过已执行 rollout 的成功/失败标签训练。即使它能够判断已执行轨迹是否失败，也不代表它能够评价人为扰动生成的候选动作。

尤其是：

\[
A^{(k)}=A+\epsilon_k
\]

这些候选可能属于训练分布之外，其风险分数没有校准依据。

方案二至少需要以下之一：

1. 训练数据包含同一状态下多个候选动作及其结果；
2. 在仿真中短暂 rollout 候选；
3. 使用动作条件化 forward model 预测候选的未来 Slot，再评价未来状态风险；
4. 加入专门的 action-outcome ranking loss。

因此，方案二不应写成“无需额外训练”的确定性结论。它可以作为后续增强，但不应成为第一版的主要贡献。

---

## 3. Naive E-SlotSSM 中可以复用的部分

| Failure-Aware 模块 | 可以复用的现有工作 | 仍需补充 |
|---|---|---|
| 视觉编码 | OpenVLA vision backbone | 无需更换 |
| 对象 Slot | 当前 Stage-1 tokenizer、bbox/mask/objectness heads | 在线持续初始化、identity 评测 |
| Slot 时间建模 | Stage-2 `EmbodiedSlotSSM`、Mamba2、next-slot head | 正式训练和在线 cache |
| 动作条件 Slot 预测 | Stage-2 next-slot head | 加入 executed action 和 robot state |
| Relation Encoder | 原仓库相关类和论文规格 | 当前主线尚未真正接入验证 |
| OpenVLA Slot 注入 | 当前 Stage-3 projector 和 LLM token injection | 正式动作训练、action chunk |
| VLA hidden | 当前 Llama final hidden 可以提取 | 明确对应哪个 action step |
| Robot state | RLDS 已包含 state/joint state | 当前 Stage-3 wrapper 尚未注入 |
| Slot deviation | Stage-1 bbox/mask + Stage-2 predicted slot | matching、归一化、EMA |
| Failure Scorer | 无 | 需要新增 |
| 恢复状态机 | 无 | 需要新增 |
| 候选动作生成 | 无 | 需要新增并限制动作安全范围 |
| 无头 LIBERO rollout | baseline 入口可以借鉴 | 需完成在线 SlotSSM 和日志保存 |
| 数据转换 | 现有 LIBERO-Mem HDF5→RLDS 工具 | 扩展失败 rollout schema |

最有价值的复用路径是：

```text
Stage-1 Slot
    ↓
Stage-2 Mamba2 时间状态
    ├─ 原 next-slot prediction
    └─ 新 action-conditioned transition head
          ↓
       object deviation
          ↓
Failure Detector / Recovery Controller
```

不建议重新开发另一套 Slot 模型。

---

## 4. 建议重新定义完整系统的因果顺序

### 4.1 决策前

已知：

\[
I_t,S_t,M_t,q_t,g
\]

生成动作：

\[
A_t
\]

同时预测计划动作的对象结果：

\[
\widehat S_{t+\Delta}
=
F_{\mathrm{slot}}
(M_t,q_t,A_t)
\]

### 4.2 执行动作

环境真正执行：

\[
a_t^{\mathrm{exec}}
\]

必须记录实际执行动作，而不只是计划动作。

### 4.3 执行后检测

得到新观测：

\[
I_{t+\Delta},S_{t+\Delta}
\]

计算：

\[
D_{t+\Delta}
=
\operatorname{Deviation}
(\widehat S_{t+\Delta},S_{t+\Delta})
\]

然后计算：

\[
p_{\mathrm{detect},t+\Delta}
=
R_{\mathrm{post}}
(M_t,a_t^{\mathrm{exec}},M_{t+\Delta},D_{t+\Delta},q_t)
\]

### 4.4 恢复决策

若风险较高：

```text
中断尚未执行的 action chunk
更新真实观测和 SlotSSM
增加异常/恢复 token
生成一个短动作
执行一步
重新观测
```

这个顺序比原伪代码更符合因果关系。

---

## 5. 建议的六个实现 Stage

## Stage-0：基础闭环与数据协议

目标：先把正常策略在线执行的接口固定下来。

需要完成：

- 无头 LIBERO rollout；
- episode reset；
- RGB、语言和 robot state 获取；
- 动作归一化与反归一化；
- action chunk 定义；
- planned action 与 executed action 区分；
- 在线 Slot 和 SlotSSM state reset/update；
- rollout 日志格式；
- 成功判定和视频保存。

输出：

```text
正常模型能够连续执行完整 episode
每一步数据可以完整回放
```

这个 Stage 必须放在 Failure Scorer 之前，因为失败数据只能从闭环 rollout 中获得。

## Stage-1：Persistent Object State

对应并复用 Naive E-SlotSSM 的 Stage-1 和 Stage-2。

### Stage-1A：对象 Slot

复用当前正在进行的正式训练：

- vision backbone；
- Slot tokenizer；
- bbox；
- mask；
- objectness。

新增验证：

- 跨帧 identity；
- episode reset；
- objectness；
- bbox/mask centroid deviation；
- 遮挡恢复。

### Stage-1B：SlotSSM

正式训练 Mamba2：

- 输入连续 Slot；
- 预测下一 Slot；
- 维护在线对象状态；
- 支持 recurrent cache；
- 评估正常轨迹 next-slot prediction。

输出：

```text
每个时间步获得稳定的 S_t 和 M_t
```

## Stage-2：正常动作策略与 Action-Conditioned Slot Dynamics

这个 Stage 同时建立正常动作基线和动作条件化对象预测。

### Stage-2A：正常动作策略

复用现有 Stage-3：

- Slot token projector；
- OpenVLA LLM 注入；
- continuous action head。

需要新增：

- robot state encoder；
- action chunk head；
- 在线推理接口；
- Slot/no-Slot/SlotSSM 消融。

### Stage-2B：动作条件化预测器

实现：

\[
\widehat S_{t+\Delta}
=
F_{\mathrm{slot}}
(M_t,q_t,a_t^{\mathrm{exec}})
\]

建议结构：

```text
per-slot state
+ action embedding
+ robot-state embedding
→ FiLM/MLP residual transition head
→ predicted next Slot
```

第一版不必重写 Mamba2，可以在冻结的 Mamba2 输出上增加 action-conditioned residual：

\[
\widehat S_{t+1}
=
\widehat S_{t+1}^{\mathrm{Mamba}}
+
\Delta S_{t+1}^{\mathrm{action}}
\]

输出包括：

- predicted next Slot；
- embedding deviation；
- bbox/centroid deviation；
- objectness；
- deviation normalization statistics。

## Stage-3：失败数据生成与 Failure Detector

### Stage-3A：失败 rollout 生成

从正常策略生成：

- 自然失败；
- 自动扰动失败；
- 成功 rollout；
- 恢复前状态。

必须记录时间标签和实际执行动作。

建议至少获得：

```text
每个任务 50～100 条成功 rollout
每个任务 100～200 条扰动或失败 rollout
```

开发期可以先选择 2～3 个任务和 3 类失败。

### Stage-3B：事后 Failure Detector

输入消融建议：

```text
A. VLA hidden
B. VLA hidden + robot state
C. SlotSSM state
D. SlotSSM + deviation
E. VLA hidden + SlotSSM + deviation + robot state
```

Detector 建议使用时间窗口，而不是单帧：

\[
[x_{t-w+1},\ldots,x_t]
\]

可以对最近 4～8 步特征使用一层小型 GRU 或 SSM。

训练与评测指标：

- AUROC；
- AUPRC；
- F1；
- 正常误停止率；
- 检测延迟；
- 扰动后多少步触发；
- 跨任务泛化。

输出：

```text
校准后的 p_detect
tau_enter / tau_exit
```

## Stage-4：恢复方案一——Failure-Aware Closed-Loop Replanning

第一版恢复不依赖候选动作。

触发后：

1. 中断剩余 action chunk；
2. 更新真实观测和 SlotSSM；
3. 为每个 Slot 加入 deviation embedding；
4. 加入 recovery-mode token；
5. 只生成或执行一个短动作；
6. 重新观测；
7. 使用迟滞状态机退出恢复。

建议恢复表示：

\[
m_i^{\mathrm{rec}}
=
m_i
+
\operatorname{MLP}
([m_i,\alpha_i,d_i,e_{\mathrm{rec}}])
\]

而不是简单使用 \(\beta_i m_i\)。

核心对比：

```text
继续原 chunk
直接重新调用 VLA
直接重新调用 + 短执行
短执行 + recovery token
短执行 + recovery token + deviation Slot
```

这样才能证明 Slot deviation 真正贡献了恢复效果。

## Stage-5：候选动作风险选择

这是增强版，不应阻塞主线。

新增：

- 安全约束下的候选生成；
- candidate action encoder；
- pre-action risk scorer；
- 候选多样性约束；
- 重复动作惩罚；
- 可选短分支仿真或 learned outcome model。

更合理的候选评分方式是：

\[
\widehat S_{t+\Delta}^{(k)}
=
F_{\mathrm{slot}}
(M_t,q_t,A_t^{(k)})
\]

然后根据预测未来状态评分：

\[
r_k
=
R_{\mathrm{pre}}
(M_t,A_t^{(k)},\widehat S_{t+\Delta}^{(k)},g)
\]

而不是使用当前已经发生的 \(D_t\) 直接评价所有候选。

如果没有 candidate outcome 数据，方案二应明确标为探索性模块。

## Stage-6：完整实验与论文消融

至少比较：

| 版本 | Slot | 时间状态 | Deviation | Detector | Recovery |
|---|---:|---:|---:|---:|---:|
| OpenVLA | 否 | 否 | 否 | 否 | 否 |
| Naive E-SlotSSM | 是 | 是 | 否 | 否 | 否 |
| Hidden Failure Baseline | 否/可选 | 否 | 否 | 是 | 直接重试 |
| Failure-Aware SlotSSM | 是 | 是 | 是 | 是 | 方案一 |
| Candidate Risk | 是 | 是 | 是 | 是 | 方案二 |

需要分别报告：

- 正常成功率；
- 失败检测性能；
- 条件恢复成功率；
- 最终任务成功率；
- 误触发代价；
- 平均额外控制步；
- 平均 VLA 调用次数；
- 推理延迟。

---

## 6. 推荐的实际实施顺序

结合当前进度，推荐路线：

```text
当前全量 Stage-1 Slot 训练
        ↓
Slot 可视化与 identity 验证
        ↓
正式 Stage-2 Mamba2
        ↓
无头 LIBERO 在线闭环
        ↓
正式 Stage-3 action chunk
        ↓
Action-conditioned Slot Predictor
        ↓
自动生成带时间标签的失败 rollout
        ↓
Post-transition Failure Detector
        ↓
恢复方案一
        ↓
候选风险方案二
```

关键依赖关系：

```text
没有正常闭环策略
→ 无法采集可信失败 rollout

没有动作条件化预测器
→ deviation 只能表示视觉变化，不能表示动作后果异常

没有准确失败时间
→ Failure Scorer 很容易只学习任务难度

没有候选 outcome 监督
→ Candidate Scorer 不一定具备反事实排序能力
```

---

## 7. 工作量判断

原文中的“两周计划”适合完成 smoke prototype，但不太可能得到可信、完整的实验结果。

比较现实的时间估计：

| Stage | 工程与训练时间 |
|---|---:|
| Stage-0 无头闭环 | 2～5 天 |
| Stage-1 Slot/SlotSSM | 当前训练 + 2～4 天 |
| Stage-2 动作与 predictor | 4～8 天 |
| Stage-3 失败数据与 detector | 5～10 天 |
| Stage-4 恢复方案一 | 3～7 天 |
| Stage-5 候选风险 | 5～10 天 |
| Stage-6 消融评测 | 5～14 天 |

较现实的里程碑：

- 2 周：正常闭环 + predictor + Failure Detector smoke；
- 3～4 周：方案一得到可重复恢复结果；
- 5～7 周：方案二和较完整消融；
- 6～8 周：形成可信论文实验。

---

## 8. 最终建议

这个方向值得继续，但第一版应将核心方法定义为：

> 使用动作条件化 Slot transition model 预测执行后的对象状态，将预测对象状态与真实对象状态的偏差作为事后失败检测依据；检测到异常后，通过 deviation-conditioned Slot 和短视野闭环重新规划进行恢复。

“同一个 Failure Scorer 同时检测已发生失败并对未执行候选动作排序”目前在因果条件和监督数据上不够成立。

更稳妥的设计是：

```text
共享：
Slot / SlotSSM / action encoder / robot-state encoder

区分：
post-transition failure detector
pre-action candidate risk head
```

方案一应作为主线；方案二应作为建立在可靠 forward model、候选 outcome 数据或 ranking supervision 上的增强实验。

---

## 9. 2026-07-20 实施更新：Stage-0 在线闭环

Stage-0A 与 Stage-0B 的机械闭环现已完成：

- `run_openvla_baseline.py` 已支持无头 LIBERO、MP4、逐步 JSONL，以及 planned/executed action、机器人状态和仅用于标签的仿真对象真值；
- 新增 `experiments/robot/libero/run_slotssm_online.py`，把已训练的 Stage-1、Stage-2、Stage-3 接入真实 LIBERO `env.step`；
- episode 开始时会清空图像历史；早期历史采用最早观测进行因果左填充，不使用未来帧；
- 当前使用 32 帧滚动窗口重算 Slot 和 Mamba2，每次预测 16 个未来动作、默认只执行第一个动作后重新观测；
- 每个控制步保存 current slots、predicted-next slots、dynamics state、归一化动作块和反归一化动作块；
- 标准 LIBERO 环境没有 LIBERO-Mem 数据中的 oracle object-stage cue，因此当前明确使用 `no active object subgoal`，元数据记录 `oracle_subgoal_available=false`。

已经完成两项真实无头测试：

1. 1 步初始化闭环，所有 Slot、Mamba 和动作张量均为有限值；
2. 5 步连续重规划，历史长度依次为 1、2、3、4、5，每步均成功写入仿真动作和内部特征。

测试产物：

- `artifacts/stage0b-slotssm-5step.mp4`
- `artifacts/stage0b-slotssm-5step-rollout/metadata.json`
- `artifacts/stage0b-slotssm-5step-rollout/steps.jsonl`
- `artifacts/stage0b-slotssm-5step-rollout/features/step_*.npz`

这里的“跑通”仅表示 Stage-1 → Stage-2 → Stage-3 → LIBERO 的在线数据和控制路径成立，并不代表任务成功率已经达标。5 步动作会随新观测变化，但变化较小；已知原因包括早期窗口左填充、缺少在线 oracle cue，以及离线消融已经显示 Stage-3 对 predicted-next slot 的利用很弱。

下一步应先完成完整 episode 的 A/B 基线评测（原 OpenVLA 对比 SlotSSM），记录成功率、推理延迟和动作轨迹。随后再实现 action-conditioned Slot predictor，将执行动作和机器人状态加入 transition model；这是采集 Failure-Aware 训练数据并构造有因果意义 deviation 的前置条件。

### 9.1 完整 episode A/B 结果

在相同的 `libero_spatial task 0 / episode 0` 初始状态上：

| 策略 | 结果 | 策略步数 | 墙钟时间 |
|---|---:|---:|---:|
| 原始 OpenVLA | 成功 | 98 | 78.92 秒 |
| 当前 Stage-3 SlotSSM 动作头 | 失败 | 220（上限） | 161.49 秒 |

原始 OpenVLA 的平移动作标准差约为 `[0.491, 0.214, 0.528]`，并将目标碗移动约 12.6 cm。SlotSSM 动作头对应标准差仅约 `[0.091, 0.003, 0.029]`，长时间输出近似固定方向动作，未移动目标物体。这表明当前 Stage-3 出现均值策略塌缩，不能作为 Failure-Aware 失败数据采集的正常策略。

因此 Failure-Aware 主线调整为：使用已经成功的原始 OpenVLA 执行动作；Stage-1 Slot、Stage-2 和新的 action-conditioned predictor 作为旁路世界模型。当前 Stage-3 保留为 Embodied-SlotSSM 复现结果和后续单独改进项，不阻塞失败检测主线。

### 9.2 Action-conditioned predictor 预检

数据接口已经修复并验证：LIBERO-Mem TFDS 实际字段是 7 维 `joint_state`，而原配置错误地写成不存在的 `robot_state`；`RLDSDatasetV3` 现在可以通过 `load_proprio=True` 返回 `[T, 7]` 的归一化关节状态。

新增 `scripts/train_failure_action_predictor.py`：

```text
current Slot
+ frozen Stage-2 dynamics
+ frozen Stage-2 predicted-next Slot
+ executed action (7D)
+ joint state (7D)
        ↓
residual action-conditioned predicted-next Slot
```

2-step preflight 已通过：可训练参数 3,286,556，梯度有限，checkpoint 成功保存。第 2 步 conditioned Smooth-L1 为 0.037088，同一 batch 的冻结 Stage-2 为 0.037206；该差异只验证训练路径，尚不构成效果结论。

### 9.3 Action-conditioned predictor 正式训练与 held-out 评测

正式训练已完成 10,000 steps，最终 checkpoint：

```text
/data2/lty/naive-e-slotssm-runs/failure-action-predictor-h32-s10k/action-predictor-step-010000.safetensors
```

在未参与训练的最后 5% trajectory 上评测 50 个 32 帧窗口：

| 模型/消融 | MAE | Smooth-L1 | Slot identity retrieval |
|---|---:|---:|---:|
| Persistence | 0.531043 | 0.352177 | 0.79145 |
| 原 Stage-2 | 0.127453 | 0.030129 | 0.97125 |
| Action-conditioned predictor | **0.090505** | **0.017535** | **0.99605** |
| 打乱 action | 0.090518 | 0.017538 | 0.99605 |
| 打乱 proprio | 0.090654 | 0.017566 | 0.99540 |

相对原 Stage-2，predictor 的 MAE 下降约 29%，Smooth-L1 下降约 42%。但是，打乱 action 后 Smooth-L1 只变化约 0.000003，打乱 proprio 后也只变化约 0.000032。这说明模型主要学习了依赖视觉状态和 Stage-2 输出的残差修正，几乎忽略 action/proprio。

因此，这个 checkpoint 是有效的 next-slot refinement，但尚不是因果意义上的 action-conditioned transition model。根因是成功示范数据中动作基本由状态决定，同一状态没有多个不同动作及其不同后果；仅靠行为克隆 trajectory，网络可以绕过动作输入获得低 loss。

在进入 Failure Detector 前必须增加仿真干预数据：从相同或可复现的 simulator state 分叉执行原动作和扰动动作，记录 planned/executed action、执行前后机器人状态、执行前后图像和对象真值。训练时应加入 matched counterfactual/ranking 约束，使正确动作预测与正确 next-slot 配对，并要求错误动作条件产生不同 outcome。只有动作打乱或反事实动作评测出现稳定退化后，才能把 predictor deviation 用作 Failure-Aware 信号。

### 9.4 同状态多动作的仿真干预采集

新增 `experiments/robot/libero/collect_counterfactual_transitions.py`。采集器使用两个隔离环境：主环境只执行原始 OpenVLA，不进行状态恢复；分支环境加载主环境的 MuJoCo flat state，并从规范化的恢复观测分别执行 nominal、六个 XYZ 正负平移、两个 Z 轴旋转和夹爪翻转。每个分支默认持续 4 个控制步并保存所有中间图像、机器人状态和仿真对象真值。

双环境设计是必要的：MuJoCo flat state 不包含 robosuite observable buffer。若在主环境中恢复 state，会改变后续视觉闭环并使原本成功的 OpenVLA 轨迹失败。分支环境中的每次 restore 都进行图像 SHA-256 一致性检查，同时显式重置 robosuite episode counter。

第一批正式数据 `libero_spatial task 0 / episode 0` 已完成：

- 主 OpenVLA 在第 98 个策略步成功；
- 20 个独立 fork state；
- 每个 state 10 个 matched action branches，共 200 branches；
- 保存 789 个分支后果帧，少于理论 800 是因为部分分支提前成功并终止；
- 200/200 恢复图像哈希一致；
- 每种 branch 恰好 20 条；
- 数据总量约 56 MB。

平移干预相对 nominal 的平均末端位置差约 0.8～1.2 cm，最终图像平均像素差约 1.9～3.0/255；旋转和夹爪干预也产生可观测后果。正式数据目录：

```text
/data2/lty/naive-e-slotssm-runs/counterfactual-data/libero_spatial_t0_e0_h4_dual
```
