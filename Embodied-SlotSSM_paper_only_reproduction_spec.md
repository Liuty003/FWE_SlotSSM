# Embodied-SlotSSM 论文级复现说明

> 目标：仅依据论文 **Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective**（arXiv:2511.11478v3）复现论文中的 Naive Embodied-SlotSSM（Naive E-SlotSSM）。
>
> 本文档面向 Codex。实现过程中禁止搜索、复制或参考作者发布的 Embodied-SlotSSM / LIBERO-Mem 仓库代码。可以使用公开的基础依赖与上游模型实现，例如 LIBERO、OpenVLA、PyTorch、Transformers、Mamba/SSM 库，但作者仓库不得作为实现依据。

---

## 0. 复现边界与信息可信度

论文没有提供足够信息来实现逐参数、逐行完全一致的复现。代码中必须对信息来源进行分级标注，禁止把推测伪装成论文原设定。

统一使用以下标记：

- **[PAPER]**：论文正文、公式、图、表或附录明确给出。
- **[DERIVED]**：可由论文公式和张量关系直接推导。
- **[ASSUMPTION]**：论文缺失细节，为保证代码可运行而采用的实现选择。
- **[UNKNOWN]**：论文未说明，暂时保留为配置项或待确认项。

代码中的所有 `[ASSUMPTION]` 必须：

1. 写入配置文件；
2. 在运行日志中打印；
3. 可以通过命令行覆盖；
4. 在最终实验报告中单独列出。

---

## 1. 论文要复现的核心目标

### 1.1 问题设定

输入包括：

- 视觉观测序列：\(o_{1:t}\)；
- 当前任务语言指令：\(l\)；
- 训练阶段的专家动作：\(a_{1:t}\)；
- Naive E-SlotSSM 使用的对象级 oracle 子目标：\(g_t^{(j)}\)。

输出为当前机器人动作：

\[
\hat a_t \sim P_\theta\left(
 a_t\mid
 \{r_t^{(j)}\}_{j=1}^{L},
 \{d_t^{(j)}\}_{j=1}^{K},
 l
\right)
\]

其中：

- \(K=16\) 个对象 slot；
- \(L=16\) 个关系 token；
- \(d_t^{(j)}\) 是第 \(j\) 个对象的任务相关动态表示；
- \(r_t^{(j)}\) 是关系编码器输出的关系 token。

### 1.2 核心思想

模型将视觉序列拆成三类互补信息：

1. **当前对象表示**：Slot Attention 输出 \(s_t^{(j)}\)；
2. **对象时间动态**：SlotSSM 输出预测下一 slot \(\hat s_{t+1}^{(j)}\)；
3. **当前对象关系**：Relation Encoder 使用对象动态表示查询原始视觉 patch 特征，输出关系 token。

最终动作解码器接收：

- 16 个 slot dynamics token；
- 16 个 relation token；
- 当前任务语言 embedding。

因此动作解码器前的对象相关 token 数量应为 32，这与论文实验表中的 Naive SlotSSM token 数 32 一致。

---

## 2. 论文明确给出的实验设定

### 2.1 数据集

复现包含两个评测集合：

1. **LIBERO-Goal**：评估一般的 Markovian 操作性能；
2. **LIBERO-Mem**：评估对象级非 Markovian 记忆能力。

### 2.2 LIBERO-Mem 数据规模

**[PAPER]** 每个任务：

- 轨迹长度约 200–700 帧；
- 120 条轨迹；
- 100 条训练轨迹；
- 20 条验证轨迹；
- 固定场景布局；
- 对象初始位姿随机化；
- 每个时间步包含对象实例 ID、实例 mask 和每个对象的子目标完成标志。

### 2.3 LIBERO-Mem 任务

| ID | 任务 | 子目标结构 | 记忆类型 |
|---|---|---|---|
| T1 | 拿起碗，再放回盘子 1 次 | bowl lifted → bowl on plate | OM |
| T2 | 拿起瓶子，再放回盘子 1 次 | bottle lifted → bottle on plate | OM |
| T3 | 拿起碗并放回盘子 3 次 | (bowl lifted → bowl on plate) ×3 | OM, OS |
| T4 | 拿起瓶子并放回盘子 3 次 | (bottle lifted → bottle on plate) ×3 | OM, OS |
| T5 | 拿起碗并放回盘子 5 次 | (bowl lifted → bowl on plate) ×5 | OM, OS |
| T6 | 拿起碗并放回盘子 7 次 | (bowl lifted → bowl on plate) ×7 | OM, OS |
| T7 | 借助空盘交换两个碗 | bowl 1 on plate 3 → bowl 2 on plate 1 → bowl 1 on plate 2；允许另一条对称路径 | OM, OR |
| T8 | 借助空盘交换三个碗 | 四阶段顺序放置 | OM, OR |
| T9 | 把碗放进最近的篮子，再把该篮子移到桌面中央 | In → On | OM, OO |
| T10 | 把碗放进最近的篮子，再把空篮子移到桌面中央 | In → On | OM, OO |

> 注意：论文表格中“拿起碗 5 次”一行重复写成 T4，应在实现中按顺序修正为 T5，并在日志中记录该论文排版问题。

### 2.4 论文最终超参数

**[PAPER]**：

```yaml
num_slots: 16
observation_chunk_size: 32
temporal_prediction_horizon: 32
contrastive_temperature: 1.0
eval_rollouts_per_task: 20
```

**[PAPER]** 其余超参数沿用引用工作的默认值，但论文没有逐项列出。

---

## 3. 总体架构

```text
RGB observation o_t
        │
        ▼
Frozen visual encoder
        │
        ├─────────────────────────────────────────────┐
        │                                             │
        ▼                                             │
Raw visual patch features v_t                         │
        │                                             │
        ▼                                             │
Slot Attention                                        │
        │                                             │
        ▼                                             │
Current slots s_t[1:K]                                │
        │                                             │
        ▼                                             │
Slot-based SSM ───────────────► predicted next slots ŝ_{t+1}[1:K]
        │                                             │
        │                    ┌────────────────────────┘
        │                    │
        │                    ▼
        │               Past MLP
        │                    │
        │                    ▼
        │        local latent window ẑ_{t-p:t+q}
        │          （训练辅助分支，动作头不直接读取）
        │
        ├── s_t + ŝ_{t+1} + oracle subgoal g_t
        │                    │
        │                    ▼
        │                Slot Fusion
        │                    │
        │                    ▼
        │             slot dynamics d_t[1:K]
        │                    │
        │                    ├───────────────┐
        │                    │               │
        │                    ▼               │
        │            Relation Encoder ◄──────┘ raw visual patches v_t
        │                    │
        │                    ▼
        │             relation tokens r_t[1:L]
        │
Task prompt l ─► Frozen prompt encoder
        │
        ▼
[r_t[1:L], d_t[1:K], prompt tokens]
        │
        ▼
Trainable/fine-tuned VLA action decoder
        │
        ▼
Δx, Δθ, Δgrasp / OpenVLA action tokens
```

---

## 4. 推荐工程目录

```text
embodied_slotssm_repro/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── base.yaml
│   ├── libero_goal.yaml
│   ├── libero_mem.yaml
│   └── ablations/
├── scripts/
│   ├── prepare_libero_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize_slots.py
│   └── inspect_checkpoint.py
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── chunk_sampler.py
│   │   ├── subgoal_parser.py
│   │   ├── subgoal_monitor.py
│   │   └── slot_object_matching.py
│   ├── models/
│   │   ├── embodied_slotssm.py
│   │   ├── visual_encoder.py
│   │   ├── slot_attention.py
│   │   ├── temporal_contrastive.py
│   │   ├── slot_ssm.py
│   │   ├── past_mlp.py
│   │   ├── slot_fusion.py
│   │   ├── relation_encoder.py
│   │   └── action_decoder.py
│   ├── losses/
│   │   ├── action_loss.py
│   │   ├── temporal_window_loss.py
│   │   └── total_loss.py
│   ├── evaluation/
│   │   ├── predicates.py
│   │   ├── structured_goal.py
│   │   ├── rollout.py
│   │   └── metrics.py
│   └── utils/
│       ├── state.py
│       ├── seeds.py
│       ├── logging.py
│       └── tensor_checks.py
└── tests/
    ├── test_slot_attention.py
    ├── test_temporal_identity.py
    ├── test_slot_ssm_independence.py
    ├── test_past_mlp.py
    ├── test_relation_encoder.py
    ├── test_episode_state_reset.py
    └── test_structured_goal.py
```

---

## 5. 数据预处理

### 5.1 每个时间步的数据结构

建议统一成：

```python
StepSample = {
    "rgb": Tensor[3, H, W],
    "action": Tensor[A],
    "language_instruction": str,
    "object_instance_ids": list[str],
    "object_masks": Tensor[M, H, W],
    "object_states": dict,
    "subgoal_flags": dict[str, bool],
    "oracle_object_subgoals": dict[str, str],
    "episode_id": str,
    "timestep": int,
    "done": bool,
}
```

### 5.2 观测窗口

**[PAPER]** 训练 observation chunk 和时间预测 horizon 都设为 32 帧。

实现连续片段：

```text
[o_t, o_{t+1}, ..., o_{t+31}]
```

需同时返回：

- 32 帧 RGB；
- 对应动作；
- 对象 mask；
- 子目标状态；
- 有效时间步 mask；
- episode 边界标志。

### 5.3 跨 chunk 状态

论文使用 observation chunking，但没有明确描述训练时是否跨 chunk 传递 Slot Attention 状态和 SSM 隐状态。

采用以下实现：

- **训练**：[ASSUMPTION] 使用按 episode 排序的连续 chunk sampler；
- 前一个 chunk 的最终 slot 和 SSM hidden 作为下一个 chunk 的初始状态；
- chunk 边界执行 `detach()`，形成 truncated BPTT；
- episode 边界必须重置所有状态；
- **评测**：完整 episode 在线执行，始终传递状态，禁止每 32 帧重置。

同时提供：

```yaml
state_carry_mode:
  train: sequential_detached
  eval: full_episode
```

### 5.4 视觉预处理

**[PAPER]** 视觉输入由预训练视觉编码器产生 raw visual features \(v_t\)，论文图中视觉编码器被冻结。

实现要求：

- 使用 OpenVLA 对应的预训练视觉编码器；
- 保留所有 patch token，不只使用 pooled token；
- 输出：

```text
visual_features: [B, T, N_patch, D_visual]
```

- 加载后默认 `requires_grad=False`；
- 保存视觉编码器版本和权重标识到 checkpoint metadata。

### 5.5 oracle 对象级子目标

对每个相关对象构造文本，例如：

```text
"bowl 1 on plate 3"
"bowl 2 on plate 1"
"basket 1 in center"
```

无当前子目标的背景 slot 使用：

```text
"no active object subgoal"
```

论文没有说明对象 slot 与实例 ID 的具体匹配方式。实现以下方法：

1. 将 Slot Attention 权重恢复为空间注意力图；
2. 与 ground-truth instance mask 计算 IoU；
3. 使用 Hungarian matching 完成 slot ↔ object instance 一一匹配；
4. 未匹配 slot 视为背景；
5. 将对应对象的 oracle 子目标 embedding 分配给该 slot。

该部分标记为 **[ASSUMPTION]**，因为论文只说明存在实例 mask、实例 ID 和 oracle 子目标，未给出匹配算法。

---

## 6. Slot Attention

### 6.1 输入输出

输入：

```text
v_t: [B, N_patch, D_visual]
```

输出：

```text
s_t: [B, K=16, D_slot]
attention_maps: [B, K, N_patch]
```

### 6.2 论文公式

对于第 \(i\) 个 slot 和第 \(j\) 个视觉 patch：

\[
a_{i,j}=\frac{q_i k_j^\top}{\sqrt{D_{enc}}}
\]

首先对固定 patch 在 slot 维度做竞争 softmax：

\[
\tilde a_{i,j}
=\frac{\exp(a_{i,j})}
{\sum_{l=1}^{K}\exp(a_{l,j})}
\]

然后对每个 slot 在 patch 维度重新归一化：

\[
w_{i,j}
=\frac{\tilde a_{i,j}}
{\sum_{m=1}^{N_{patch}}\tilde a_{i,m}+\epsilon}
\]

聚合视觉 value：

\[
u_i=\sum_{j=1}^{N_{patch}}w_{i,j}v_j
\]

使用 GRU 更新 slot：

\[
s_i^{(r+1)}=\operatorname{GRU}(u_i,s_i^{(r)})
\]

> 论文公式（7）中的 \(K/N\) 求和上下标存在明显符号混用。代码应采用标准 Slot Attention：先在 slot 维度 softmax，再在输入 patch 维度归一化。

### 6.3 时间初始化

**[PAPER]**：

\[
s_t^{(0)}=
\begin{cases}
\operatorname{RandomInit}(),&t=0\\
s_{t-1}^{(R)},&t>0
\end{cases}
\]

实现要求：

- episode 第 0 帧随机初始化 slot；
- 后续帧用上一帧最后一次 refinement 输出初始化；
- 禁止在同一 episode 内逐帧重新随机初始化；
- episode 切换时清空 slot 状态。

### 6.4 未说明的超参数

```yaml
slot_attention:
  num_slots: 16                  # [PAPER]
  slot_dim: null                 # [UNKNOWN]，默认等于 visual hidden dim
  num_iterations: 3             # [ASSUMPTION]，可配置
  num_heads: 1                  # [ASSUMPTION]，论文称 multi-head，默认可改为 4
  mlp_hidden_dim: null           # [UNKNOWN]
  epsilon: 1.0e-8
```

建议第一轮实现将 `slot_dim = visual_dim`，减少不必要的维度投影。

---

## 7. 时间对比损失

### 7.1 论文目标

锚点为：

\[
s_i^t
\]

正样本：同一视频附近帧中对应的 slot：

\[
s_i^{t+\delta}
\]

负样本：来自其他视频或其他位置的 slot。

温度：

\[
\tau=1
\]

### 7.2 实现定义

论文没有给出“对应 slot”和“不同位置”的精确采样算法。采用：

- 正样本：同一 episode、同一 slot index、时间距离满足 `1 <= |delta| <= positive_horizon`；
- 负样本：
  - 同 batch 其他 episode 的所有 slot；
  - 同 episode 同一帧的其他 slot index；
  - 同 episode邻近帧的其他 slot index；
- 排除 padding、episode 边界和当前锚点本身。

多正样本 InfoNCE：

\[
\mathcal L_{contrast}
=-\sum_{i,t}
\log
\frac{
\sum_{p\in P(i,t)}\exp(\operatorname{cos}(s_i^t,p)/\tau)
}{
\sum_{x\in P(i,t)\cup N(i,t)}
\exp(\operatorname{cos}(s_i^t,x)/\tau)
}
\]

配置：

```yaml
temporal_contrastive:
  enabled: true
  temperature: 1.0              # [PAPER]
  positive_horizon: 4           # [ASSUMPTION]
  use_same_episode_other_slots_as_negatives: true
  use_cross_episode_negatives: true
```

需要记录两个诊断指标：

```text
positive_cosine_similarity
negative_cosine_similarity
```

训练后应满足前者显著高于后者。

---

## 8. Slot-based SSM 动态编码器

### 8.1 论文状态空间公式

\[
h_t=A(e_t)h_{t-1}+B(e_t)e_t
\]

\[
y_t=C(e_t)h_t
\]

每个对象 slot 单独维护状态：

\[
h_t=\operatorname{concat}(h_t^{(1)},\ldots,h_t^{(K)})
\]

块对角矩阵：

\[
A_t=\operatorname{diag}
\left(\{A(s_t^{(k)})\}_{k=1}^{K}\right)
\]

\[
B_t=\operatorname{diag}
\left(\{B(s_t^{(k)})\}_{k=1}^{K}\right)
\]

\[
C_t=\operatorname{diag}
\left(\{C(s_t^{(k)})\}_{k=1}^{K}\right)
\]

### 8.2 推荐实现

无需显式构造巨大块对角矩阵。将 slot 维度并入 batch：

```python
# slots: [B, T, K, D]
x = slots.permute(0, 2, 1, 3)      # [B, K, T, D]
x = x.reshape(B * K, T, D)         # 每个 slot 形成一条独立时间序列
y, h = shared_selective_ssm(x)
y = y.reshape(B, K, T, D).permute(0, 2, 1, 3)
```

这等价于：

- 每个 slot 拥有独立 hidden state；
- 所有 slot 共享 SSM 参数；
- SSM 时间更新中不发生跨 slot 混合；
- 跨对象交互由后续 Relation Encoder 承担。

### 8.3 预测下一 slot

论文图和正文使用：

\[
\hat s_{t+1}^{(j)}
\]

实现：

```python
ssm_output_t = y[:, t, j]
predicted_next_slot = next_slot_head(ssm_output_t)
```

其中 `next_slot_head` 可为线性层或两层 MLP。

该 head 的具体结构在论文中未说明，标记为 **[ASSUMPTION]**。

### 8.4 视觉与文本条件

论文 Figure 2 的 Slot-based SSM 图中包含：

- Visual Conditioning；
- Textual Conditioning。

正文没有说明具体条件化结构。

实现两种可切换模式：

```yaml
ssm_conditioning_mode: figure_complete
```

#### 模式 A：`equation_only`

仅使用当前 slot：

```python
x_t_j = slot_t_j
```

#### 模式 B：`figure_complete`（默认）

将 pooled visual feature 和 prompt embedding 投影后作为 FiLM 条件：

```python
gamma, beta = condition_mlp(concat(pooled_visual_t, prompt_embedding))
conditioned_slot = layer_norm(slot_t_j) * (1 + gamma) + beta
```

然后送入 SSM。

该 FiLM 形式是 **[ASSUMPTION]**，代码必须支持关闭，以便做消融。

### 8.5 配置

```yaml
slot_ssm:
  implementation: mamba_or_selective_ssm   # [PAPER] 引用 Mamba/SlotSSM
  model_dim: null                          # [UNKNOWN]
  state_dim: 16                            # [ASSUMPTION]
  num_layers: 2                            # [ASSUMPTION]
  shared_across_slots: true                # [DERIVED]
  cross_slot_mixing_inside_ssm: false      # [DERIVED from block-diagonal formulation]
  conditioning_mode: figure_complete       # [ASSUMPTION]
```

---

## 9. Past MLP 与局部时间窗口预测

### 9.1 论文公式

对每个对象 slot \(j\)：

\[
\left\{\hat z_{t+\delta}^{(j)}\right\}_{\delta=-p}^{q}
=
W_{pred}
\left(
\hat s_{t+1}^{(j)}\Vert s_t^{(j)}
\right)
\]

- Past MLP 参数在所有 slot 间共享；
- 对每个 slot 独立应用；
- 预测当前时刻附近的过去和未来静态 latent；
- 该分支用于训练 transient memory；
- 动作解码器没有直接读取整段 \(\hat z\)。

### 9.2 预测窗口长度歧义

论文写：

\[
P=p+q
\]

同时索引范围为：

\[
\delta=-p,\ldots,q
\]

按包含端点和 \(\delta=0\) 的数学计数，输出数量为 \(p+q+1\)。论文附录又明确写 \(P=32\)。

实现必须显式使用 offset 列表，默认：

```yaml
window_offsets: [-16, -15, ..., -1, 0, 1, ..., 15]  # 共 32 帧
```

对应：

```text
p = 16
q = 15
num_window_targets = 32
```

同时提供可选模式：

```yaml
window_mode: paper_32_inclusive_asymmetric
```

禁止在代码中隐式假设 `p=q=16` 后输出 33 帧。

### 9.3 监督目标 \(z\)

论文对 \(z_{t+\delta}^{(j)}\) 的具体生成方法和是否 stop-gradient 没有说明。

默认采用：

```python
z_target = stop_gradient(slot_projection(s_{t+delta}^{(j)}))
```

其中：

- `s_{t+delta}^{(j)}` 来自同一次前向过程中的 Slot Attention；
- 依靠跨帧 slot 初始化保持 slot index 对齐；
- `slot_projection` 默认恒等映射；
- stop-gradient 防止预测目标和预测器共同漂移。

该做法标记为 **[ASSUMPTION]**。

### 9.4 Past MLP 结构

```python
input_dim = 2 * D_slot
output_dim = 32 * D_slot

PastMLP:
    Linear(input_dim, 4 * D_slot)
    GELU
    Linear(4 * D_slot, output_dim)
    reshape -> [B, K, 32, D_slot]
```

具体层数和隐藏维度为 **[UNKNOWN/ASSUMPTION]**。

### 9.5 窗口预测损失

论文没有给出窗口损失公式。

实现一个可配置组合：

\[
\mathcal L_{window}
=
\lambda_{mse}\operatorname{MSE}(\hat z,z)
+
\lambda_{cos}
\left(1-\operatorname{cos}(\hat z,z)\right)
\]

默认：

```yaml
window_loss:
  mse_weight: 1.0
  cosine_weight: 0.0
  mask_invalid_offsets: true
```

使用 MSE 作为最保守的 latent regression 默认值，并在实验报告中明确标记为论文未给出的实现选择。

---

## 10. Slot Fusion

### 10.1 输入

对每个 slot：

\[
s_t^{(j)},\quad
\hat s_{t+1}^{(j)},\quad
g_t^{(j)}
\]

输出：

\[
d_t^{(j)}
\]

其中 \(g_t^{(j)}\) 是 oracle 对象级文本子目标 embedding。

### 10.2 论文未说明内部结构

默认实现：

```python
x = concat([
    layer_norm(current_slot),
    layer_norm(predicted_next_slot),
    projected_object_subgoal,
], dim=-1)

d = residual_projection(current_slot) + fusion_mlp(x)
d = layer_norm(d)
```

配置：

```yaml
slot_fusion:
  type: concat_mlp_residual       # [ASSUMPTION]
  output_dim: llm_hidden_dim
  hidden_multiplier: 4
  dropout: 0.0
```

需支持消融：

```text
current only
current + next
current + subgoal
current + next + subgoal
```

---

## 11. Relation Encoder

### 11.1 论文信息

**[PAPER]**：

- 轻量级 Relation Encoder；
- 对 slot latents \(\{d_t^{(j)}\}_{j=1}^{K}\) 与 raw visual features \(v_t\) 做 cross-attention；
- 输出 \(L=16\) 个 relation tokens；
- 当前默认 \(K=16\)，所以关系 token 数也为 16。

### 11.2 最直接的实现

使用 slot dynamics 作为 query，视觉 patch 作为 key/value：

```python
Q = Wq(d_t)                   # [B, 16, D]
K = Wk(v_t)                   # [B, N_patch, D]
V = Wv(v_t)                   # [B, N_patch, D]
R = CrossAttention(Q, K, V)   # [B, 16, D]
```

再执行：

```python
R = LayerNorm(d_t + R)
R = LayerNorm(R + FFN(R))
```

这会让每个对象动态 token 从当前图像中重新提取精细空间信息。

配置：

```yaml
relation_encoder:
  num_relation_tokens: 16       # [PAPER]
  query_source: slot_dynamics   # [DERIVED/ASSUMPTION]
  num_layers: 1                 # [ASSUMPTION]，论文称 lightweight
  num_heads: 8                  # [ASSUMPTION]
  use_ffn: true
```

论文没有说明是否使用独立的可学习 relation queries。默认不增加额外 queries，因为 `d_t` 自身有 16 个 token，直接作为 query 最符合“对 slot latents 与 raw visual features 做 cross-attention”的文字描述。

---

## 12. Prompt Encoder 与 oracle 子目标编码

### 12.1 全局任务指令

全局 prompt：

```text
"Pick up the bowl and place down"
```

**[PAPER]** Figure 2 中 Prompt Encoder 冻结。

输出：

```text
prompt_tokens: [B, N_text, D_llm]
pooled_prompt: [B, D_llm]
```

用途：

- prompt tokens 输入最终 VLA action decoder；
- pooled prompt 可用于 SSM textual conditioning；
- 同一个语言编码器用于编码对象级 oracle 子目标，或使用共享文本 embedding 层。

### 12.2 对象子目标

```text
object_subgoal_embedding: [B, T, K, D_text]
```

编码后投影到 Slot Fusion 所需维度。

---

## 13. VLA Action Decoder

### 13.1 输入 token

每个时间步构造：

```text
relation_tokens:      [B, 16, D_llm]
slot_dynamics_tokens: [B, 16, D_llm]
prompt_tokens:        [B, N_text, D_llm]
```

推荐顺序：

```text
[prompt tokens] + [relation tokens] + [slot dynamics tokens]
```

同时提供 position/type embedding：

```text
token_type = prompt / relation / slot_dynamic
```

论文公式只规定条件变量，没有规定 token 拼接顺序，因此该顺序是 **[ASSUMPTION]**。

### 13.2 Backbone

**[PAPER]** Naive E-SlotSSM 实验以 OpenVLA 为主要 VLA 基础。

实现要求：

- 加载预训练 OpenVLA action decoder；
- 视觉编码器冻结；
- Prompt Encoder 冻结；
- 新增模块可训练；
- Figure 2 将 LLM Action Decoder 标记为可学习，因此默认允许微调 action decoder；
- 微调方式（全量、LoRA、部分层）论文未说明。

默认配置：

```yaml
action_decoder:
  backbone: openvla
  finetune_mode: lora             # [ASSUMPTION，资源友好]
  use_standard_action_tokenizer: true
  action_loss: cross_entropy
```

为更接近论文中“2×A100 finetune”，需要提供：

```yaml
finetune_mode: full
```

### 13.3 动作形式

Figure 2 显示动作包含：

```text
Δx, Δθ, Δgrasp
```

具体旋转表示、归一化范围和动作离散方式未说明。采用 OpenVLA/LIBERO 原有动作接口，禁止自行改变动作定义。

---

## 14. 完整前向过程

```python
def forward_chunk(batch, recurrent_state):
    # batch.rgb: [B, T=32, 3, H, W]

    visual = frozen_visual_encoder(batch.rgb)
    # [B, T, N_patch, D_visual]

    prompt_tokens, pooled_prompt = frozen_prompt_encoder(
        batch.language_instruction
    )

    slots = []
    slot_maps = []
    slot_init = recurrent_state.prev_slots

    for t in range(T):
        s_t, attn_t = slot_attention(
            visual[:, t],
            initial_slots=slot_init,
        )
        slots.append(s_t)
        slot_maps.append(attn_t)
        slot_init = s_t

    slots = stack(slots, dim=1)   # [B,T,K,D]

    conditioned_slots = apply_optional_visual_text_conditioning(
        slots,
        visual,
        pooled_prompt,
    )

    ssm_outputs, next_hidden = slot_ssm(
        conditioned_slots,
        recurrent_state.ssm_hidden,
    )

    predicted_next_slots = next_slot_head(ssm_outputs)
    # [B,T,K,D]

    predicted_window = past_mlp(
        concat([slots, predicted_next_slots], dim=-1)
    )
    # [B,T,K,32,D]

    object_subgoal_embeddings = build_oracle_subgoal_embeddings(
        slot_maps,
        batch.object_masks,
        batch.oracle_object_subgoals,
    )

    slot_dynamics = slot_fusion(
        slots,
        predicted_next_slots,
        object_subgoal_embeddings,
    )
    # [B,T,K,D_llm]

    relation_tokens = []
    for t in range(T):
        r_t = relation_encoder(
            queries=slot_dynamics[:, t],
            key_values=visual[:, t],
        )
        relation_tokens.append(r_t)

    relation_tokens = stack(relation_tokens, dim=1)

    action_logits = action_decoder(
        prompt_tokens=prompt_tokens,
        relation_tokens=relation_tokens,
        slot_dynamics_tokens=slot_dynamics,
    )

    losses = compute_losses(
        action_logits=action_logits,
        action_targets=batch.action,
        slots=slots,
        predicted_window=predicted_window,
        valid_mask=batch.valid_mask,
        episode_ids=batch.episode_id,
    )

    new_state = RecurrentState(
        prev_slots=slots[:, -1].detach(),
        ssm_hidden=detach_state(next_hidden),
    )

    return action_logits, losses, new_state
```

---

## 15. 总损失

论文明确给出 temporal contrastive loss，并描述 Past MLP 的过去/未来重建监督，但没有给出完整总损失及权重。

实现：

\[
\mathcal L
=
\lambda_{action}\mathcal L_{action}
+
\lambda_{contrast}\mathcal L_{contrast}
+
\lambda_{window}\mathcal L_{window}
\]

默认：

```yaml
loss:
  action_weight: 1.0
  contrastive_weight: 0.1    # [ASSUMPTION]
  window_weight: 1.0         # [ASSUMPTION]
```

必须在训练初期分别记录三项 loss 的数值和梯度范数，避免辅助损失压制动作学习。

建议自动报警条件：

```text
contrastive loss 为 NaN
正负样本数为 0
slot 方差接近 0（slot collapse）
不同 slot attention map 过度重合
SSM hidden 范数持续爆炸
window loss 远高于 action loss 且不下降
```

---

## 16. 训练策略

### 16.1 冻结与训练模块

根据 Figure 2：

| 模块 | 默认状态 |
|---|---|
| 预训练视觉编码器 | Frozen |
| Prompt Encoder | Frozen |
| Slot Attention | Trainable |
| Slot-based SSM | Trainable |
| Past MLP | Trainable |
| Slot Fusion | Trainable |
| Relation Encoder | Trainable |
| LLM Action Decoder | Trainable / fine-tuned |

### 16.2 两阶段训练建议

论文没有明确两阶段训练。为了稳定复现，使用可关闭的两阶段方案：

#### 阶段 A：对象时序表示预热

训练：

- Slot Attention；
- SlotSSM；
- Past MLP；
- temporal contrastive loss；
- window loss。

动作头冻结或使用较低学习率。

#### 阶段 B：联合动作微调

训练：

- 所有新增模块；
- VLA action decoder；
- 三项联合损失。

配置：

```yaml
training:
  use_representation_warmup: true       # [ASSUMPTION]
  warmup_epochs: 5                      # [ASSUMPTION]
  mixed_precision: bf16
  gradient_clip_norm: 1.0
  truncated_bptt_chunk: 32
```

论文使用 2×NVIDIA A100。显存不足时使用：

- bf16；
- gradient checkpointing；
- LoRA；
- 梯度累积；
- 预计算 frozen visual features。

### 16.3 随机种子

**[PAPER]** 需要设置：

- Python seed；
- NumPy seed；
- PyTorch seed。

同时设置：

```python
torch.cuda.manual_seed_all(seed)
```

记录：

- 数据顺序 seed；
- 环境 rollout seed；
- 对象位姿 seed；
- 模型初始化 seed。

---

## 17. 在线推理状态机

```python
state = RecurrentState.empty()

for t in episode:
    visual_t = visual_encoder(rgb_t)

    if t == 0:
        init_slots = random_init()
        ssm_hidden = zeros()
    else:
        init_slots = state.prev_slots
        ssm_hidden = state.ssm_hidden

    slots_t = slot_attention(visual_t, init_slots)
    ssm_t, next_hidden = slot_ssm.step(slots_t, ssm_hidden)
    next_slots_t = next_slot_head(ssm_t)

    oracle_subgoals_t = environment_or_annotation_subgoal_monitor(...)
    g_t = encode_and_assign_subgoals_to_slots(...)

    d_t = slot_fusion(slots_t, next_slots_t, g_t)
    r_t = relation_encoder(d_t, visual_t)

    action_t = action_decoder(prompt, r_t, d_t)
    env.step(action_t)

    state.prev_slots = slots_t
    state.ssm_hidden = next_hidden
```

推理阶段无需执行 Past MLP，除非需要可视化或诊断。动作公式没有直接使用其输出。

---

## 18. 子目标监控与评测

### 18.1 结构化目标

支持三个操作符：

```text
And(...)
Sequence(...)
Or(...)
```

### 18.2 Sequence

- 只检查当前尚未完成的下一个子目标；
- 当前子目标满足后追加到 `satisfied`；
- 顺序不能跳跃；
- 所有子目标完成后任务成功。

### 18.3 Or

- 每条分支是一条允许的子目标序列；
- 根据已经完成的前缀排除不一致分支；
- 任意合法序列全部完成即成功。

### 18.4 过度重复

**[PAPER]** 最后一个子目标完成后，机器人再次改变目标状态，视为 over-repetition failure。

例如要求完成 3 次抓放，完成第 3 次后继续执行第 4 次，应判失败。

### 18.5 指标

**任务成功率**：

\[
\text{Task Success Rate}
=
\frac{\text{成功 rollout 数}}{20}
\]

**子目标完成率**：

\[
\text{Subgoal Completion Rate}
=
\frac{\text{完成子目标数}}{\text{总子目标数}}
\]

每个任务运行 20 个独立 seed。

---

## 19. 论文结果作为复现 sanity check

### 19.1 LIBERO-Goal

| Task | SlotVLA h=1 | SlotVLA h=8 | Naive E-SlotSSM |
|---|---:|---:|---:|
| bowl in stove | 45% | 95% | 100% |
| bowl in cabinet | 75% | 90% | 90% |
| plate on front of stove | 70% | 80% | 95% |
| stove turned on | 70% | 95% | 100% |
| bottle on cabinet | 20% | 90% | 90% |
| bowl on plate | 0% | 90% | 90% |
| middle drawer open | 5% | 25% | 45% |
| cheese in bowl | 25% | 55% | 75% |
| bottle on rack | 10% | 70% | 75% |
| top drawer open → bowl in | 0% | 65% | 70% |
| **Average** | **32.0%** | **75.5%** | **83.0%** |

### 19.2 LIBERO-Mem 子目标完成率

| Task | Dense baseline h=1 | SlotVLA h=1 | SlotVLA h=8 | Naive E-SlotSSM |
|---|---:|---:|---:|---:|
| T1 | 50.0% | 0% | 50.0% | 50.0% |
| T2 | 0% | 0% | 0% | 0% |
| T3 | 0% | 0% | 0% | 33.3% |
| T4 | 0% | 0% | 0% | 0% |
| T5 | 0% | 0% | 0% | 14.3% |
| T6 | 0% | 0% | 0% | 0% |
| T7 | 0% | 0% | 0% | 0% |
| T8 | 0% | 0% | 0% | 0% |
| T9 | 0% | 0% | 0% | 30% |
| T10 | 0% | 0% | 0% | 20% |
| **Average** | **5.0%** | **0%** | **5.0%** | **14.8%** |

这些结果只作为量级参考。论文缺少大量训练细节，因此结果不一致时，应先检查 `[ASSUMPTION]` 项，不能直接判定代码错误。

---

## 20. 必须实现的可视化与诊断

### 20.1 Slot attention 可视化

每隔固定步数输出：

```text
RGB
每个 slot 的 attention heatmap
slot 对应 bbox
slot ↔ instance ID 匹配
```

论文定性分析重点观察 gripper 和 bowl slot 是否跨时间保持一致。

### 20.2 时间一致性

记录：

```text
sim(s_t^j, s_{t+1}^j)
sim(s_t^j, s_{t+1}^{k != j})
slot identity switch rate
```

### 20.3 Slot 独立动态测试

构造单元测试：

1. 固定所有 slot；
2. 只扰动第 3 个 slot 输入；
3. 在一次纯 SlotSSM 更新后，仅第 3 个 slot 的 hidden/output 应直接变化；
4. Relation Encoder 之后允许其他 relation token 受影响。

### 20.4 Past MLP

对一个人造线性运动对象，检查模型是否能够：

- 重建过去；
- 预测短期未来；
- 正确 mask episode 边界之外的 offset。

### 20.5 状态重置

两个 episode 连续执行时，第二个 episode 的第 0 帧不得继承前一个 episode 的 slot 或 SSM hidden。

---

## 21. 关键消融实验

至少完成：

| 实验 | 目的 |
|---|---|
| 无 SlotSSM，仅当前 slot | 检验对象历史的作用 |
| SlotVLA h=1 | 单帧 slot baseline |
| SlotVLA h=8 | 短时窗口拼接 baseline |
| 去掉 temporal contrastive loss | 检验 slot 身份一致性 |
| 去掉 Past MLP/window loss | 检验局部时间辅助监督 |
| 去掉 Relation Encoder | 检验精细场景关系 |
| 去掉 oracle subgoal | 显示 Naive E-SlotSSM 对 oracle 的依赖 |
| equation_only SSM conditioning | 检验图中视觉/文本条件化 |
| current slot only / current+next | 检验 \(\hat s_{t+1}\) 的贡献 |

---

## 22. 论文没有说明、必须谨慎处理的部分

以下内容均不可声称“严格遵循论文原实现”：

1. Slot dimension；
2. Slot Attention refinement 次数、head 数和 MLP 维度；
3. SSM/Mamba 的具体版本、层数、state dimension、卷积宽度；
4. Visual/Textual Conditioning 的实现；
5. \(\hat s_{t+1}\) 的输出 head；
6. Past MLP 的层数和隐藏维度；
7. \(z\) 目标的确切定义；
8. 窗口预测损失函数；
9. \(P=p+q\) 与索引数量的 off-by-one 矛盾；
10. temporal contrastive 正负样本的精确采样；
11. Slot Fusion 的内部结构；
12. Relation Encoder 的层数、head 数和 query 设计；
13. 对象 mask 如何与 slot 绑定；
14. OpenVLA 的微调方式、学习率和优化器；
15. 各损失权重；
16. batch size、epoch 数、训练步数和学习率调度；
17. 训练 chunk 是否跨边界携带 hidden state；
18. 论文 Figure 2 中 “persistent memory” 的独立实现边界；
19. oracle 子目标在测试时的生成与切换时机；
20. 动作旋转参数化。

Codex 实现这些部分时，必须在代码注释中引用本节编号，并将具体选择写入 `resolved_assumptions.yaml`。

---

## 23. 默认配置草案

```yaml
experiment:
  name: naive_e_slotssm_paper_only
  seed: 0
  dataset: libero_mem

model:
  num_slots: 16
  slot_dim: null
  visual_encoder_frozen: true
  prompt_encoder_frozen: true

slot_attention:
  num_iterations: 3
  num_heads: 4
  epsilon: 1.0e-8
  recurrent_init_across_frames: true

slot_ssm:
  implementation: mamba
  num_layers: 2
  state_dim: 16
  shared_across_slots: true
  cross_slot_mixing_inside_ssm: false
  conditioning_mode: figure_complete

past_mlp:
  enabled: true
  window_offsets: [-16, -15, -14, -13, -12, -11, -10, -9,
                   -8, -7, -6, -5, -4, -3, -2, -1,
                    0, 1, 2, 3, 4, 5, 6, 7,
                    8, 9, 10, 11, 12, 13, 14, 15]
  stop_gradient_target: true

slot_fusion:
  type: concat_mlp_residual

relation_encoder:
  num_relation_tokens: 16
  query_source: slot_dynamics
  num_layers: 1
  num_heads: 8

oracle_subgoal:
  enabled: true
  slot_object_matching: hungarian_iou
  background_text: "no active object subgoal"

action_decoder:
  backbone: openvla
  finetune_mode: lora
  use_standard_action_tokenizer: true

loss:
  action_weight: 1.0
  contrastive_weight: 0.1
  window_weight: 1.0
  contrastive_temperature: 1.0

training:
  chunk_size: 32
  sequential_state_carry: true
  detach_state_at_chunk_boundary: true
  precision: bf16
  gradient_clip_norm: 1.0
  representation_warmup_epochs: 5

evaluation:
  rollouts_per_task: 20
  carry_state_full_episode: true
  report_task_success: true
  report_subgoal_completion: true
  over_repetition_is_failure: true
```

所有未带 `[PAPER]` 依据的默认值都必须视为可替换初始设定。

---

## 24. Codex 执行顺序

### 阶段 1：搭建最小数据闭环

1. 读取 LIBERO 轨迹；
2. 返回连续 32 帧 chunk；
3. 对齐 RGB、动作、mask、object ID、subgoal flags；
4. 实现 episode 状态重置；
5. 实现结构化子目标 evaluator；
6. 单元测试 T1/T3/T7/T9 的 Sequence/Or 逻辑。

### 阶段 2：实现对象定位

1. 冻结视觉编码器；
2. 实现 Slot Attention；
3. 实现跨帧 recurrent slot initialization；
4. 可视化每个 slot 的 attention；
5. 实现 temporal contrastive loss；
6. 验证同一对象跨帧相似度。

### 阶段 3：实现对象动态

1. 实现按 `B*K` 展开的共享 SSM；
2. 实现 hidden state carry 和 episode reset；
3. 实现 next-slot head；
4. 实现 Past MLP；
5. 实现 offset masking 和 window loss；
6. 使用人造运动数据验证。

### 阶段 4：实现动作条件化

1. 实现 slot ↔ object mask 匹配；
2. 编码 oracle object-level subgoal；
3. 实现 Slot Fusion；
4. 实现 Relation Encoder；
5. 将 16 relation + 16 dynamics token 接入 OpenVLA action decoder；
6. 验证动作 token shape 和 loss。

### 阶段 5：训练与评测

1. 先跑 T1；
2. 再跑 T3；
3. 再跑 T7 和 T9；
4. 确认状态跨 32 帧 chunk 延续；
5. 扩展至完整 LIBERO-Mem；
6. 最后测试 LIBERO-Goal；
7. 运行 20 seeds；
8. 输出任务成功率、子目标完成率和置信区间。

---

## 25. 完成标准

复现项目只有同时满足以下条件才算完成：

- [ ] K=16 slot；
- [ ] 32 帧 observation chunk；
- [ ] slot 跨帧递归初始化；
- [ ] temporal contrastive loss，\(\tau=1\)；
- [ ] 每个 slot 独立维护 SSM hidden；
- [ ] 共享 SlotSSM 参数；
- [ ] Past MLP 预测 32 帧局部窗口；
- [ ] Past MLP 输出不直接输入动作头；
- [ ] Slot Fusion 融合 \(s_t,\hat s_{t+1},g_t\)；
- [ ] oracle object-level subgoal 启用；
- [ ] Relation Encoder 输出 16 个关系 token；
- [ ] 动作头同时接收 16 relation token、16 slot dynamics token 和全局任务语言；
- [ ] episode 边界重置状态；
- [ ] 完整 episode 评测时持续传递状态；
- [ ] 支持 Sequence、Or、And 子目标；
- [ ] over-repetition 判失败；
- [ ] 每个任务 20 个 rollout；
- [ ] 可视化 slot attention 和身份跟踪；
- [ ] 所有论文未说明的选择写入 `resolved_assumptions.yaml`。

---

## 26. 最重要的注意事项

1. **Past MLP 是辅助训练分支。** 它预测的 \(z\) 窗口没有出现在动作公式（11）的直接条件中。实际动作路径使用 \(s_t\)、\(\hat s_{t+1}\)、oracle 子目标、relation tokens 和全局语言。
2. **不能在每个 chunk 重置记忆。** 这样会破坏长程任务的核心设定。
3. **slot index 并不天然等于对象 ID。** 需要跨帧初始化、对比损失和可视化验证；oracle 子目标分配还需要实例 mask 对齐。
4. **块对角 SSM 不负责显式跨对象交互。** 对象关系主要由 Relation Encoder 建模。
5. **论文结果属于弱基线。** LIBERO-Mem 平均子目标完成率只有 14.8%，T7/T8 等任务仍为 0；复现目标应先对齐方法和量级，不应假定模型已经解决长程对象记忆。
6. **论文依赖 oracle 子目标。** 复现第一版必须保留该设定。自主子目标识别属于后续扩展。
7. **P=32 存在索引歧义。** 代码必须使用显式 offset 数组并记录选项。
8. **禁止通过作者仓库补全细节。** 所有缺失项只允许使用明确标记的实现假设，并通过消融验证。

