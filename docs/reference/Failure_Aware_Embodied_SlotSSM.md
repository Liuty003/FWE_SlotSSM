# Failure-Aware Embodied-SlotSSM：基于持续对象 Slot 的 VLA 失败检测与恢复方案

## 0. 方案概述

本项目以 **Embodied-SlotSSM** 为基础，在其持续对象建模与动作生成框架上加入一个轻量的失败评分器，并设计两种由对象 slot 参与的恢复机制。

整个系统希望使用同一套对象级表示服务三个阶段：

1. **正常操作**：持续对象 slot 为 VLA 提供任务相关的对象级视觉表示，使动作模型能够持续关注目标物体、夹爪、目标容器和关键环境物体。
2. **失败检测**：融合 VLA 内部动作特征、持续对象 slot、机器人内部状态和对象状态偏差，输出单一失败概率 \(p_{\mathrm{fail}}\)。
3. **失败恢复**：检测到高失败风险后，系统中断当前 action chunk，利用异常对象 slot 改变恢复阶段的动作条件，并采用短视野、高频闭环的执行方式继续完成任务。

本项目不引入失败类型分类、对象级失败标签、结构化恢复目标或人工恢复文本。失败评分器只输出一个标量：

\[
p_{\mathrm{fail},t}\in[0,1]
\]

恢复阶段包含两个方案：

- **方案一：Slot-Guided Slow Replanning**  
  使用对象偏差重新加权 slot，缩短 action chunk，提高 VLA 调用频率，从当前状态进行慢速闭环恢复。

- **方案二：Slot-Conditioned Candidate Risk Selection**  
  在方案一基础上生成少量候选动作，复用同一个失败评分器评估每个候选动作的风险，并执行风险最低的短动作前缀。

项目的核心主张是：

> 持续对象 slot 既能够提高正常操作中的对象感知能力，也能够提供对象级异常信号；同一个 slot 条件化风险模型可以同时用于执行过程中的失败检测和恢复阶段的动作选择。

--- 

# 1. 研究问题

## 1.1 当前 VLA 的局限

现有 VLA 通常根据当前图像、语言指令和机器人内部状态直接输出动作。该范式在正常任务中能够产生有效行为，但在失败场景中存在以下问题：

1. 当前 action chunk 内发生抓取偏移、滑落或放置偏差时，策略可能继续执行剩余动作。
2. 失败后直接重新调用同一个 VLA，可能再次生成相似动作。
3. 全局图像特征缺少稳定的对象身份表示，难以判断哪个对象发生了异常变化。
4. 单帧视觉难以表达“目标物体本应移动但实际没有移动”这类时间偏差。
5. 基于失败文本、失败类型或恢复子目标的方案需要额外标注，公开数据通常不包含这些标签。

## 1.2 本项目要回答的问题

本项目集中回答三个问题：

### 问题一：对象 slot 能否提高正常操作能力？

持续 slot 能够维护目标物体、夹爪、容器和干扰物体的对象身份，使动作模型通过交叉注意力读取完整对象集合。

### 问题二：对象 slot 能否提高失败检测能力？

对象 slot 沿时间更新，可以表示对象的历史状态和运动变化。动作条件化 slot 预测可以进一步得到“预期对象变化”和“实际对象变化”的偏差。

### 问题三：对象 slot 能否参与失败后的动作恢复？

恢复阶段根据对象偏差调整 slot 权重，使动作模型重点读取发生异常的任务相关对象。进阶版本进一步利用同一个失败评分器比较候选动作风险。

---

# 2. 设计原则

## 2.1 保持失败监督简单

Failure Scorer 只输出：

\[
p_{\mathrm{fail},t}
\]

训练只依赖轨迹级或时间窗口级成功/失败弱标签。

暂时不输出：

- 失败类型；
- 失败对象；
- 恢复模式；
- 恢复子任务；
- 对象关系标签；
- 结构化恢复目标。

这些附加输出会增加标注和训练难度，也会使方法依赖公开数据中不存在的监督信息。

## 2.2 复用 Embodied-SlotSSM

项目以 Embodied-SlotSSM 为主要基础，尽量保留其已有模块：

- 视觉编码器；
- Slot Attention；
- 时间一致的对象 slot；
- SlotSSM；
- Relation Encoder；
- 基于对象表示的动作解码器。

新增模块应保持轻量：

- 动作条件化 slot 预测头；
- 单标量 Failure Scorer；
- 恢复阶段 slot 重加权；
- 可选的候选动作风险排序。

## 2.3 Slot 在三个阶段承担不同作用

| 阶段 | Slot 的作用 |
|---|---|
| 正常操作 | 提供持续对象表示，供动作模型交叉注意力读取 |
| 失败检测 | 提供对象状态、时间变化和预测偏差信号 |
| 恢复执行 | 根据任务相关性和异常程度重新加权，引导 VLA 关注关键对象 |

## 2.4 恢复依赖短视野闭环

失败发生后采用更短的 action chunk：

\[
H_{\mathrm{rec}} < H_{\mathrm{normal}}
\]

系统每执行少量动作就重新观测、更新 slot 和计算失败风险。

该策略降低一次性执行长动作造成的错误累积，也适合当前两周左右的开发周期。

---

# 3. 系统总体结构

整体数据流如下：

```text
RGB 图像 I_t
    │
    ▼
视觉编码器 E_vision
    │
    ├──────────────► 图像 patch 特征 F_t
    │
    ▼
Slot Attention（上一帧 slot 初始化）
    │
    ▼
当前对象 slot S_t
    │
    ▼
SlotSSM 时间更新
    │
    ├──────────────► 持续对象状态 M_t
    │
    ├──────────────► 动作条件化未来 slot 预测
    │
    ▼
Relation Encoder
    │
    ▼
对象关系特征 R_t
    │
    ▼
VLA 动作解码器
    │
    ├──────────────► action chunk A_t
    │
    └──────────────► VLA 动作隐特征 H_t
                         │
持续 slot M_t ───────────┤
机器人状态 q_t ─────────┤
slot 预测偏差 D_t ──────┤
                         ▼
                Failure Scorer
                         │
                         ▼
                  p_fail,t
                         │
           ┌─────────────┴─────────────┐
           │                           │
       正常执行                    进入恢复阶段
                                   │
                         方案一 / 方案二
```

---

# 4. 输入与输出定义

## 4.1 系统输入

在时刻 \(t\)，系统接收：

\[
X_t = \{I_t, g, q_t, S_{t-1}, M_{t-1}\}
\]

其中：

- \(I_t\)：当前 RGB 图像；
- \(g\)：任务语言指令；
- \(q_t\)：机器人内部状态；
- \(S_{t-1}\)：上一帧 Slot Attention 输出；
- \(M_{t-1}\)：上一时刻 SlotSSM 隐状态。

机器人内部状态可包含：

- 机械臂关节角；
- 末端位姿；
- 夹爪开合状态；
- 可选的关节速度；
- 可选的末端速度。

## 4.2 正常动作输出

动作模型输出长度为 \(H_{\mathrm{normal}}\) 的 action chunk：

\[
A_t =
[a_t,a_{t+1},\ldots,a_{t+H_{\mathrm{normal}}-1}]
\]

动作形式取决于 Embodied-SlotSSM 所使用的基础策略，可采用：

- 末端位置增量；
- 末端旋转增量；
- 夹爪动作；
- 或机器人关节动作。

## 4.3 失败评分输出

Failure Scorer 只输出：

\[
p_{\mathrm{fail},t}
=
R_\phi(H_t^{\mathrm{VLA}},M_t,D_t,q_t)
\]

其中：

- \(H_t^{\mathrm{VLA}}\)：当前动作对应的 VLA 最后一层隐特征；
- \(M_t\)：持续对象 slot；
- \(D_t\)：对象 slot 预测偏差；
- \(q_t\)：机器人内部状态。

---

# 5. 基于 Embodied-SlotSSM 的对象建模

## 5.1 视觉特征提取

当前图像经过视觉编码器：

\[
F_t = E_{\mathrm{vision}}(I_t)
\]

其中：

\[
F_t\in \mathbb{R}^{N_p\times d_v}
\]

\(N_p\) 是视觉 patch 数量，\(d_v\) 是视觉特征维度。

优先复用 Embodied-SlotSSM 已有视觉编码器和预处理流程，避免更换视觉主干带来的额外调试成本。

---

## 5.2 时间一致的 Slot Attention

当前帧 slot 由上一帧 slot 初始化：

\[
S_t =
\operatorname{SlotAttention}(F_t,S_{t-1})
\]

其中：

\[
S_t=\{s_t^1,s_t^2,\ldots,s_t^K\}
\]

\(K\) 为 slot 数量。

该设计使同一对象在相邻帧中尽量保持相同的 slot 索引。

训练时继续使用 Embodied-SlotSSM 原有的时间一致性约束，例如相邻帧同一对象 slot 的对比学习或匹配损失。

建议第一版保持原论文的 slot 数量和维度，减少超参数搜索。

---

## 5.3 SlotSSM 时间更新

每个对象 slot 沿时间更新：

\[
m_t^i =
\operatorname{SlotSSM}(m_{t-1}^i,s_t^i)
\]

其中：

- \(s_t^i\)：当前帧观察到的对象表示；
- \(m_t^i\)：融合历史信息后的持续对象状态。

所有持续对象状态组成：

\[
M_t=\{m_t^1,m_t^2,\ldots,m_t^K\}
\]

SlotSSM 主要记录：

- 对象外观和身份；
- 对象最近位置和运动趋势；
- 对象是否被夹爪带动；
- 对象是否发生突然位移；
- 遮挡前后的对象状态；
- 与任务执行相关的短期历史。

第一版优先直接复用 Embodied-SlotSSM 的更新方式。

---

## 5.4 Slot 间交互

Slot 之间通过 Embodied-SlotSSM 现有的对象混合或关系模块交换信息。

两周版本不建议新写完整稀疏图网络。可以使用以下轻量形式：

\[
\widetilde M_t =
M_t +
\operatorname{SelfAttn}(M_t)
\]

或者直接复用其 Relation Encoder。

后续扩展可以加入 top-\(k\) 稀疏关系混合，只保留以下对象关系：

- 夹爪与目标物体；
- 目标物体与目标容器；
- 空间距离较近的对象；
- 运动变化相关的对象；
- 当前任务相关性较高的对象。

---

## 5.5 Relation Encoder

利用持续对象 slot 查询当前图像 patch：

\[
R_t =
\operatorname{CrossAttn}
(
Q=M_t,
K=F_t,
V=F_t
)
\]

得到融合对象历史和当前视觉细节的关系特征：

\[
R_t\in\mathbb{R}^{K\times d_r}
\]

动作模型保留全部 slot，通过交叉注意力读取对象集合。

正常阶段不对 slot 做硬筛选，以免丢失夹爪、目标容器、障碍物等间接相关对象。

---

# 6. 正常操作模块

## 6.1 动作模型输入

正常操作阶段，VLA 接收：

\[
\{g,q_t,M_t,R_t\}
\]

可选地保留原始图像 token：

\[
\{F_t,g,q_t,M_t,R_t\}
\]

推荐通过一个 gated cross-attention adapter 将 slot 注入动作解码器：

\[
H_t' =
H_t +
\gamma
\operatorname{CrossAttn}
(
Q=H_t,
K=M_t,
V=M_t
)
\]

其中 \(\gamma\) 为可训练门控参数，初始值设置较小，使模型初始行为接近原 Embodied-SlotSSM。

## 6.2 任务相关对象注意力

动作模型内部会根据语言指令和动作 token 对全部 slot 进行交叉注意力。

可记录平均注意力权重：

\[
\alpha_t^i
=
\operatorname{MeanAttn}
(
H_t^{\mathrm{action}},m_t^i
)
\]

\(\alpha_t^i\) 表示第 \(i\) 个对象对当前任务动作的重要性。

该权重不需要单独监督，可以直接从动作解码器的交叉注意力中提取。

其作用包括：

- 正常操作中的对象聚焦；
- 恢复阶段的任务相关性先验；
- 可视化当前动作主要读取哪些对象。

---

# 7. 动作条件化 Slot 预测

## 7.1 目的

单纯依赖当前 slot 无法直接表达“对象本应发生什么变化”。

增加一个轻量的动作条件化 slot 预测头：

\[
\widehat S_{t+1}
=
D_\psi(M_t,E_A(A_t),E_q(q_t))
\]

它学习：

> 给定当前对象状态、机器人状态和当前动作，对象在下一时刻应当如何变化。

该模块只使用公开轨迹中已有的数据：

\[
(S_t,A_t,S_{t+1})
\]

无需失败类型或恢复标签。

---

## 7.2 预测结构

对每个 slot：

\[
\widehat s_{t+1}^i
=
s_t^i +
f_\psi
(
m_t^i,
E_A(A_t),
E_q(q_t)
)
\]

第一版可以使用两层 MLP：

```text
slot_state ─┐
action_emb ─┼─► concat ─► MLP ─► delta_slot
robot_emb ──┘
```

也可以输入 action chunk 的前若干步或池化表示：

\[
E_A(A_t)=
\operatorname{MLP}
(
\operatorname{Flatten}(A_t)
)
\]

两周版本建议只预测下一时刻或短窗口末端：

\[
\widehat S_{t+1}
\quad \text{或}\quad
\widehat S_{t+\Delta}
\]

避免多步递归预测造成训练和实现复杂度。

---

## 7.3 Slot 预测损失

基础形式：

\[
\mathcal L_{\mathrm{slot\_pred}}
=
\frac{1}{K}
\sum_{i=1}^{K}
\left\|
\widehat s_{t+1}^i-s_{t+1}^i
\right\|_2^2
\]

如果 slot 已归一化，也可以使用余弦距离：

\[
\mathcal L_{\mathrm{slot\_pred}}
=
\frac{1}{K}
\sum_{i=1}^{K}
\left[
1-
\cos
(
\widehat s_{t+1}^i,
s_{t+1}^i
)
\right]
\]

为了减小 slot 顺序漂移影响，可以沿用 Embodied-SlotSSM 已有 slot 对齐方式。

---

## 7.4 对象级偏差

动作执行后得到真实新 slot：

\[
S_{t+1}
\]

计算每个对象的预测偏差：

\[
d_{t+1}^i
=
1-
\cos
(
\widehat s_{t+1}^i,
s_{t+1}^i
)
\]

或：

\[
d_{t+1}^i
=
\left\|
\widehat s_{t+1}^i-s_{t+1}^i
\right\|_2
\]

所有偏差组成：

\[
D_{t+1}
=
[d_{t+1}^1,\ldots,d_{t+1}^K]
\]

抓取失败示例：

```text
预期：
夹爪上升，目标物体 slot 随夹爪共同变化。

实际：
夹爪上升，目标物体 slot 基本不变。

结果：
目标物体对应 slot 的预测偏差显著增大。
```

Failure Scorer 不需要知道“抓取失败”这一语义标签，只需要利用偏差信号判断当前轨迹是否具有较高失败风险。

---

# 8. Slot-Conditioned Failure Scorer

## 8.1 输出设计

Failure Scorer 只输出一个标量：

\[
p_{\mathrm{fail},t}
=
\sigma(z_t)
\]

不增加其他辅助分类头。

该设计适合轨迹级弱标签，并保持训练稳定。

---

## 8.2 VLA 内部动作特征

从动作解码器最后一层提取与当前 action chunk 对应的隐特征：

\[
H_t^{\mathrm{VLA}}
=
\{h_{t,1},h_{t,2},\ldots,h_{t,H}\}
\]

通过池化得到：

\[
\bar h_t
=
\operatorname{Pool}(H_t^{\mathrm{VLA}})
\]

可选池化方式：

- mean pooling；
- 最后一个动作 token；
- attention pooling；
- 对动作时间维进行一层轻量 Transformer。

第一版建议 mean pooling，便于快速实现和稳定训练。

---

## 8.3 Slot 与 VLA 特征融合

使用 VLA 动作特征查询持续 slot：

\[
C_t =
\operatorname{CrossAttn}
(
Q=H_t^{\mathrm{VLA}},
K=M_t,
V=M_t
)
\]

池化得到：

\[
\bar c_t=\operatorname{Pool}(C_t)
\]

将 slot 偏差编码为：

\[
e_t^D =
\operatorname{MLP}_D(D_t)
\]

机器人状态编码为：

\[
e_t^q =
\operatorname{MLP}_q(q_t)
\]

最终风险特征：

\[
z_t^{\mathrm{risk}}
=
[
\bar h_t,
\bar c_t,
e_t^D,
e_t^q
]
\]

Failure Scorer：

\[
p_{\mathrm{fail},t}
=
\sigma
\left(
\operatorname{MLP}_{\mathrm{risk}}
(
z_t^{\mathrm{risk}}
)
\right)
\]

---

## 8.4 弱监督标签

优先使用公开失败 rollout 或在公开环境中运行已有模型得到的成功/失败轨迹。

轨迹标签：

\[
y_\tau=
\begin{cases}
0,&\text{成功轨迹}\\
1,&\text{失败轨迹}
\end{cases}
\]

对于轨迹中的时间步，可采用 SAFE 风格的弱监督方式：

\[
y_t=y_\tau
\]

训练损失：

\[
\mathcal L_{\mathrm{fail}}
=
-\left[
y_t\log p_{\mathrm{fail},t}
+
(1-y_t)\log(1-p_{\mathrm{fail},t})
\right]
\]

可选地对轨迹后半段或失败发生附近时间步给予更高权重，但第一版先保持统一权重。

---

## 8.5 风险阈值

当：

\[
p_{\mathrm{fail},t}>\tau
\]

系统中断当前 action chunk。

阈值 \(\tau\) 在验证集上根据以下目标选择：

- 提高失败检测率；
- 控制正常轨迹误停止率；
- 保留足够的检测提前量。

第一版可直接通过验证集网格搜索：

\[
\tau\in\{0.3,0.4,0.5,0.6,0.7\}
\]

---

# 9. 恢复方案一：Slot-Guided Slow Replanning

## 9.1 核心思路

检测到失败风险后，系统进入恢复模式：

1. 中断当前 action chunk；
2. 根据任务相关性和对象异常偏差重新加权 slot；
3. 调用同一个 VLA 生成新的短 action chunk；
4. 只执行一到两步；
5. 重新观测、更新 slot 和计算 \(p_{\mathrm{fail}}\)；
6. 风险下降后继续短视野执行；
7. 连续多次低风险后返回正常模式。

该方案不需要恢复文本、恢复类型或恢复动作标签。

---

## 9.2 正常阶段的任务相关性

从动作模型交叉注意力中提取：

\[
\alpha_t^i
\]

满足：

\[
\sum_{i=1}^{K}\alpha_t^i=1
\]

\(\alpha_t^i\) 表示第 \(i\) 个 slot 对当前任务和动作的重要程度。

---

## 9.3 恢复阶段的异常权重

对象偏差为：

\[
d_t^i
\]

先做归一化：

\[
\widetilde d_t^i
=
\frac{d_t^i-\mu_D}{\sigma_D+\epsilon}
\]

恢复权重定义为：

\[
\beta_t^i
=
\operatorname{softmax}_i
\left(
\log(\alpha_t^i+\epsilon)
+
\lambda_D\widetilde d_t^i
\right)
\]

其中：

- \(\alpha_t^i\)：任务相关性；
- \(\widetilde d_t^i\)：对象异常程度；
- \(\lambda_D\)：异常偏差权重。

恢复 slot：

\[
M_t^{\mathrm{rec}}
=
\{
\beta_t^i m_t^i
\}_{i=1}^{K}
\]

也可以使用门控残差形式：

\[
\widetilde m_t^i
=
m_t^i
+
\lambda_R\beta_t^i W_Rm_t^i
\]

门控残差通常更稳定，可以保留原 slot 信息。

---

## 9.4 恢复阶段动作生成

恢复模式下重新调用动作模型：

\[
A_t^{\mathrm{rec}}
=
\pi_\theta
(
g,q_t,M_t^{\mathrm{rec}},R_t
)
\]

正常模式：

\[
H_{\mathrm{normal}}=8\sim16
\]

恢复模式：

\[
H_{\mathrm{rec}}=1\sim2
\]

如果基础 VLA 固定输出较长 action chunk，系统只执行前 \(H_{\mathrm{rec}}\) 步：

\[
A_{t,\mathrm{exec}}^{\mathrm{rec}}
=
A_t^{\mathrm{rec}}[0:H_{\mathrm{rec}}]
\]

---

## 9.5 模式切换

### 进入恢复模式

满足任一条件：

\[
p_{\mathrm{fail},t}>\tau_{\mathrm{enter}}
\]

或：

\[
\max_i d_t^i>\tau_D
\]

第一版建议主要使用 \(p_{\mathrm{fail}}\) 触发，slot 偏差只作为输入，避免双阈值引入过多超参数。

### 退出恢复模式

连续 \(N_{\mathrm{safe}}\) 次满足：

\[
p_{\mathrm{fail},t}<\tau_{\mathrm{exit}}
\]

其中：

\[
\tau_{\mathrm{exit}}<\tau_{\mathrm{enter}}
\]

形成迟滞机制，避免正常/恢复模式频繁切换。

建议：

```text
tau_enter = 0.6
tau_exit  = 0.4
N_safe    = 3
```

具体数值通过验证集调整。

---

## 9.6 方案一中 Slot 的明确作用

正常操作和恢复操作读取的 slot 不同：

### 正常模式

\[
M_t
\]

动作模型按照任务语义读取完整对象集合。

### 恢复模式

\[
M_t^{\mathrm{rec}}
\]

动作模型重点读取：

- 与任务相关的对象；
- 状态偏离预期的对象；
- 与这些对象存在关系的其他 slot。

因此，恢复阶段的 VLA 条件发生了变化。

该变化来源于对象历史和动作后果，能够缓解失败后再次产生完全相同行为的问题。

---

## 9.7 方案一优点

- 工程改动较小；
- 不需要候选动作采样；
- 不需要恢复动作标签；
- slot 在恢复中发挥直接作用；
- 适合抓取偏差、轻微滑落、目标位移和放置偏差；
- 适合两周内完成主线代码。

## 9.8 方案一局限

- 新动作仍由同一个 VLA 单次生成；
- 对系统性错误的规避能力有限；
- VLA 在异常状态分布上的能力决定恢复上限；
- 复杂失败可能连续触发恢复模式。

---

# 10. 恢复方案二：Slot-Conditioned Candidate Risk Selection

## 10.1 核心思路

方案二在方案一基础上增加少量候选动作：

1. 根据异常 slot 构建恢复表示；
2. VLA 生成 \(K_c\) 个短候选 action chunk；
3. 为每个候选提取动作相关隐特征；
4. 复用同一个 Failure Scorer 计算候选失败风险；
5. 可选地加入与上一次失败动作的相似度惩罚；
6. 选择风险最低的候选；
7. 只执行候选动作前一到两步；
8. 重新观测并再次评分。

建议候选数量：

\[
K_c=3
\]

该数量能够控制推理成本，也便于两周内完成。

---

## 10.2 候选动作生成

候选动作可以通过以下方式得到：

### 方法一：动作解码采样

从 VLA 动作分布中进行多次采样：

\[
A_t^{(k)}
\sim
\pi_\theta
(
\cdot
\mid
g,q_t,M_t^{\mathrm{rec}},R_t
)
\]

### 方法二：在动作输出附近加入小扰动

先得到主候选：

\[
A_t^{(1)}
\]

构造：

\[
A_t^{(k)}
=
A_t^{(1)}
+
\epsilon_k
\]

其中 \(\epsilon_k\) 为受限的小幅平滑扰动。

### 方法三：不同随机噪声或采样温度

如果动作模型支持扩散、流匹配或随机动作解码，可以使用不同初始噪声得到候选。

第一版优先使用模型已有的采样接口。若模型为确定性输出，则使用受限动作扰动。

---

## 10.3 候选相关隐特征

为了使同一个 Failure Scorer 能区分候选动作，评分器输入必须包含候选动作信息。

优先方案：

\[
H_t^{(k)}
=
H_{\mathrm{action}}
(
A_t^{(k)}
\mid
g,q_t,M_t^{\mathrm{rec}},R_t
)
\]

其中 \(H_t^{(k)}\) 是候选动作 token 对应的最后一层隐特征。

若当前代码无法获得候选动作对应的解码隐特征，可以增加一个轻量动作编码器：

\[
e_A^{(k)}
=
E_A(A_t^{(k)})
\]

评分器输入改为：

\[
p_{\mathrm{fail},t}^{(k)}
=
R_\phi
(
H_t^{\mathrm{context}},
M_t^{\mathrm{rec}},
D_t,
q_t,
e_A^{(k)}
)
\]

其中 \(E_A\) 可以使用两层 MLP。

---

## 10.4 复用同一个 Failure Scorer

正常执行时：

\[
p_{\mathrm{fail},t}
=
R_\phi
(
H_t^{\mathrm{exec}},
M_t,
D_t,
q_t
)
\]

恢复候选评分时：

\[
r_t^{(k)}
=
R_\phi
(
H_t^{(k)},
M_t^{\mathrm{rec}},
D_t,
q_t
)
\]

同一个评分器承担两个任务：

1. 判断当前准备执行或正在执行的动作是否具有较高失败风险；
2. 比较多个恢复候选动作的相对风险。

该统一形式是方案二的主要创新点。

---

## 10.5 候选动作评分

基础评分：

\[
J_k=r_t^{(k)}
\]

选择：

\[
k^*=\arg\min_k J_k
\]

执行：

\[
A_t^*=A_t^{(k^*)}
\]

---

## 10.6 失败动作重复惩罚

为了降低恢复时重复上一次失败动作的概率，记录失败 action chunk：

\[
A_t^{\mathrm{failed}}
\]

定义相似度：

\[
\operatorname{Sim}
(
A_t^{(k)},
A_t^{\mathrm{failed}}
)
=
\exp
\left(
-\frac{
\|A_t^{(k)}-A_t^{\mathrm{failed}}\|_2^2
}{
\sigma_A^2
}
\right)
\]

最终评分：

\[
J_k
=
r_t^{(k)}
+
\lambda_{\mathrm{repeat}}
\operatorname{Sim}
(
A_t^{(k)},
A_t^{\mathrm{failed}}
)
\]

选择：

\[
k^*=\arg\min_k J_k
\]

第一版可以将该项作为可选组件，并设计消融实验。

---

## 10.7 方案二中的短视野闭环

即使候选动作长度为 \(H\)，恢复阶段只执行前：

\[
H_{\mathrm{rec}}=1\sim2
\]

步。

执行后重新计算：

\[
I_{t+\Delta}
\rightarrow
S_{t+\Delta}
\rightarrow
M_{t+\Delta}
\rightarrow
D_{t+\Delta}
\rightarrow
p_{\mathrm{fail},t+\Delta}
\]

这样可避免一次候选评分错误导致长时间偏离。

---

## 10.8 方案二训练问题

方案二不要求额外的候选动作优劣标签。

Failure Scorer 仍然通过成功/失败 rollout 训练。

部署时利用其风险输出进行候选排序。

如果后续有充足时间，可以加入排序损失：

\[
\mathcal L_{\mathrm{rank}}
=
\max
\left(
0,
m+r^+-r^-
\right)
\]

其中：

- \(r^+\)：失败动作风险；
- \(r^-\)：成功动作风险；
- \(m\)：间隔。

两周版本不将排序损失设为必要模块。

---

## 10.9 方案二优点

- 贡献二和贡献三共享同一个风险模型；
- 能够降低单次 VLA 重规划重复失败的概率；
- slot 同时参与候选生成和候选评分；
- 不需要人工恢复标签；
- 候选数量较少，工程量可控。

## 10.10 方案二局限

- 推理成本约为单次 VLA 的 \(K_c\) 倍；
- 候选动作多样性影响恢复效果；
- Failure Scorer 通过轨迹弱标签训练，其候选排序能力需要实验验证；
- 如果候选动作都属于同一种系统性错误，风险排序无法产生有效恢复动作。

---

# 11. 两种恢复方案的关系

方案二完整继承方案一：

```text
方案一：
异常 slot 重加权
+ 短 action chunk
+ 高频闭环重规划

方案二：
异常 slot 重加权
+ 少量候选动作
+ 同一 Failure Scorer 排序
+ 可选失败动作重复惩罚
+ 短 action chunk
+ 高频闭环重规划
```

建议论文和代码中将二者定义为：

- **基础恢复模块**：方案一；
- **风险选择增强模块**：方案二。

实验中可以逐步增加组件：

1. 原始 Embodied-SlotSSM；
2. 检测失败后直接重调 VLA；
3. 方案一；
4. 方案二；
5. 方案二 + 重复动作惩罚。

---

# 12. 完整执行算法

## 12.1 正常与恢复统一伪代码

```python
mode = "normal"
safe_count = 0
failed_action = None

for t in episode:
    image = env.get_rgb()
    robot_state = env.get_robot_state()

    patch_features = vision_encoder(image)

    slots = slot_attention(
        patch_features,
        prev_slots=slots
    )

    slot_memory = slot_ssm(
        slots,
        prev_state=slot_memory
    )

    relation_tokens = relation_encoder(
        slot_memory,
        patch_features
    )

    # 使用上一动作预测当前 slot，并得到对象偏差
    if previous_action is not None:
        predicted_slots = slot_predictor(
            previous_slot_memory,
            previous_action,
            previous_robot_state
        )
        slot_deviation = compute_slot_deviation(
            predicted_slots,
            slots
        )
    else:
        slot_deviation = zeros(num_slots)

    if mode == "normal":
        action_chunk, action_hidden = vla(
            instruction,
            robot_state,
            slot_memory,
            relation_tokens
        )

        p_fail = failure_scorer(
            action_hidden,
            slot_memory,
            slot_deviation,
            robot_state
        )

        if p_fail <= tau_enter:
            execute(action_chunk[:H_normal])
        else:
            mode = "recovery"
            failed_action = action_chunk

    if mode == "recovery":
        task_attention = extract_slot_attention(action_hidden)

        recovery_slots = reweight_slots(
            slot_memory,
            task_attention,
            slot_deviation
        )

        if recovery_method == "scheme_1":
            recovery_action, recovery_hidden = vla(
                instruction,
                robot_state,
                recovery_slots,
                relation_tokens
            )
            selected_action = recovery_action

        elif recovery_method == "scheme_2":
            candidate_actions = sample_candidates(
                vla,
                instruction,
                robot_state,
                recovery_slots,
                relation_tokens,
                num_candidates=3
            )

            candidate_scores = []

            for candidate in candidate_actions:
                candidate_hidden = get_candidate_hidden(
                    candidate,
                    instruction,
                    robot_state,
                    recovery_slots,
                    relation_tokens
                )

                risk = failure_scorer(
                    candidate_hidden,
                    recovery_slots,
                    slot_deviation,
                    robot_state
                )

                repeat_penalty = action_similarity(
                    candidate,
                    failed_action
                )

                score = risk + lambda_repeat * repeat_penalty
                candidate_scores.append(score)

            selected_action = candidate_actions[argmin(candidate_scores)]

        execute(selected_action[:H_recovery])

        # 下一次循环更新观测并重新评分
        if p_fail < tau_exit:
            safe_count += 1
        else:
            safe_count = 0

        if safe_count >= N_safe:
            mode = "normal"
            safe_count = 0
            failed_action = None

    previous_action = selected_action or action_chunk
    previous_slot_memory = slot_memory
    previous_robot_state = robot_state
```

实际实现时，模式退出判断需要使用执行新动作后的下一帧风险值。

---

# 13. 训练方案

## 13.1 阶段一：复现 Embodied-SlotSSM

目标：

- 确认公开代码能够训练和推理；
- 确认 slot、SlotSSM、Relation Encoder 和动作解码器接口；
- 确认正常任务成功率达到合理水平。

损失沿用原方法：

\[
\mathcal L_{\mathrm{base}}
=
\mathcal L_{\mathrm{action}}
+
\lambda_{\mathrm{slot}}\mathcal L_{\mathrm{slot}}
+
\lambda_{\mathrm{temp}}\mathcal L_{\mathrm{temporal}}
\]

第一阶段尽量不修改主干。

---

## 13.2 阶段二：训练动作条件化 slot 预测头

冻结或部分冻结主干：

\[
\mathcal L_{\mathrm{pred}}
=
\mathcal L_{\mathrm{slot\_pred}}
\]

训练数据直接来自公开成功轨迹和失败 rollout 中的：

\[
(S_t,A_t,S_{t+1})
\]

优先先在成功轨迹上训练，使模型学会正常动作下的对象变化。

随后可混入失败轨迹，使预测器适应更广泛状态。

---

## 13.3 阶段三：训练 Failure Scorer

冻结：

- 视觉编码器；
- Slot Attention；
- SlotSSM；
- 基础 VLA 主干。

训练：

- slot/VLA 特征融合层；
- 偏差编码器；
- Failure Scorer MLP。

损失：

\[
\mathcal L_{\mathrm{fail}}
=
\operatorname{BCE}
(
p_{\mathrm{fail}},y
)
\]

为了处理成功/失败数据不平衡，可以使用：

- class weight；
- balanced sampler；
- focal loss。

第一版优先使用 weighted BCE。

---

## 13.4 阶段四：恢复模块

### 方案一

无需额外训练。

只需实现：

- slot 任务相关性提取；
- slot 偏差重加权；
- 正常/恢复模式切换；
- action chunk 截断；
- 高频闭环执行。

### 方案二

优先无需额外训练。

只需实现：

- 候选动作采样；
- 候选相关隐特征提取或动作编码；
- Failure Scorer 复用；
- 候选动作风险排序；
- 可选的动作相似度惩罚。

---

# 14. 总损失函数

主干联合训练时可写为：

\[
\mathcal L
=
\mathcal L_{\mathrm{action}}
+
\lambda_{\mathrm{slot}}
\mathcal L_{\mathrm{slot}}
+
\lambda_{\mathrm{temp}}
\mathcal L_{\mathrm{temporal}}
+
\lambda_{\mathrm{pred}}
\mathcal L_{\mathrm{slot\_pred}}
+
\lambda_{\mathrm{fail}}
\mathcal L_{\mathrm{fail}}
\]

两周版本建议分阶段训练，降低梯度相互干扰。

推荐顺序：

1. 加载或复现 Embodied-SlotSSM；
2. 冻结主干训练 slot predictor；
3. 冻结主干和 predictor 训练 Failure Scorer；
4. 部署方案一；
5. 在方案一稳定后部署方案二。

---

# 15. 数据使用方案

## 15.1 不建立新数据集

本项目不设计新的任务集合，也不进行大规模人工标注。

优先使用：

- Embodied-SlotSSM 对应的公开数据和训练设置；
- LIBERO-Mem 等公开操作数据；
- SAFE 或其他公开失败 rollout；
- 公开仿真环境中由已有策略运行得到的成功/失败轨迹。

在公开环境中运行当前模型并保存 rollout，属于模型评测和训练样本生成，不需要设计新的数据集协议。

---

## 15.2 训练数据需要保存的字段

每个时间步建议保存：

```text
episode_id
task_id
rgb
robot_state
language_instruction
action_chunk
executed_action
slot_features
slot_memory
relation_features
vla_action_hidden
next_slot_features
success_label
done
```

失败恢复评测还需保存：

```text
p_fail
recovery_mode
slot_deviation
slot_attention
selected_candidate
candidate_scores
failed_action
recovery_step_count
```

---

## 15.3 成功/失败标签

优先使用环境自动成功判定：

\[
y_{\mathrm{episode}}\in\{0,1\}
\]

Failure Scorer 标签可转换为：

\[
y_{\mathrm{fail}}=1-y_{\mathrm{success}}
\]

如果公开 rollout 提供明确失败时间，可以使用更精确的时间标签。

如果只提供轨迹级标签，保持 SAFE 风格弱监督。

---

# 16. 实验设计

## 16.1 研究问题一：Slot 是否提高正常操作？

对比：

1. 原始 VLA；
2. Embodied-SlotSSM；
3. Embodied-SlotSSM + 当前项目的 slot adapter；
4. Embodied-SlotSSM + slot predictor。

指标：

- 正常任务成功率；
- 对象位置变化后的成功率；
- 干扰物体增加后的成功率；
- 遮挡情况下的成功率；
- 长序列任务成功率。

该实验主要验证 slot 对正常操作的贡献。

---

## 16.2 研究问题二：Slot 是否提高失败检测？

对比：

1. 图像特征 + MLP；
2. VLA 隐特征 + MLP，接近 SAFE；
3. slot 特征 + MLP；
4. VLA 隐特征 + slot；
5. VLA 隐特征 + slot + robot state；
6. VLA 隐特征 + slot + robot state + slot deviation。

指标：

- AUROC；
- AUPRC；
- F1；
- 失败检测率；
- 正常轨迹误停止率；
- 平均检测提前量；
- 不同失败类型上的检测性能。

Failure Scorer 仍然只输出单一 \(p_{\mathrm{fail}}\)。

---

## 16.3 研究问题三：Slot 是否提高恢复成功率？

对比：

1. 不恢复，继续执行当前 action chunk；
2. 中断后直接重新调用原 VLA；
3. 直接重新调用原 VLA + 短 action chunk；
4. 方案一：slot 重加权 + 短视野闭环；
5. 方案二：方案一 + 候选风险选择；
6. 方案二 + 重复动作惩罚。

指标：

- 条件恢复成功率；
- 最终任务成功率；
- 平均恢复次数；
- 平均额外动作步数；
- 平均 VLA 调用次数；
- 相同失败重复率；
- 恢复后的二次失败率；
- 恢复推理时间。

---

## 16.4 关键消融

### Slot 时间建模

- 单帧 slot；
- 时间持续 slot；
- 时间持续 slot + 动作条件化预测。

### Failure Scorer 输入

- 只用 VLA 隐特征；
- VLA 隐特征 + slot；
- VLA 隐特征 + slot + deviation；
- 完整输入。

### 恢复阶段 slot 权重

- 不重加权；
- 只使用任务相关性 \(\alpha\)；
- 只使用对象偏差 \(d\)；
- 使用任务相关性和对象偏差 \(\beta\)。

### 执行视野

\[
H_{\mathrm{rec}}\in\{1,2,4,8\}
\]

### 候选数量

\[
K_c\in\{1,3,5\}
\]

---

# 17. 预期贡献

## 贡献一：面向正常操作与失败恢复的统一持续对象表示

基于 Embodied-SlotSSM 建立持续对象 slot，动作模型在正常阶段读取完整对象集合，恢复阶段利用对象异常偏差调整对象注意力。

## 贡献二：Slot 条件化的单标量失败评分器

融合 VLA 动作隐特征、持续对象 slot、机器人内部状态和动作条件化对象预测偏差，只使用弱监督标签输出：

\[
p_{\mathrm{fail}}
\]

## 贡献三：基于统一风险评分的对象级恢复

方案一利用异常 slot 引导短视野闭环重规划。

方案二进一步复用同一个 Failure Scorer 对多个候选恢复动作进行风险排序，使失败检测模块同时承担恢复动作选择功能。

---

# 18. 两周实现计划

## 第 1～2 天：梳理和跑通 Embodied-SlotSSM

完成：

- 环境安装；
- 数据加载；
- 模型推理；
- 获取 slot；
- 获取 SlotSSM 状态；
- 获取 Relation Encoder 输出；
- 获取 VLA 动作隐特征；
- 确认 action chunk 执行接口。

输出：

```text
smoke test
slot shape log
action hidden shape log
单条 rollout
```

---

## 第 3～4 天：实现动作条件化 Slot Predictor

完成：

- action encoder；
- robot state encoder；
- slot prediction head；
- slot deviation 计算；
- predictor 训练脚本；
- deviation 可视化。

验证：

- 成功轨迹中预测误差较低；
- 人工扰动或失败轨迹中误差上升；
- 不同 slot 的偏差存在差异。

---

## 第 5～7 天：实现 Failure Scorer

完成：

- VLA hidden pooling；
- slot cross-attention；
- deviation encoder；
- robot state encoder；
- Failure Scorer MLP；
- 成功/失败训练；
- AUROC、F1、阈值评测。

首要对比：

```text
SAFE-style hidden only
vs.
hidden + slot
vs.
hidden + slot + deviation
```

---

## 第 8～9 天：实现恢复方案一

完成：

- 正常/恢复状态机；
- action chunk 中断；
- slot 任务相关性提取；
- slot deviation 重加权；
- 短 chunk 执行；
- 高频重新观测；
- 恢复退出条件。

验证：

- 抓取偏移；
- 轻微滑落；
- 目标物体小范围移动；
- 放置偏移。

---

## 第 10～11 天：实现恢复方案二

完成：

- 三候选生成；
- 候选 hidden 提取；
- 同一 Failure Scorer 评分；
- 低风险候选选择；
- 可选重复动作惩罚；
- 候选评分日志。

如果候选 hidden 获取困难，使用动作编码器作为备选实现。

---

## 第 12～13 天：实验和消融

优先完成：

1. 正常操作成功率；
2. 失败检测对比；
3. 直接重试与方案一对比；
4. 方案一与方案二对比；
5. slot deviation 消融；
6. 短 chunk 消融。

---

## 第 14 天：整理结果和代码

完成：

- 配置文件；
- README；
- 推理命令；
- 训练命令；
- 评测命令；
- 结果表；
- 框架图；
- 失败恢复案例可视化；
- 论文方法部分初稿。

---

# 19. 代码模块建议

```text
project/
├── models/
│   ├── embodied_slotssm.py
│   ├── slot_predictor.py
│   ├── failure_scorer.py
│   ├── slot_reweight.py
│   └── candidate_action_encoder.py
│
├── recovery/
│   ├── recovery_controller.py
│   ├── scheme1_slow_replan.py
│   ├── scheme2_candidate_risk.py
│   └── action_similarity.py
│
├── rollout/
│   ├── collect_rollouts.py
│   ├── normal_rollout.py
│   ├── recovery_rollout.py
│   └── rollout_buffer.py
│
├── train/
│   ├── train_slot_predictor.py
│   ├── train_failure_scorer.py
│   └── train_base_policy.py
│
├── eval/
│   ├── eval_normal_success.py
│   ├── eval_failure_detection.py
│   ├── eval_recovery.py
│   └── metrics.py
│
├── configs/
│   ├── base.yaml
│   ├── failure_scorer.yaml
│   ├── recovery_scheme1.yaml
│   └── recovery_scheme2.yaml
│
└── scripts/
    ├── smoke_slot_features.py
    ├── smoke_failure_scorer.py
    ├── smoke_recovery_scheme1.py
    └── smoke_recovery_scheme2.py
```

---

# 20. 第一版超参数建议

```yaml
num_slots: 8
slot_dim: follow_embodied_slotssm
normal_chunk_len: 8
recovery_exec_len: 1
num_candidates: 3

failure_threshold_enter: 0.6
failure_threshold_exit: 0.4
safe_steps_to_exit: 3

slot_deviation_weight: 1.0
recovery_residual_weight: 0.5
repeat_action_penalty: 0.1

failure_scorer_hidden_dim: 256
failure_scorer_num_layers: 2
dropout: 0.1

slot_predictor_hidden_dim: 256
action_embedding_dim: 128
robot_state_embedding_dim: 64
```

这些参数只作为初始值，最终通过验证集调整。

---

# 21. 风险与降级方案

## 21.1 Slot 身份不稳定

表现：

- 相邻帧对象 slot 交换；
- slot deviation 大量误报；
- 恢复权重跳变。

处理：

- 使用 Embodied-SlotSSM 的时间初始化和对比损失；
- 对 slot deviation 做指数平滑；
- 使用 Hungarian matching 对齐预测 slot 和真实 slot；
- 恢复权重使用残差门控，避免完全覆盖原 slot。

---

## 21.2 Failure Scorer 只学习轨迹难度

表现：

- 某些困难任务从一开始就输出高风险；
- 缺少明确的失败时间变化。

处理：

- 对同一任务内成功与失败轨迹进行平衡采样；
- 加入时间窗口；
- 对失败轨迹靠近结束的帧提高采样概率；
- 评估相对风险变化：

\[
\Delta p_t
=
p_{\mathrm{fail},t}
-
p_{\mathrm{fail},t-\Delta}
\]

---

## 21.3 Slot Predictor 对正常轨迹拟合过强

表现：

- 失败状态中所有 slot 都出现大偏差；
- 无法定位关键对象。

处理：

- 对偏差做任务相关性加权；
- 只使用 top-\(k\) 异常 slot；
- 使用标准化偏差；
- 混入公开失败 rollout 训练 predictor。

---

## 21.4 方案一重复失败

处理：

- 降低恢复 action chunk 长度；
- 增加恢复阶段调用频率；
- 提高异常 slot 权重；
- 启用方案二候选筛选；
- 记录并惩罚上一失败动作。

---

## 21.5 方案二候选缺乏多样性

处理：

- 使用不同采样噪声；
- 在动作空间加入受限扰动；
- 对末端位置、旋转和夹爪动作分别扰动；
- 增加候选差异性约束；
- 从 3 个候选增加到 5 个候选。

---

# 22. 最小可行版本

考虑两周开发周期，必须优先完成：

\[
\boxed{
\text{Embodied-SlotSSM}
+
\text{动作条件化 Slot Predictor}
+
\text{单标量 Failure Scorer}
+
\text{恢复方案一}
}
\]

方案二作为增强版本，在方案一稳定后加入。

最小可行实验包括：

1. Embodied-SlotSSM 正常成功率；
2. SAFE 风格隐特征基线；
3. 隐特征 + slot；
4. 隐特征 + slot + deviation；
5. 直接重试；
6. 方案一恢复；
7. 可选的方案二恢复。

---

# 23. 最终方法表述

本项目提出一个基于 Embodied-SlotSSM 的失败感知与恢复框架。系统利用持续对象 slot 为正常动作生成提供对象级表征，并通过动作条件化 slot 预测获得对象状态偏差。一个轻量 Failure Scorer 融合 VLA 动作隐特征、持续对象 slot、机器人内部状态和 slot 偏差，只输出单一失败概率。检测到高失败风险后，系统中断当前 action chunk。

恢复方案一根据任务相关性和对象异常程度重新加权 slot，并采用短 action chunk 和高频闭环重新规划。

恢复方案二在方案一基础上生成少量候选动作，并复用同一个 Failure Scorer 对候选动作进行风险排序，选择低风险且与历史失败动作差异较大的候选执行。

整个方法不依赖失败类型、恢复文本或结构化恢复目标标签，能够在公开数据和公开仿真环境的弱监督条件下实现。
