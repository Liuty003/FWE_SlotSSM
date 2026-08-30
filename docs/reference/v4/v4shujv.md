可以。你现在最值得做的不是继续写“数据采集脚本”，而是做一个统一的：

[
\boxed{
\text{Counterfactual Recovery Data Factory}
}
]

以后无论 42 条、420 条还是 4000 条，流程都应该变成：

[
\boxed{
\text{OFT rollout}
\rightarrow
\text{自动挖 failure}
\rightarrow
\text{自动恢复 fork}
\rightarrow
\text{自动生成 teacher}
\rightarrow
\text{自动审计 handoff}
\rightarrow
\text{自动写训练集}
}
]

人工只负责**抽查质量和处理少量异常 case**，不能负责逐条裁剪。

---

# 一、我建议把数据生成拆成 4 个自动模块

## 1. Failure Miner：自动找自然失败和 (t_f)

你现在已经有完整 OFT rollout，所以不要人工看视频定位。

对每条轨迹离线计算一组通用事件：

[
E_t=
{
E_t^{grasp},
E_t^{slip},
E_t^{place},
E_t^{collateral},
E_t^{stall}
}.
]

然后自动得到：

[
\boxed{
t_f=
\min{t:E_t^{failure}=1}
}
]

例如抓空可以定义成：

[
\text{close occurred}
\land
\Delta z_{eef}>d_{lift}
\land
\Delta z_{obj}<\epsilon
]

持续 (K) 帧后确认：

[
E_t^{grasp}=1.
]

滑落：

[
|p_t^{obj}-p_t^{eef}|
---------------------

|p_{t-k}^{obj}-p_{t-k}^{eef}|

> \epsilon.
> ]

stall：

[
\underbrace{
|\phi(x_t)-\phi(x_{t-k})|<\epsilon_p
}*{\text{任务几何没进展}}
\land
\underbrace{
\sum*{i=t-k}^{t}|a_i|>\epsilon_a
}_{\text{机器人一直在动作}}
]

就标记为失败。

所以每条 rollout 自动输出：

```text
task_id
source_id
stage
failure_type
failure_onset
nearest_snapshot
replay_start
confidence
```

人工只看例如 10%：

[
\boxed{\text{automatic mining + spot check}}
]

这一步应该彻底自动化。

---

# 二、Snapshot 也不用人工裁

你现在每 8 步已经有 MuJoCo snapshot，这非常适合自动恢复精确 failure state。

假设检测：

[
t_f=137
]

最近的 checkpoint 是：

[
t_s=136.
]

程序自动：

[
x_{136}
\xrightarrow{
a_{136}^{OFT}
}
x_{137}.
]

所以：

[
\boxed{
x_{t_f}
=

\operatorname{Replay}
(
x_{t_s},
a_{t_s:t_f-1}
)
}
]

这样任何 failure onset 都可以自动重建。

以后几千条任务也是：

```text
failure miner
      ↓
t_f
      ↓
nearest snapshot
      ↓
replay OFT actions
      ↓
exact failure fork
```

不需要人工存一个个 failure snapshot。

---

# 三、Teacher 也应该做成自动 branch generator

这里你已经有很好的基础。

对于：

[
x_f
]

根据 failure type 自动调用相应 teacher primitive。

例如：

[
\text{failed grasp}
\rightarrow
{
\text{retract},
\text{realign},
\text{descend},
\text{close},
\text{lift}
}
]

生成多个几何候选：

[
\mathcal B(x_f)
===============

{
\tau_f^1,\tau_f^2,\ldots,\tau_f^M
}.
]

例如偏移：

[
\Delta p_i
\in
{
(0,0),
(\pm d_x,0),
(0,\pm d_y)
}.
]

然后自动执行全部 branch。

只保留满足：

[
\boxed{
\text{local recovery predicate}=1
}
]

并且：

[
\boxed{
\text{strict replay}=1
}
]

的 branch。

所以：

```text
failure fork
   ↓
自动生成 M 个 teacher candidates
   ↓
并行执行
   ↓
success filter
   ↓
strict replay filter
   ↓
candidate teacher pool
```

这里也不应该人工选择“这条看起来比较好”。

---

# 四、最关键：Handoff 不要逐帧人工找

这个地方最容易把你拖死。

应该做成 **coarse-to-fine automatic handoff audit**。

假设 teacher trajectory 长：

[
T=64.
]

不要一开始测：

[
0,1,2,\cdots,64.
]

先粗采样：

[
t=
0,8,16,24,32,40,48,56,64.
]

每个状态自动：

[
x_t
\xrightarrow{\pi_{OFT}}
success/failure.
]

得到例如：

[
1,\ 1,\ 1,\ 1,\ 0,\ 1,\ 0,\ 0,\ 0.
]

这里：

* (1)：Recovery 还应该控制；
* (0)：OFT 可以接管。

然后只对边界附近加密。

例如：

[
24\rightarrow1,\quad32\rightarrow0
]

就审计：

[
25,26,\ldots,32.
]

同时：

[
32\rightarrow0,\quad40\rightarrow1
]

说明这里存在非单调区域，也加密。

最终自动获得：

[
\boxed{
{(x_t,y_t^{handoff})}
}
]

完全不需要手动“裁 handoff”。

---

# 五、千万别直接对 handoff 做二分搜索

因为你已经发现：

[
1,1,0,1,0,0
]

这种情况存在。

所以 resumability：

[
R(x_t)
]

并不保证：

[
R(x_t)
\le
R(x_{t+1}).
]

因此严格二分搜索不成立。

正确方法是：

[
\boxed{
\text{coarse scan}
\rightarrow
\text{transition detection}
\rightarrow
\text{local dense audit}
}
]

这样既省 rollout，又保留非单调性。

---

# 六、甚至可以自动找“稳定 handoff”

我建议最终 Recovery action 不一定裁到第一个成功点。

定义 audit：

[
r_t=
\begin{cases}
1,&OFT\text{ 能完成}\
0,&OFT\text{ 失败}
\end{cases}
]

然后定义稳定 handoff：

[
\boxed{
t_h
===

\min
\left{
t:
\sum_{j=0}^{K-1}r_{t+j}=K
\right}
}
]

例如：

[
K=3.
]

如果：

[
0,0,\boxed{1},0,1,1,1
]

第一个 resumable state 不作为 action trajectory 终点。

真正 handoff：

[
\boxed{
1,1,1
}
]

这一段的第一个位置。

这也是程序自动完成。

---

# 七、规模再大以后，连 handoff audit 数量都能继续减少

第一批 20 个 failure sources 全部 coarse-to-fine audit。

积累一些数据后，训练一个 preliminary gate：

[
\hat q_\psi(x_t)
================

P(
\pi_{OFT}\text{ 可接管}
).
]

下一批 teacher trajectory 不需要平均审计所有位置。

优先 audit：

### 1. Gate 最不确定的

[
\left|
\hat q_t-0.5
\right|
\rightarrow0.
]

### 2. 预测发生 ownership transition 的

[
|\hat q_t-\hat q_{t-k}|>\delta.
]

### 3. teacher 几何关键阶段

例如：

[
\text{close},
\text{lift},
\text{object stabilized}.
]

这样变成一个 **active handoff auditing**：

[
\boxed{
\text{cheap gate prediction}
\rightarrow
\text{select informative states}
\rightarrow
\text{expensive OFT rollout audit}
}
]

但注意：最终监督标签仍然来自真实：

[
x_t\xrightarrow{\pi_{OFT}}\text{outcome}.
]

Gate 只用于决定**哪里值得花计算量审计**。

---

# 八、Normal 数据更容易，根本不用专门采

Normal negative 可以直接从现有 rollout 自动抽。

不要存 20 条轨迹中的所有几万帧。

对任务阶段分层：

[
s_t\in
{
\text{approach mug},
\text{grasp mug},
\text{transport mug},
\ldots
}.
]

然后每个 source、每个阶段抽：

[
N_s
]

个窗口。

即：

[
\boxed{
P(x\mid normal)
===============

\sum_s
P(x\mid s)P(s)
}
]

人为控制 (P(s))，保证不同 task phase 相对均衡。

这样避免：

```text
大量 approach frames
淹没
grasp / switch / place frames
```

---

# 九、我还建议你稍微改一下数据格式：不要真的维护“四个独立数据集”

底层只维护一个统一的 **state-centric dataset**。

每一 row：

[
D_i=
(
source_i,
t_i,
z_i,
y_i^{gate},
m_i^{action},
A_i^R,
audit_i
).
]

其中：

[
m_i^{action}\in{0,1}
]

决定有没有 recovery action supervision。

例如：

| 状态                | (y^{gate}) | (m^{action}) |
| ----------------- | ---------: | -----------: |
| normal            |          0 |            0 |
| natural failure   |          1 |          0/1 |
| teacher recovery  |          1 |            1 |
| resumable         |          0 |            0 |
| DAgger correction |          1 |            1 |

这样训练代码只需要：

[
\mathcal L
==========

\lambda_g
\mathcal L_g
+
\lambda_a
m_i^{action}
\mathcal L_a.
]

数据生成复杂，但训练数据接口非常简单。

这个结构会优雅很多。

---

# 十、DAgger 也可以完全自动成为一个 queue

训练出 checkpoint (k) 后：

[
\pi_R^{(k)}
]

自动在 train sources 上 rollout。

检测异常：

[
\mathcal Q=
{
x:
\text{recovery failed}
\lor
g_t\text{ uncertain}
\lor
\text{handoff failed}
}.
]

把状态进入一个 DAgger queue：

```text
model rollout
    ↓
failure / deviation detector
    ↓
DAgger queue
    ↓
restore state
    ↓
teacher correction
    ↓
handoff audit
    ↓
dataset merge
```

于是：

[
\boxed{
\mathcal D_{k+1}
================

\mathcal D_k
\cup
\operatorname{Annotate}
(
\mathcal Q_k
)
}
]

后续扩规模几乎就是算力问题，而不是人工问题。

---

# 十一、最终我建议你真正实现的是这套流水线

```text
                    OFT rollouts
                         │
                         ▼
                ① Failure Miner
                         │
                 failure manifest
                         │
                         ▼
              ② Fork Reconstructor
                         │
                exact failure state
                         │
                         ▼
             ③ Teacher Branch Generator
                         │
                  M candidates
                         │
                success/replay filter
                         │
                         ▼
             ④ Handoff Audit Engine
                         │
               coarse scan every 8
                         │
                transition detection
                         │
                  local dense scan
                         │
                         ▼
                certified recovery
                         │
                         ▼
              ⑤ Dataset Builder
              ┌──────────┼──────────┐
              │          │          │
           normal      gate       action
           rows         rows        rows
                         │
                         ▼
                    train V4.3
                         │
                         ▼
                   online rollout
                         │
                         ▼
                  ⑥ DAgger Queue
                         │
                         └──────→ teacher/audit
```

---

## 十二、你们现在最值得写的代码

我甚至建议**先不要继续写训练代码**。

先把下面 5 个东西做成稳定接口：

[
\boxed{
\begin{aligned}
1.&\quad \texttt{mine_natural_failures.py}\
2.&\quad \texttt{restore_failure_fork.py}\
3.&\quad \texttt{generate_recovery_branches.py}\
4.&\quad \texttt{audit_oft_resumability.py}\
5.&\quad \texttt{build_ownership_dataset.py}
\end{aligned}
}
]

整个过程做到：

```bash
python build_recovery_dataset.py \
    --rollout_root ... \
    --split train
```

最后自动吐出：

```text
failure_manifest.json
teacher_manifest.json
handoff_manifest.json
gate_dataset.jsonl
action_dataset.jsonl
quality_report.json
```

那么你以后把：

[
42
]

换成：

[
4200
]

理论上只需要增加机器运行时间。

---

## 我认为这里最重要的研究设计变化

你原来的思维还是：

> “我需要收集这些类型的数据。”

更适合规模化的方法是：

[
\boxed{
\text{我定义什么事件能够自动产生这些监督？}
}
]

对应关系就是：

[
\begin{aligned}
\text{Natural rollout}
&\xrightarrow{\text{causal event detector}}
t_f
\
x_f
&\xrightarrow{\text{teacher search}}
\tau_R
\
x_t\in\tau_R
&\xrightarrow{\text{counterfactual OFT rollout}}
y_t^{ownership}
\
\pi_R\text{ rollout}
&\xrightarrow{\text{deviation mining}}
\text{DAgger states}.
\end{aligned}
]

这样你的数据集其实成了**方法的一部分**：一个自动产生 recovery supervision 和 controller-ownership supervision 的 counterfactual pipeline。

这个方向我认为比“人工整理 20 条 teacher，再训练模型”强很多，也更适合后面扩到几百上千个 rollout。
