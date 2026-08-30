可以。我们把现在的方法彻底定型，不再来回增加模块。

我建议最终采用一个**最小但逻辑闭环的版本**：

> **冻结 OpenVLA-OFT 作为正常策略；用 Action-aware SlotSSM 建立对象级时序状态；Gate 判断是否需要恢复；Recovery Head 直接从 SlotSSM 生成恢复动作。训练时增加 Future-Slot Prediction，专门强迫 Slot 表示和机器人动作发生对应。**

其中 **Future Prediction Head 只在训练时存在，部署时删除**。Candidate Verifier、Dynamics rollout、FSM 都不进入最终模型。

---

# 0. 先看最简单的模型结构

你当前每次策略决策输出一个 (H=8) 步 action chunk。模型可以画成：

```text
 RGB + Wrist RGB + Instruction + Proprio + Previous Action
                         │
                         ▼
                ┌─────────────────┐
                │  OpenVLA-OFT    │   ← 冻结
                └─────────────────┘
                    │          │
                    │          └──────────────► Normal Action
                    │                              Aᴺ
                    ▼
                Visual Tokens
                    │
                    ▼
              Slot Attention
                    │
                 z₁ ... zK
                    │
                    ▼
          ┌────────────────────┐
          │ Action-aware       │ ◄── Previous Action
          │ SlotSSM            │ ◄── Proprio
          └────────────────────┘
                    │
              S = {s₁...sK}
               /            \
              /              \
             ▼                ▼
      Failure Gate      Cross-Attention
             │          OFT feature ↔ Slots
             │                │
             │                ▼
             │          Recovery Head
             │                │
             │                ▼
             │          Recovery Action
             │                Aᴿ
             │
             ▼
      normal / recovery
             │
             ▼
       最终 Action Chunk
             │
             ▼
        LIBERO 执行
             │
             ▼
        下一次重新观测
```

训练的时候额外挂一条：

```text
                  Sₙ
                   │
            + Ground-truth
             Action Chunk
                   │
                   ▼
        ┌──────────────────┐
        │ Future Prediction│   ← 只训练时存在
        └──────────────────┘
                   │
                   ▼
             predicted Sₙ₊₁
                   │
                 Loss
                   │
                   ▼
               true Sₙ₊₁
```

这条支路的唯一任务就是：

[
\boxed{
\text{让 SlotSSM 必须理解：
机器人执行这个 Action 后，对象应该怎样变化}
}
]

---

# 1. 首先统一符号

为了后面公式不乱，我定义一次。

| 符号                           | 含义                                          |
| ---------------------------- | ------------------------------------------- |
| (n)                          | 第 (n) 次策略决策                                 |
| (H)                          | action chunk 长度，你当前设定 (H=8)                 |
| (I_n^m)                      | 第 (n) 次决策的主相机 RGB                           |
| (I_n^w)                      | wrist camera RGB                            |
| (l)                          | language instruction                        |
| (p_n)                        | 当前 proprioception，例如 EEF pose、gripper state |
| (A_n)                        | 第 (n) 次决策输出的整个 action chunk                 |
| (a_{n,h})                    | action chunk 中第 (h) 个低层动作                   |
| (h_n)                        | OpenVLA-OFT 输出的融合视觉语言 feature/token         |
| (V_n)                        | 从 OFT 中取出的视觉 tokens                         |
| (K)                          | Slot 数量                                     |
| (z_n^k)                      | 当前图像提取出的第 (k) 个 instantaneous slot          |
| (s_n^k)                      | SlotSSM 维护的第 (k) 个时序 object state           |
| (S_n)                        | 所有 SlotSSM states 的集合                       |
| (g_n)                        | Failure Gate 输出的 failure/recovery 概率        |
| (\tau)                       | Gate threshold                              |
| (A_n^N)                      | OFT 正常动作                                    |
| (A_n^R)                      | Recovery Head 输出的恢复动作                       |
| (^\star)                     | ground-truth / teacher supervision          |
| (\theta,\phi,\psi,\rho,\eta) | 不同网络模块的可训练参数                                |

后面全部按照这些符号来。

---

# 2. 第一步：OpenVLA-OFT 做正常策略

输入：

[
O_n=
{
I_n^m,
I_n^w,
l,
p_n
}
]

其中 (O_n) 表示第 (n) 个 policy step 的 observation。

冻结的 OFT 做：

[
\boxed{
(h_n,A_n^N)
===========

\operatorname{OFT}(I_n^m,I_n^w,l,p_n)
}
]

其中：

* (\operatorname{OFT})：冻结的 OpenVLA-OFT；
* (h_n)：内部视觉语言融合 feature；
* (A_n^N)：normal action chunk；
* (N)：Normal；
* (n)：当前 policy decision。

在你这个版本里面：

[
\boxed{\operatorname{OFT}\ \text{不训练}}
]

所以正常策略原有能力尽量保持不变。

---

# 3. 第二步：Slot Attention 从视觉中找对象

我们不用再额外搞一个大型视觉 encoder。

直接从 OFT 的视觉 feature 里拿：

[
V_n
]

然后：

[
\boxed{
Z_n=
SA_\theta(V_n)
}
]

其中：

[
Z_n=
{
z_n^1,z_n^2,\ldots,z_n^K
}
]

这里：

* (SA)：Slot Attention；
* (\theta)：Slot Attention 的可学习参数；
* (V_n)：当前 visual tokens；
* (K)：slot 数；
* (z_n^k)：第 (k) 个当前视觉 slot。

比如它可能逐渐学成：

```text
slot 1 → book
slot 2 → gripper
slot 3 → shelf
slot 4 → bowl
slot 5 → background object
...
```

但此时 (z_n^k) 只描述**当前帧**。

我们还需要时间信息。

---

# 4. 第三步：最核心的 Action-aware SlotSSM

这是现在整个方法的核心。

先把上一段机器人动作编码：

[
\boxed{
e_{n-1}^{A}
===========

E_A(A_{n-1})
}
]

以及 proprio 编码：

[
\boxed{
e_n^p
=====

E_p(p_n)
}
]

其中：

* (E_A)：action encoder，小 MLP 即可；
* (A_{n-1})：上一段实际执行的 action chunk；
* (e_{n-1}^A)：上一动作的 embedding；
* (E_p)：proprio encoder；
* (p_n)：当前 proprio；
* (e_n^p)：proprio embedding。

然后每一个 slot 的时序更新：

[
\boxed{
s_n^k
=====

SSM_\phi
\left(
s_{n-1}^k,,
[z_n^k;e_{n-1}^{A};e_n^p]
\right)
}
]

这里：

* (s_{n-1}^k)：上一个时刻第 (k) 个对象的 memory；
* (z_n^k)：当前看到的该对象 feature；
* (e_{n-1}^{A})：机器人刚刚执行了什么动作；
* (e_n^p)：机器人现在自己的状态；
* ([,;,])：向量拼接；
* (SSM_\phi)：SlotSSM；
* (\phi)：SlotSSM 参数；
* (s_n^k)：更新后的 object state。

全部 slot：

[
\boxed{
S_n=
{s_n^1,\ldots,s_n^K}
}
]

---

# 5. 为什么 Previous Action 一定要进 SlotSSM？

因为这里我们不希望 SlotSSM 只学：

[
\text{前一帧图像}
\rightarrow
\text{后一帧图像}
]

而是希望它理解：

[
\boxed{
\text{Previous object state}
+
\text{robot action}
+
\text{new observation}
\rightarrow
\text{new object state}
}
]

举个抓书例子：

上一时刻：

```text
book 在桌上
gripper 到 book 附近
```

机器人执行：

```text
close gripper
+
lift upward
```

那么正常情况下：

```text
book slot 应该：
接近 gripper
+
跟着 gripper 向上移动
```

失败情况下：

```text
gripper 上去了

但是

book slot 没有上去
```

所以：

[
S_{n-1},A_{n-1},S_n
]

三者联合起来，才真正能够描述：

> **这个动作到底对这个物体产生了什么效果。**

---

# 6. 但只有 Previous Action 输入还不够

这是你之前最大的疑问：

> “即使把 action 输入 SlotSSM，它真的会使用 action 吗？”

答案是：

**不保证。**

网络完全可能忽略 (A_{n-1})。

因此我们需要最关键的训练约束：

[
\boxed{\text{Future Slot Prediction}}
]

---

# 7. Future-Slot Prediction：真正建立 Action ↔ Slot 对应

训练的时候，在当前状态：

[
S_n
]

我们已经知道真实执行的 action：

[
A_n^\star
]

于是做：

[
\boxed{
\hat S_{n+1}
============

F_\eta
(
S_n,
E_A(A_n^\star)
)
}
]

其中：

* (F_\eta)：很小的 future prediction head；
* (\eta)：它的参数；
* (A_n^\star)：数据集中的真实 action chunk；
* (E_A(A_n^\star))：action embedding；
* (S_n)：当前 object states；
* (\hat S_{n+1})：预测执行该动作以后 object states 会变成什么样；
* (\hat{})：预测值。

真实执行动作之后，我们会看到下一帧：

[
I_{n+1}^m,I_{n+1}^w
]

然后通过同样的 SlotAttention + SlotSSM 得到：

[
S_{n+1}.
]

所以直接监督：

[
\boxed{
\mathcal L_{\text{future}}
==========================

\frac{1}{K}
\sum_{k=1}^{K}
\left|
\hat s_{n+1}^k
--------------

sg(s_{n+1}^k)
\right|_2^2
}
]

每个字符：

* (\mathcal L_{\text{future}})：future-state prediction loss；
* (K)：slot 数；
* (k)：第 (k) 个 slot；
* (\hat s_{n+1}^k)：预测的未来第 (k) 个 slot；
* (s_{n+1}^k)：实际下一帧得到的 slot；
* (sg(\cdot))：stop-gradient，不让 target branch 被这个 loss 拉着跑；
* (|\cdot|_2^2)：平方 (L_2) 距离。

所以整个监督变成：

```text
当前 book slot
      +
执行“向上抓取”动作
      │
      ▼
预测 book 应该向上
      │
      ▼
实际下一帧 book slot
      │
      ▼
     Loss
```

这就是你一直想要的：

[
\boxed{
\text{Action}
\leftrightarrow
\text{Slot state change}
}
]

---

# 8. 这个 Future Head 部署时完全不用

特别强调：

[
F_\eta
]

不是 world model，不是 verifier，也不是部署时 rollout planner。

训练结束：

```text
Future Prediction Head
        ×
      删除
```

它存在的作用只有一个：

[
\boxed{
\text{把 action information 压进 SlotSSM representation}
}
]

类似一种 auxiliary task。

---

# 9. Slot Attention 自己怎么监督？

Future loss 本身不能保证 Slot Attention 一定学出好的 object decomposition。

所以建议再保留一个非常标准的 slot reconstruction loss。

SlotAttention 从：

[
V_n
]

产生：

[
Z_n
]

然后一个很小的 decoder 重建原始 visual features：

[
\boxed{
\hat V_n=D_\omega(Z_n)
}
]

其中：

* (D_\omega)：slot decoder；
* (\omega)：decoder 参数；
* (Z_n)：所有 slot；
* (\hat V_n)：重建出来的视觉 feature。

Loss：

[
\boxed{
\mathcal L_{\text{slot}}
========================

\frac{1}{N_v}
\left|
\hat V_n-
sg(V_n)
\right|_F^2
}
]

其中：

* (N_v)：visual token 数；
* (|\cdot|_F^2)：矩阵 Frobenius squared norm；
* (V_n)：OFT visual tokens；
* (\hat V_n)：slots 重建出的 visual tokens。

这个 Decoder：

[
D_\omega
]

也只训练时用。

部署直接扔。

---

# 10. 到这里 SlotSSM 学到了什么？

现在它同时受到两个约束。

### Object-centric 约束

[
\mathcal L_{\text{slot}}
]

要求：

> 每些 slots 要能够解释当前视觉场景。

### Action-effect 约束

[
\mathcal L_{\text{future}}
]

要求：

> 当前 slots + action 必须能够解释下一时刻对象状态。

所以最终：

[
\boxed{
S_n
===

\text{object-centric}
+
\text{temporal}
+
\text{action-aware}
}
]

这才是我们真正想给 recovery policy 的 representation。

---

# 11. 第四步：Failure Gate

Gate 不再只看当前图像。

它重点看：

[
\boxed{
S_{n-1},S_n,A_{n-1}
}
]

也就是：

> 刚才机器人做了什么，以及对象实际上发生了什么。

为了写公式方便，定义一个 action-effect representation：

[
\boxed{
r_n
===

Pool_{k=1}^{K}
\left(
[
s_n^k;
s_n^k-s_{n-1}^k;
e_{n-1}^{A}
]
\right)
}
]

其中：

* (r_n)：当前 action-effect summary；
* (Pool)：对全部 slots 做 attention pooling；
* (s_n^k)：当前第 (k) 个 slot；
* (s_n^k-s_{n-1}^k)：这个对象刚刚发生了什么变化；
* (e_{n-1}^{A})：上一动作；
* ([,;,])：拼接。

然后：

[
\boxed{
g_n
===

\sigma
\left(
G_\psi
(
[r_n;\bar h_n;e_n^p]
)
\right)
}
]

其中：

* (G_\psi)：Failure Gate MLP；
* (\psi)：Gate 参数；
* (r_n)：object transition + previous action 信息；
* (\bar h_n)：OFT feature (h_n) pooling 后的全局 feature；
* (e_n^p)：proprio embedding；
* (\sigma)：sigmoid；
* (g_n\in[0,1])：进入 recovery 的概率。

最终：

[
\boxed{
b_n=
\mathbf 1[g_n>\tau]
}
]

其中：

* (b_n\in{0,1})；
* (\mathbf1[\cdot])：indicator function；
* (\tau)：Gate threshold。

如果：

[
b_n=0
]

正常执行。

如果：

[
b_n=1
]

进入 recovery。

---

# 12. Gate 到底在学习什么？

例如上一 chunk 是：

```text
move toward book
close gripper
lift
```

对应：

[
A_{n-1}
]

正常：

[
s_n^{book}-s_{n-1}^{book}
]

应该表现出：

> book 位置随 gripper 改变。

抓空：

[
s_n^{book}-s_{n-1}^{book}
\approx 0
]

而机器人 proprio 明明显示 EEF 已经上升。

因此 Gate 学：

[
\boxed{
(A_{n-1},\Delta S_n)
\rightarrow
failure
}
]

这里：

[
\Delta S_n=S_n-S_{n-1}
]

就是对象状态变化。

所以 failure detection 和 action-object correspondence 是同一个故事。

---

# 13. 第五步：Recovery Action Head

这是另外一个关键：

**Recovery Head 不直接只看 OFT feature。**

否则 SlotSSM 又可能没用了。

我们明确强制：

[
\boxed{
\text{Recovery action query}
\rightarrow
\text{Cross-Attention}
\rightarrow
\text{Object Slots}
}
]

先生成 recovery queries：

[
\boxed{
Q_n
===

Q_\rho(h_n,l,p_n)
}
]

其中：

* (Q_\rho)：Recovery Head 中生成 action queries 的网络；
* (\rho)：整个 Recovery Head 参数；
* (h_n)：OFT feature；
* (l)：instruction；
* (p_n)：proprio；
* (Q_n)：用于产生 recovery action 的 query tokens。

然后：

[
\boxed{
C_n
===

CrossAttn
(
Q_n,S_n,S_n
)
}
]

这实际上对应：

[
Q=Q_n,
\qquad
K=S_n,
\qquad
V=S_n.
]

其中：

* (Q)：Query；
* (K)：Key；
* (V)：Value；
* (S_n)：object states；
* (C_n)：读完 object states 后的 recovery features。

最后：

[
\boxed{
\hat A_n^R
==========

R_\rho(C_n,p_n)
}
]

其中：

* (R_\rho)：Recovery Action Decoder；
* (\hat A_n^R)：预测 recovery action chunk；
* (C_n)：Slot-conditioned feature；
* (p_n)：机器人当前自身状态。

展开：

[
\hat A_n^R
==========

[
\hat a_{n,1}^R,
\ldots,
\hat a_{n,H}^R
].
]

你当前：

[
H=8.
]

---

# 14. 这一步为什么保证 Slot 指导 Action？

因为 recovery 的数据通路被设计成：

[
\boxed{
h_n
\rightarrow
Query
\rightarrow
CrossAttention(S_n)
\rightarrow
Action
}
]

也就是说 recovery action token 必须主动读取：

[
s_n^1,\ldots,s_n^K.
]

这叫**结构上的 coupling**。

前面的：

[
(S_n,A_n)\rightarrow S_{n+1}
]

则叫**训练目标上的 coupling**。

因此有两重约束：

[
\boxed{
\begin{aligned}
\text{Representation coupling:}&\quad
(S_n,A_n)\rightarrow S_{n+1}\
\text{Control coupling:}&\quad
S_n\rightarrow A_n^R
\end{aligned}
}
]

这是整个设计最核心的地方。

---

# 15. 最终 action 非常简单

正常动作：

[
A_n^N
]

恢复动作：

[
A_n^R
]

Gate：

[
b_n.
]

最终：

[
\boxed{
A_n
===

(1-b_n)A_n^N+b_nA_n^R
}
]

其中：

* (A_n)：最终执行的 action chunk；
* (A_n^N)：normal action；
* (A_n^R)：recovery action；
* (b_n=0)：正常；
* (b_n=1)：recovery。

由于 (b_n) 是 0/1，所以实际上不是平均两个动作：

[
b_n=0
\Rightarrow
A_n=A_n^N
]

[
b_n=1
\Rightarrow
A_n=A_n^R.
]

执行后：

```text
Aₙ
 ↓
LIBERO
 ↓
new RGB / wrist / proprio
 ↓
SlotSSM update
 ↓
重新判断
```

形成闭环。

---

# 16. 那到底哪些东西需要训练？

现在其实没有那么多。

## 永远冻结

[
\boxed{\text{OpenVLA-OFT}}
]

包括：

* backbone；
* normal action head。

---

## 真正部署时需要训练并保留的只有三个

### ① SlotAttention + SlotSSM

参数：

[
\theta,\phi
]

负责：

[
RGB\ history
+
Action\ history
\rightarrow
S_n.
]

### ② Failure Gate

参数：

[
\psi
]

负责：

[
S_{n-1},S_n,A_{n-1}
\rightarrow
g_n.
]

### ③ Recovery Head

参数：

[
\rho
]

负责：

[
h_n,S_n,p_n
\rightarrow
A_n^R.
]

---

## 训练时有、部署时删除

两个非常小的辅助东西：

[
\boxed{
F_\eta
}
]

Future-Slot Predictor；

以及：

[
\boxed{
D_\omega
}
]

Slot reconstruction decoder。

所以部署模型实际上仍然只有：

```text
Frozen OFT
    +
SlotAttention/SlotSSM
    +
Gate
    +
Recovery Head
```

非常干净。

---

# 17. 训练分成两个 Stage 就够了

我不建议五阶段训练。

就是：

[
\boxed{
Stage~1:\ Representation~Pretraining
}
]

然后：

[
\boxed{
Stage~2:\ Recovery~Joint~Training
}
]

---

# 18. Stage 1：先训练 Action-aware SlotSSM

这一阶段**根本不需要 failure label**。

只需要连续机器人轨迹。

每条训练数据有：

[
\boxed{
{
I_n^m,
I_n^w,
p_n,
A_{n-1},
A_n^\star,
I_{n+1}^m,
I_{n+1}^w,
p_{n+1}
}
}
]

也就是说：

```text
当前 RGB
当前 wrist RGB
当前 proprio

上一 action
当前真实 action

下一时刻 RGB
下一时刻 wrist RGB
下一 proprio
```

这就够了。

---

# 19. Stage 1 的监督从哪里来？

有两个 self-supervised supervision。

第一：

[
V_n
\rightarrow
Slots
\rightarrow
\hat V_n
]

监督：

[
V_n.
]

产生：

[
\mathcal L_{\text{slot}}.
]

第二：

[
S_n+A_n^\star
\rightarrow
\hat S_{n+1}
]

监督：

[
S_{n+1}.
]

产生：

[
\mathcal L_{\text{future}}.
]

总 loss：

[
\boxed{
\mathcal L_{\text{stage1}}
==========================

\mathcal L_{\text{slot}}
+
\lambda_f
\mathcal L_{\text{future}}
}
]

其中：

* (\mathcal L_{\text{stage1}})：Stage 1 总损失；
* (\mathcal L_{\text{slot}})：object-centric reconstruction loss；
* (\mathcal L_{\text{future}})：action-conditioned future prediction loss；
* (\lambda_f)：控制 future loss 权重的超参数。

所以 Stage 1 学到：

[
\boxed{
\text{物体是谁}
+
\text{物体随时间怎么变}
+
\text{action 会怎样影响物体}
}
]

---

# 20. Stage 1 在 LIBERO-Plus 用什么数据？

这里数据很多，不需要靠你目前那几十条 recovery trajectories。

可以使用：

### 原始 LIBERO demonstrations

大量正常成功轨迹。

再加：

### training perturbation rollout

自己生成与最终 evaluation 不重叠的：

* Robot Initial State；
* Camera；
* Layout；

等 perturbation。

LIBERO-Plus 官方本身也采用了这种训练/评测分开的设计：最终 10,030 个任务属于 test-only benchmark；论文另外构造了超过 20,000 条成功训练轨迹，并使用与 evaluation 不同的扰动参数。([arXiv][1])

所以你也应该：

```text
Training perturbation
        ≠
LIBERO-Plus official test perturbation
```

不能拿官方最终 test instance 做 recovery training。

---

# 21. Stage 1 最好还加入 OFT 自己跑出来的数据

因为普通 demonstration 基本都是：

```text
正确 action
→
正确 object motion
```

但我们最终关心：

```text
action
→
object 没按预期运动
```

所以我建议：

[
D_{\text{stage1}}
=================

D_{\text{demo}}
\cup
D_{\text{OFT-rollout}}
]

其中：

* (D_{\text{demo}})：成功 demonstrations；
* (D_{\text{OFT-rollout}})：OFT 在训练 perturbation 下自己执行出来的数据。

后者里面既有成功，也有失败。

Future-Slot prediction 都可以照常训练：

[
(S_n,A_n)\rightarrow S_{n+1}.
]

因为：

> **它不需要知道成功还是失败，只需要学习“这个 action 实际造成了什么变化”。**

这是一个很大的优点。

LIBERO-Plus 官方公开的 generalized training trajectories 本身筛选的是成功轨迹，因此你自己的 OFT failure rollout 对我们这个 failure-recovery 任务仍然是必要补充。([arXiv][1])

---

# 22. Stage 2：联合训练 Gate + Recovery

Stage 1 完成以后：

```text
SlotAttention
+
SlotSSM
```

已经有一个不错的 initialization。

Stage 2 加：

```text
Failure Gate
+
Recovery Head
```

同时：

> **SlotAttention / SlotSSM 不完全冻结，而是小学习率一起微调。**

于是 Stage 2 是真正联合训练：

[
\boxed{
SlotSSM
+
Gate
+
Recovery
}
]

---

# 23. Gate 需要什么数据？

每个决策时刻需要：

[
\boxed{
y_n^g\in{0,1}
}
]

其中：

[
y_n^g=0
]

表示：

> 当前不用 recovery。

[
y_n^g=1
]

表示：

> 当前应该进入 recovery。

数据可以来自 OFT 在 training perturbation 环境中的 rollout。

---

# 24. Gate label 怎么做最合理？

我建议不要纯手工看视频标。

因为你在 simulator，可以利用 **branching**。

例如 OFT 运行到状态：

[
x_n.
]

保存 simulator state。

然后从这个 state 分两个分支：

```text
                    state xₙ
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
        继续跑 OFT        执行 recovery
              │                 │
              ▼                 ▼
           failure           success
```

如果：

[
\text{OFT continuation fails}
]

而：

[
\text{Recovery + OFT succeeds}
]

那么这个状态就是非常干净的：

[
\boxed{
y_n^g=1
}
]

因为你已经通过仿真实验证明：

> 这里确实需要 intervention。

反过来，如果继续 OFT 就能成功：

[
\boxed{
y_n^g=0
}
]

这种 Gate label 比单纯：

> “我觉得这里看起来失败了”

干净很多。

---

# 25. Gate Loss

Gate 输出：

[
g_n\in[0,1].
]

标签：

[
y_n^g\in{0,1}.
]

使用 class-balanced BCE：

[
\boxed{
\mathcal L_{\text{gate}}
========================

*

## w_+y_n^g\log g_n

w_-(1-y_n^g)\log(1-g_n)
}
]

其中：

* (\mathcal L_{\text{gate}})：Gate loss；
* (y_n^g)：真实 recovery label；
* (g_n)：模型预测 recovery probability；
* (w_+)：positive sample 权重；
* (w_-)：negative sample 权重；
* (\log)：自然对数。

为什么需要：

[
w_+,\ w_-?
]

因为正常帧通常远多于 failure 帧。

否则模型很容易学成：

> 永远 normal。

---

# 26. Recovery Action 数据怎么采？

这里你之前已经做出来的 **5 个 FSM recovery candidates** 仍然很有用。

但注意：

[
\boxed{
FSM现在只负责生产训练数据
}
]

最终模型里完全没有 FSM。

例如找到 failure state：

```text
failure state xₙ
      │
      ├─ center FSM ── success
      ├─ left FSM   ── failure
      ├─ right FSM  ── success
      ├─ upper FSM  ── failure
      └─ lower FSM  ── failure
```

我们得到两条 successful recovery trajectories。

---

# 27. 但是不能把两个不同 recovery action 都直接当 GT

否则又回到你之前担心的问题：

```text
left success
+
right success
      ↓
L1 regression
      ↓
平均动作
```

所以第一版我建议非常简单：

> **同一个 failure source，只选择一个 canonical successful recovery trajectory。**

比如：

1. 最终任务成功；
2. recovery 步数最短；
3. action magnitude 最小。

按照这个规则选唯一一个：

[
A_n^{R\star}.
]

这样对于每个 failure state：

[
\boxed{
x_n\rightarrow A_n^{R\star}
}
]

只有一个监督目标。

因此：

**Candidate Verifier 可以完全删掉。**

---

# 28. Recovery Action Loss

模型预测：

[
\hat A_n^R
==========

[
\hat a_{n,1}^R,
\ldots,
\hat a_{n,H}^R
].
]

teacher action：

[
A_n^{R\star}
============

[
a_{n,1}^{R\star},
\ldots,
a_{n,H}^{R\star}
].
]

Loss：

[
\boxed{
\mathcal L_{\text{rec}}
=======================

\frac{1}{H}
\sum_{h=1}^{H}
\left|
\hat a_{n,h}^R
--------------

a_{n,h}^{R\star}
\right|_1
}
]

其中：

* (\mathcal L_{\text{rec}})：recovery action loss；
* (H)：chunk length；
* (h)：chunk 内第 (h) 个 action；
* (\hat a_{n,h}^R)：预测动作；
* (a_{n,h}^{R\star})：teacher recovery action；
* (|\cdot|_1)：L1 distance。

这个 loss 只对：

[
y_n^g=1
]

的 recovery samples 计算。

---

# 29. Stage 2 为什么还要继续 Future Prediction？

因为如果只训练：

[
\mathcal L_{\text{gate}}
+
\mathcal L_{\text{rec}}
]

SlotSSM 有可能慢慢变成：

> 单纯适合分类 / action regression 的 feature。

所以继续保留：

[
\mathcal L_{\text{future}}
]

作为 regularization。

让它始终保持：

[
\boxed{
\text{action-aware object dynamics}
}
]

---

# 30. 最终 Stage 2 总 Loss

我建议第一版就：

[
\boxed{
\mathcal L_{\text{stage2}}
==========================

\lambda_g\mathcal L_{\text{gate}}
+
\lambda_r\mathcal L_{\text{rec}}
+
\lambda_f\mathcal L_{\text{future}}
+
\lambda_s\mathcal L_{\text{slot}}
}
]

每个字符：

* (\mathcal L_{\text{stage2}})：第二阶段总 loss；
* (\mathcal L_{\text{gate}})：failure detection loss；
* (\mathcal L_{\text{rec}})：recovery action imitation loss；
* (\mathcal L_{\text{future}})：action-conditioned future slot loss；
* (\mathcal L_{\text{slot}})：slot reconstruction loss；
* (\lambda_g)：Gate loss 权重；
* (\lambda_r)：Recovery loss 权重；
* (\lambda_f)：Future loss 权重；
* (\lambda_s)：Slot reconstruction loss 权重。

不需要第五、第六个 loss。

---

# 31. 哪些 loss 更新哪些网络？

这个非常重要。

可以理解成：

```text
L_slot
  │
  ▼
SlotAttention

L_future
  │
  ├── SlotAttention
  ├── SlotSSM
  └── Future Head

L_gate
  │
  ├── Gate
  └── SlotSSM（小学习率）

L_rec
  │
  ├── Recovery Head
  └── SlotSSM（小学习率）


OpenVLA-OFT
      ×
   不更新
```

换成表格：

| 模块               | Stage 1 | Stage 2 | 部署 |
| ---------------- | ------: | ------: | -: |
| OpenVLA-OFT      |      冻结 |      冻结 |  ✓ |
| SlotAttention    |      训练 |      微调 |  ✓ |
| SlotSSM          |      训练 |      微调 |  ✓ |
| Failure Gate     |       — |      训练 |  ✓ |
| Recovery Head    |       — |      训练 |  ✓ |
| Future Predictor |      训练 |      训练 |  × |
| Slot Decoder     |      训练 |  训练/可冻结 |  × |

所以真正 inference graph 并不复杂。

---

# 32. 你的训练数据最终其实只有三类

### A. 普通 transition 数据

用于：

[
\mathcal L_{\text{slot}}
+
\mathcal L_{\text{future}}
]

形式：

```text
obs_n
action_n
obs_n+1
```

来源：

* LIBERO demo；
* generalized training trajectories；
* OFT rollout；
* failure rollout；
* recovery rollout。

只要有：

[
O_n,A_n,O_{n+1}
]

就能用。

---

### B. Failure/Normal 数据

用于：

[
\mathcal L_{\text{gate}}.
]

形式：

```text
history
+
previous action
+
current observation
+
failure label 0/1
```

---

### C. Successful recovery trajectory

用于：

[
\mathcal L_{\text{rec}}.
]

形式：

```text
failure observation
+
history
+
successful recovery action chunk
```

就这三类。

---

# 33. 你现在在 LIBERO-Plus 最适合怎么搞

第一阶段我建议甚至先只针对：

[
\boxed{\text{Robot Initial State perturbation}}
]

因为你的目标目前是 failure recovery，不需要第一版同时把 Camera、Noise、Light、Language 全做进去。

LIBERO-Plus 当前官方 benchmark 一共覆盖七类 perturbation；其 10,030 个最终任务是 test-only。([arXiv][1])

所以你的数据流程最好是：

```text
                Original LIBERO
                     │
                     ▼
          自己生成 training perturbation
            Robot Initial States
                     │
          ┌──────────┴───────────┐
          ▼                      ▼
    successful data          OFT rollout
                                 │
                           success / failure
                                 │
                                 ▼
                          failure states
                                 │
                  restore simulator state
                                 │
                  ┌──────────────┼──────────────┐
                  ▼              ▼              ▼
              recovery 1     recovery 2     recovery ...
                  │              │              │
               success        failure        success
                  │                             │
                  └─────────选最佳一个──────────┘
                                 │
                                 ▼
                        recovery training data
```

最后所有 checkpoint 和超参数都确定以后：

```text
Official LIBERO-Plus
test-only tasks
       │
       ▼
Final evaluation
```

不要反过来用 test tasks 收集 recovery data。官方论文也明确把最终 benchmark 和 generalized training set 分开构造。([arXiv][1])

---

# 34. 从头到尾训练流程可以压缩成这样

如果你真正准备开始代码实现，我会按这个顺序。

### Stage 1

准备：

[
(O_n,A_n,O_{n+1})
]

训练：

```text
SlotAttention
     +
SlotSSM
     +
Future Prediction
```

目标：

[
\boxed{
S_n+A_n\rightarrow S_{n+1}
}
]

得到 action-aware object memory。

---

### 收集 Recovery Dataset

冻结 OFT rollout：

```text
training perturbation
        ↓
OFT
        ↓
failure state
        ↓
save simulator state
        ↓
branch recovery FSM
        ↓
挑选 successful recovery
```

得到：

[
\boxed{
D_R
}
]

以及 Gate label。

---

### Stage 2

加载 Stage 1 SlotSSM。

一起训练：

```text
              SlotSSM
             /       \
            /         \
         Gate      Recovery Head
                      │
                    Action
```

Loss：

[
\boxed{
L_{\text{gate}}
+
L_{\text{rec}}
+
L_{\text{future}}
+
L_{\text{slot}}
}
]

OFT始终冻结。

---

# 35. 推理时就更简单了

最终 deployment：

```text
RGB + wrist + instruction + proprio
                  │
                  ▼
             Frozen OFT
              /       \
             /         \
      normal action   visual feature
                         │
                         ▼
                  SlotAttention
                         │
                      SlotSSM
                         │
                   ┌─────┴─────┐
                   ▼           ▼
                 Gate       Recovery Head
                   │           │
                   │       recovery action
                   │           │
                   └─────┬─────┘
                         ▼
                     execute
                         │
                         ▼
                  next observation
```

没有：

* Dynamics rollout；
* Candidate Verifier；
* FSM；
* Future Predictor；
* slot decoder。

所以部署模型其实并不重。

---

# 36. 最重要的是：现在 Action 和 Slot 到底在哪里“对应”？

现在终于可以非常明确地回答你最开始那个问题。

不是靠一个地方，而是靠 **三个连接**。

### 第一层：Action → Slot Memory

[
\boxed{
s_n^k
=====

SSM(s_{n-1}^k,z_n^k,A_{n-1},p_n)
}
]

上一动作直接参与 slot state 更新。

---

### 第二层：Action → Future Slot

[
\boxed{
(S_n,A_n)
\rightarrow
S_{n+1}
}
]

Future loss 强制 SlotSSM 表示包含 action-effect 信息。

这是**训练层面的对应**。

---

### 第三层：Slot → Recovery Action

[
\boxed{
A_n^R
=====

\pi_R(h_n,S_n,p_n)
}
]

而且：

[
h_n
\xrightarrow{Query}
CrossAttention
\xleftarrow{Key,Value}
S_n.
]

这是**结构层面的对应**。

所以最终形成一个很漂亮的闭环：

[
\boxed{
A_{n-1}
\rightarrow
S_n
\rightarrow
A_n^R
}
]

同时训练时又有：

[
\boxed{
S_n+A_n
\rightarrow
S_{n+1}
}
]

因此整套方法可以用一个循环来概括：

[
\boxed{
\text{Action}
\rightarrow
\text{Object State Change}
\rightarrow
\text{Failure Detection}
\rightarrow
\text{Recovery Action}
\rightarrow
\text{New Object State}
}
]

---

# 37. 这是不是 end-to-end？

现在可以非常准确地说：

### Recovery 部分

是：

[
\boxed{
Observation
\rightarrow
SlotSSM
\rightarrow
Recovery Action
}
]

整个过程都是 neural network，而且 Stage 2 中：

[
SlotSSM+Gate+Recovery
]

可以联合反向传播。

所以可以叫：

> **end-to-end learned recovery policy**

但因为 OpenVLA-OFT 是冻结的，所以整套系统不能说成：

> fully end-to-end jointly trained VLA。

论文里更严谨的说法是：

> **an end-to-end neural recovery policy built upon a frozen VLA backbone**

我认为这样反而更好，因为实验可以非常干净地证明：

[
\boxed{
\text{同一个 OFT}
\quad vs\quad
\text{同一个 OFT + Ours}
}
]

性能提升只能来自 recovery。

---

# 38. 最后把论文方法浓缩成三个模块

如果最后 Method Figure 只画三个模块，我甚至建议就写：

[
\boxed{
\text{Action-Aware Object Memory}
}
]

SlotAttention + SlotSSM + Future Prediction supervision。

↓

[
\boxed{
\text{Failure Detection}
}
]

根据：

[
(A_{n-1},S_{n-1},S_n)
]

检测动作效果是否异常。

↓

[
\boxed{
\text{Object-Conditioned Recovery}
}
]

通过 cross-attention：

[
S_n\rightarrow A_n^R
]

直接生成 recovery action。

整个 paper 的核心 statement 就可以是：

> **We learn an action-aware object-centric memory that captures how robot actions transform object states. The learned temporal object states are then jointly used to detect execution failures and directly condition recovery action generation.**

这版我认为已经比之前的“Gate + Candidate Verifier + FSM/Recovery Adapter”清楚很多：**SlotSSM 不再只是一个额外 feature extractor，而是连接“过去动作效果”和“未来恢复动作”的中间状态。**

[1]: https://arxiv.org/html/2510.13626v1 "LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models"
