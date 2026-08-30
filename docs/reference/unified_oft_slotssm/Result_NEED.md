如果目标是 **ICRA 2027**，我建议你们不要把精力放在“跑很多零散失败 case”，而是围绕一个非常清楚的核心问题组织实验：

> **在同一个强 VLA base policy 上，SlotSSM 是否能利用对象级时序状态，在失败发生后生成有效修复动作，并显著提高最终任务成功率？**

ICRA 2027 截稿是 **2026 年 9 月 15 日**，而且只有 8 页，所以实验最好少而硬。([IEEE ICRA 2027][1])

## 我认为最理想的主结果表

先做 **LIBERO-Plus Robot Initial States**，因为官方已经证明这一类对 VLA 非常困难；OFT 在 Robot 上只有 **31.9%**，OFT+ 甚至只有 **30.3%**，因此这里有大量真实 failure 可以恢复。([GitHub][2])

主表我会设计成：

| Method                           | Robot L1 | L2 | L3 | L4 | L5 | Overall |
| -------------------------------- | -------: | -: | -: | -: | -: | ------: |
| OpenVLA-OFT                      |          |    |    |    |    |         |
| OpenVLA-OFT+                     |          |    |    |    |    |         |
| OFT + recovery data vanilla SFT  |          |    |    |    |    |         |
| **Ours: OFT + SlotSSM Recovery** |          |    |    |    |    |         |

这里最关键的不是你们现在那 **19 条**。

如果用于正式论文主结果，应该：

> **先确定一个官方子集，然后这个子集里的任务全部跑，不能根据 OFT 是否失败再筛。**

例如可以定：

**方案 A，最强：**

```text
LIBERO-Plus
Robot Initial States
× 4 suites
× 全 difficulty
```

完整跑 Robot 类。

如果算力/时间不够，我更建议：

**方案 B，ICRA 最低可接受但仍然比较干净：**

```text
LIBERO-Plus
Robot Initial States
× LIBERO-10
× 全部 difficulty
```

或者：

```text
Robot / LIBERO-10 / Level 4–5
```

但必须**提前规定 L4–5 全部跑**，不能：

```text
先跑 OFT+
↓
挑19条失败
↓
只在这19条报告 Ours success rate
```

这种结果只能叫 **recovery diagnostic**，不能叫 overall task success rate。

LIBERO-Plus 官方本身就是一 task instance 一 rollout，并给每个任务提供 perturbation category 和 difficulty，所以天然适合这种完整切片评估。([GitHub][2])

---

# 第二个指标比 overall success rate 还重要

你们是 recovery 论文，所以一定要报告：

[
\boxed{
\text{Recovery Success Rate}
============================

\frac{\text{原策略失败、被我们最终救回的任务}}
{\text{原策略失败任务}}
}
]

例如 393 个 Robot 任务：

```text
OFT+：
成功 120
失败 273

加入 Ours 后：
273 个原失败任务中
救回来 115 个
```

那么：

```text
Base task success = 120 / 393 = 30.5%

Recovery success = 115 / 273 = 42.1%

Final success = 235 / 393 = 59.8%
```

这三个数一起出现，非常有说服力：

> 原策略成功多少 → 剩下多少失败 → 我们能救回来多少 → 最终提升多少。

这也符合近期 failure-recovery 工作的实验逻辑。例如 FAR 同时强调 retry/recovery 后的任务成功提升，并报告 simulation 和 real-world 的最终 success improvement，而不是只展示几个成功恢复视频。([arXiv][3])

---

# 第三个：一定要证明不是“失败数据 SFT 就行”

这是我认为你们**最重要的 controlled baseline**：

```text
OpenVLA-OFT
       │
       ├── 相同 recovery dataset
       │         ↓
       │     vanilla SFT
       │
       └── 相同 recovery dataset
                 ↓
             SlotSSM Ours
```

最终：

| Method                     | Training data          | Final SR |
| -------------------------- | ---------------------- | -------: |
| OFT                        | 原数据                    |      31% |
| OFT+                       | Plus expert data       |      30% |
| **OFT + Recovery SFT**     | 你们 failure/recovery 数据 |      45% |
| **OFT + SlotSSM Recovery** | **完全相同的数据**            |  **60%** |

如果得到类似结果，那么才能说：

> **时序对象状态建模产生了额外收益。**

否则 reviewer 很可能会问：

> “是不是因为你们只是额外训练了一批 recovery data？”

尤其近期 recovery 工作已经明确把 recovery trajectory 作为训练信号，因此“有 recovery 数据”本身已经不能算很强的新贡献。FAR 会把成功恢复轨迹重新加入训练；EgoRecovery 也直接训练 recovery behavior。([arXiv][3])

---

# 第四个：SlotSSM 必须做消融

我至少会放这四个：

| Variant                 | Object slots | Temporal memory | Recovery |
| ----------------------- | -----------: | --------------: | -------: |
| Base OFT                |            × |               × |        × |
| Single-frame Slot       |            ✓ |               × |        ✓ |
| Temporal global feature |            × |               ✓ |        ✓ |
| **SlotSSM**             |            ✓ |               ✓ |        ✓ |

你真正想证明的是：

[
\text{single frame}
<
\text{object-centric temporal state}
]

否则别人会问：

> “直接看当前图像重新规划不就行了吗？为什么需要 SlotSSM？”

你们需要拿数据回答：

```text
Single-frame recovery     43%
Temporal global feature   47%
SlotSSM                   58%
```

这才真正支撑论文 title 里的 SlotSSM。

---

# 第五个：按 failure type 分解

这个我很建议做，而且不用非常大。

把 OFT/OFT+ 的失败标成：

| Failure                          | # cases | Vanilla retry | SFT | Ours |
| -------------------------------- | ------: | ------------: | --: | ---: |
| Grasp miss                       |         |               |     |      |
| Unstable grasp / slip            |         |               |     |      |
| Drop                             |         |               |     |      |
| Incorrect placement              |         |               |     |      |
| Collision / trajectory deviation |         |               |     |      |

这张表能直接回答：

> **SlotSSM 到底擅长修什么？**

我最希望看到的是：

```text
Grasp miss       +25pp
Slip/drop        +35pp
Wrong placement  +28pp

Perception failure  +3pp
```

这反而比“所有 failure 都提升”更可信，因为你们的方法机制本身就应该更适合 **object-state transition failure**。

LIBERO-Plus 本身指出 Robot Initial States 和 camera perturbations 是当前 VLA 的薄弱区域，因此选 Robot 作为主实验场景是有 benchmark 动机的。([GitHub][4])

---

# 第六个：failure detector 如果也是贡献，就必须单独评测

如果你们还有 failure gate，就不能只给最终 success rate。

至少：

| Metric                  | Result |
| ----------------------- | -----: |
| Precision               |        |
| Recall                  |        |
| F1                      |        |
| Detection latency       |        |
| False intervention rate |        |

尤其是：

### False intervention rate

原来 OFT 能成功的 episode：

```text
Base originally succeeds
↓
你的 gate 不应该乱触发 recovery
```

例如：

```text
Base successful episodes = 120
Ours破坏其中2条

False intervention = 1.7%
```

这个很重要。

否则一个方法可能：

> 什么情况都触发 recovery，多试几遍自然提高成功率。

审稿人会质疑这一点。

---

# 第七个：做一个真正的 held-out test

这是我非常建议你们从现在开始就锁死的。

例如：

```text
Failure / recovery 数据
        ↓
Train / validation
        ↓
模型、threshold、checkpoint全部确定
        ↓
最后才打开 test IDs
        ↓
一次性跑正式结果
```

不要：

```text
跑19条
↓
发现不好
↓
改anchor
↓
再跑这19条
↓
继续调
```

然后最后把这 19 条当 test。

那实际上已经成 development set 了。

你们前面那 **9 个 selected failure sources** 就完全可以保留作为：

> development / diagnostic cases

然后另外锁一批 **完全没用于开发的 Robot instances** 做 final test。

这是你们现在最应该立刻做的事情。

---

# 如果有时间，再加 Layout

我会把优先级设成：

[
\boxed{\text{Robot} > \text{Layout} > \text{Camera}}
]

LIBERO-Plus 官方的七类扰动中，Robot 是 manipulator initial pose variation，Layout 则改变目标位置或增加干扰物。([GitHub][2])

SlotSSM 的 object-centric temporal modeling 对：

```text
抓不到
抓偏
物体滑落
物体移动
放置失败
```

解释最自然。

所以比较漂亮的是：

```text
Main:
Robot Initial States

Generalization:
Objects Layout

Diagnostic:
selected grasp/drop/placement failures
```

我反而不会花大量精力跑：

```text
Light
Background
Language
Noise
```

因为即使结果提升，和“SlotSSM failure recovery”的方法动机也没那么强。

---

# 实机有没有必要？

**有会明显加分，但我不会为了实机牺牲上面主实验的完整性。**

如果时间够，我只做 3–4 个典型 failure：

```text
1. grasp miss
2. unstable grasp / slip
3. object drop
4. wrong placement
```

每类：

```text
10–20 trials
```

对比：

```text
Base VLA
Vanilla retry
Ours
```

即可。

近期 FAR 同时做 simulation + real-world，并报告分别约 **17.6pp 和 11.7pp** 的平均提升；EgoRecovery 也把 real-world failure recovery 作为核心验证，因此你的方向加入一个小规模真实机器人实验会明显强化“真实 recovery 能力”的论证。([arXiv][3])

而且 ICRA 官方强烈鼓励提交 accompanying video，最长 180 秒，这种 failure→detection→recovery 的方法尤其适合视频展示。([IEEE ICRA 2027][5])

---

# 如果让我给你定一套“够冲 ICRA”的实验包

我会定成下面这个，不继续无限扩：

**主实验**

* LIBERO-Plus **Robot**
* 至少一个完整 suite，最好四个 suite
* OFT / OFT+ / same-data Recovery-SFT / Ours
* Overall success + recovery success
* 按 difficulty L1–L5 分解

**泛化**

* LIBERO-Plus **Layout**
* Ours vs strongest baseline

**消融**

* no temporal memory
* no object slots
* no SlotSSM
* full model

**Failure analysis**

* grasp miss
* slip/drop
* wrong placement
* collision/deviation

**公平性**

* train/val/test 严格隔离
* test 在 checkpoint 和 threshold 完全确定后才读取
* 所有方法使用相同 recovery training data

**实机**

* 3–4 failure types × 10–20 trials，如果来得及。

---

我尤其不建议最后论文主结果变成：

> “我们挑了 19 个 OFT+ 会失败的任务，其中修好了 14 个。”

**这个只能当 case study。**

真正能撑起 ICRA 主结果的形式应该是：

> **“在预先定义的完整 Robot benchmark 上，OFT+ 从 X% 提高到 Y%；在 base-policy 的 N 个自然失败中恢复了 Z%；使用完全相同 recovery 数据的 vanilla SFT 只能达到 W%。”**

如果这个结果再配上明确的 SlotSSM ablation，即使你们只把 failure scope 限制在 **grasp / drop / placement 这类 object-centric execution failure**，论文故事已经相当完整。

[1]: https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/ "IEEE ICRA 2027 | Call for Technical Papers"
[2]: https://github.com/sylvestf/LIBERO-plus/blob/main/README.md "LIBERO-plus/README.md at main · sylvestf/LIBERO-plus · GitHub"
[3]: https://arxiv.org/abs/2607.01111 "FAR: Failure-Aware Retry for Test-Time Recovery and Continual Policy Improvement"
[4]: https://github.com/sylvestf/LIBERO-plus/blob/main/index.html?utm_source=chatgpt.com "LIBERO-plus/index.html at main · sylvestf/LIBERO-plus · GitHub"
[5]: https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/?utm_source=chatgpt.com "IEEE ICRA 2027 | Call for Technical Papers"
