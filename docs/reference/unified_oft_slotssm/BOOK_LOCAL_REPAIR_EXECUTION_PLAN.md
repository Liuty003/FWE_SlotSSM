# Book → Caddy：局部重抓闭环执行计划

更新：2026-08-09

这是当前唯一的短期实验主线。目标不是解决所有 LIBERO-Plus Robot 初始位姿
扰动，也不是重新训练 OpenVLA-OFT+ 的通用抓取规划；目标是先在一个可控任务族
中，完成**自然空抓后的自主局部重抓闭环**。

## 1. 固定问题

- 任务：`pick up the book and place it in the back compartment of the caddy`。
- 策略：`Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata`（OpenVLA-OFT+）。
- 基准：LIBERO-Plus `libero_10` 的 Robot Initial States 任务。
- 主要失败源：runner task `504`，`initstate 276`。
- 稳定对照：runner task `505`，`initstate 286`（三次均成功）。

第一版失败定义：夹爪执行闭合并开始抬升/离开 pregrasp 区域后，book 没有与夹爪
共同运动；策略未进入有效 transport 阶段。此定义不使用线上 simulator state。

不纳入第一版动作恢复训练：抓错语义对象、初始接近方向完全错误、不可逆碰撞、
或需要重新规划整个任务的失败。这些可保留为后续泛化/检测集。

同样排除 `task497`、`task518`、`task519`：三者虽经三次审计和一次双相机重录均为
失败，但其 520 个 policy action 均保持 closed gripper，没有出现“接近后闭合、物体
未随动”的局部空抓事件；它们属于初始执行模式异常，不与本版 missed-grasp gate 混训。

## 2. 当前状态

- [x] `task504` 原批量评测失败。
- [x] `task504` 全量双相机/action 录制：失败，520 steps。
- [x] `task504` 两次独立不保存 PNG 的重放：均失败，520 steps。
- [x] `task505` 全量成功录制与两次独立重放：三次均成功，均为 219 steps。
- [x] 三次 source 稳定性筛选并重录：训练候选中得到 8 个 stable NORMAL、12 个 stable FAILURE
  source；另有翻转 source 被排除，不进入训练。
- [x] `task504` timestep 78（首次闭合前）的 offline MuJoCo fork snapshot。
- [x] task504 的 recovery branch collector：在 `t=78` 的 6 个局部分支中，
  `xy080_z085`、`xy080_z110`、`xy100_z085`、`xy100_z110` 由双指接触与书本抬升
  `2.3–3.3 cm` 验证为成功；`xy120_*` 为失败对照。分支仅使用离线 pose/contact
生成和标注，模型输入仍严格限于 RGB、语言、proprio 与已执行动作历史。

分支记录的 RGB、proprio 与 action 严格同属动作前时刻；发现过的旧版
post-action proprio 记录不进入训练清单，均由其 fork snapshot 重生成。
- [x] 双相机自然轨迹：20 个 stable source 均已重录，且结果与稳定性标签一致。
- [x] episode/source-disjoint manifest v2：24 source、156 branches、110 verified
  successful branches、4,730 action windows、374 gate windows；恢复正例 source
  按 source-disjoint 划入 train/val/test，不存在跨 split source 泄漏。
- [x] v2 特征预计算真实冒烟：OFT+ 双相机输入 → 16 个 SlotSSM token → 8-step
  recovery target，单样本形状与线上输入约束均通过。
- [x] train split 的 action feature 预计算：1,720 个窗口，产物为
  `features-v2/action-train.safetensors`。
- [x] val/test action feature 预计算：258 / 172 个窗口；task504 test 文件只用于评估。
- [x] recovery-head pilot：仅用 train 训练、用 val 选 checkpoint；在 held-out task504
  离线 action 模仿中，loss 由 frozen normal head 的 0.489 降至 0.080（83.6% 相对
  降幅）。这不是 simulator success result。
- [x] 扩展版 recovery head：使用 3,354 个 train、516 个 validation action window
  训练 3,000 step，best validation loss 为 0.0650（`recovery-head-v2-expanded/`）。
  validation 集相对 pilot 已扩充，故该数值不与旧 0.0425 直接横向比较；test source
  task504、task511 均未参与该训练或 checkpoint 选择。
- [x] 扩展版 held-out offline action evaluation：在 task504+task511 的 860 个窗口上，
  frozen normal head loss 为 0.3602、recovery loss 为 0.0677（81.2% 相对下降）。
  这仍是 offline imitation，不能替代在线成功率。
- [x] 扩展版 oracle-trigger 在线反例：task504（step78）和 task511（step60）各接管
  50 policy step 后均运行至 520 step 失败；因此扩展版不得称为闭环改善。两条失败
  rollout 的 on-policy state 已保存，正在针对其 step68/86 状态生成 correction branch。
- [x] 第一轮 on-policy correction：仅用 train source task513 的 step68 correction
  branch（16/18 verified success）加入训练，并以 first-step-weight=4 重训。相同 oracle
  protocol 下，task504 在 309 帧恢复成功，task511 仍在 520 step 失败。因此当前在线
  held-out 结果是 1/2，表明 correction 能修复部分闭环偏移、但尚未跨 failure state
  泛化；task511 的 branch 仍严格保留为 test-only 诊断数据。
- [x] 阶段感知 recovery adapter：在原有 SlotSSM/OFT 状态融合上增加
  `backoff → align → descend → close → lift/verify` 五分类阶段头与 soft phase
  embedding，利用 teacher branch timestep 标注训练。使用 4,730 个 train action
  window、516 个 validation window 训练 3,000 step，best validation action loss
  为 0.0700；task504/task511 的 held-out oracle-trigger 在线检验正在并行运行。
- [x] task504 oracle-trigger 闭环：在已知失败 fork 的 policy step 78 强制切到 learned
  recovery head 50 步，随后交回 OFT+；270 帧内任务成功。该视频证明 recovery action
  可闭环执行，但触发时刻来自 oracle，不能称为 autonomous recovery。
- [x] task501 oracle-trigger 反例：相同流程在 step 55 接管后未能在 350 步内完成；
  recovery 首步出现与 teacher 相反的明显下探，表明离线 action-window 模仿有 on-policy
  covariate shift。不得把 task504 单例成功表述为泛化结果。
- [x] gate 功能验证：train/val 分别只有 9 / 1 个 RECOVERY 时刻；binary gate 未检出
  唯一 validation 正例（recall=0）。checkpoint 仅作接口测试，**禁止接入闭环**。
- [ ] 扩充独立 failure source 与 RECOVERY 事件后，重新训练 gate，再做在线闭环评测。
  剩余 11 个初始状态的三次稳定性审计已完成：`495/505/507/520` 成败翻转，
  `517` 全程闭爪，均排除；`514/524` 虽稳定失败但以初始或绝大部分时间闭爪为主，
  暂不混入 missed-grasp 动作监督。当前正补录 `511/513/516/521` 四个稳定失败、且
  有开合爪过程的自然双相机轨迹；四条均已复录为 520-step failure。`511` 在首次
  闭合前的 fork 上已得到 18 条 branch（16 条 verified local regrasp success、2 条
  failure 对照）。`513` 与 `521` 的 branch 正在生成；`516` 的早期闭合模式异常，
  先保留给审计，不盲目混入训练。

数据根目录：

`/data2/lty/naive-e-slotssm-runs/oft-plus-book-local-repair-v1/`

## 3. 模型：只做局部修复

保留现有统一架构：

```text
main/wrist RGB + language + proprio + executed-action history
    ├─ frozen OpenVLA-OFT+ → H_act
    └─ task-conditioned interaction slots → SlotSSM state Z_t

H_act cross-attends Z_t
    ├─ frozen normal head(H_act)
    ├─ recovery head(H_act + residual)
    └─ execution/failure gate
```

第一版新增的训练语义：

1. interaction roles：`book`、`gripper`、`caddy`；离线 segmentation/pose 只可作
   role teacher，线上只能由图像和语言产生 role token。
2. execution phase：`PREGRASP`、`CLOSING`、`HELD`、`TRANSPORT`、`PLACED`、
   `STALLED`。
3. failure gate：学习 `missed_grasp`，而不是泛化的“任何失败”。
4. recovery action：`backoff → lateral_align → regrasp → lift_verify`；每次只执行
   一个动作后重观测。

normal OFT+ backbone 和 normal action head 保持冻结。第一轮仅训练 role/slot
模块、SlotSSM 适配、cross-attention、gate 和 recovery head。

## 4. 执行顺序

### A. 先确保 source 可复现

1. 对每个候选 task 以相同 checkpoint、初始 state、控制参数独立跑三次。
2. 第一轮只接纳 3/3 同一结果的成功/失败 source。
3. 若成败翻转，先审计 recorder 与官方 evaluator 的动作、图像翻转、wait steps、
   随机种子和环境 state；该 source 暂不进入 action supervision。

验收：task504 是稳定失败；pilot 当前为 12 个稳定失败 source、8 个稳定成功
source。正常 source 数量少于原定 10，第一轮 gate 训练显式按该限制报告，不以重复
window 代替新增 source。

### B. 录制可 fork 的自然轨迹

每个 source 保存：主/腕 RGB、proprio、OFT planned action chunk、environment
executed action、MP4 和每个候选 fork 的 MuJoCo state snapshot。

fork 优先级：

1. 第一次对 book 的接近结束；
2. 夹爪闭合后；
3. 第一段抬升后 book 未随动。

验收：每个 failure source 至少有一个可精确恢复的 snapshot；线上字段不含 pose、
contact 或 segmentation。

### C. 离线 teacher recovery branches

teacher 可以读取 simulator pose/contact，但仅在离线生成数据时使用。每个 fork
生成以局部横移、深度和高度为变量的 branch 集：

```text
backoff (2–5 cm) → book-side lateral offset → descend → close → lift verification
```

以 IK/operational-space 控制执行；对横移、下降高度、闭合时间和抬升幅度加入小扰动。
保存成功及失败 branch、teacher-only contact/pose 标签和最终 task outcome。

验收：当前有 102 个 branch，其中 62 条经双指接触和书本抬升验证为成功，达到闭环
pilot 的 **branch 数量**门槛；其余 40 条为失败 branch。成功 branch 分别来自
task494、501、504、506、510、523 六个 source。失败 branch 保留作 outcome/gate
监督，但绝不充当 action teacher。这里仍只有 6 个可恢复 source，故仅可做
end-to-end 工程 smoke 和 protocol 验证，不能作为正式论文规模结果。

### D. 数据集与训练

split 的最小单位为 `source episode + fork`，所有从同一 snapshot 生成的 branches
必须位于同一 split。推荐 pilot split：70/15/15。

训练顺序：

1. 冻结 OFT+、预计算 `H_act`、interaction slots 与因果 `Z_t`。
2. 训练 execution phase / missed-grasp gate；正常轨迹用于控制 false positive。
3. 仅用成功 branch 训练 recovery head 的 normalized OFT actions。
4. 联合微调 gate、cross-attention 和 recovery head：

   `L = L_phase + L_gate + I(success_branch) L_action + L_normal_consistency + L_outcome`

5. teacher-gate replay → shadow mode → hard routing online rollout。

在当前 action-only pilot 后，必须插入一轮 on-policy correction：从 learned recovery
head 的 oracle-trigger rollout 采集其实际 RGB/proprio/action history，在同一 state 上以
teacher local regrasp 给出纠正动作；训练时混合 teacher branch 与 correction windows。
这是处理 task501 所见 distribution shift 的必要步骤，不可用更多同一 teacher window
替代。

已采集两组该类纠正样本：task501 的 6 条成功 branch 仅作为 validation-distribution
诊断，不能进入训练；task494 的 6 条成功 branch 可进入训练。它们来自 learned head
实际到达的 state，而不是原始 teacher trajectory，因此 builder 会把它们严格限定为
action supervision，不将其错误标成自然轨迹上的 gate 触发时刻。

### E. 闭环验收

在 held-out stable sources 上，按相同 initial state 对比：

1. OFT+ baseline；
2. OFT+ direct retry；
3. OFT+ + local SlotSSM recovery。

必须报告：任务成功率、失败后的恢复成功率、gate precision/recall/F1、正常成功
轨迹误接管率、检测延迟、额外动作数。第一版成功标准是恢复系统能提高 task504
任务族的 held-out 成功率，同时不显著破坏正常 source。

## 5. 样本量门槛

| 阶段 | stable normal sources | stable failure sources | successful recovery branches |
| --- | ---: | ---: | ---: |
| 闭环 pilot | 10 | 10 | >= 50 |
| 单任务正式结果 | >= 30 | >= 30 | >= 150 |

window 数量不能代替 source 数量；报告中必须以 held-out source 为单位。

## 6. 最新可复现数据（2026-08-08）

- 数据根：`/data2/lty/naive-e-slotssm-runs/oft-plus-book-local-repair-v1/`。
- 可用于训练的 branch 根：`branches-v2/`、`branches-v2-extra/`、
  `branches-v2-extra2/`、`branches-v2-extra3/`、`on-policy-correction-branches-v1/`；它们均显式标注
  `observation_timing: pre_action`。后者只提供 recovery action，绝不作为 natural
  gate timing 的标签来源。
  旧 `branches/` 存在 post-action proprio 对齐错误，仅作审计，不得读取。
- 训练清单：`manifests-v2/`；`action_windows.jsonl` 只包含 verified successful
  branches，`branch_outcomes.jsonl` 保留成功和失败，`gate_windows.jsonl` 包含
  NORMAL/RECOVERY 监督，`source_splits.json` 为 split 的唯一真源。
- `manifests-v2/report.json` 的当前可复现计数：24 stable source（8 NORMAL / 16
  FAILURE）、156 branches（110 successful）、4,730 action windows、374 gate windows；
  split 为 train/val/test = 17/2/5 source。新 `511` 为 held-out test source，`513/521`
  为 train source；该划分按 source 完整隔离。
- v2 预计算入口：`scripts/precompute_book_regrasp_v2_features.py`。它只读取
  manifest 中的 RGB、wrist RGB、语言、proprio 与 executed-action history；单样本
  OFT+→SlotSSM 冒烟输出在 `features-v2-smoke/action-one.safetensors`。
- recovery action 训练入口：`scripts/train_book_regrasp_v2_recovery.py`；它强制
  train/val 特征分离、冻结 OFT+ normal head，并且不接收 held-out test feature。
  现有 `features-v2/action-{train,val}.safetensors` 是加入 on-policy correction 前的
  旧版预计算产物；下一次训练前必须按更新后的 manifest 全量重算，不能混用。
- held-out offline action 评估：`recovery-head-v2-pilot/test-action-report.json`；该
  指标仅衡量 recovery action teacher 的模仿误差，不能表述为恢复成功率。
- gate 功能验证：`gate-v2-functional-pilot/report.json`；因为只有 9 个 train
  正例且 validation 的唯一正例漏检，此 checkpoint 禁止作为在线触发器使用。
