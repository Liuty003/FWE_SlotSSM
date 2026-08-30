# Failure-Aware Embodied SlotSSM：项目交接（2026-08-06）

本文件面向没有任何上下文的接手者。先读本文件，再读
`OPENVLA_OFT_SLOTSSM_UNIFIED_RECOVERY_PLAN.md`；不要把旧的候选宏、人工
触发 demo 误认为当前统一模型的最终证据。

## 1. 项目目标

目标不是在 OpenVLA 旁边用人工规则“补救”，而是在**同一模型**中实现：

1. OpenVLA-OFT 给出原始主策略动作特征；
2. Stage-1 object slots 持续从主/腕双相机观测提取对象表示；
3. Mamba2 SlotSSM 将 slots 变为因果时序状态；
4. OFT action-token hidden cross-attend 到该时序状态；
5. 同一 adapter 有三个输出：冻结的 normal action head、训练的 recovery
   action head、三状态 failure gate；
6. gate 输出 `NORMAL / RECOVERY / RECOVERY_COMPLETE`。NORMAL 执行主策略
   action chunk；RECOVERY 立即截断 chunk、每次仅执行一动作再观察；DONE
   将控制权交回主策略。

线上输入只有：任务语言、主视角 RGB、腕视角 RGB、robot proprio 和历史。
不得使用 simulator object pose、接触、分割或人工 failure timestep。接触和
object pose 只可用于离线 branch teacher label / 审计。

## 2. 正确的模型接口

令 `H_act ∈ R^(56×4096)` 为官方 OFT 的 action-token hidden，`Z_t ∈
R^(16×512)` 为 SlotSSM 当前状态：

```
main/wrist RGB + task + proprio ──► OpenVLA-OFT ──► H_act
main/wrist RGB history ─► Stage-1 slots ─► Mamba2 ─► Z_t
H_rec = H_act + CrossAttention(H_act, Z_t)

normal_head(H_act)  ───────────────────────────────────────────► normal chunk
recovery_head(H_rec) ──────────────────────────────────────────► recovery chunk
gate(mean(H_act), mean(Z_t)) ──────────────────────────────────► 3-state mode
```

`normal_head` 从官方 OFT action head 加载且冻结；`recovery_head` 从它深拷贝
初始化；cross-attention 输出投影零初始化，故初始化时 recovery 与 normal 等价。

**动作尺度注意：** OFT action head 输出的是归一化动作，环境执行的是
`model._unnormalize_actions(...)` 后的动作。早期 `v1` recovery checkpoint 因把
环境尺度当成 head 监督尺度而无效；绝不能在线使用。有效版本均为 `v2`。

## 3. 有效代码与运行环境

- 主仓库：`/data/lty/code/naive-e-slotssm`
- 大文件 / 结果：`/data2/lty/naive-e-slotssm-runs`
- OFT 官方源：`/data/lty/code/openvla-oft-official`，commit `e4287e9`
- 官方 checkpoint：`/data2/lty/models/openvla-7b-oft-finetuned-libero-10`
- 正确环境：`/data/lty/code/naive-e-slotssm/.conda/envs/openvla-oft`
- 单元测试环境仍可用：`.conda/envs/openvla-a`

核心文件：

- `recovery/models/openvla_oft_slotssm_adapter.py`：统一 adapter；
- `recovery/tests/test_openvla_oft_slotssm_adapter.py`：初始化等价性等 3 个测试；
- `scripts/smoke_unified_oft_slotssm_bridge.py`：真实 OFT↔Stage1/2 bridge smoke；
- `scripts/precompute_oft_recovery_window_features.py`：冻结真实 OFT/SlotSSM 特征；
- `scripts/train_oft_slotssm_recovery_smoke.py`：恢复头训练；
- `scripts/train_oft_slotssm_gate_smoke.py`：gate 训练并合并 checkpoint；
- `scripts/run_oft_slotssm_unified_episode.py`：在线统一 runner；
- `scripts/collect_oft_nominal_trajectories.py`：官方 OFT baseline / normal 数据采集。

`prismatic` 名称被旧 SlotSSM 代码和官方 OFT 共用。在线 runner 已显式处理
namespace；不要简单替换 import 或混用旧 action head（旧版本是 5 action chunk，
OFT 必须是 8）。

## 4. 已完成且可复现实验

### 模型、数据、训练

- 官方 OFT Task3 ep0 已成功运行；
- 真实 bridge 已验证：OFT patch `1×512×2176` → Stage1 `1×16×512` → Mamba2
  时序 state `1×16×512`，OFT hidden `1×56×4096`；
- Task6 采集了 6 条双相机 recovery branches，转换为 66 windows：
  `RECOVERY=55`, `DONE=11`，其中末端稳定 regrasp 正样本 44；
- Task6 官方 OFT 成功轨迹 ep0--3 共 945 步，转换为 116 NORMAL windows；
- recovery head 在 44 正样本窗口上正确 OFT-scale 训练 400 steps，loss `0.0362`；
- gate 在 182 个同源窗口（116/55/11）上训练并保存统一 adapter。

有效 checkpoint：

```
/data2/lty/naive-e-slotssm-runs/oft-slotssm-recovery-head-smoke-v2/adapter_recovery.safetensors
/data2/lty/naive-e-slotssm-runs/oft-slotssm-unified-smoke-v2/adapter_unified.safetensors
```

### 在线闭环证据（仅功能证据）

- Task6 ep7：统一 runner 成功，250 steps；gate step 152 起进入 RECOVERY 共 47
  steps。此 ep7 是 recovery 数据源，**有训练源泄漏，只能证明闭环可运行**。
  视频：`oft-slotssm-unified-online-full-v1/task6-episode7.mp4`。
- Task6 ep6：未进入正常/恢复训练源，统一 runner 成功，215 steps，gate step
  152--167 RECOVERY 8 steps。视频：
  `oft-slotssm-unified-online-heldout-v1/task6-episode6.mp4`。
- 但官方 OFT 同一 Task6 ep6 也成功（208 steps），故没有成功率提升证据。

当前 Task6 官方 OFT ep0--9 全部成功，不能作为主“失败恢复优于 OFT”评测。

### 发现的自然失败

- Task9（把两只杯子放进微波炉）官方 OFT ep0--9：仅 ep4 失败，基线 9/10；
  ep4 视频：
  `/data2/lty/naive-e-slotssm-runs/openvla-oft-baseline-screen-task9-v1/task9-episode4/rollout-failed.mp4`。
- 人工检阅结论：杯子抬得过高，撞在微波炉上沿并卡住。这是很有价值的
  **transport collision / stagnation** 失败，不是空抓。
- 但 Task9 尚未为 OFT 重录可 fork 的 `steps.jsonl`，也尚未训练 Task9 adapter。

### 正在运行 / 最近的筛选

Task7 ep0--3 官方 OFT 均成功：

- ep0：248 steps；
- ep1：268 steps；ep2：270 steps；ep3：240 steps；
- ep1--3 视频位于
  `/data2/lty/naive-e-slotssm-runs/openvla-oft-parallel-screen-task7-v1/`。

Task7 ep4--9、Task8 ep0--9 尚未真正完成筛选。此前尝试后台启动失败的原因是
shell 结束清理子进程；若需并行，使用单个持久 `bash` 命令启动子进程并 `wait`，或
通过持续 exec session 管理，不要 fire-and-forget。

## 5. 当前最重要的下一步（按顺序）

### A. 完成失败场景筛选并导出全部视频

1. 并行跑 Task7 ep4--9 和 Task8 ep0--9 的官方 OFT；GPU0 可能是其他用户，优先
   GPU1--7。每个回合必须保存 JSON 和 MP4。
2. 汇总每个 task 的成功率、失败 episode、失败视频路径和人工失败类型。
3. 选择至少一个有多个自然失败的任务作主任务。若 Task7/8 仍几乎全成功，Task9 ep4
   可以做 collision case study，但不足以单独报告成功率提升；需要扩展到 harder
   benchmark / 标准鲁棒性协议，不能人为破坏已成功轨迹。

### B. Task9 ep4：建立真正 OFT failure→recovery 数据

1. 修改/新建 OFT rollout recorder，使失败回合也保存：
   `steps.jsonl`（含每一步 **environment executed_action**、OFT raw/planned action、
   双相机 RGB、proprio）和 MP4。失败不能进 NORMAL train set，但可以用于 fork。
2. 从撞上沿前的多个 timestep（例如接近微波炉口、初次接触、连续无进展）fork 分支。
3. teacher 离线产生候选：`后撤 → 下移 → 低高度重入 → 松爪`。保留多个成功/失败
   candidate，记录接触/是否进入微波炉/最终任务成功，仅作 label。
4. 分支 writer 已支持 wrist RGB：`recovery/data/branch_trajectory.py`。
5. 将成功 branch 转为 8-step OFT normalized recovery target；收集独立 failure episode
   与 nominal successful episode，按**episode**而非 window 划分 train/val/test。

### C. 训练与评价规范

1. Task6 checkpoint 不应直接迁移后宣称 Task9 能力；允许每任务单独训练。
2. gate 必须有 NORMAL、RECOVERY、DONE 三类，并做 held-out episode 指标：precision,
   recall, F1, false-positive rate, detection delay。
3. 对照至少包括：官方 OFT、OFT direct retry、完整统一模型；相同 initial state。
4. 报告端到端成功率、恢复成功率、额外步数、碰撞/卡死率；视频必须覆盖 baseline
   failure 与统一模型 recovery。
5. 必做消融：去 Mamba2（当前帧 slots）、去 slots（仅 OFT feature）、无 cross-attn、
   不截断 action chunk。

## 6. 可直接运行的命令

单个官方 OFT baseline（有 MP4）：

```bash
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false \
  .conda/envs/openvla-oft/bin/python scripts/run_openvla_oft_single_episode.py \
  --gpu 1 --task-id 9 --episode-id 4 --open-loop-steps 8 \
  --output-dir /data2/lty/naive-e-slotssm-runs/example
```

统一 Task6 runner：

```bash
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false \
  .conda/envs/openvla-oft/bin/python scripts/run_oft_slotssm_unified_episode.py \
  --gpu 1 --task-id 6 --episode-id 6 \
  --output-dir /data2/lty/naive-e-slotssm-runs/example-unified
```

测试：

```bash
.conda/envs/openvla-a/bin/pytest -q recovery/tests/test_openvla_oft_slotssm_adapter.py
git diff --check
```

## 7. 禁止的误表述

- 不要把 Task6 ep7 成功称为 held-out：它是 recovery training source；
- 不要称当前 gate 为泛化 failure detector：它只做过同源 window-level fit；
- 不要使用 v1 recovery/unified checkpoint；
- 不要把离线 privileged contact/pose/teacher branch 信息接入部署输入；
- 不要把人工选择 candidate 或固定 failure timestep 的旧视频称为自主恢复。

## 8. 交接时的工作区注意事项

工作树本来就是 dirty；用户的无关修改必须保留。当前项目相关新增文件多数尚未提交。
修改代码用 `apply_patch`；大下载、运行数据和视频均放 `/data2/lty/`。每次长运行前先看
`nvidia-smi`， GPU 空闲时可并行。
