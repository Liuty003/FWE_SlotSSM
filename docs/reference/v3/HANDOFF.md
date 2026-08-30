# Recovery-VLA / V3 工作交接

更新时间：2026-08-15（Asia/Shanghai）

## 目标与当前判断

目标不是为 LIBERO-Plus 手写一个只能成功一次的教师，而是训练一个部署时只依赖
`RGB + wrist RGB + language + proprio + 已执行动作历史` 的 recovery VLA。

V3 使用冻结的 OpenVLA-OFT 双视角表征，经 SlotAttention 与 causal
ActionAware SlotSSM 建模；gate 决定是否切至 recovery head，后者输出一个
8×7 action chunk。在线路径**不允许**使用 simulator state、object pose、contact、FSM
或候选编号。它们仅可用于离线产生、筛选和验证教师轨迹。

先前模型在线完全不触发的根因已经确认：旧 canonical teacher 是从孤立的
“恢复后/抬起后”状态开始，缺少正常失败策略到 recovery 的真实切换边界。因此，
离线 gate 指标很好，但 task 501 在线 65 次决策中 recovery 触发为 0。

当前改为只接纳完整的：

`OFT 失败策略正常前缀 -> 离线恢复分支 -> 成功完成任务`

其中完整拼接回放的 `metadata.success` 必须为 `true`。

## 关键代码

- `recovery/models/v3_action_aware_recovery.py`：V3 模型。
- `recovery/data/v3_sequence_features.py`：冻结特征契约与校验。
- `scripts/collect_v3_book_prefix_teacher.py`：生成完整前缀 recovery teacher；只落盘在线允许的输入。
- `scripts/save_oft_recovery_fork_snapshot.py`：精确重放失败前缀并保存仅离线可用的 MuJoCo fork。
- `scripts/collect_book_regrasp_fsm_branches.py`、`scripts/collect_book_placement_search.py`：离线重抓取/放置搜索。已修正为 LIBERO-Plus 的 import 优先级高于 OFT 内置 legacy LIBERO。
- `scripts/precompute_v3_oft_sequence_features.py`：冻结 OFT 特征。已优化：同一个 source 的每个 physical decision 只推理一次，重叠序列仅复用缓存，输出数值定义不变。
- `scripts/filter_v3_sequence_features.py`：按完整物理 source 过滤特征，支持新连续教师替换旧孤立教师。
- `scripts/merge_v3_sequence_features.py`、`scripts/split_v3_sequence_features.py`：合并和严格 source-disjoint 切分。
- `scripts/train_v3_action_aware_recovery.py`：两阶段训练。
- `scripts/evaluate_v3_action_aware_recovery.py`：gate recall/FPR 与 normalized recovery-action L1。
- `scripts/run_v3_oft_action_aware_recovery.py`：在线 LIBERO-Plus runner，写出 `decisions.jsonl` 供审计。

## 数据与可复现路径

所有运行产物根目录：

`/data2/lty/naive-e-slotssm-runs/v3-action-aware-slotssm-v1/plus-robot-book`

### 基线数据

- `raw/`：官方 LIBERO-Plus Robot book task 491–524、episode 0 的 OFT nominal rollout。
  - 34 个物理 source；10 成功、24 失败。
  - `features/all-oft-rollouts.safetensors`：217 sequences；失败 rollout 的 gate label 均为 `-1`（不被当作 NORMAL）。
- 历史成功 completion 已 promotion 为 `canonical-teachers/task{501,504,511}-episode0-full-completion`，但其孤立形式不再用于 501/511。

### 已通过完整连续回放的 prefix teachers

| source | 轨迹路径 | 结果 | 用途 |
| --- | --- | --- | --- |
| task 501, fork t=49 | `prefix-teachers/task501-t0049-fsmxyp0-placement08` | success=true，187 transitions，EEF fork error 0.561 mm | 严格 held-out validation |
| task 511, fork t=44 | `prefix-teachers/task511-t0044-fsmxyp25-placement04` | success=true，191 transitions，EEF fork error 0.397 mm | training |

两者都已通过：

`scripts/audit_v3_rollout_sources.py --require-recovery-labels`

对应冻结特征：

- `features/task501-prefix-recovery.safetensors`：16 sequences。
- `features/task511-prefix-recovery.safetensors`：16 sequences。

### 明确拒绝的轨迹（不要加入训练）

- task 511 `...placement03`：独立 placement success，但拼接真实前缀后 `metadata.success=false`。
- task 504 `prefix-teachers/task504-t0055-fsmxyp25-placement17`：完整拼接后 `success=false`。

这些结果表明“分支独立成功”不等价于“可部署状态上的完整 recovery 成功”。

### 当前 V2 特征集

`features/all-with-prefix-recovery-v2.safetensors`

构成：nominal rollouts + task 504 的旧 canonical teacher + task 501/511 的连续 prefix teacher。
旧 task 501/511 孤立 teacher 已通过 source filter 删除。

固定 seed=8 的切分：

- 训练：`features/train-source-split-seed8-prefix-v2.safetensors`
  - 197 sequences，27 physical sources。
  - labels：`-1=1280`、`NORMAL=149`、`RECOVERY=147`；recovery action targets=147。
  - task 511 在训练集。
- 验证：`features/val-source-split-seed8-prefix-v2.safetensors`
  - 57 sequences，7 physical sources。
  - labels：`-1=256`、`NORMAL=100`、`RECOVERY=100`；recovery action targets=100。
  - task 501 在验证集。

因此没有 source/window 泄漏，且 validation 的恢复切换边界没有被 Stage 2 学到。

## 训练状态

### 已完成的旧实验（仅作历史参考）

- `stage1-book-source-split-v2`：旧教师集 2000 steps，best val total 5.33928、future 0.000612642。
  - action shuffle MSE gap 7.956e-6（约 1.3%，方向正确但弱）。
- `stage2-book-source-split-v1`：旧教师集 2000 steps。
  - source-held-out 离线：recall 0.975、FPR 0.0139、normalized recovery action L1 0.0696。
  - 但 task 501 在线 gate 从未触发（65 decisions、0 recovery），因数据缺少真实 handoff 边界。

### 当前 V2 Stage 1（已完成）

输出目录：

`stage1-book-prefix-v2/`

命令要点：Stage 1、2000 steps、batch size 4、LR 1e-4、GPU 2、seed 20260815。

最终 step 2000：

- train total 4.21274；slot 4.21240；future 0.00034098。
- val total 5.33989；slot 5.33919；future 0.00070821。

可供下一阶段加载的 checkpoint：

`stage1-book-prefix-v2/train_best.safetensors`

## 下一步（按顺序执行）

1. 以 `stage1-book-prefix-v2/train_best.safetensors` 初始化 Stage 2，在 V2 split 上训练完整 2000 steps。
   - Stage 2 才训练 gate 与 recovery head；不要用 Stage 1 的随机下游 head 做模型选择。
2. 运行 `scripts/evaluate_v3_action_aware_recovery.py` 于 V2 validation。重点报告：valid gate decisions、recall、FPR、precision/F1、normalized recovery-action L1。
3. 做 action-conditioning shuffle audit；要求新的表征对打乱动作历史有可见且稳定的损失差距，不能只复述旧的弱 1.3% gap。
4. 用 `scripts/run_v3_oft_action_aware_recovery.py` 在 LIBERO-Plus task 501 episode 0 在线运行。
   - 保存 `decisions.jsonl`。
   - 检查实际 gate 概率范围、是否在失败前缀附近触发、以及最终 success；离线好看不等同于在线有效。
5. 继续为 task 504 生成完整连续成功 teacher。现有 t=55 分叉与 verified-lift 是可复用起点，但 placement17 的完整拼接失败，不能使用。应搜索更鲁棒的 residual/分叉时刻，而不是放宽 `success=true` 条件。
6. 如有新的完成轨迹，冻结特征时用当前优化后的 `precompute_v3_oft_sequence_features.py`；按 source 替换旧 isolated teacher 后重新切分、重训、重测。

## 环境与运行注意事项

- Python：`.conda/envs/openvla-oft/bin/python`
- OFT root：`/data/lty/code/openvla-oft-official`
- LIBERO-Plus root：`/data/lty/code/LIBERO-plus`
- OFT checkpoint：`/data2/lty/models/openvla-7b-oft-finetuned-libero-plus-mixdata`
- 导入顺序必须是 `LIBERO-plus -> OFT -> repo root`，否则可能误用 OFT vendored legacy LIBERO。
- 先用 `nvidia-smi` 看空闲显存；只在余量充足的卡上启动新作业。OFT 特征提取约占 11–16GB，V3 训练约 3GB。
- 任何失败 nominal rollout 的 gate 标签必须为 `-1`，而不是 NORMAL；任何完整 teacher 必须经过 audit，并且最终 `metadata.success=true`。

