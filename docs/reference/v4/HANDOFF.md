# Recovery-VLA V4.3 项目交接文档

> 最后更新：2026-08-17 12:02（Asia/Shanghai）
> 工作目录：`/data/lty/code/naive-e-slotssm`
> 当前版本：V4.3，设计讨论以本目录的 `v4_3.md` 为准；本文后面的 V4.2/r4/r5 内容是历史记录。
> 当前最重要的事实：V4.2 hidden-residual 模型即使修到 r6、增大 residual scale 并强化关键动作权重，仍未在 task523 闭环恢复成功。V4.3 已转向“只恢复到 Frozen OFT 能接手的局部状态”，并使用直接动作头和锁存执行器。

## 紧急更新：V4.3 已开工（2026-08-17 12:02）

### 已经确定的方向

1. **恢复目标缩短**：teacher 不再负责完成整项任务，只执行 fork 到经实验认证的 OFT-resumable handoff；之后严格交还 Frozen OFT。
2. **动作接口改变**：恢复分支不再预测 OFT hidden residual，而是从与 gate 共享的 relation context 直接预测 checkpoint-normalized `8x7` 动作。NORMAL 分支仍是 Frozen OFT 的 exact action bypass。
3. **执行器改变**：gate 一次触发后锁存恢复状态，不允许逐 decision 抖动退出；MVP 恢复上限 52 simulator steps、每次只执行预测 chunk 的前 4 步后重观测，结束后 cooldown 16 步。
4. **暂不使用 diffusion**：目前只有 5 条认证局部恢复轨迹，没有证据说明动作条件分布多模态；先做 direct BC 基线，diffusion 只作为后续有证据的对照。

### Handoff 审计结果

所有源都是 LIBERO-plus 中同一语义任务“pick up the book and place it in the back compartment of the caddy”的不同 initial state。审计方法是：精确重放 canonical teacher 到候选 handoff，然后只运行 Frozen OFT，禁止 recovery adapter。

| task | fork | 最早已认证 handoff | 局部恢复长度 | OFT suffix | 状态 |
|---:|---:|---:|---:|---:|---|
| 501 | 49 | 90 | 41 | 127 | train |
| 506 | 49 | 101 | 52 | 167 | train |
| 510 | 49 | 93 | 44 | 111 | train |
| 511 | 44 | 93 | 49 | 109 | val |
| 523 | 49 | 94 | 45 | 108 | val |
| 504 | 55 | 无 | - | - | 110/134/151 均失败；另测 183 运行中 |
| 521 | 49 | 无 | - | - | 80/153 均失败；另测 193 运行中 |

这给出了一个关键实验证据：例如 task523 原 teacher recovery 接近完整后半段，而恢复到 step94 后 Frozen OFT 已能成功，局部恢复仅需 `94-49=45` 步。当前 5/7 个源在 regrasp/close 后即可交还；task504/521 不能当作当前局部恢复目标硬塞入训练，除非更晚 handoff 审计成功。

版本化 manifest：

```text
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/
  v43-short-recovery-r1/handoff-manifest.json
```

生成脚本：`scripts/build_v43_handoff_manifest.py`。审计报告位于同一运行根目录的 `handoff-audit-r1/*/report.json`。

### V4.3 已落地代码

```text
recovery/models/v43_local_recovery.py
    V43LocalRecovery：复用 V4.2 Slot/3D relation/shared context，新增 direct 8x7 action head。
    RecoveryLatch：固定恢复时长、chunk 边界截断、cooldown。

scripts/build_v43_handoff_manifest.py
    从 nominal-only OFT 审计报告构建认证 handoff manifest。

scripts/precompute_v4_oft_relation_features.py
    新增 --handoff-manifest 和 --recovery-sampling-stride；只冻结认证 fork→handoff，
    可每4步采样但保留8步动作目标，且目标不得跨 handoff。

scripts/split_v43_handoff_features.py
    按 manifest 明确指定的 task/source 划分 train/val。

scripts/train_v43_local_recovery.py
    direct action L1 + gate + 可选 slot loss；optimization batch 对 recovery/NORMAL 1:1 采样；
    支持单独提供不同采样契约的 NORMAL shards。

scripts/run_v4_relational_online.py
    同时支持 V4.2 与 V4.3 checkpoint；V4.3 使用锁存、4步 receding horizon、52步 handoff。

recovery/tests/test_v43_local_recovery.py
    direct head、exact NORMAL action route、gate flicker、handoff chunk cap、cooldown 测试。
```

截至本次更新，V4/V4.3 定向测试为 **21 passed**，相关脚本 `py_compile` 通过。

### 数据与当前运行状态

现有 V4.2 r6 是 13 sources / 317 decision rows（train 219、val 98），只能作为历史/初始化/normal negatives；它包含 teacher 长恢复尾部，不应直接作为 V4.3 恢复目标。

V4.3 认证恢复数据预计约 50 个 stride-4 decision rows（实际数以冻结报告为准）：3 个 train recovery sources、2 个 val recovery sources。NORMAL negatives 已从 5 个成功 nominal sources 单独做 source split：

```text
.../v43-short-recovery-r1/normal-train.safetensors  # 61 rows / 3 sources
.../v43-short-recovery-r1/normal-val.safetensors    # 44 rows / 2 sources
```

短恢复 feature freeze 已完成并严格验收：52 rows / 5 sources，`gate_labels=[1]`，`recovery_mask.sum=52`，target shape 为 `[52,8,7]`：

```text
.../v43-short-recovery-r1/recovery-5sources-stride4.safetensors
.../v43-short-recovery-r1/recovery-train-stride4.safetensors  # 31 rows / task501,506,510
.../v43-short-recovery-r1/recovery-val-stride4.safetensors    # 21 rows / task511,523
```

三个正式 direct-BC seed（800 steps）已完成：

| seed | best val total | best-checkpoint gate recall@0.5 | normal FPR@0.5 | recovery L1 |
|---:|---:|---:|---:|---:|
| 20260817 | 0.46995 | 0.9524 | 0.0682 | 0.26955 |
| 20260818 | 0.71984 | 0.6667 | 0 | 0.32470 |
| 20260819 | 0.48201 | 0.9048 | 0 | 0.27348 |

当前 online 首选 seed20260819：它比 seed17 略差一点 recovery L1，但 held-out normal FPR 为0；task523 的 mean gate=0.9756、action L1=0.2701。路径：

```text
.../v43-short-recovery-r1/direct-stride4-seed20260819/best.safetensors
.../v43-short-recovery-r1/direct-stride4-seed20260819/eval-val.json
```

**尚未直接上线 seed19。** 新发现的契约问题是：最早 handoff 长度为41–52不等，而当前 executor 统一锁存52步；短于52的来源没有训练末尾动作。正在审计所有认证 source 的统一 `fork+52` handoff。只有全部成功后，才能用 `--use-fixed-latch-handoff` 重新冻结固定52步数据并训练部署候选。不要拿当前 variable-window checkpoint 直接声称固定52步闭环有效。

### 下一个 AI 必须继续做的顺序

1. 等 `recovery-5sources-stride4` 冻结完成，检查 rows、每 source 范围、gate 全为 1、target shape `8x7`、sampling/handoff provenance。
2. 用 `split_v43_handoff_features.py` 得到 recovery train(501/506/510) 和 val(511/523)，再与已经分开的 NORMAL shards 一起喂 `train_v43_local_recovery.py`。
3. 初始化可使用 `stage1-slot-exact-r5-scale020/best-total.safetensors`，但只把它当 slot/frontend 初始化；direct action head 必须新初始化。先做小步 smoke，再跑正式训练。
4. 离线必须报告按 source 的 direct-action L1，尤其是 val task523 fork 附近；不要只看混合平均 loss。
5. 首个闭环只测 held-out task523 fork49：paired nominal + V4.3，确认一次 gate 触发会连续执行到52步而非每8步重新开关，然后 Frozen OFT 接手。成功后再扩展多个 seed/source。
6. 若 direct BC 离线拟合但闭环漂移，下一步是收集 V4.3 自己 rollout 的偏离状态做 DAgger；不要先堆 diffusion、SlotSSM 或继续调 residual scale。

### 新增踩坑提醒

- 认证 handoff 是**具体 source/initial state 的实验事实**，不是阶段编号的普遍规律；不能假设所有 task 在 close/lift 后 OFT 都能接手。
- 52 是当前 5 个认证源的最大局部恢复长度，不是理论最优。执行最后一个 recovery chunk 时必须 cap，不能越过 handoff 再执行4/8步。
- `recovery_sampling_stride=4` 与 collector 的 `open_loop_steps=8` 是不同概念：前者是离线重观测采样间隔，后者是采集时 OFT nominal chunk 协议。
- task504/521 当前应作为“未被现有 Frozen OFT 覆盖的困难状态”保留，不能标成失败标签后混入 direct recovery BC。
- 多进程同时从 `/data2` 载入 7B checkpoint 会大量卡在 `wait_on_page_bit_common`；这台机器并行 GPU 多不等于模型冷加载能无限并行。feature freeze/online audit 建议最多同时 2 个模型加载进程。

下文从“紧急更新：r4 已失效”开始均为 V4.2 历史，保留用于追溯静默协议错误和旧实验，但不得覆盖本节的 V4.3 决策。

## 紧急更新：r4 已失效，当前应接 r5（2026-08-16 21:51）

在首次 r4 paired online 中发现：

- task523 fork49：threshold 0.5 时 nominal/V4 都失败，V4 0 triggers；threshold 0.4 时 V4 38 triggers 仍失败。
- task499 normal：nominal 成功（213 steps），r4 V4 在 step48/56 false trigger，随后累计 32 triggers 并失败。r4 因此没有保持 normal safety。

由此追到两个新的静默协议错误：

1. freezer/旧 online runner 使用 PIL 近似 crop；正式 OFT executor 使用 JPEG roundtrip、TF Lanczos3 resize、`sqrt(0.9)` crop-and-resize。两者会产生不同动作轨迹。
2. freezer 用 post-`process_action` 的 `executed_action` 监督 frozen decoder；decoder 实际输出 pre-`process_action` 的 `planned_action`。gripper 经过二值化和符号反转，仅这一维就可人为增加约 `2/7` L1。

修复已落地在 `precompute_v4_oft_relation_features.py` 和 `run_v4_relational_online.py`：

- 新 preprocessing 与官方 OFT 对同一图片逐像素完全一致：`pixel_equal=true, max_abs=0`。
- normal decoder-alignment L1 从 r4 的约 0.3717 降到 r5 的 **0.01090**。
- online `--gpu` 现在在任何 torch import 前设置；同时修复本仓库旧 `experiments` 包遮蔽官方 OFT 包的问题。
- TensorFlow 只做图像预处理并禁用 GPU，避免抢占 PyTorch 显存。
- merge/train/online 强制检查 `decoder_action_contract` 和 `image_preprocessing` provenance，拒绝旧 r4 adapter。
- V4 tests 当前为 **15 passed**。

r5 数据：

```text
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/features-v4_2-exact-r5/all-exact-13sources.safetensors
train: .../features-v4_2-exact-r5/train-stratified-r5.safetensors
val:   .../features-v4_2-exact-r5/val-stratified-r5.safetensors
decoder admission: .../features-v4_2-exact-r5/decoder-alignment-admission-r5.json
```

- 313 rows / 13 sources。
- train：217 rows / 9 sources，118 normal / 99 recovery。
- val：96 rows / 4 sources，57 normal / 39 recovery。
- Stage1 已完成 1200 steps；best validation Slot loss 2.75182（step1000）：
  `.../stage1-slot-exact-r5-scale020/best-total.safetensors`。
- Stage2 smoothing 0/0.05/0.10 正在 GPU3/6/7 并行重训，输出：
  `.../stage2-exact-r5-scale020-smooth000|005|010`。

若后文与本紧急更新冲突，以本节为准。当前下一步是等 r5 Stage2 完成、统一离线评估、再用修正后的 exact runner 重跑 task499/task523 paired online。

## 0. 给下一个 AI 的最短执行摘要

我们要做的不是继续修 V1--V3 的 SlotSSM，而是在冻结 OpenVLA-OFT 原能力的前提下，增加一个以机器人坐标系 3D 关系为表示的失败检测和恢复残差模块。

截至本文更新时间：

1. V4.2 模型、RGB-D 数据链路、两阶段训练、离线评估和 paired online runner 已经贯通。
2. 已修复两个会让旧结果失真的静默错误：recovery action 坐标系错误、validation recovery loss 错误稀释。旧 Book Stage2 checkpoint 因此作废。
3. 第一轮修正版 r3 已证明 held-out normal episode 完全旁路且成功，但在 task504 failure fork 上无论降阈值还是放大 residual 都未恢复成功。
4. recovery 数据已经从 3 个 physical sources 扩展到 7 个；新增的 task506/510/521/523 都经过 simulator 严格 action-trace replay 并再次成功。
5. 扩展数据 r4 的 Stage1 和三组 Stage2 都已完成；Stage2 区别是 gate label smoothing 为 0/0.05/0.10，三组均完整训练 1600 steps 并正常退出。
6. 3 smoothing × 3 checkpoint 共 9 组统一离线评估已完成。当前主候选是 `smooth005/best-total`，下一步是在 held-out normal 与 failure prefixes 上跑 paired online。若仍不能恢复，再考虑 decoder LoRA；不要再只调 gate threshold 或推理期 residual scale。

## 1. 我们到底要做什么

### 1.1 研究目标

目标是构建一个 Recovery-VLA：

- 正常状态下保持冻结 OpenVLA-OFT 的原始策略能力，adapter 不应扰动正常动作。
- 失败或偏离状态下识别需要恢复，并利用当前任务相关的机器人中心 3D 关系生成 hidden residual correction。
- correction 仍由 **同一个冻结 OFT action decoder** 解码，不能另造第二套动作生成器。
- 最终必须在 simulator 闭环中证明：相同 seed、相同 init state、相同 prefix 下，nominal baseline 失败，而 V4.2 恢复成功；同时 held-out normal episodes 不退化。

### 1.2 V4.2 的核心架构

简化数据流：

```text
RGB + wrist RGB + depth + language + proprio
                    |
             frozen OpenVLA-OFT
              /              \
      visual tokens       action hidden H_t
            |                    |
       Slot Attention            |  (query)
            |                    |
  robot-frame 3D object slots    |
            |                    |
 gripper-object / object-object relations
            \____________________/
                     |
       CrossAttn(Q=H_t, K/V=relations)
                     |
              shared context C_t
               /             \
          failure gate    recovery residual
               \             /
              H'_t = H_t + alpha * delta_t
                     |
          SAME frozen OFT action decoder
```

硬约束：

- OFT backbone、proprio projector、原 action decoder 全冻结。
- gate 和 recovery head 必须共享由 relation cross-attention 得到的 `C_t`。
- 训练时使用 GT gate 选择 recovery loss；推理时使用 hard gate。
- gate 未触发时必须返回原始 hidden/action，逐元素 exact bypass，不允许 soft blending 暗中改变 normal policy。
- instance segmentation 只允许作为 Stage1 slot 训练监督；online inference 不得读取 simulator mask。
- 关系位置来自 RGB-D、camera calibration 和 proprio，不得在 online 使用 simulator privileged object poses。
- V4 Stage1 Slot 是新实现，只借用了“Slot Attention”概念，没有复用 V1--V3 的 SlotSSM/world model 代码。

设计全文：

- `docs/reference/v4/v4_2.md`：当前实现规范，优先级最高。
- `docs/reference/v4/v4初步.md`：V4 初稿，仅作背景。
- `docs/reference/v4/证明自己创新性.md`：controlled representation study 和论文证据路线。
- `docs/reference/v4/PROGRESS_2026-08-16.md`：r3 阶段进度快照，已被本文后续 r4 进展部分更新。
- `docs/reference/v3/HANDOFF.md`：V3 历史与失败经验；不要直接把 V3 架构搬回来。

### 1.3 最终验收标准

主模型成立至少需要：

1. 多个 held-out normal episodes：V4.2 success 不低于 nominal，未触发决策通过 exact bypass。
2. 多个 held-out real failure forks：paired nominal 失败，V4.2 有可复现的 recovery success。
3. source-level train/validation/test 隔离；共享同一 initial state/prefix 的不同 fork 不能跨 split。
4. 报告 gate precision/recall/Brier、conditional recovery action error 和闭环 success，而不只报平均 loss。
5. 主模型真实恢复成功后，再做 `global / slot / relation_2d / relation_3d` controlled ablation、counterfactual relation、language grounding 和 viewpoint robustness。

当前只满足第 1 项的单个 normal episode案例；主结论尚未成立。

## 2. 环境、模型和重要路径

### 2.1 代码与运行环境

```text
主仓库          /data/lty/code/naive-e-slotssm
OpenVLA-OFT     /data/lty/code/openvla-oft-official
LIBERO-plus     /data/lty/code/LIBERO-plus
Python          /data/lty/code/naive-e-slotssm/.conda/envs/openvla-oft/bin/python
主运行根目录    /data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book
```

建议所有命令从主仓库执行，并显式使用上述 Python。不要依赖 shell 当前激活的 conda 环境。

### 2.2 Book 主模型

```text
OFT checkpoint
/data2/lty/models/openvla-7b-oft-finetuned-libero-plus-mixdata

action head
/data2/lty/models/openvla-7b-oft-finetuned-libero-plus-mixdata/action_head--150000_checkpoint.pt

proprio projector
/data2/lty/models/openvla-7b-oft-finetuned-libero-plus-mixdata/proprio_projector--150000_checkpoint.pt

action normalization key
libero_10
```

必须校验的 SHA-256：

```text
action head:
adc70a2da8cadd90b5f5cdec1771058130e62e48ede4c8cd351a8c35ee740330

proprio projector:
491e30d2aaa1d9ff33521a93a511b1828f8deb33515dd2b43b470ade74569221
```

注意：早期 task6 canonical 实验使用另一套 `libero-10` checkpoint 和 `libero_10_no_noops` normalization key。它与当前 plus-robot-book 数据不能混用。

### 2.3 GPU 和 CPU 的分工

- LIBERO/MuJoCo physics rollout 主要在 CPU 上执行。
- OFT feature extraction / online policy inference / adapter training 在 GPU 上执行。
- rollout 时 GPU 利用率通常是短 burst，中间大量时间等待 CPU simulator，所以只看瞬时 `nvidia-smi` 容易误判“没有用 GPU”。
- `CUDA_VISIBLE_DEVICES` 必须在导入 torch/OFT 之前设置。runner 已按 `--gpu` 处理；新脚本也必须保持这一点。
- 机器有 8 卡，可以把独立 collection、feature freeze、training、online eval 分配到不同卡并行，但不要在同一个输出目录并发写。

## 3. 代码地图

### 3.1 模型和几何

```text
recovery/models/v4_relational_recovery.py
    V4.2 Slot、3D relation、action-conditioned cross-attention、gate、residual adapter。

recovery/data/v4_geometry.py
    depth/camera/world/robot frame 几何和 instance-label 投影。

recovery/data/v4_relation_features.py
    V4 feature tensor schema、读取和验证。
```

### 3.2 数据采集与重放

```text
scripts/collect_v4_oft_rgbd_rollouts.py
    nominal OFT RGB-D rollout collector。

scripts/collect_v4_book_prefix_teacher.py
    Book canonical prefix + teacher branch collector。

scripts/collect_v4_task6_canonical_teacher.py
    task6 专用 collector；checkpoint/norm 与 Book 不同。

scripts/collect_v4_action_trace_replay.py
    将已有成功 action trace 严格重放成 V4 RGB-D/instance 数据；本轮新写。
```

### 3.3 Feature、训练和评估

```text
scripts/precompute_v4_oft_relation_features.py
    冻结 OFT visual/action hidden、RGB-D points、训练标签；执行 action normalization 和 provenance 校验。

scripts/merge_v4_relation_features.py
    合并 shards；检查 decoder provenance、source split group 和 concrete rollout identity。

scripts/split_v4_relation_features.py
    按 source_id 切分；使用 --stratify-gate 保证 normal/recovery sources 分布。

scripts/train_v4_relational_recovery.py
    Stage1 slot / Stage2 joint trainer；Hungarian slot matching、weighted validation、独立 best checkpoints。

scripts/evaluate_v4_relational_recovery.py
    离线 gate threshold、Brier、normal/recovery action error 评估。

scripts/run_v4_relational_online.py
    paired nominal/V4 online rollout；支持 canonical prefix、hard gate、exact bypass、SHA 校验、trace/video。
```

### 3.4 测试

```text
recovery/tests/test_v4_geometry.py
recovery/tests/test_v4_relational_recovery.py
```

基本回归命令：

```bash
.conda/envs/openvla-oft/bin/python -m pytest \
  recovery/tests/test_v4_geometry.py \
  recovery/tests/test_v4_relational_recovery.py -q
```

本文交接前最后一次运行结果为 13 tests 全部通过；交接后改代码应先重跑。

## 4. 数据协议和 provenance：不能破坏

### 4.1 rollout schema

新 collector 使用 `v4_rgbd_rollout_v2`，metadata 至少记录：

- `task_suite/task_id/episode_id`
- `success/transitions`
- `wait_steps/open_loop_steps/decision_stride`
- `env_seed`
- `rollout_id`
- `libero_root/oft_root/checkpoint provenance`
- camera calibration
- online permitted inputs / training-only inputs

precompute 对 v2 强制要求 `open_loop_steps == decision_stride`。旧 v1 为兼容历史数据仍能读取，但缺失字段必须显式按 legacy 处理，不能假装它有完整 provenance。

### 4.2 source identity

这里有两个不同概念：

- `source_id`：benchmark initial-state split group，例如同一个 task/episode。共享 nominal prefix 的多个 forks 必须共用它，因此绝不能跨 train/val/test。
- `source_instance_id`：具体 rollout/fork 的 identity，包含 fork/source/recovery branch、schema、wait/open-loop、env/checkpoint 等。merge 用它拒绝真正的重复 concrete fork。

同一个 `source_id` 有多个不同 `source_instance_id` 不是碰撞，而是合法的多 fork。不过多个 fork 会重复共享 prefix 样本，可能造成加权偏置；以后大量扩 fork 时应做 prefix 去重或 sample weighting，不能通过拆 source_id 来“解决”。

旧 fixed r1 没有完整 source_instance identity。当前 merge 通过显式 legacy decision stride 接纳，并在 report 的 `legacy_inputs_without_source_identity` 中记录；不要悄悄取消这条审计信息。

### 4.3 action 坐标系

这是已经踩过的最严重坑：

- frozen OFT decoder 的训练目标在 checkpoint-normalized action coordinates。
- 旧 freezer 把 nominal action 正确归一化，却把 physical recovery target 直接送进 decoder loss。
- 当前 `precompute_v4_oft_relation_features.py` 已使用 checkpoint action statistics 对 recovery target 做同样转换。
- 因此修复前的 Book Stage2 checkpoints 全部无效，不得继续引用或与 r4 混训。

gripper action 还要保持 OFT/LIBERO 的 `process_action` 约定；不要只修前 6 维而破坏第 7 维。

## 5. 已完成工作与有效结果

### 5.1 关键代码修复

已经完成：

1. recovery target 坐标统一到 frozen decoder coordinates。
2. Slot instance-mask supervision 改成 Hungarian one-to-one matching，避免多个物体复用同一 slot。
3. split 增加 `--stratify-gate`，避免 train 几乎没有 recovery source。
4. feature / decoder provenance 在 precompute、merge、split、train 中贯通；错误 decoder 的负测试会失败。
5. validation loss 按有效 gate/slot/recovery 样本数加权，不再把 normal-only batch 当 recovery loss 0。
6. trainer 分开保存 `best-total.safetensors`、`best-gate.safetensors`、`best-recovery.safetensors`；`best.safetensors` 只是 best-total alias。
7. 增加 gate label smoothing 实验。
8. online runner 重写为相同 seed/init/prefix 的 paired baseline，并检查 hard-gate exact bypass。
9. 修复/明确 `env.seed(0)`：必须在任何 `reset()` / `set_init_state()` 前调用。漏掉它曾造成约 0.497 m fork mismatch；正确重放可到毫米甚至数纳米级。
10. online runner 支持 `--prefix-rollout` 和 `--prefix-steps`。
11. evaluator/online 支持 residual scale override，仅用于诊断。
12. rollout v2、concrete source identity、legacy merge 审计已经落地。
13. 新增成功 action trace 严格 replay collector。

### 5.2 第一轮修正版 fixed r1 / r3（历史有效诊断）

数据：

```text
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/features-v4_2-fixed/all-fixed-r1.safetensors
```

- 198 decisions，9 sources。
- 147 normal / 51 recovery。
- 只有 3 个 recovery sources：task501 fork49、task504 fork55、task511 fork44。
- Stage1 Hungarian best validation Slot loss：2.7564（step 600）。
- r3 最佳 Stage2：`stage2-joint-stratified-smooth010-weighted-r3`，训练 residual scale 0.1。

离线 held-out validation：

- normal gate probability mean：0.0482。
- recovery gate probability mean：0.4184。
- Brier：0.1388。
- threshold 0.5：TP=7、FN=11、TN=48、FP=0，precision=1.0、recall=0.389。
- nominal-to-teacher L1：0.3710。
- recovery-to-teacher conditional L1：0.2339。
- inference scale 0.2 时 conditional L1：0.2178。

在线目录：

```text
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/online-r3
```

在线结果：

- held-out normal task491 episode0：nominal 与 V4 都成功，均 165 env steps；21 decisions，0 triggers，所有未触发 decision exact bypass。
- held-out failure task504 fork55：nominal 300 steps 失败。
- V4 threshold 0.5 无触发失败；threshold 0.1 一次触发失败；threshold 0.05、scale 0.1 七次触发失败；threshold 0.05、scale 0.2 十二次触发仍失败。

结论：normal preservation 已有真实证据；recovery 尚无成功。对 r3 继续只扫 threshold/scale 没有价值。

## 6. 当前 r4 数据扩展

### 6.1 新增成功 action traces 的来源

原始成功报告位于：

```text
/data2/lty/naive-e-slotssm-runs/oft-plus-book-local-repair-v1/autonomous-full-recovery-data-v1/taskX-episode0/taskX-episode0.json
```

审计出的候选为 task506/510/513/521/523 episode0。它们不能原样喂 `collect_v4_book_prefix_teacher.py`，因为没有独立 recovery/completion branch identity；因此使用 `collect_v4_action_trace_replay.py`。

严格重放输出：

```text
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/action-trace-teachers-r1
```

已接纳：

| Task | transitions | first recovery/fork | max EEF replay drift | replay success |
|---|---:|---:|---:|---|
| 506 | 240 | 49 | 2.45e-9 m | true |
| 510 | 232 | 49 | 2.31e-9 m | true |
| 521 | 219 | 49 | 0.0010017 m | true |
| 523 | 233 | 49 | 0.0007756 m | true |

task513 被拒绝：虽然 EEF tracking 基本跟随，但最终 simulator success 无法重现。它的目录没有完整 metadata，绝不能加入 feature merge。

重放精度坑：source JSON 的 NORMAL 动作必须保持 float64；RECOVERY 动作来源本来是 float32，必须保持 float32。统一粗暴 cast 会改变长时闭环结果。

### 6.2 r4 feature 和 split

新增四个 recovery sources 的冻结特征：

```text
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/features-v4_2-expanded-r4/new-recovery-4tasks.safetensors
```

- 115 rows / 4 sources。
- task506 30 rows、510 29、521 27、523 29。

合并数据：

```text
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/features-v4_2-expanded-r4/all-expanded-13sources.safetensors
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/features-v4_2-expanded-r4/all-expanded-13sources.json
```

- 313 rows / 13 sources。
- recovery physical sources 从 3 增到 7。

当前 stratified split：

```text
train: /data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/features-v4_2-expanded-r4/train-stratified-r4.safetensors
val:   /data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/features-v4_2-expanded-r4/val-stratified-r4.safetensors
report:/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/features-v4_2-expanded-r4/train-stratified-r4.split.json
```

- train：238 rows / 10 sources，139 normal / 99 recovery。
- validation：75 rows / 3 sources，36 normal / 39 recovery。
- validation source IDs：task523 recovery (`1012023744717756897`)、task499 normal (`8332168195546953122`)、task511 recovery (`9214430861981279965`)。
- train recovery sources 包括 task501/504/506/510/521，共 5 个；validation recovery sources 共 2 个。

重要限制：当前只有 train/validation 两份。validation 正在用于 smoothing/checkpoint/threshold 选择，因此它不是最终 held-out test。正式结论前必须新增 source-level test split，且 test 至少含多个 recovery 与 normal sources。

## 7. r4 训练和离线评估状态

### 7.1 Stage1

输出：

```text
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/stage1-slot-expanded-r4-scale020
```

- 使用 residual scale 0.2 创建兼容 config。
- 训练 1200 steps。
- validation Slot best 约为 step1100 的 2.7462；step1200 约 2.7497。
- Stage2 init 使用 `best-total.safetensors`。

### 7.2 Stage2 三组并行实验

共同配置：joint stage、1600 steps、batch 16、lr `5e-5`、`lambda_gate=1`、`lambda_recovery=1`、`lambda_slot=0.1`、residual scale 0.2、每 100 steps eval、每 400 steps save、seed 20260816。

输出目录：

```text
smooth 0.00  .../plus-robot-book/stage2-expanded-r4-scale020-smooth000
smooth 0.05  .../plus-robot-book/stage2-expanded-r4-scale020-smooth005
smooth 0.10  .../plus-robot-book/stage2-expanded-r4-scale020-smooth010
```

三者曾分别在 GPU 5/6/7 并行运行，均已完成 1600 steps 并以 exit code 0 退出。可用以下命令确认没有遗留训练：

```bash
pgrep -af 'train_v4_relational_recovery.py.*stage2-expanded-r4'
nvidia-smi
```

最终训练末段：

- smooth000 的 best total 出现在较早阶段；step500 为 `val_total=0.4722, gate=0.0596, recovery=0.1372`。到 step1600 recovery 降到 0.1090，但 gate 波动使 total 回升到 0.6361。
- smooth005 step1600：`val_total=0.5597, gate=0.1698, recovery=0.1143`。
- smooth010 step1600：`val_total=0.6459, gate=0.2693, recovery=0.1010`。
- smooth005/010 的日志里 gate loss 使用不同 label smoothing target，不能直接用其 smoothed BCE 跨实验比较 calibration。
- 三个目录都有 `best-total/best-gate/best-recovery/last`。

训练脚本没有把每一步历史写成独立 log 文件（stdout 是启动会话 pipe），checkpoint 也没有 safetensors metadata。完整 checkpoint 选择不要靠文件时间猜，以下统一 evaluator 结果才是交接依据。

### 7.3 统一离线评估结果

已在同一 validation feature set 上评估 9 个候选，JSON 保存在各 Stage2 目录，命名为：

```text
eval-best-total.safetensors.json
eval-best-gate.safetensors.json
eval-best-recovery.safetensors.json
```

关键候选：

| Candidate | normal p mean | recovery p mean | Brier | recovery L1 | threshold 0.5 (TP/FP/TN/FN) |
|---|---:|---:|---:|---:|---:|
| smooth000 / best-total | 0.0521 | 0.9966 | 0.01730 | 0.13717 | 39 / 1 / 35 / 0 |
| smooth005 / best-total | 0.0724 | 0.9704 | 0.02152 | 0.11430 | 39 / 2 / 34 / 0 |
| smooth010 / best-gate | 0.0674 | 0.9208 | **0.01317** | 0.15220 | 38 / 0 / 36 / 1 |
| smooth010 / best-recovery | 0.1441 | 0.9478 | 0.03265 | **0.10097** | 39 / 3 / 33 / 0 |

共同的 nominal action L1 on recovery 是 0.37169。因此：

- `smooth005/best-total` 把 recovery L1 降到 0.11430，净改善 0.25738，gate 仍有 100% recall，作为主在线候选最均衡。
- `smooth000/best-total` gate 更保守，threshold 0.5 只有 1 FP，适合做 normal-safety 对照。
- `smooth010/best-recovery` correction imitation 最好，但 normal probability 和 FP 更高，只应作为 recovery-aggressive 诊断候选。
- `smooth010/best-gate` Brier 最低，threshold 0.5 无 FP 但有 1 FN，可作 gate-specialized 对照。

这组结果比 r3 明显改善，但仍只是在两个 recovery validation sources和一个 normal validation source上测得；不能称为 test，更不能代替 simulator online recovery success。

## 8. 接下来按这个顺序做

### Step 1：训练和回归测试（已完成）

三组训练已正常结束。2026-08-16 19:40 运行 V4 tests，结果为 `13 passed in 4.44s`。

### Step 2：统一离线比较（已完成）

已经对每个 adapter dir 评估：

- `best-total.safetensors`
- `best-gate.safetensors`
- `best-recovery.safetensors`

复现命令模板：

```bash
CUDA_VISIBLE_DEVICES=4 .conda/envs/openvla-oft/bin/python \
  scripts/evaluate_v4_relational_recovery.py \
  --features /data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/features-v4_2-expanded-r4/val-stratified-r4.safetensors \
  --adapter-dir /data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/stage2-expanded-r4-scale020-smooth000 \
  --checkpoint best-total.safetensors \
  --action-head /data2/lty/models/openvla-7b-oft-finetuned-libero-plus-mixdata/action_head--150000_checkpoint.pt \
  --thresholds 0.05 0.1 0.2 0.3 0.4 0.5 \
  --device cuda \
  --output /data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/stage2-expanded-r4-scale020-smooth000/eval-best-total.json
```

结果和当前候选结论见 7.3。后续增加模型时仍应同时看：

- normal/recovery raw probability 分布和 Brier。
- threshold confusion matrix，尤其 normal false positive 与 recovery recall。
- conditional recovery-to-teacher L1，以及相对 nominal-to-teacher 的净改善。
- 不要只按 `val_total` 选，更不要把不同 smoothing 的 training BCE 直接横比。

### Step 3：跑 paired online matrix（当前第一优先级）

优先评估：

1. held-out normal：task499 episode0（当前 validation normal source）。
2. held-out recovery：task523 prefix49 和 task511 prefix44。
3. 可以再用 train recovery task 做 in-distribution sanity check，但不能当泛化证据。

task523 可直接使用严格 replay rollout 做 canonical prefix：

```bash
CUDA_VISIBLE_DEVICES=4 .conda/envs/openvla-oft/bin/python \
  scripts/run_v4_relational_online.py \
  --task-suite libero_10 \
  --task-id 523 --episode 0 \
  --adapter-dir /data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/stage2-expanded-r4-scale020-smooth005 \
  --adapter-checkpoint best-total.safetensors \
  --prefix-rollout /data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/action-trace-teachers-r1/task523-episode0 \
  --prefix-steps 49 \
  --wait-steps 10 \
  --gate-threshold 0.5 \
  --gpu 4 \
  --max-env-steps 300 \
  --output-dir /data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/online-r4/task523-fork49
```

runner 默认 `--paired-baseline`，应保留。检查最终 report：两条 rollout 的 init/prefix 是否一致、baseline 是否确实失败、V4 是否触发、exact bypass 是否通过、V4 是否成功。

task511 可使用已有 prefix-teacher rollout 的前 44 actions，但它是旧 v1 数据，先核对 metadata/task/init/wait；不要因路径相似就默认 identity 一致。task499 normal 不需要 prefix。

在线 threshold 应根据 validation gate distribution 选定后冻结。可以做少量预注册阈值敏感性，但不能在 test rollout 上不断调到成功。

### Step 4：建立真正的 train/val/test

当前 13 sources 对正式三分仍偏少。优先继续收集独立 successful recovery sources和 normal sources，然后按 `source_id` 做 stratified 三分：

- train 和 val 用于模型、smoothing、checkpoint、threshold 选择。
- test 只在所有选择冻结后运行。
- train/val/test 各自应有多个 recovery physical sources；不能把同 initial state 的不同 fork 分到不同集合。

如果暂时无法扩数据，也必须明确当前 online task511/523 已参与 validation，结果只能称 validation/diagnostic，不能称 held-out test。

### Step 5：若 r4 仍无 online recovery success

按优先级：

1. 检查 recovery correction 是否在正确 failure moment 连续触发、解码动作是否在 teacher 方向上、gripper 维是否正确。
2. 增加更多独立 recovery sources，尤其不同 failure geometry，而不是复制同一个 prefix。
3. 检查训练 action chunk 与闭环 open-loop 8 steps 的时序一致性。
4. 若 expanded data + scale-matched training 下 frozen decoder 仍无法复现 teacher，再按 `v4_2.md` 做小规模 decoder LoRA ablation。LoRA 是诊断 frozen decoder expressivity 的下一步，不应未经 r4 评估直接上。
5. 暂不要重加 SlotSSM/world model/verifier/第二 decoder；这会改变 V4.2 的核心故事且没有当前证据支持。

### Step 6：主模型成功后再做论文 ablation

受控保持其他训练和数据不变，对比：

- `global`
- `slot`
- `relation_2d`
- `relation_3d`（ours）

随后做 relation-controlled recovery、counterfactual relation、language-relation grounding、viewpoint robustness、normal OFT vs always-on relation vs gated relation。完整论证见 `证明自己创新性.md`。在主模型还不能闭环恢复时优先做这些 ablation 不会解决核心问题。

## 9. 已知问题和高风险踩坑点

1. **旧 Stage2 作废**：action coordinate bug 修复前的 checkpoint 不可引用。
2. **当前 val 不是 test**：它已用于 smoothing、checkpoint 和 threshold 选择。
3. **env seed 顺序**：必须 `env.seed(0)` 后再 reset/set init；否则 fork drift 巨大。
4. **paired 条件**：same seed、same init state、same wait steps、same exact prefix；任一不同都不能归因于 recovery adapter。
5. **动作精度**：action-trace replay 的 NORMAL float64、RECOVERY float32 约定会影响长时成功。
6. **动作 normalization**：Book 使用 `libero_10`；task6 是另一 checkpoint/key，禁止混合。
7. **source leakage**：split 用 `source_id`，去重 concrete fork 用 `source_instance_id`；不要交换二者职责。
8. **shared prefix 重复加权**：大量同 source forks 会重复 normal prefix，应去重或加权，不要拆 split group。
9. **schema v1/v2**：v2 必须有 wait/open-loop/provenance；legacy merge 要显式记录缺失信息。
10. **task513 不可用**：无成功 replay metadata，不能 precompute/merge。
11. **成功标签要严格**：只接受 simulator `success=true` 的完整 trajectory，不接受“EEF 看起来接近”。
12. **online 禁止 privileged input**：instance mask 和 object pose 仅训练监督/诊断，不得进入 inference。
13. **hard bypass**：gate 低于阈值必须精确返回 frozen OFT hidden/action；不能用近似 residual=0 替代检查。
14. **GPU 利用率误判**：simulator 在 CPU，GPU 是 burst；用 PID/memory 和持续采样判断。
15. **输出目录并发**：每个 job 独立目录；不完整目录不要覆盖重跑后再误 merge。
16. **模型选择**：`best-total`、`best-gate`、`best-recovery` 代表不同目标，必须统一 evaluator 后选择。
17. **smoothing 指标**：不同 label smoothing 的训练 BCE 不可直接横向比较；比较 raw probability、Brier、confusion 和在线结果。
18. **residual scale**：r4 训练时已经是 0.2。推理 override 只作诊断，不能靠 test-time 放大代替重新训练。
19. **Book task ID**：491、499、506、523 等是 LIBERO-plus patched benchmark task IDs，不是标准 LIBERO suite 内常规 0--9 index。
20. **工作树很脏**：大量 V1--V4 文件为未提交或用户已有修改。严禁 `git reset --hard`、`git checkout --` 或批量清理；只改任务相关文件。

## 10. 数据与结果索引

### 10.1 当前应使用

```text
# r4 replay rollouts
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/action-trace-teachers-r1

# r4 features / reports / split
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/features-v4_2-expanded-r4

# r4 Stage1
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/stage1-slot-expanded-r4-scale020

# r4 Stage2 candidates
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/stage2-expanded-r4-scale020-smooth000
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/stage2-expanded-r4-scale020-smooth005
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/stage2-expanded-r4-scale020-smooth010
```

### 10.2 历史诊断可用，但不是最终模型

```text
# 修正坐标后的第一轮 fixed features
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/features-v4_2-fixed/all-fixed-r1.safetensors

# r3 online diagnostic
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/online-r3

# 原三条 recovery prefix teachers
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/prefix-teachers
```

### 10.3 不应使用

- recovery target 坐标修复前生成的 Book Stage2 checkpoints/features。
- `action-trace-teachers-r1/task513-episode0` 中没有 metadata 的不完整输出。
- 任何 `success=false` 或文件数量/steps/transitions 不完整的 rollout。
- task500 早期不完整的 `normal-v4_2/task500-episode0`；完整 retry 在 `normal-v4_2-retry500/task500-episode0`，对应 feature 已另行正确冻结。

## 11. 交接后建议的工作记录方式

每轮实验至少保存：

- 完整 CLI/config、git diff 或代码版本标识。
- 数据输入清单、source IDs、source_instance IDs、split report。
- checkpoint/action-head/projector SHA。
- 离线 evaluator JSON。
- online paired report、trace、video、seed/init/prefix identity。
- 明确结论属于 train、validation、diagnostic 还是 untouched test。

如果本文件与 `PROGRESS_2026-08-16.md` 冲突，以本文件中 r4 内容为准；如果本文件与 `v4_2.md` 的架构定义冲突，以 `v4_2.md` 为准，并同步修订本文件。

## 12. 当前一句话结论

V4.2 的工程链路和 normal safety 已基本站住，真正的瓶颈是 recovery 数据覆盖与 frozen-decoder correction 在闭环中的有效性；r4 已用 7 个 recovery sources、训练时 scale 0.2 和三组 gate smoothing 完成第一次有意义的扩数据重训，离线主候选为 `smooth005/best-total`。下一位 AI 应立即做 paired online recovery 验证，再决定是否需要 decoder LoRA，而不是回头修旧 SlotSSM 或继续盲调阈值。

---

## 13. 2026-08-18 V4.3 / Book34 最新交接（权威更新）

> 本节晚于上面的 V4.2/r4 记录。凡是模型候选、数据规模、在线结果或“下一步”与前文冲突，**以本节为准**。前文保留用于解释历史和工程协议，不代表当前实验状态。

### 13.1 当前目标与 benchmark

当前第一目标不是继续做架构 ablation，而是把 Recovery-VLA 在 LIBERO-plus 的 34 个取书任务上做起来：

```text
benchmark: LIBERO-plus patched libero_10
task IDs: 491--524（共 34 个）
language family: pick up the book and place it in the back compartment of the caddy
episode: 当前统一使用 episode0
```

用户已经明确要求：现阶段把 34 条上的成功率拉高优先于 relation/global、LoRA、decoder expressivity 等小实验。当前工作假设是**主要瓶颈为成功 recovery action 数据覆盖**；除非新增覆盖仍无效，不要擅自把主线切回架构搜索。

### 13.2 当前模型：V4.3

当前 V4.3 部署协议：

- normal 时 exact frozen OpenVLA-OFT bypass，Recovery-VLA 不应改写 normal action。
- recovery 输入包含 frozen OFT visual tokens、OFT action hidden/query、proprio、Slot 表征和由 RGB-D/3D token points 构造的 relation context。
- direct recovery head 为 `pooled_chunk_mlp_v3`，输出 normalized `8 x 7` action chunk。
- 当前 recovery head 约 555,576 参数；V4.3 总模块约 14.32M 参数，active recovery path 约 8.26M。对照 OFT action decoder 约 151.12M 参数。
- 在线当前采用 recovery latch：horizon 62、minimum 40、execution horizon 4。具体在线 runner/config 必须以 r2 产物为准，不要从旧 V4.2 命令猜参数。

gate/handoff 的概念方向：二者本质都在回答“Frozen OFT 当前是否仍应拥有控制权”，后续应保持一个 competence score/控制权分数的干净表述。不要重新引入两个相互打架的独立判别器。当前线上仍使用 r2 gate；本轮只微调 direct recovery head。

核心代码：

```text
recovery/models/v43_relational_local_recovery.py
scripts/train_v43_local_recovery.py
scripts/run_v43_relational_online.py（若文件名有变，先 rg V43 runner，禁止套用旧 V4.2 runner）
scripts/precompute_v4_oft_relation_features.py
scripts/merge_v4_relation_features.py
```

`scripts/train_v43_local_recovery.py` 已加入：

- `--gate-only-finetune`
- `--recovery-head-only-finetune`：只训练 `direct_action.*`，且 batch 100% 从 recovery target rows 采样；要求 init checkpoint，并与 gate-only 互斥。

`scripts/precompute_v4_oft_relation_features.py` 已加入：

- `--success-only`
- `--include-prefork-gate-rows`
- `--task-ids`
- `--failed-prefix-gate-positive-steps`
- `--fixed-recovery-window-steps`

最后一项当前用于成功 teacher 的 62-step 固定恢复窗口。它要求 `--mark-all-recovery`、与 handoff manifest 互斥，并保证 8-action target 不跨出窗口。

### 13.3 已冻结的真实在线基线

Fresh OpenVLA-OFT 在 Book34 上成功：

```text
10 / 34
```

当前正式 V4.3 r2 在线成功：

```text
16 / 34
```

r2 在线结果目录：

```text
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/online-v43-book34-expanded-r2-uniform62
```

r2 checkpoint：

```text
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/v43-book34-expanded-r2/direct-v3-gw4-lr1e4-seed20260818/best.safetensors
```

相对 Fresh OFT：

- failure -> success：task495、496、497、503、506、510、518。
- success -> failure：仅 task493。
- 其余 baseline successes 保持 success。

因此 V4.3 已有净提升，不是只有离线 L1 的模型；但距离 34 条覆盖仍很远。

### 13.4 gate-only r3 诊断结论：不部署

gate-only r3 的用途只是定位瓶颈：

- 修复了 task493/499 的误触发，两者 success 且 zero trigger。
- 使原先不触发的失败 task502/508/516/517/519/520 开始触发，但这些 task 仍全部失败。
- task495 的 trigger 从 step0 推迟到 step8，action head 未变却从成功退化为失败。

结论：gate recall 不是当前唯一瓶颈，新增触发无法弥补 action coverage；不要部署 r3 gate，继续保留 r2 gate。

诊断目录：

```text
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/v43-book34-expanded-r3/gate-only-*
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/online-v43-book34-r3-lr1e3last-th080-targeted-r1
```

task516 在 step248 才触发已经不可恢复，即使用 oracle macro 也失败；以后收 task516 必须更早 fork，不能把晚触发失败误判成 action head 单独的问题。

### 13.5 截至 r6 的 recovery 数据和训练

最初 r2 recovery train：77 rows / 6 train sources；实际 recovery targets 为 63。r2 recovery val：38 rows / 3 sources；实际 recovery targets 为 30。

成功 teacher 扩展过程中已新增并严格重放/冻结的主要 task 包括：

```text
495, 496, 502, 503, 504, 514, 517, 518, 519, 521
```

其中 r4 的 7 条新成功 teacher：task495/496/502/503/517/518/519；另纳入已有成功 task504/521；r5 新增 task514。每条当前以 fixed 62-step window 冻结为 14 rows。

teacher、replay 和 feature 位置：

```text
# r4 成功 oracle teachers
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/oracle-onpolicy-recovery-teachers-r4

# r4 strict action-trace replay
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/action-trace-teachers-book34-r4

# r5 task514 teacher / replay
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/oracle-data-expansion-r5
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/action-trace-teachers-book34-r5/task514-episode0

# fixed-window features（包含上述新来源及 task504/521）
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/features-v43-book34-r4-window62
```

当前已经训练完成的最大数据集 r6：

```text
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/v43-book34-expanded-r6-data/recovery-train-expanded.safetensors
217 rows / 15 source_ids
```

当前 recovery-head-only 最佳 checkpoint：

```text
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/v43-book34-expanded-r6-data/recovery-head-only-lr1e4-seed20260818/best.safetensors
```

训练协议：init 上一轮 best，只训练 `direct_action.*`，1200 steps、batch32、lr=1e-4、100% recovery-target sampling，validation 固定使用 r2 recovery val。

结果：best 在 step800，held-out recovery validation L1 为 `0.0586743012`；原始 r2 约为 `0.0669128`，离线降低约 12.3%。last step1200 为约 `0.0590976`，因此必须使用 `best.safetensors`，不要误用 last。

### 13.6 r7 最新结果（2026-08-18，已完成严格重放）

r7 一次运行了 8 个 oracle geometry teacher 候选：

```text
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/oracle-data-expansion-r7
```

最终只有 2/8 成功：

| task | fork | geometry offset | source frames | simulator success |
|---|---:|---:|---:|---|
| 508 | 16 | X -0.025 m | 209 | true |
| 509 | 49 | Y -0.025 m | 264 | true |

其余 task508 X+0.025、task509 Y+0.025、task512 Y-0.025、task513 zero、task516 X-0.025、task520 X-0.025 均为 `success=false`，禁止并入 action imitation 数据。

两条成功 trace 已于 2026-08-18 完成严格 RGB-D/instance replay，且再次复现 simulator success：

```text
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/action-trace-teachers-book34-r7/task508-xm25-episode0
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/action-trace-teachers-book34-r7/task509-ym25-episode0
```

重放核验：

- task508：209 transitions，fork16，max EEF replay drift = 0.0 m。
- task509：264 transitions，fork49，max EEF replay drift = 0.0 m。
- schema 均为 `v4_rgbd_rollout_v2`，wait_steps=10，open_loop/decision_stride=8，env_seed=0。

**当前精确停点**：r7 两条 replay 已完成，但尚未做 feature freeze、尚未 merge 到 r6、尚未启动下一轮 head-only training。因此 r6 的 217 rows / 15 sources 仍是磁盘上当前最大已训练数据集。

### 13.7 下一位 AI 立即执行的顺序

1. 对 r7 task508/task509 strict replay 分别运行 `precompute_v4_oft_relation_features.py`，使用与 r4/r5 完全一致的 OFT checkpoint/provenance、`--mark-all-recovery --fixed-recovery-window-steps 62`。预期每条约 14 recovery rows，但必须以生成报告为准。
2. 用 `merge_v4_relation_features.py` 将这两个 shard 与 r6 `recovery-train-expanded.safetensors` 合并到新目录（建议命名 `v43-book34-expanded-r7-data`）。如果均为 14 rows，预期约 245 rows / 17 sources；不要在未核验 report 前硬写这个数字。
3. 启动 `--recovery-head-only-finetune`：init 使用 r6 `best.safetensors`，其他超参保持 1200 steps、batch32、lr1e-4、seed20260818、同一个 r2 recovery val。
4. 训练完成后至少保存 best step/L1，并做一次 Book34 online matrix 才能判断真实成功率。离线 L1 下降不能替代 online success。
5. 数据主线继续扩大，但停止低收益的单点 XY offset 盲扫。优先从每个 task 的成功轨迹构造**可控、可恢复的扰动状态**，再由成功 teacher 完成恢复；目标是快速增加 physical task coverage，而不是为同一 source 堆很多几乎相同的 fork。

### 13.8 数据扩展中的失败事实与策略调整

以下 task 已经测试多个固定 offset 仍失败：

- task508：offset 0、Y±0.025、X+0.025 失败；X-0.025 成功。
- task516：offset 0、Y±0.025、X-0.025 等均失败，且晚 fork248 不可恢复；要改用更早、可恢复 perturbation。
- task520：offset 0、Y±0.025、X-0.025 等均失败。
- task509：offset 0、Y+0.025 失败；Y-0.025 成功。
- task512、513 当前尝试也失败。

这说明固定 anchor macro + 手工 offset 的成功区域稀疏。r7 的 2/8 不是两个 tensor samples，而是 473 个连续 action transitions，冻结后约产生 28 个 8-action supervised chunks；但 physical coverage 只增加两个 task，产出率仍不理想。后续不要把“GPU 跑满”误当成“数据策略有效”。

### 13.9 仍然必须遵守的工程约束

- 只接受最终 simulator `success=true` 且 strict replay 再次 success 的 recovery action trace。
- normal action 在 replay 保持来源 float64，recovery action 保持来源 float32；不要统一 cast。
- env 必须先 `env.seed(0)`，再 reset/set_init_state；wait_steps/open_loop_steps 必须写 metadata 并核对。
- source split 使用 `source_id`；同 source 的不同 concrete fork 用 `source_instance_id` 去重。不同 fork 不可跨 split。
- inference 禁止 instance segmentation、object pose 或 simulator privileged state；instance 仅用于训练 Slot 标签。
- 当前 online 使用 r2 gate + 新 direct action head。加载 head-only checkpoint 时要确认它包含并保留 r2 其余权重，不能只抽出 direct head 后漏掉 gate/representation 参数。
- simulator rollout 的 MuJoCo/env step 大量运行在 CPU；GPU 主要用于 OFT inference/feature freeze/training，利用率会呈 burst。判断作业是否运行应同时看 PID、显存和输出增长。
- 工作树有大量未提交 V4 文件，严禁 reset/checkout/清理用户修改。

### 13.10 当前一句话结论

截至 2026-08-18：V4.3 r2 已把 Book34 从 Fresh OFT 的 10/34 提升到 16/34；扩大 recovery 数据并只微调 direct head 后，217-row/15-source r6 将 held-out recovery L1 从约 0.0669 降到 0.05867；r7 又得到并严格重放了 task508/task509 两个新成功来源，但尚未冻结和重训。下一步应立刻完成 r7 freeze -> merge -> head-only train -> Book34 online 验证，同时将数据生成从低效 offset 盲扫转向成功轨迹上的可恢复扰动构造。

---

## 14. 2026-08-19 Book34 快速收尾：task-ID overfit selector 达到 29/34

用户将最终目标明确改为：固定 LIBERO-plus Book34（tasks491--524、episode0）达到至少 27/34 或 28/34 即可，允许 task-specific overfit，不再要求这一阶段证明泛化。

### 14.1 r7 数据和训练已经完成

r7 task508/task509 strict replay 已各冻结 14 rows：

```text
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/features-v43-book34-r7-window62/task508-xm25.safetensors
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/features-v43-book34-r7-window62/task509-ym25.safetensors
```

合并数据：

```text
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/v43-book34-expanded-r7-data/recovery-train-expanded.safetensors
245 rows / 17 sources
```

head-only 训练：

```text
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/v43-book34-expanded-r7-data/recovery-head-only-lr1e4-seed20260818
```

r7 best 在 step100，固定 r2 recovery val L1=`0.0591391772`，略差于 r6 best 的 `0.0586743012`。因此仅按离线指标不能替换 r6；这不影响下面的 task-specific 快速收尾策略。

### 14.2 29/34 selector

已有两组真实 simulator success 证据：

- V4.3 r2 Book34 在线成功 16 个 task。
- 完整 `success=true`、有 `steps.jsonl`、有非空 MP4 的 strict verified action trace 覆盖 20 个 task。
- 两者 task-ID 并集为 29 个 task。

最终 selector 规则：固定 episode0 上，如果 task 有 verified task-specific action trace，就直接执行该 trace；否则执行冻结的 V4.3 r2。该策略显式使用 task ID，是允许的过拟合，不得包装成泛化结果。

统一审计脚本：

```text
scripts/build_book34_overfit_selector.py
```

统一结果报告：

```text
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/book34-overfit-selector-v1/report.json
```

结果：

```text
29 / 34 = 85.2941%
```

成功 task：

```text
491,492,494,495,496,497,498,499,500,501,502,503,504,505,506,507,
508,509,510,511,514,515,517,518,519,521,522,523,524
```

剩余失败 task：

```text
493,512,513,516,520
```

报告逐 task 保存实际选择模式、r2 report 或 strict trace 路径、trace metadata、steps 和 MP4。构建脚本会拒绝没有 executable steps 或非空 MP4 的 teacher，因此不是仅凭目录名统计。

### 14.3 teacher 低成功率的最终诊断

r7 失败视频确认：task508 的错误 offset、task512、task516 会把竖书碰倒后空手搬运；task513、task509 错误 offset、task520 主要是放置到 caddy 外沿或错误位置。代码原因是旧 geometry FSM 在闭夹后只按时间和 EEF pose 无条件进入 lift/transport/release，没有 grasp verification 或 placement verification。

r7 所有 8 条 teacher 都有 MP4。此后若再生成 teacher，成功和失败都必须保留 MP4；接纳数据必须检查 `success=true`、完整 steps、MP4 非空且帧数合理。

### 14.4 当前终点

用户要求的 Book34 >=80% 已通过明确允许的 task-specific overfit selector 达成（29/34，85.29%）。如果不再要求单一神经网络、未知 episode 或新 initstate 泛化，本任务可以在此收尾。不要继续为了 5 个剩余失败 task 消耗时间，也不要将 29/34 写成纯 V4.3 neural model 的成功率；纯 r2 neural online 仍是 16/34。

## 15. 2026-08-19 新主线：mug + pudding 42-task V4.3 训练

> 本节晚于第14节，是当前继续工作的权威状态。Book34 已收尾；当前 benchmark 已切换到 LIBERO-plus 的另一个语义组。用户最新明确要求：本任务必须交付一个**训练得到、可在线复现成功率的 checkpoint**，不能再把 task-ID selector、散落 action trace 或 privileged controller 当最终模型。允许过拟合固定 42 个 task，但最终数字必须由同一训练 checkpoint 在线跑出。

### 15.1 当前目标和 baseline

语义：

```text
put the white mug on the plate and put the chocolate pudding to the right of the plate
```

范围：

```text
LIBERO-plus patched libero_10
task IDs 525--566，共42条
episode0
```

Fresh OpenVLA-OFT+ baseline 为 `20/42 = 47.62%`。成功 task：

```text
525,526,527,528,529,530,531,532,533,534,535,536,537,539,540,546,547,552,553,563
```

失败 task：

```text
538,541,542,543,544,545,548,549,550,551,554,555,556,557,558,559,560,561,562,564,565,566
```

原始 baseline 结果：

```text
/data2/lty/naive-e-slotssm-runs/libero-plus/oft-plus-robot-levels1-4-v1
/data2/lty/naive-e-slotssm-runs/libero-plus/oft-plus-robot-level5-v1
```

更大的 `Robot Initial States` 审计已确认完整：393/393 JSON、393/393 非空 MP4、无 task/episode 重复或缺失。Fresh OFT 总成功 `149/393=37.91%`；分 level 为 L1 `54/64`、L2 `55/77`、L3 `29/100`、L4 `9/83`、L5 `2/69`。本组因为已有20个成功且距80%只差14个，被选作 Book 后的下一组。

### 15.2 已有轨迹和严格 teacher 构造

重要发现：之前的 global OFT sweep 已为 525--566 全42条保存逐步动作、RGB/wrist RGB、proprio 和每8步 MuJoCo snapshot：

```text
/data2/lty/naive-e-slotssm-runs/oft-plus-robot-global-v1/forks/task{525..566}-episode0
```

新增转换脚本：

```text
scripts/build_oft_trajectory_action_trace.py
```

它把已经执行过的 LIBERO gripper 命令 `-1/+1` 显式逆变换回 strict replay 所需的 policy gripper 坐标 `1/0`，禁止静默混淆动作协议。20条 baseline success 被转换到：

```text
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/mug-pudding/action-trace-inputs-v1
```

随后用 `scripts/collect_v4_action_trace_replay.py` 重放为 V4 RGB-D + wrist RGB-D + instance schema。20条中17条严格复现成功，task529/533/539 虽历史报告 success，但重新渲染后 final predicate 未复现，已按严格标准剔除。有效 teacher tasks：

```text
525,526,527,528,530,531,532,534,535,536,537,540,546,547,552,553,563
```

有效 teacher 位置：

```text
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/mug-pudding/strict-teachers-v1
```

共3364 simulator transitions；17/17 有 `success=true`、完整 steps/RGB/depth/instance 和非空 `rollout.mp4`。task525 的最大逐步 EEF replay drift 为 `4.39e-9m`；其他接纳轨迹也通过 collector 的逐步 `<1cm` 硬阈值。失败的部分输出目录不能用于训练，因为没有 metadata.json。

### 15.3 V4.3 feature 与 split

按固定 local recovery window 62 steps、recovery sampling stride 4、action target 8 steps 冻结：

```text
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/mug-pudding/features-v43-v1/recovery-all.safetensors
```

结果：

```text
238 rows / 17 physical source_ids
每 source 14 rows
所有 rows gate_label=RECOVERY，且都有 recovery action target
```

source-disjoint split，seed `20260819`：

```text
train: recovery-train.safetensors = 182 rows / 13 sources
val:   recovery-val.safetensors   = 56 rows / 4 sources
```

这里把成功 nominal OFT 的前62步动作显式当作当前 task family 的 recovery teacher。它是本轮第一版数据扩展，不等于已经证明62步 handoff 最优；若闭环显示 Frozen OFT 接手过早，应把窗口扩到约94/126步并重新 freeze，而不是在错误 handoff 上盲调网络。

### 15.4 已训练 checkpoint

训练脚本：

```text
scripts/train_v43_local_recovery.py
```

从 Book V4.3 r2 初始化，但本轮不是 head-only：relation representation、统一 competence gate 和 direct action head 一起适配当前语义。训练配置：3000 steps、batch32、lr1e-4、gripper weight4、seed20260819、relation_3d。输出：

```text
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/mug-pudding/v43-trained-v1/full-ft-lr1e4-seed20260819
```

必须使用：

```text
best.safetensors
```

best 在 step1400：

```text
held-out val total       0.1444382056
held-out recovery loss  0.0274604947
held-out gate loss      0.1169777108
```

step1 的 held-out recovery loss 为 `0.172396`，下降约84%。后期 train loss 继续下降但 val 变差，因此不能用 `last.safetensors` 代替 best。

### 15.5 当前正在运行的线上验证

在线 runner 仍为：

```text
scripts/run_v4_relational_online.py
```

本轮必须使用 `wait_steps=10`，因为 teacher/source rollout 也是 wait10；不能沿用 Book 命令中的 wait20。第一轮先测：

```text
task525: train source
task563: source-disjoint validation teacher
task538: Fresh OFT failure
```

输出根目录：

```text
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/mug-pudding/online-v1
```

当前 deployment probe 参数：

```text
gate threshold 0.5
execution horizon 4
recovery horizon 62
minimum recovery steps 62
cooldown 1000
wait steps 10
```

`cooldown=1000` 的目的，是先验证一次从 step0 附近触发的局部 recovery 后交还 Frozen OFT，避免同一 episode 不断重复激活。最终参数必须以在线报告为准；离线 L1 不能替代在线成功率。

### 15.6 放弃作为最终方案的分支

为了快速探测上限，曾新增：

```text
scripts/run_mug_pudding_privileged_controller.py
scripts/build_mug_pudding_privileged_checkpoint.py
scripts/build_mug_pudding_overfit_checkpoint.py
```

privileged Cartesian controller 读取 simulator object position、只通过标准7D robot action执行，不 teleport 状态；其中 v2 在 task538 成功。但用户随后明确要求像 V4.3 一样采 teacher 并训练 checkpoint，因此这些文件/结果只能作为后续 teacher 生成器或诊断，**不得作为最终模型成功率**。调试结果位于：

```text
/data2/lty/naive-e-slotssm-runs/libero-plus/mug-pudding-checkpoint-v1
```

另一个 OFT oracle prompt-stage-switch 在 task538/541 上均未触发第一子目标，已停止；不要继续把时间花在只切 prompt 的批量实验上。

### 15.7 下一步顺序

1. 等 task525 online-v1 完成，读取 `report.json`，先看 success、gate probability、首次 trigger、62步后状态和视频是否完整。
2. 同协议跑 task563 与 task538。若 train/val teacher task 都失败，直接判定 BC rollout drift/handoff 协议问题，不要立刻扫42条。
3. 若 teacher tasks 成功且 task538 有改善，立即跑 525--566 全42条 online matrix，保存每条 report + MP4，给出同一 `best.safetensors` 的真实成功率。
4. 若动作前62步合理但 OFT 接手过早，使用同17条 strict teacher 重新冻结 94或126步窗口，重训并延长 latch；不要重新采原始轨迹。
5. 若在 teacher task 上前几十步就漂移，收集本 checkpoint 自己 rollout 的偏离状态，使用 privileged Cartesian teacher 或成功 OFT trace 的可恢复扰动做 DAgger。新增 teacher 必须 success=true、steps 完整、MP4 非空、严格 replay 后才入库。
6. 42条目标仍是至少34/42（80.95%）。达标数字必须来自一个训练 checkpoint 的完整 online matrix；baseline success 与新模型 success 不能事后取 union。

### 15.8 当前最重要的坑

- 当前 task family 使用 Plus OFT checkpoint 和 normalization key `libero_10`；不要混入早期 canonical task6 的旧 `/openvla-7b-finetuned-libero-10` / `libero_10_no_noops` 数据。
- task525--566 虽与历史 `libero_10 task6` 语义相同，但 Plus task ID、initstate、view、checkpoint 和 wait协议不同，不能因为语言相同就直接混源。
- strict replay 的 source action 已是 executed gripper `-1/+1`；转换回 policy trace 时必须用 `(1-executed)/2`，否则 replay 会把开合反转。
- 只有包含 metadata.json 且 `success=true` 的 strict teacher 才能被 feature precompute 接纳。task529/533/539 的不完整 strict 输出必须排除。
- 当前17个 teacher 都来自原本就成功的 OFT task；它们扩大状态覆盖但还没有失败态 DAgger。若第一次 checkpoint 对22个 baseline failures 没提升，下一步的真正数据瓶颈是失败分布上的 teacher，而不是继续在17条 easy sources 上增加 epoch。
