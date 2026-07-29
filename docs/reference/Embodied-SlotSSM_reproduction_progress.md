# Embodied-SlotSSM 复现工作说明与当前进展

> 更新时间：2026-07-17  
> 当前仓库：`/data/lty/code/naive-e-slotssm`  
> 运行环境：8 × RTX 4090 无头服务器，Conda 环境 `openvla-a`

## 1. 目标与当前结论

本项目的目标是在 OpenVLA 基础上复现论文 **Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective** 中的 Naive Embodied-SlotSSM，并最终在 LIBERO / LIBERO-Mem 仿真中比较：

1. 原始 OpenVLA；
2. OpenVLA + 对象 Slot；
3. OpenVLA + SlotSSM；
4. 完整的 Embodied-SlotSSM，包括关系 token、对象子目标和时间记忆。

目前已经完成了三个阶段的最小链路验证：

```text
LIBERO-Mem RGB 序列
        ↓
OpenVLA 冻结视觉编码器
        ↓
Stage-1 对象 Slot
        ↓
Stage-2 Mamba2 SlotSSM
        ↓
Slot token 投影并注入 OpenVLA LLM
        ↓
连续 7 维动作头
```

当前能够进行一次完整的前向传播、动作损失计算、反向传播和权重保存。但这仍然是 **smoke test（冒烟测试）**，用于证明代码路径可运行，不是已经训练完成的策略，也还没有得到仿真成功率。

## 2. 论文中的三个训练 Stage 分别做什么

### 2.1 Stage-1：训练对象 Slot 表示

Stage-1 的任务是把一帧 RGB 图像中的对象压缩成固定数量的对象 token。

主要流程：

```text
RGB 图像
  ↓
冻结的 OpenVLA 视觉编码器
  ↓
视觉 patch features
  ↓
Slot Attention / object-centric tokenizer
  ↓
16 个对象 Slot
  ├── bbox head：预测对象边界框
  ├── objectness head：判断该 Slot 是否对应有效对象
  └── mask head：预测对象分割 mask
```

其中：

- `bbox` 是 bounding box，即对象在图像中的矩形边界框；
- `mask` 是逐像素对象区域，比 bbox 更精细；
- Slot 的目标不是直接生成机器人动作，而是建立稳定、紧凑、尽可能具有对象身份含义的视觉表示；
- 不同时间帧中，同一个 Slot 最好持续代表同一个对象，否则后面的时间模型难以学习对象状态变化。

Stage-1 训练完成后得到对象 Slot tokenizer、bbox head 和 mask head 的权重。

### 2.2 Stage-2：训练 SlotSSM 对象时间记忆

Stage-2 接收连续多帧的对象 Slot，并学习对象状态如何随时间变化。

主要流程：

```text
连续图像序列
  ↓
Stage-1 Slot tokenizer
  ↓
[s1, s2, ..., st]
  ↓
对象级 SSM（当前采用 Mamba2）
  ↓
时间动态表示 / 下一时刻 Slot 预测
```

其训练目标是让模型根据历史对象状态预测后续对象状态，例如：

- 碗是否已经被拿起；
- 碗当前在哪个盘子上；
- 某个动作阶段是否已经完成；
- 同一操作已经重复了多少次。

论文强调的核心正是：不能只看当前画面，而要维护对象级的历史状态。Stage-2 不直接负责输出最终动作，它生成供动作解码器使用的对象时间记忆。

### 2.3 Stage-3：让 SlotSSM 影响动作生成

Stage-3 把 Stage-1 和 Stage-2 得到的对象表示接入 VLA 动作解码器。

论文目标路径应当包括：

```text
当前对象 Slot + SlotSSM 时间动态 + oracle 对象子目标
                        ↓
                    Slot Fusion
                        ↓
              16 个 slot dynamics token

slot dynamics 查询原始视觉 patches
                        ↓
                 Relation Encoder
                        ↓
              16 个 relation token

[16 个 dynamics token, 16 个 relation token, 语言指令]
                        ↓
                  VLA action decoder
                        ↓
                     机器人动作
```

因此，完整论文版本的 Stage-3 应该使对象动态、对象关系和语言共同影响动作。

我们当前跑通的是一个较小但真实连接 OpenVLA 的版本：

```text
16 个最终时刻 SlotSSM token
        ↓
512 → 4096 投影层
        ↓
插入 OpenVLA/Llama 的输入 embedding 序列
        ↓
冻结的 OpenVLA 7B LLM
        ↓
连续 7 维动作头
```

它已经证明 SlotSSM token 可以实际改变 OpenVLA 的动作输出，但还没有实现论文中的完整 Slot Fusion、Relation Encoder、oracle subgoal 和 32 个对象相关 token。

## 3. 原开源代码存在的主要问题

### 3.1 代码不是开箱即用的完整发布

原仓库包含不少研究过程代码和未清理脚本，三个训练阶段无法直接顺序运行。部分代码更接近作者内部实验快照，而不是具有完整配置、依赖说明、数据准备和评测入口的复现工程。

### 3.2 Stage-1 的问题

- 存在作者机器上的私有路径和默认 checkpoint 路径；
- 部分依赖导入是强制的，即使当前阶段根本不需要，也会阻止程序启动；
- 保存步数、最终 checkpoint 保存和恢复训练逻辑不够可靠；
- 默认日志系统依赖外部服务，不适合先做本地无头 smoke test；
- 数据中的 NumPy 图像可能是只读数组，直接转换为 PyTorch tensor 会产生警告；
- 原 mask/object loss 对当前 LIBERO-Mem 数据不够稳定，需要调整；
- 视觉、语言模型和投影器中有 Stage-1 不需要的模块，会占用额外显存。

### 3.3 Stage-2 的问题

- 原训练脚本中存在未定义或作用域不完整的权重变量；
- 依赖私有 checkpoint 路径；
- 模型构造、权重加载和冻结顺序不够清晰；
- 对 Mamba、Flash Attention 等可选依赖缺少健壮的兼容处理；
- `FlashMHA` 不可用时仍被用于 `isinstance()`，会触发：

  ```text
  TypeError: isinstance() arg 2 must be a type or tuple of types
  ```

- 没有一个独立、低成本的 Stage-2 最小验证入口，无法快速判断 Slot → SSM → loss → backward 是否正常；
- 代码缺乏对张量形状、设备和梯度有限性的系统检查。

### 3.4 Stage-3 的问题

Stage-3 原始脚本的问题最严重，包括：

- 导入了仓库中不存在的 `EmbodiedDecodedSlotSSM`；
- 含有作者机器上的 `/cm/shared/...` 私有权重路径；
- 出现先 `del weights`、随后又执行 `load_state_dict(weights)` 的明显错误；
- 部分代码重复或处于不完整状态；
- 没有明确可用的 wrapper 将 OpenVLA、Slot tokenizer、SlotSSM、token projector 和 action head 组合起来；
- 单卡加载完整 OpenVLA 7B 容易 OOM；
- 普通 DDP 只复制整个模型，不能解决单卡容纳不了模型的问题；
- 8 卡模型并行时，Accelerate 的设备分配与 Mamba2/Triton 的当前 CUDA device 可能不一致，从而报错：

  ```text
  ValueError: Pointer argument (at 0) cannot be accessed from Triton
  ```

- 整体加载 Stage-2 小 checkpoint 到完整 OpenVLA wrapper 时会显示大量 `missing keys`。这通常不代表 Stage-2 权重加载失败，而是 checkpoint 本来就不包含冻结的 OpenVLA 7B 参数，但日志容易造成误解。

### 3.5 数据和评测方面的问题

- 原代码没有提供从原始 LIBERO-Mem HDF5 到当前训练入口所需 RLDS 数据的完整可重复流程；
- 数据路径、任务命名和元信息依赖作者环境；
- 缺乏独立的 Slot 可视化与 Slot identity 检查；
- 缺少可直接运行的无头 LIBERO 仿真评测入口；
- 当前仓库也没有提供能够直接复现论文表格成功率的一键配置；
- 论文的部分结构细节和超参数本身没有完全公开，因此不能把工程假设称为作者原始实现。

## 4. 我们已经完成的工作

### 4.1 环境与 OpenVLA 基线

- 建立并使用 Conda 环境：

  ```text
  /data/lty/code/naive-e-slotssm/.conda/envs/openvla-a
  ```

- 下载并加载本地 OpenVLA LIBERO-Spatial checkpoint；
- 验证 OpenVLA 基础模型能够运行；
- 针对 8 × 4090 且显卡同时被其他用户占用的情况，建立按剩余显存动态分配模型层的方案；
- 采用单进程 model parallel，而不是会在每张卡复制完整模型的普通 DDP。

### 4.2 数据准备

- 准备了 LIBERO-Mem HDF5 数据和 `metainfo.json`；
- 建立了 HDF5 → RLDS 的本地转换工具；
- 已生成当前任务的 RLDS 数据：

  ```text
  /data2/lty/libero-mem-rlds/libero/rlds_datasets/libero_mem/1.0.0
  ```

- 当前已用于训练的数据主要是一个 LIBERO-Mem 任务、100 条轨迹；
- 数据 transform 已补充连续动作 `continuous_actions`，供 Stage-3 连续动作训练使用。

### 4.3 Stage-1：对象 Slot

我们修复了 Stage-1 训练入口，包括：

- 修正模型类和导入；
- 移除私有路径依赖；
- 支持从已有 Slot checkpoint 恢复；
- 减少不必要的 LLM/投影器显存占用；
- 修正训练步数和最终 checkpoint 保存；
- 支持无外部日志服务运行；
- 将 mask 损失改进为 BCE + Dice 组合；
- 增加稳定性测试、正式小规模训练、refine 训练和可视化脚本。

已经得到的最终 Stage-1 checkpoint：

```text
/data2/lty/naive-e-slotssm-runs/stage1-adapters/
openvla-7b-finetuned-libero-spatial+libero_mem+b1+lr-1e-05
--refine-bce-dice-h16-s2k--image_aug--multiview1000--object/
pretrained_slots-wo-filters_s16.safetensors
```

并生成了 Slot/bbox/mask 可视化：

```text
/data2/lty/naive-e-slotssm-runs/stage1-visualization-refined/slot_predictions.png
```

当前观察是：Slot identity 已具备一定稳定性，但 bbox 和 mask 仍较粗糙，而且只在单任务数据上训练，存在明显过拟合风险。

### 4.4 Stage-2：对象级 Mamba2 SlotSSM

- 增加纯 PyTorch `ReferenceDiagonalSSM`，用于不依赖 CUDA kernel 的结构验证；
- 使用 CUDA 12.1 安装并启用真正的 `mamba-ssm==2.2.2`；
- 修复 FlashMHA 可选依赖导致的 `isinstance` 报错；
- 新增独立 Stage-2 smoke test；
- 验证 Slot 序列能够进入 Mamba2；
- 验证 loss、反向传播和 checkpoint 保存均正常。

成功结果：

```text
Stage-2 mamba2 SSM loss: 44.414455
Gradients finite: True
```

checkpoint：

```text
/data2/lty/naive-e-slotssm-runs/stage2-smoke/mamba2-slot-ssm.safetensors
```

注意：这只是一次 smoke 优化结果，尚未进行论文规模的 Stage-2 正式训练。

### 4.5 Stage-3：轻量动作头验证

我们先建立了不经过完整 7B LLM 的轻量动作路径，用于确认 SlotSSM 能否产生动作并回传梯度。

成功结果：

```text
Stage-3 action L1 loss: 0.366177
Stage-3 Slot consistency loss: 40.924580
Predicted action shape: (1, 16, 7)
Action-head gradients finite: True
```

checkpoint：

```text
/data2/lty/naive-e-slotssm-runs/stage3-smoke/slotssm-action-head.safetensors
```

该结果只验证轻量连续动作头，不属于 OpenVLA LLM 微调。

### 4.6 Stage-3：SlotSSM 注入真实 OpenVLA

我们新增了一个明确的 wrapper：

```text
OpenVLASlotSSMActionPath
```

它负责组合：

- OpenVLA 基础模型；
- Stage-1 Slot tokenizer；
- Stage-2 Mamba2 SlotSSM；
- Slot 维度投影层；
- OpenVLA LLM；
- 连续 7 维动作头。

同时完成：

- 8 卡按剩余显存动态 model parallel；
- 自动识别 vision、token embedding、最终 LLM 层所在设备；
- 将辅助 Slot/Mamba 模块放置到剩余显存较多的 GPU；
- 修复 Mamba2 Triton kernel 的 current CUDA device 与张量所在设备不一致问题；
- 加入设备一致性检查；
- 加入 Slot token 置零消融，验证动作是否真的依赖 SlotSSM；
- 验证投影层和动作头梯度均为有限值。

最新成功结果：

```text
OpenVLA SlotSSM action L1 loss: 0.392595
Predicted action shape: (1, 7)
Mean action change when SlotSSM tokens are zeroed: 0.223633
Injection/action gradients finite: True
```

checkpoint：

```text
/data2/lty/naive-e-slotssm-runs/stage3-openvla-smoke/
slotssm-openvla-action-path.safetensors
```

这个结果表明：

1. `RGB → Slot → Mamba2 → OpenVLA LLM → action` 全链路已经打通；
2. SlotSSM token 置零后，动作平均变化约 `0.2236`，说明它们确实影响了动作输出；
3. 该变化目前不代表性能提升，因为注入层和动作头还没有经过正式训练；
4. 单次 L1 loss 不能用于衡量 LIBERO 成功率。

## 5. 当前实现与完整论文方法之间仍有的差距

| 项目 | 当前状态 | 完整复现要求 |
|---|---|---|
| OpenVLA 基线 | 已跑通 | 在统一测试集记录成功率 |
| Stage-1 Slot | 小规模训练完成 | 扩大到完整多任务数据并量化质量 |
| Stage-2 SlotSSM | Mamba2 smoke 跑通 | 正式训练、验证下一 Slot 和长期记忆 |
| Slot → OpenVLA 注入 | 已跑通 | 正式 Stage-3 训练 |
| 连续动作头 | 单步 smoke 跑通 | 动作 chunk、归一化和训练/验证闭环 |
| Relation Encoder | 尚未接入 | 生成论文中的 16 个 relation tokens |
| Slot Fusion | 尚未完整实现 | 融合当前 Slot、预测 Slot 和子目标 |
| Oracle subgoal | 尚未接入 | 从 LIBERO-Mem 标注构造对象子目标状态 |
| Past MLP / temporal auxiliary branch | 尚未复现完整 | 按论文训练时间窗口辅助目标 |
| 32 个对象相关 token | 当前只有 16 个 SlotSSM token | 16 dynamics + 16 relation |
| 多任务 LIBERO-Mem | 当前主要为单任务 | 下载、转换并训练全部任务 |
| 无头仿真 | 尚未完成 SlotSSM 评测 | 实现 reset、历史缓存、动作执行和成功率统计 |
| 消融实验 | 已有 token-zero smoke | baseline / Slot / SlotSSM / full model 对比 |

## 6. 接下来建议的工作顺序

### 6.1 第一优先级：把 Stage-3 smoke 扩展为正式训练

需要新增正式训练脚本，至少支持：

- 加载固定的 Stage-1 checkpoint；
- 加载固定的 Stage-2 checkpoint；
- 冻结 OpenVLA、视觉编码器、Stage-1 和 Stage-2；
- 先只训练 `slot_projector + action_head`；
- 使用连续动作监督；
- 梯度累积、bf16、定期保存和断点续训；
- 训练/验证数据分离；
- 日志记录 action loss、梯度范数、Slot 置零差异；
- 避免通过 `output_hidden_states=True` 保存所有 LLM 层输出，降低模型并行显存；
- 为共享 GPU 环境保留动态 `max_memory` 和安全余量。

这是距离“可以进仿真看动作”的最短路径。

### 6.2 第二优先级：建立无头 LIBERO 最小闭环

先用最简单任务完成：

```text
仿真 reset
  ↓
获取 RGB + 语言指令
  ↓
维护最近 N 帧 Slot/SSM 状态
  ↓
模型输出 7 维动作
  ↓
仿真 step
  ↓
直到成功或超时
```

需要处理：

- EGL 或其他无头渲染后端；
- 摄像头预处理与训练时一致；
- 动作反归一化；
- gripper 维度和符号约定；
- episode 开始时清空 SlotSSM 状态；
- 记录视频、动作、模型输出和最终成功标志。

第一轮目标不是达到高成功率，而是确认模型可以在无头仿真中连续执行完整 episode。

### 6.3 第三优先级：正式训练 Stage-2

当前 Stage-2 checkpoint 只来自一次 smoke step。应当：

- 使用更多轨迹和更多时间窗口；
- 建立 train/validation split；
- 监控下一 Slot 预测误差；
- 检查 Slot identity 是否随时间漂移；
- 比较纯 PyTorch reference SSM 与 Mamba2；
- 保存 best/last checkpoint；
- 对不同历史长度做消融。

### 6.4 第四优先级：补齐论文结构

在基础版本能够进入仿真后，再逐步加入：

1. Slot Fusion；
2. LIBERO-Mem oracle object subgoal；
3. Relation Encoder；
4. 16 个 dynamics token + 16 个 relation token；
5. Past MLP 和论文中的时间辅助目标；
6. 完整多任务 LIBERO-Mem。

每加入一个模块都应保留前一个版本作为消融组，否则无法判断性能变化来自哪里。

### 6.5 第五优先级：正式对比实验

建议最终至少保存四套可重复配置：

```text
A. 原始 OpenVLA
B. OpenVLA + 当前对象 Slot
C. OpenVLA + SlotSSM
D. 完整 Embodied-SlotSSM
```

统一比较：

- LIBERO-Goal 成功率；
- LIBERO-Mem 各任务成功率；
- 长序列任务与重复操作任务的性能；
- Slot token 置零；
- 去掉 SSM；
- 去掉 relation token；
- 去掉 oracle subgoal；
- 推理显存、速度和每个 episode 的耗时。

## 7. 当前最重要的判断

目前可以确认的不是“论文已经复现成功”，而是：

- 原仓库代码确实存在多处会阻止直接运行的问题；
- 在该仓库上继续修补是可行的，但 Stage-3 和最终评测需要较大程度的工程重构；
- Stage-1 对象 Slot、Stage-2 Mamba2、Stage-3 OpenVLA 动作注入都已经分别和串联跑通；
- SlotSSM token 已经能够真实影响 OpenVLA 的动作输出；
- 下一步应优先完成正式 Stage-3 训练和无头 LIBERO 最小 rollout；
- 在得到基础闭环后，再补 Relation Encoder、Slot Fusion、oracle subgoal 等论文完整模块，风险最低，也最容易做清晰的前后对比。

## 8. 相关文件

论文复现规格：

```text
Embodied-SlotSSM_paper_only_reproduction_spec.md
```

原三阶段脚本：

```text
vla-scripts/finetune-1_slotvla_slot-wo-filter.py
vla-scripts/finetune-2_slotssm_pretrain.py
vla-scripts/finetune-3_slotssm_to_action.py
```

我们新增的主要验证脚本：

```text
scripts/smoke_slot_stage2_ssm.py
scripts/smoke_slot_stage3_action.py
scripts/smoke_slot_stage3_openvla.py
scripts/visualize_slot_stage1.py
```

对应启动脚本：

```text
scripts/run_slot_stage1_smoke.sh
scripts/run_slot_stage1_stability.sh
scripts/run_slot_stage1_train_1k.sh
scripts/run_slot_stage1_refine_2k.sh
scripts/run_slot_stage1_visualize.sh
scripts/run_slot_stage2_smoke.sh
scripts/run_slot_stage3_action_smoke.sh
scripts/run_slot_stage3_openvla_smoke.sh
```
