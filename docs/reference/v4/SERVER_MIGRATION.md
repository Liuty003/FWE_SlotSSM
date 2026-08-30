# 新服务器迁移清单

本文记录继续当前 LIBERO-Plus recovery 工作及复现论文结果时必须迁移的代码、模型、数据和环境信息。迁移后运行：

```bash
python scripts/verify_server_migration.py
```

脚本全部通过后，再启动训练或评测。

## 1. 推荐目录结构

尽量在新服务器保持当前路径：

```text
/data/lty/code/naive-e-slotssm
/data/lty/code/openvla-oft-official
/data/lty/code/LIBERO-plus
/data2/lty/models
/data2/lty/naive-e-slotssm-runs
```

原因是部分历史脚本、LIBERO 配置和已有 manifest 保存了绝对路径。若新服务器不能使用这些挂载点，应通过命令行参数重定向；对不支持重定向的历史入口，可建立兼容软链接。不要批量修改已有 JSON/manifest 中的 provenance。

## 2. 必须迁移：代码

### 主仓库

- GitHub：`git@github.com:Liuty003/FWE_SlotSSM.git`
- 分支：`main`
- 迁移时使用本文件所在提交或更新提交。
- GitHub 仓库不包含模型、运行数据、本地 Conda 环境和 MP4 大文件。

### OpenVLA-OFT 官方代码

- 路径：`/data/lty/code/openvla-oft-official`
- 上游：`https://github.com/moojink/openvla-oft.git`
- 固定提交：`e4287e94541f459edc4feabc4e181f537cd569a8`
- 当前只有未跟踪的 `MUJOCO_LOG.TXT`，不需要迁移。

### LIBERO-Plus

- 路径：`/data/lty/code/LIBERO-plus`
- 上游：`https://github.com/sylvestf/LIBERO-plus.git`
- 固定提交：`4976dc30028e805ff8094b55501d532c48fec182`
- 上游仓库之外还必须保留 `baseline_libero_config/`。
- `libero/libero/assets` 当前是指向迁移资产目录的软链接，clone 后需要重新创建。

## 3. 必须迁移：基础模型和仿真资产

| 内容 | 当前路径 | 大小 |
|---|---|---:|
| OFT+ 7B 基础模型 | `/data2/lty/models/openvla-7b-oft-finetuned-libero-plus-mixdata` | 15.99 GB |
| LIBERO-Plus 解压资产 | `/data2/lty/naive-e-slotssm-runs/libero-plus/assets/extracted` | 10.19 GB |

OFT 模型目录必须整体迁移，不能只复制四个 LLM shard。以下内容同样必需：

- `action_head--150000_checkpoint.pt`
- `proprio_projector--150000_checkpoint.pt`
- 当前 `config.json` 与 `dataset_statistics.json`
- tokenizer、processor 和本地 modeling/configuration 文件

当前 `config.json` 包含已同步的 LIBERO normalization，不能用重新下载的原始配置覆盖。

迁移资产后恢复软链接：

```bash
ln -s /data2/lty/naive-e-slotssm-runs/libero-plus/assets/extracted/inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets \
  /data/lty/code/LIBERO-plus/libero/libero/assets
```

如果 clone 后路径已经存在，应先检查它是什么，不能直接覆盖。

## 4. 必须迁移：当前工作和论文主资产

| 优先级 | 路径 | 大小 | 用途 |
|---|---|---:|---|
| P0 | `/data2/lty/naive-e-slotssm-runs/libero-plus-global-v1` | 1.012 TB | 10,030 条稠密审计、LIBERO-10 Gate/Recovery、teacher、当前模型和 recovery family |
| P0 | `/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1` | 102.81 GB | Book、Mug/Pudding、Black Bowl 的最终模型、成功 teacher、训练数据、最终评测和视频 |
| P0 | `/data2/lty/naive-e-slotssm-runs/oft-plus-robot-global-v1` | 20.18 GB | OFT+ robot initial-state task forks 和基线 provenance |
| P0 | `/data2/lty/naive-e-slotssm-runs/libero-plus/oft-plus-robot-levels1-5-v1` | 71.88 MB | 用户指定的 OFT+ LIBERO-Plus 基线 |
| P0 | `/data2/lty/naive-e-slotssm-runs/libero-plus/oft-plus-robot-levels1-4-v1` | 63.46 MB | 历史脚本直接引用的 level 1–4 基线 |
| P0 | `/data2/lty/naive-e-slotssm-runs/libero-plus/oft-plus-robot-level5-v1` | 15.61 MB | 历史脚本直接引用的 level 5 基线 |
| P0 | `/data2/lty/naive-e-slotssm-runs/libero-plus/mug-pudding-checkpoint-v1` | 2.95 MB | Mug/Pudding privileged teacher checkpoint |

上述资产连同代码、基础模型和解压资产约为 **1.16 TB（十进制）/1.06 TiB**。为避免重跑 10,030 条采集，这一档不能删减。

### 强烈建议一并迁移

| 路径 | 大小 | 用途 |
|---|---:|---|
| `/data2/lty/naive-e-slotssm-runs/oft-plus-book-local-repair-v1` | 52.99 GB | 清理后保留的 Book teacher、FSM、checkpoint、报告和评测 |
| `/data2/lty/naive-e-slotssm-runs/v3-action-aware-slotssm-v1` | 6.34 GB | 清理后保留的 V3 原始轨迹、canonical teacher 和模型 |

加入这两项后约为 **1.22 TB**。它们不是继续 V4.4.2 所必需，但属于论文方法演化、历史对照和未来消融的重要资产。

## 5. 可以不迁移

- `/data/lty/code/naive-e-slotssm/.conda`：两个环境共约 18.35 GB，不建议直接跨机器复制。
- `/data/lty/code/naive-e-slotssm/artifacts/checkpoints/openvla-7b-finetuned-libero-spatial`：约 15.1 GB，只服务早期非 OFT 实验。
- `/data2/lty/models` 下除 `openvla-7b-oft-finetuned-libero-plus-mixdata` 外的旧模型：只有重做早期 OpenVLA/SlotSSM 实验才需要。
- `/data2/lty/naive-e-slotssm-runs/libero-plus/assets/assets.zip`：解压资产已迁移时属于重复副本。
- 已在 `STORAGE_CLEANUP_2026-08-30.md` 中标明的旧特征、失败 RGB-D 和废弃版本。
- 缓存、日志、`__pycache__`、W&B 本地缓存和临时 MP4。

如果目标是完整保存所有历史实验，而不只是继续论文主线，则可直接迁移清理后的整个 `/data2/lty/naive-e-slotssm-runs`，当前为约 1.236 TB。

## 6. 环境重建基线

当前已验证环境：

- Ubuntu 20.04，kernel 5.15
- NVIDIA driver 550.144.03
- RTX 4090 24 GB
- Python 3.9.21
- PyTorch 2.2.0 + CUDA 12.1
- torchvision 0.17.0、torchaudio 2.2.0
- transformers：`moojink/transformers-openvla-oft@bc339d9ad707454c0c115970db43c260067c61ab`
- dlimp：`moojink/dlimp_openvla@040105d256bd28866cc6620621a3d5f7b6b91b46`
- accelerate 0.29.3、safetensors 0.4.5、peft 0.11.1
- numpy 1.26.4、timm 0.9.10、einops 0.8.0
- mujoco 2.3.7、robosuite 1.4.1、libero 0.1.0
- imageio 2.37.2、imageio-ffmpeg 0.6.0
- mamba-ssm 2.2.2、causal-conv1d 1.4.0、triton 2.2.0

重建时优先遵循固定提交的 OpenVLA-OFT 环境说明，再安装两个本地仓库：

```bash
pip install -e /data/lty/code/openvla-oft-official
pip install -e /data/lty/code/LIBERO-plus
```

`mamba-ssm` 和 `causal-conv1d` 含 CUDA 扩展，必须在新服务器的 PyTorch/CUDA 组合下重新编译。不要把旧环境中的 `.so` 当成可移植文件。

在线运行前设置：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH=/data/lty/code/LIBERO-plus/baseline_libero_config
```

完整 OFT 在线推理需要至少约 20 GB 空闲显存，推荐独占 24 GB GPU。仅训练预提取特征上的 Gate/Recovery/Resume head，4 GB 可启动，8 GB 更稳妥。

## 7. 推荐传输顺序

先在新服务器 clone 三个代码仓库并 checkout 固定提交，再用 `rsync` 传大文件。以下命令在新服务器执行，将 `old-server` 换成旧服务器 SSH 地址：

```bash
mkdir -p /data/lty/code /data2/lty/models /data2/lty/naive-e-slotssm-runs/libero-plus

git clone git@github.com:Liuty003/FWE_SlotSSM.git /data/lty/code/naive-e-slotssm
git clone https://github.com/moojink/openvla-oft.git /data/lty/code/openvla-oft-official
git -C /data/lty/code/openvla-oft-official checkout e4287e94541f459edc4feabc4e181f537cd569a8
git clone https://github.com/sylvestf/LIBERO-plus.git /data/lty/code/LIBERO-plus
git -C /data/lty/code/LIBERO-plus checkout 4976dc30028e805ff8094b55501d532c48fec182

rsync -aH --numeric-ids --partial --info=progress2 \
  old-server:/data/lty/code/LIBERO-plus/baseline_libero_config/ \
  /data/lty/code/LIBERO-plus/baseline_libero_config/

rsync -aH --numeric-ids --partial --info=progress2 \
  old-server:/data2/lty/models/openvla-7b-oft-finetuned-libero-plus-mixdata/ \
  /data2/lty/models/openvla-7b-oft-finetuned-libero-plus-mixdata/

rsync -aH --numeric-ids --partial --info=progress2 \
  old-server:/data2/lty/naive-e-slotssm-runs/libero-plus-global-v1/ \
  /data2/lty/naive-e-slotssm-runs/libero-plus-global-v1/

rsync -aH --numeric-ids --partial --info=progress2 \
  old-server:/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/ \
  /data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/

rsync -aH --numeric-ids --partial --info=progress2 \
  old-server:/data2/lty/naive-e-slotssm-runs/oft-plus-robot-global-v1/ \
  /data2/lty/naive-e-slotssm-runs/oft-plus-robot-global-v1/

rsync -aH --numeric-ids --partial --info=progress2 \
  old-server:/data2/lty/naive-e-slotssm-runs/libero-plus/assets/extracted/ \
  /data2/lty/naive-e-slotssm-runs/libero-plus/assets/extracted/

for baseline_name in oft-plus-robot-levels1-5-v1 oft-plus-robot-levels1-4-v1 oft-plus-robot-level5-v1 mug-pudding-checkpoint-v1; do
  rsync -aH --numeric-ids --partial --info=progress2 \
    "old-server:/data2/lty/naive-e-slotssm-runs/libero-plus/${baseline_name}/" \
    "/data2/lty/naive-e-slotssm-runs/libero-plus/${baseline_name}/"
done
```

不要在首次传输时使用 `rsync --delete`。大目录可重复运行同一命令断点续传。

## 8. 校验值

关键文件 SHA-256：

```text
07cf2af20bf1a7132376ef181f5a7be91fd71437008a4afa1baaa03d4fd605fd  official-manifest.json
857dd111a58e8ed084444ccd846c5a1e766a84f2461ef0f57b7e694bbb022e82  baseline-inventory.json
6cac2553419f87bfeeb88becb298411b36f6841a74f2dc03702c21e822e6bede  native-slot-h1-action/best.safetensors
9751d0a0d4fb5e88600476a854719c0781d131f5f143ab1db9263be93a2b29c7  native-slot-h1-gate/best.safetensors
0ec095ce16193c9e321f5943994cf3e4dc0f3aa8f1e5802f3f8b17ccee2b2af2  family-gate-deployment-v1.json
a5acdcdabdcf7b429174cbed4e880ace5e487d7a0523f36bd5a9966450e4e693  Book34 r2 best.safetensors
82c5a27d4ebfa5af078bc7c3b83a7ab50ee6a56cf1de297cf32f3d5ba72410c0  Book34 selector report.json
de8026889700cbefc396550a40cc0540eafc4032b41ea66cf03bf301e5f05834  Mug/Pudding best.safetensors
1e370c7dc073ee5de6db28c87692616de2879cc3aacb6a0da55fa3247ee77172  Black Bowl best.safetensors
```

## 9. 新服务器容量建议

- 只恢复当前工作：至少 1.5 TB 可用空间。
- 继续完成 10,030 条并加入 DP、ACT 等多基线：建议至少 4 TB 工作空间。
- 如果每种基础策略都保存同等规模的 dense audit，4 TB 仍可能不够，应让不同策略共享官方初态、任务索引和静态资产，只为每种策略保存其动作、状态与必要观测。

现有 `full-audit-v1` 是 OpenVLA-OFT 的评测/恢复数据底座，不应直接冒充 DP 或 ACT 的训练演示数据；迁移到新基础策略时仍需保持公平输入协议并重新运行对应 baseline。
