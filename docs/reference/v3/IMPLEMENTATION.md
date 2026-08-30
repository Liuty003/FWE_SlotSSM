# V3 实现入口

`howtodo.md` 定义方法；本文只记录已落地的代码与首轮运行顺序。

## 已实现的部署图

`recovery/models/v3_action_aware_recovery.py` 实现：

```text
frozen OFT visual tokens -> SlotAttention -> ActionAwareSlotSSM
frozen OFT action hidden + (S[n-1], S[n], previous executed chunk) -> binary gate
frozen OFT action hidden -> action queries -> cross-attention(S[n]) -> recovery 8x7 action
```

`FutureSlotPredictor` 与 `SlotFeatureDecoder` 只出现在训练 checkpoint；
`deploy_*.safetensors` 不包含它们。线上入口是
`scripts/run_v3_oft_action_aware_recovery.py`，没有 FSM、candidate verifier、anchor、
simulator pose/contact 或 simulator state 输入。

## Stage 1：因果 representation 数据与训练

先在**自建 training perturbation**收集可回放 OFT trajectory，必须保存双 RGB：

```bash
python scripts/collect_oft_nominal_trajectories.py \
  --task-suite libero_10 --task-ids '<training-task-ids>' \
  --episodes '<training-episodes>' --save-frame-images \
  --output-root /data2/lty/naive-e-slotssm-runs/v3/training-oft

# 不加载 OFT 即可检查训练准入；现有未保存双 RGB 的历史 JSONL 会被明确拒绝。
python scripts/audit_v3_rollout_sources.py \
  --input-root /data2/lty/naive-e-slotssm-runs/v3/training-oft \
  --output /data2/lty/naive-e-slotssm-runs/v3/training-oft/admission.json

python scripts/precompute_v3_oft_sequence_features.py \
  --input-root /data2/lty/naive-e-slotssm-runs/v3/training-oft \
  --output /data2/lty/naive-e-slotssm-runs/v3/features/transitions.safetensors \
  --task-suite libero_10 --gpu 0

python scripts/split_v3_sequence_features.py \
  --features /data2/lty/naive-e-slotssm-runs/v3/features/transitions.safetensors \
  --train-output /data2/lty/naive-e-slotssm-runs/v3/features/train.safetensors \
  --val-output /data2/lty/naive-e-slotssm-runs/v3/features/val.safetensors

python scripts/train_v3_action_aware_recovery.py \
  --stage stage1 \
  --train-features /data2/lty/naive-e-slotssm-runs/v3/features/train.safetensors \
  --val-features /data2/lty/naive-e-slotssm-runs/v3/features/val.safetensors \
  --output-dir /data2/lty/naive-e-slotssm-runs/v3/stage1 --device cuda
```

Stage 1 后先做最低限度因果准入；只要 shuffle 不变差，就不进入 Stage 2：

```bash
python scripts/evaluate_v3_action_conditioning.py \
  --features /data2/lty/naive-e-slotssm-runs/v3/features/val.safetensors \
  --checkpoint /data2/lty/naive-e-slotssm-runs/v3/stage1/train_best.safetensors \
  --output /data2/lty/naive-e-slotssm-runs/v3/stage1/action-conditioning.json
```

每个 feature row 都是完整 causal source sequence；`split` 严格按 `source_ids` 切分，
拒绝 window-level 随机划分。
对于带 task-suite/task-id/episode-id provenance 的数据，`source_ids` 按物理初始 episode
分组；因此同一 episode 的 nominal 与 recovery branch 不会被拆到 train/val 两侧。
早于这一规则冻结的 feature 不需重跑 OFT，可用
`reindex_v3_feature_source.py --features ... --trajectory-metadata ... --output ...`
只修正 source identity；tensor 内容不变。
同一 rollout 同时存在 `steps.jsonl` 和 `transitions.jsonl` 时只读取前者，避免同源重复；
未成功的 nominal `steps.jsonl` 仍可参与 Stage 1 的 future/重建损失，但其 gate label
强制为 `-1`，绝不作为 NORMAL。

历史 `oft-nominal-seed-v1` 只记录了 planned OFT action。它可用
`--allow-legacy-planned-actions` 做代码/特征 smoke，但生成的 metadata 会明确标成
compatibility-only，不能进入最终 action-effect 或失败检测结果。

注意：另一些历史目录可能有 `executed_action`，但未保存两路逐帧 PNG（只保存 MP4 或
simulator state）。它们同样不能用于 v3；不能以 privileged state、视频补帧或单路图像
替换在线输入。`audit_v3_rollout_sources.py` 是进入特征提取前的硬性准入检查。

并行 collector 共用本地 checkpoint 时，先用一个非并行进程完成配置初始化；其后可在每个
worker 加 `--skip-checkpoint-config-sync`，避免多个 worker 同时重写 checkpoint 的
`config.json`。

## Stage 2：canonical recovery teacher

对每个 failure source 只保留一条 canonical successful recovery trajectory（最终完成、
最短、动作幅度最小的顺序选择）。从这些 teacher rollout 生成 feature 时使用：

已有审计过的 Task6 branch 可通过下列 collector 完成「failure prefix → teacher macro →
frozen OFT completion」的闭环；只有输出 metadata 的 `success: true` 才能进入下一步：

```bash
python scripts/collect_v3_task6_canonical_teacher.py \
  --source-rollout '<recorded-nominal-source>' \
  --recovery-branch '<selected-raw-transitions.jsonl>' \
  --output-dir /data2/lty/naive-e-slotssm-runs/v3/canonical-teachers/sourceX \
  --skip-checkpoint-config-sync
```

该 collector 会拒绝 failure fork 复现误差过大的 source，并保留失败 teacher 的审计记录，
但 feature extraction 的 `--mark-all-recovery` 只接受显式成功的 canonical source。

```bash
python scripts/precompute_v3_oft_sequence_features.py \
  --input-root /data2/lty/naive-e-slotssm-runs/v3/canonical-recovery-teachers \
  --output /data2/lty/naive-e-slotssm-runs/v3/features/recovery.safetensors \
  --task-suite libero_10 --mark-all-recovery --gpu 0

python scripts/merge_v3_sequence_features.py \
  --inputs /data2/lty/naive-e-slotssm-runs/v3/features/transitions.safetensors \
           /data2/lty/naive-e-slotssm-runs/v3/features/recovery.safetensors \
  --output /data2/lty/naive-e-slotssm-runs/v3/features/all.safetensors
```

然后重新 source-disjoint split，并训练：

```bash
python scripts/train_v3_action_aware_recovery.py \
  --stage stage2 --init-checkpoint /data2/lty/naive-e-slotssm-runs/v3/stage1/train_best.safetensors \
  --train-features '<train.safetensors>' --val-features '<val.safetensors>' \
  --output-dir /data2/lty/naive-e-slotssm-runs/v3/stage2 --device cuda
```

`--mark-all-recovery` 只能用于 canonical teacher；它不会从多条成功 candidate 做 L1 平均。

## 上线

```bash
python scripts/run_v3_oft_action_aware_recovery.py \
  --task-suite libero_10 --task-id '<held-out-task>' --episode-id 0 \
  --v3-checkpoint /data2/lty/naive-e-slotssm-runs/v3/stage2/deploy_best.safetensors \
  --v3-config /data2/lty/naive-e-slotssm-runs/v3/stage2/config.json \
  --output-dir /data2/lty/naive-e-slotssm-runs/v3/online/taskX --save-online-images
```

在进入正式 recovery 训练前，必须先做 action-shuffle / matched-counterfactual 检验；
future prediction 不能只因视觉自回归而变好。
