# V4 任务代码、数据、模型与评测索引

最后更新：2026-08-25

本文是 V4 系列实验的长期资产入口。历史固定任务方法见 [v4流程.md](./v4流程.md)，LIBERO-Plus 10,030 条通用恢复方法见 [LIBERO_PLUS_10030_RECOVERY_METHOD.md](./LIBERO_PLUS_10030_RECOVERY_METHOD.md)，历史过程与细节见 [HANDOFF.md](./HANDOFF.md)。模型和大规模数据仍保存在 `/data2/lty/naive-e-slotssm-runs`；本文进入代码仓库，用于防止路径、版本和成绩口径丢失。

## 统一登记规范

以后每完成一个任务族，都在本文新增一节，并固定登记以下内容：

1. 任务语义、task ID 范围、评测 episode 和任务数。
2. 冻结的基础策略、任务专用 teacher（若有）及二者职责。
3. 数据根目录、ownership manifest、normal/failure/recovery/DAgger 特征。
4. 最终部署模型，以及保留但不部署的中间模型。
5. 训练、采集、评测和数据构造的代码入口。
6. 最终评测目录、视频位置、成功率及人工修正口径。
7. 最终模型和汇总报告的 SHA-256。
8. 适用范围：纯神经模型、固定任务 selector、特权 teacher 不能混称。

新任务建议采用统一目录：

```text
/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/<task-family>/
  baseline/
  failure-mining/
  recovery-teacher/
  ownership-dataset/
    features/
  training/<version>/
  online-eval/<version>/
  artifact-report.json
```

`artifact-report.json` 应至少保存 task IDs、命令/配置、数据路径、checkpoint、评测路径、成功任务、失败任务和成绩口径。目录名称允许兼容已有历史产物，不为统一命名而移动或破坏旧数据。

## 公共基础策略

- OpenVLA-OFT：`/data2/lty/models/openvla-7b-oft-finetuned-libero-plus-mixdata`
- 原则：基础策略冻结；V4.3 只训练 relation-3D、ownership gate 和 recovery action head。

## Book34：把书放入后部隔间

### 范围与最终口径

- task IDs：491–524，共 34 个固定 task，episode 0。
- 资产根目录：`/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book`
- 旧版局部修复根目录：`/data2/lty/naive-e-slotssm-runs/oft-plus-book-local-repair-v1`
- 最终交付方案是固定 Book34 task-ID overfit selector，不是单一泛化神经模型。
- selector 规则：存在 simulator-verified task-specific action trace 时使用该轨迹，否则使用冻结的 V4.3 r2。
- selector 成绩：29/34，85.2941%。成功 task：491、492、494–511（不含 493）、514、515、517–519、521–524；失败 task：493、512、513、516、520。
- 纯 V4.3 r2 神经在线成绩：16/34。不得把 selector 的 29/34 表述成纯神经恢复成功率或跨初始化泛化成绩。

### 最终模型与 selector

- r2 神经 checkpoint：`/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/v43-book34-expanded-r2/direct-v3-gw4-lr1e4-seed20260818/best.safetensors`
- r2 配置和训练报告：与上述 checkpoint 同目录下的 `config.json`、`report.json`。
- selector 构造脚本：`scripts/build_book34_overfit_selector.py`
- selector 汇总报告：`/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/plus-robot-book/book34-overfit-selector-v1/report.json`
- 已验证轨迹主要位于：`.../plus-robot-book/action-trace-teachers-book34-r2` 和 `.../plus-robot-book/action-trace-teachers-book34-r4`。每个 task 目录保存 `metadata.json`、`steps.jsonl` 和 `rollout.mp4`；实际选择以 selector 报告逐 task 的 `selected_artifact` 为准。

### 数据和保留模型

- r2 数据：`.../plus-robot-book/v43-book34-expanded-r2/normal-train.safetensors`、`normal-val.safetensors` 及同目录 recovery train/val 文件。
- r6 checkpoint：`.../plus-robot-book/v43-book34-expanded-r6-data/recovery-head-only-lr1e4-seed20260818/best.safetensors`
- r7 checkpoint：`.../plus-robot-book/v43-book34-expanded-r7-data/recovery-head-only-lr1e4-seed20260818/best.safetensors`
- r7 新特征：`.../plus-robot-book/features-v43-book34-r7-window62/task508-xm25.safetensors`、`task509-ym25.safetensors`。
- r7 合并数据：`.../plus-robot-book/v43-book34-expanded-r7-data/recovery-train-expanded.safetensors`。
- r7 离线结果不优于 r6，因此不是最终部署模型。

### 主要代码入口

- 在线执行：`scripts/run_v4_relational_online.py`
- 关系特征：`scripts/precompute_v4_oft_relation_features.py`
- V4.3 训练：`scripts/train_v43_local_recovery.py`
- handoff manifest：`scripts/build_v43_handoff_manifest.py`
- 特征拆分：`scripts/split_v43_handoff_features.py`
- prefix teacher：`scripts/collect_v4_book_prefix_teacher.py`
- 动作轨迹回放：`scripts/collect_v4_action_trace_replay.py`
- 最终 selector：`scripts/build_book34_overfit_selector.py`

### 校验值

- r2 `best.safetensors`：`a5acdcdabdcf7b429174cbed4e880ace5e487d7a0523f36bd5a9966450e4e693`
- selector `report.json`：`82c5a27d4ebfa5af078bc7c3b83a7ab50ee6a56cf1de297cf32f3d5ba72410c0`

## Mug + Pudding：白杯放盘子、布丁放盘子右侧

### 范围与最终口径

- 指令：`put the white mug on the plate and put the chocolate pudding to the right of the plate`
- task IDs：525–566，共 42 个固定 task，episode 0。
- 最终资产根目录：`/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/mug-pudding/counterfactual-factory-unified-v1`
- 严格模拟器结果：31/42，73.8%。
- 按人工视频判断将 task557 记为成功时：32/42，76.2%。二者均保留，默认报告严格模拟器结果。
- 18 个 recovery 主评测中 11/18 成功；20 个 normal safety 中 18/20 成功，且 gate 没有误触发；4 个 special 中 task545、548 成功，task557、564 失败。

### 最终模型

- 最终 checkpoint：`/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/mug-pudding/counterfactual-factory-unified-v1/training/v43-relation3d-insample-dagger-action-v3/best.safetensors`
- 配置和训练报告：同目录 `config.json`、`report.json`。
- gate timing v2 中间模型：`.../training/v43-relation3d-insample-gate-timing-v2/best.safetensors`
- 初始 v1 中间模型：`.../training/v43-relation3d-insample-v1/best.safetensors`

### 数据与 teacher

- ownership manifest：`.../ownership-dataset/handoff_manifest.json`
- ownership 质量报告：`.../ownership-dataset/quality_report.json`
- normal：`.../ownership-dataset/features/normal-all.safetensors`
- failed gate：`.../ownership-dataset/features/failed-gate-all.safetensors`
- gate：`.../ownership-dataset/features/gate-all.safetensors`
- 初始 recovery：`.../ownership-dataset/features/recovery-all.safetensors`
- DAgger：`.../ownership-dataset/features/dagger-all-v1.safetensors`
- 最终 recovery + DAgger：`.../ownership-dataset/features/recovery-plus-dagger-v1.safetensors`
- DAgger 数据报告：`.../ownership-dataset/dagger-sources-v1/dataset-report.json`，包含 48 个 teacher sources、3291 transitions。
- privileged Cartesian teacher：`/data2/lty/naive-e-slotssm-runs/libero-plus/mug-pudding-checkpoint-v1/checkpoints/privileged-cartesian-v3.pt`。它只用于构造恢复数据，不是最终部署模型。

### 基线与失败快照

- level 1–4 基线：`/data2/lty/naive-e-slotssm-runs/libero-plus/oft-plus-robot-levels1-4-v1`
- level 5 基线：`/data2/lty/naive-e-slotssm-runs/libero-plus/oft-plus-robot-level5-v1`
- task fork：`/data2/lty/naive-e-slotssm-runs/oft-plus-robot-global-v1/forks/task{525..566}-episode0`
- DAgger task538 smoke：`.../dagger-factory-v1-smoke`
- DAgger 全量 shards：`.../dagger-factory-v1`

### 最终评测与视频

- 18 个 recovery tasks：`.../online-eval/dagger-v3-batch08`
- 20 个 normal safety tasks：`.../online-eval/dagger-v3-normal-safety08`
- 4 个 special tasks：`.../online-eval/dagger-v3-special08`
- 每个 task 子目录均保存 `v4.mp4` 和 `report.json`，因此 cup/pudding 的最终视频已保存。
- 主评测成功 task：538、541、542、543、550、554、555、556、558、560、562。
- 主评测失败 task：544、549、551、559、561、565、566。

### 主要代码入口

- 模型：`recovery/models/v43_local_recovery.py`、`recovery/models/v4_relational_recovery.py`
- 特征与几何：`recovery/data/v4_relation_features.py`、`recovery/data/v4_geometry.py`
- 失败挖掘：`scripts/mine_mug_pudding_failures.py`
- 精确状态恢复：`scripts/restore_mug_pudding_failure_fork.py`
- privileged teacher：`scripts/collect_mug_pudding_feedback_recovery.py`
- 反事实恢复数据：`scripts/build_counterfactual_recovery_dataset.py`
- ownership 数据：`scripts/build_ownership_dataset.py`
- OFT RGB-D 采集：`scripts/collect_v4_oft_rgbd_rollouts.py`
- 特征预计算/合并：`scripts/precompute_v4_oft_relation_features.py`、`scripts/merge_v4_relation_features.py`
- 训练与离线评测：`scripts/train_v43_local_recovery.py`、`scripts/evaluate_v43_local_recovery.py`
- 在线执行：`scripts/run_v4_relational_online.py`
- DAgger factory/合并：`scripts/run_v43_dagger_factory.py`、`scripts/build_v43_dagger_dataset.py`
- 批量评测：`scripts/run_v43_batch_eval.py`

### 关键在线配置

- gate threshold：0.8
- confirmation：2
- release threshold：0.5
- minimum recovery steps：24
- recovery horizon：72
- action execution chunk：4
- cooldown：16
- post-release wait：10

### 校验值

- 最终 `best.safetensors`：`de8026889700cbefc396550a40cc0540eafc4032b41ea66cf03bf301e5f05834`

## Black Bowl + Bottom Drawer：黑碗放入底层抽屉并关抽屉

### 范围与最终口径

- 指令：`put the black bowl in the bottom drawer of the cabinet and close it`
- task IDs：409–452，共 44 个 canonical view/init-state variants，episode 0。
- 资产根目录：`/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/black-bowl-bottom-drawer`
- 冻结 OpenVLA-OFT 基线：20/44，45.5%。
- 最终混合系统：31/44，70.5%，比基线净增 11 条成功、提升 25 个百分点。
- 基线 20 条成功全部保持成功，无正常案例回退；24 条基线失败中恢复 11 条，恢复率 45.8%。
- 新恢复成功 task：421、426、433、434、436、442、446、447、449、450、452。
- 最终失败 task：412、413、415、419、425、428、429、431、435、443、445、448、451。
- 该成绩属于“学习式 ownership gate + relation-3D 目标几何 + 确定性闭环抓取 FSM + 冻结 OFT”的混合部署结果，不得表述为纯神经 recovery action head 的成绩；在线不读取 simulator privileged state。

### 最终模型与部署结构

- 最终 checkpoint：`/data2/lty/naive-e-slotssm-runs/v4-relational-recovery-v1/black-bowl-bottom-drawer/detailed-v1/model/best.safetensors`
- 配置和训练报告：同目录 `config.json`、`report.json`。
- 在线入口：`scripts/run_v4_relational_online.py --grasp-only-controller`。
- gate 连续 2 个决策超过 0.8 后接管；checkpoint 的 `predicted_target_delta` 用于闭环定位黑碗。
- 抓取 FSM 依次执行 approach、descend、close、proprioceptive grasp verification 和 14 cm lift，确认抓取后永久归还冻结 OFT 完成放置及关抽屉。
- approach/descend 位置阈值均为 1.2 cm；夹爪非空闭合阈值为 0.004；最多尝试抓取 2 次；整条轨迹最多 520 个环境步。
- `detailed-gate-v1` 至 `detailed-gate-v4` 只保留作 gate timing 诊断，没有替换最终部署 checkpoint。

### 数据与 teacher

- 基线失败 manifest：`.../black-bowl-bottom-drawer/failure-mining-v1/failure-manifest.json`，包含 44 条基线、20 条成功、17 条高置信失败和 7 条待复核失败。
- 精确失败状态：`.../black-bowl-bottom-drawer/exact-failure-forks-v1`。
- simulator-verified teacher source 报告：`.../black-bowl-bottom-drawer/ownership-dataset/sources/source-report.json`，17 个候选中保留 12 条成功恢复轨迹；privileged teacher 只用于离线构造数据。
- normal 特征：`.../black-bowl-bottom-drawer/ownership-dataset/features/normal-all.safetensors`。
- failed gate 特征：`.../black-bowl-bottom-drawer/ownership-dataset/features/failed-gate-all.safetensors`。
- recovery 特征：`.../black-bowl-bottom-drawer/ownership-dataset/features/recovery-all.safetensors`。
- 人工复核 gate 边界：`.../black-bowl-bottom-drawer/ownership-dataset/features/manual-reviewed-gate-v1.safetensors` 至 `manual-reviewed-gate-v3.safetensors`。
- detailed phase 快速 teacher 特征：`.../black-bowl-bottom-drawer/detailed-v1/features/fast-all.safetensors`。
- detailed phase DAgger 特征：`.../black-bowl-bottom-drawer/detailed-v1/features/dagger-{approach,descend,close,lift}.safetensors`。
- 15-phase 标签及训练数据：`.../black-bowl-bottom-drawer/detailed-v1/labels.safetensors`、`fast-phase-sources/report.json`、`dagger-phase-sources/report.json`。

### 最终评测与视频

- 44 条端到端评测：`.../black-bowl-bottom-drawer/evaluation/grasp-only-all44-final-v1`。
- task409–423、424–438、439–452 分别保存在 `gpu0`、`gpu1`、`gpu7` shard；每个 task 目录均包含 `report.json` 和 `v4.mp4`，共保存 44 个最终视频。
- 最终成功 task：409、410、411、414、416、417、418、420、421、422、423、424、426、427、430、432、433、434、436、437、438、439、440、441、442、444、446、447、449、450、452。
- 早期 12 条“基线结束时碗仍未入抽屉”困难集评测：`.../evaluation/grasp-only-handoff-v2`，其中 6/12 成功；它是开发阶段子集结果，不是最终整体成绩。
- 统一资产报告：`.../black-bowl-bottom-drawer/artifact-report.json`。

### 主要代码入口

- 在线执行与闭环抓取 FSM：`scripts/run_v4_relational_online.py`
- 44 条分片批量评测：`scripts/run_v43_batch_eval.py`
- 失败挖掘：`scripts/mine_black_bowl_drawer_failures.py`
- 精确状态恢复：`scripts/restore_black_bowl_drawer_failure_fork.py`
- privileged teacher：`scripts/collect_black_bowl_drawer_feedback_recovery.py`
- verified source 构造：`scripts/build_black_bowl_drawer_training_sources.py`
- detailed phase source 构造：`scripts/build_black_bowl_phase_sources.py`
- 特征预计算：`scripts/precompute_v4_oft_relation_features.py`
- V4.3 detailed/gate-only 训练：`scripts/train_v43_local_recovery.py`

### 校验值

- 最终 `best.safetensors`：`1e370c7dc073ee5de6db28c87692616de2879cc3aacb6a0da55fa3247ee77172`
- `artifact-report.json`：`b2b4d963410721844cd8dbd9fe5425b168b2bb941b483b3bb86140a856a872a1`

## LIBERO-Plus 10,030 条全局自动化流水线（2026-08-25）

### 官方范围与索引

- 全局资产根目录：`/data2/lty/naive-e-slotssm-runs/libero-plus-global-v1`。
- 冻结官方索引：`official-manifest.json`，共 10,030 条；套件为 `libero_spatial` 2402、`libero_object` 2518、`libero_goal` 2591、`libero_10` 2519。
- 分析口径为 suite × 官方 perturbation category × difficulty；恢复路由使用轨迹自动推断的 failure type，不按 task ID 人工分组。
- `official-manifest.json` SHA256：`07cf2af20bf1a7132376ef181f5a7be91fd71437008a4afa1baaa03d4fd605fd`。

### 已有基线与采集协议

- 基线盘点：`baseline-inventory.json`；当前找到兼容 OFT+ 基线 393/10030，成功 149/393，尚缺 9637；已有覆盖均属于 `libero_10 / Robot Initial States`。
- `baseline-inventory.json` SHA256：`857dd111a58e8ed084444ccd846c5a1e766a84f2461ef0f57b7e694bbb022e82`。
- 新采集同时保存正常在线观测/动作、MP4，以及仅供离线标注的 BDDL 目标真假、对象/fixture 位姿和关节状态；privileged audit 不进入在线 policy 输入。
- 全量正式采集使用 `oft_archival_rollout_v4`：每条轨迹额外保存压缩 `dense-audit.npz`，包含无损双视角 RGB、双视角 float32 depth、双视角 instance segmentation、完整 proprio、真实 action chunk 边界、逐步及终态 MuJoCo state/ctrl/camera pose、NumPy RNG 状态和官方初态；主/腕视角 MP4 均保留。
- `--resume` 只有在 metadata 标记 v4 complete、steps 和 dense bundle 均存在时才跳过；早期 v3 条目会自动重跑，避免不完整 schema 混入正式全量数据。
- 自动校验入口：`scripts/validate_libero_plus_archival_rollouts.py`；精确 fork 入口 `scripts/save_oft_recovery_fork_snapshot.py` 已支持 v4 bundle 和逐步 RNG 恢复。
- Camera Viewpoint 与 Sensor Noise 跨扰动 smoke 均通过。Sensor Noise task1374 从官方初态重放至 step20 时，逐步/最终 MuJoCo state 最大误差为 0，受噪下一帧 RGB 逐像素一致；fork manifest 记录 `exact_archival_prefix_validated=true`、`max_prefix_state_error=0.0`。
- 全部 10,030 条 BDDL 已验证可解析，目标只包含 `on`、`in`、`open`、`close`、`turnon` 五类谓词，因此可统一计算子目标进度和剩余目标。

### 自动失败路由与分层 pilot

- 自动分类输出 success、partial goal、goal regression、空抓/接触失败、搬运/放置失败、机构接触/未完成等粗粒度类型，并附 confidence、逐谓词状态与候选恢复 skill；低置信条目留待聚类抽检。
- 已知失败 `libero_10:task412:episode0` 被高置信识别为：黑碗 `in` 已完成，抽屉 `close` 未完成，候选恢复为继续剩余子目标。
- 分层 pilot：`stratified-pilot-v1`，覆盖当前存在的 141 个 suite × perturbation × difficulty 单元，4 个 worker 分配 36/35/35/35 条，使用 GPU 0、2、4、7，可断点续跑。
- pilot 已完整采集 141/141，耗时约 16.5 分钟、占用约 180 MiB；OFT+ 成功 115/141（81.6%）。按 suite：`libero_10` 80.0%、`libero_goal` 75.0%、`libero_object` 88.6%、`libero_spatial` 82.9%。
- 按扰动：Background/Camera/Language 均 95.0%，Light 85.7%，Layout 80.0%，Robot Initial States 30.0%，Sensor Noise 90.0%；这只是每个有效分层单元一条的 pipeline pilot，不作为最终成功率估计。
- 26 条失败自动分为 articulation contact 9、empty grasp/contact 9、partial goal 4、transfer/placement 4；119 条为高置信、22 条为中置信，没有 unknown/low-confidence。分类文件：`stratified-pilot-v1/failure-classification.json`。

### 主要代码入口

- 官方索引：`scripts/build_libero_plus_global_manifest.py`
- 现有基线盘点：`scripts/inventory_libero_plus_baselines.py`
- 带离线 privileged audit 的 OFT 采集：`scripts/collect_oft_nominal_trajectories.py`
- 自动失败分类：`scripts/classify_libero_plus_failures.py`
- 分层 pilot/全量多 GPU 分片：`scripts/run_libero_plus_audit_shard.py`

## LIBERO-10 自动恢复 V4.4.2（2026-08-29）

### 根目录与已完成模型

- 资产根目录：`/data2/lty/naive-e-slotssm-runs/libero-plus-global-v1/libero-10-recovery-v1`。
- H=1 recovery action：`model-v3/native-slot-h1-action/best.safetensors`；18,432 个 native-input teacher 状态，训练时不读取 RGB-D。
- 十任务共享 Gate（只作初始化/对照）：`model-v3/native-slot-h1-gate/best.safetensors`。
- 十个 family Gate：`model-v3/gates-by-family/<family>/best.safetensors`；仅 `gate.*` 与共享 Gate 不同，representation 和 recovery action 逐 tensor 完全一致。
- 2519-task 部署路由：`model-v3/family-gate-deployment-v1.json`；task metadata 只在评测器外部选择 Gate head，task ID 不进入模型输入。

### Gate 正确标定口径

- 每任务离线预测：`model-v3/gates-by-family/<family>/offline-gate-eval-t050.json`。
- 部署标定：`model-v3/gates-by-family/<family>/calibration-trajectory-fpr005.json`。
- 阈值按完整正常 source 的误接管率标定，并要求连续两个决策超过阈值；旧的单帧 FPR 标定禁止用于部署。
- 十类部署阈值范围为 0.653–0.930；当前 30 条 normal/source 每类的离散分辨率为 3.33%，标定结果均为 1/30 正常 source 发生误接管。

### Handoff / Resume 状态

- fingerprinted、自动 normalization、OFT 延迟加载 smoke：`handoff-audits-v3/smoke-release-task10-v2`。
- task10 release boundary 已认证成功：`env_steps=0`、`policy_decisions=0`、normalization=`libero_10`，并保存单帧 `rollout.mp4`；该路径在 GPU 满载时不加载 7B OFT。
- shared Resume 尚未训练；必须先完成更早 phase 的 fresh-replan 审计，生成 certified handoff manifest。
- Resume 训练代码现在拒绝完整未截断 teacher 特征，只接受 stride-1、H=1、共享同一 certified handoff manifest 的 pre-handoff negatives 与精确 boundary positives。

### 主要代码入口

- 轨迹级 Gate 标定：`scripts/analyze_libero10_gate_by_family.py`
- family Gate 部署路由：`scripts/build_libero10_family_gate_deployment_manifest.py`
- family Gate + shared action/Resume 闭环：`scripts/run_v44_dual_batch.py --family-gate-manifest ...`
- fresh-replan handoff 审计：`scripts/run_relational_oft_handoff_audit.sh`
- certified handoff 特征：`scripts/run_libero10_handoff_feature_shards.sh`
- handoff 截断 action + shared Resume 训练：`scripts/run_libero10_handoff_action_resume_training.sh`
