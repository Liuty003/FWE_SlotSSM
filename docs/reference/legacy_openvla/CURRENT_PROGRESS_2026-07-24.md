# Failure-Aware Embodied-SlotSSM 当前进度总结

更新时间：2026-07-24（Asia/Shanghai）

本文总结当前已经完成的工作、可复现产物、结论边界和下一步优先级。详细实验过程
见 `HANDOFF.md`，总体重构原则见
`Failure_Aware_Embodied_SlotSSM_root_cause_and_rebuild_plan.md`。

## 1. 当前目标

当前阶段不是立即完成跨任务、端到端 Failure-Aware SlotSSM，而是先在 LIBERO
无头仿真中证明一个完整且肉眼可见的失败恢复闭环：

```text
正常抓取
→ 明显掉碗
→ 从视觉检测失败
→ 重新定位并抓取
→ 验证已经抓稳
→ 继续原任务
→ env.check_success()=true
```


用户允许这一阶段存在单场景适配或过拟合，但错误必须真实发生，不能在错误尚未
显现时提前“纠正”。

## 2. 已经达到的最强结果

### 2.1 学习恢复动作 + 学习 phase 的未见初始状态成功

在 task9 episode11 上，冻结配置完成：

- Stage-B predicted-mask failure detection；
- mask + depth 视觉三维目标估计；
- learned geometry residual recovery actions；
- learned monotonic recovery phase transitions；
- 明显 lost grasp 后重新抓取；
- 恢复后继续 OpenVLA 名义任务；
- 最终 `env.check_success()=true`。

视频：

```text
/data2/lty/naive-e-slotssm-runs/recovery-learned-phase-final-test-v1/
task9-episode11.mp4
```

关键时序：

- step68–74：强制张开夹爪，碗明显掉落；
- step75：Stage-B detector 检出失败；
- step75：RETURN；
- step101：DESCEND；
- step144：CLOSE；
- step152：VERIFY；
- step167：旧 privileged-height verifier 判定恢复完成；
- step211：环境任务成功。

这是目前最强的未见初始状态端到端证据，但该实验的 VERIFY exit 仍读取 simulator
bowl height。

### 2.2 视觉 regrasp verifier 在线替换 bowl-height oracle

最新 episode10 online smoke 已将 VERIFY exit 换成非 privileged 视觉时序模型：

```text
/data2/lty/naive-e-slotssm-runs/recovery-visual-verifier-online-smoke-v4-settle/
task9-episode10.mp4
```

关键时序：

- step64：失败检测；
- step166–172：视觉 verifier 概率由0.629升到0.795；
- step173：概率0.812，满足连续3帧确认；
- step173–189：`VISUAL SETTLE`；
- step190：恢复完成；
- step238：`env.check_success()=true`。

视频为 H.264、256×256、20 FPS、249 frames、12.45秒。

该实验已经证明：在线恢复完成判断可以不读取 bowl pose/height。但 episode10
参与过 phase DAgger，因此它是 online smoke，不是新的 final test。

## 3. 当前系统组成

```text
OpenVLA nominal policy
        │
        ▼
Stage-B target proposal + temporal mask tracking
        │
        ├── mask centroid motion failure detector
        │
        └── predicted mask + depth → visual target xyz
                                      │
                     ┌────────────────┴───────────────┐
                     ▼                                ▼
          learned phase predictor          learned residual controller
      RETURN→DESCEND→CLOSE→VERIFY          continuous recovery actions
                     │
                     ▼
          visual temporal regrasp verifier
                     │
                     ▼
             eef-only VISUAL SETTLE
                     │
                     ▼
              resume OpenVLA task
```

它是一个分模块的 feasibility prototype，还不是最终两阶段 SlotSSM policy。

## 4. 各模块进度

### 4.1 Stage-B 目标选择与跟踪

已完成：

- 首帧用语言关系选择黑碗 proposal；
- 后续以 mask IoU / centroid 做时间身份跟踪；
- failure detector 不读取 simulator segmentation 或 object pose；
- Stage-B temporal target-mask IoU 约0.847。

失败定义已经是肉眼可见的真实 lost grasp：先看到碗被抬起，再强制张开夹爪，等待
碗掉回去后才检测和恢复。

### 4.2 视觉三维目标估计

实现：

```text
scripts/audit_recovery_visual_3d.py
```

使用 Stage-B predicted mask、agent-view depth 和 robosuite camera calibration
反投影 world xyz。采用 mask point-cloud median 和 train+validation 固定表面偏差。

在 episode9 holdout 上：

- mean error=1.87 cm；
- P90=2.78 cm；
- max=3.32 cm。

审计产物：

```text
/data2/lty/naive-e-slotssm-runs/recovery-visual-3d-audit-v2/report.json
```

### 4.3 Learned recovery residual controller

高维 proprio MLP 曾出现更低 frame MAE、但连续控制方向错误的问题，因此没有强行
上线。最终采用低维 geometry feature：

```text
visual_target_xyz - eef_xyz
+ nominal OpenVLA action
+ recovery phase one-hot
```

checkpoint：

```text
/data2/lty/naive-e-slotssm-runs/recovery-residual-smoke-v6-geometry/
recovery-residual-mlp.safetensors
```

episode9 离线：

- action MAE=0.0163；
- translation MAE=0.0267；
- 相对 nominal 改善89.3%；
- gripper sign accuracy=100%；
- sequential direction safety audit 通过。

在线已在 episode9、episode10 和 episode11 完成恢复动作。

### 4.4 Learned phase predictor

输入：

```text
visual_target_xyz - eef_xyz
+ gripper qpos/qvel
+ previous phase
```

输出：

```text
RETURN → DESCEND → CLOSE → VERIFY
```

通过 monotonic constraint 禁止倒退或跨 phase 跳转。第一版 teacher-forced 离线
准确率较高，但在线 episode10 只 DESCEND 9步就过早 CLOSE，证明单纯离线指标不够。

加入成功在线 trace 做 DAgger-style teacher adaptation 后得到：

```text
/data2/lty/naive-e-slotssm-runs/recovery-phase-predictor-v2-dagger/
recovery-phase-predictor.safetensors
```

v2 在 episode10 和未见 episode11 在线成功。不过 episode13 中它长期卡在
DESCEND，说明 phase distribution shift 仍未解决。

### 4.5 Visual regrasp verifier

实现：

```text
scripts/train_visual_regrasp_verifier.py
```

checkpoint：

```text
/data2/lty/naive-e-slotssm-runs/visual-regrasp-verifier-v10-temporal/
visual-regrasp-verifier.safetensors
```

模型输入仅使用 VERIFY 相对时序视觉和机器人状态：

- visual target/eef displacement；
- target-eef relative drift；
- target/eef motion mismatch；
- gripper qpos/qvel；
- elapsed VERIFY steps。

关键发现：失败轨迹曾短暂把碗带起约5.7 cm，随后才滑落，所以单帧 lift classifier
会误报。当前使用 probability threshold=0.75，并要求连续3帧确认。

离线结果：

- 180帧硬负失败轨迹无连续误确认；
- extended validation false positive=0、false negative=0；
- held-out 成功 trace false positive=0，确认延迟1帧；
- `offline_gate_passed=true`。

在线 episode10 已成功替换 privileged bowl-height exit。

### 4.6 VISUAL SETTLE

视觉 verifier 为避免短暂抓住后滑落，需要比旧 oracle 多观察几帧，导致确认时碗
通常高于 OpenVLA 熟悉的恢复出口分布。直接在7.7–9.4 cm 高度交还 OpenVLA 的两次
smoke 都完成重抓、但最终放置失败。

当前临时方案：

- verifier 确认后进入 `VISUAL SETTLE`；
- 只读取 eef encoder；
- 将末端降至 VERIFY-start +4 cm；
- 再恢复 OpenVLA。

这使 episode10 最终成功，但它是明确的手工安全过渡，不是 learned action。

## 5. 恢复数据

已有 recovery expert episodes：

```text
0, 1, 2, 3, 4, 8, 9
```

有效 recovery BC episodes：

```text
0, 1, 2, 4, 9
```

manifest：

```text
/data2/lty/naive-e-slotssm-runs/recovery-expert-manifest-v3.json
```

当前 split：

- train：0、2、4；
- validation：1；
- test：9。

数据包含 RGB、depth、Stage-B target mask、proprio、nominal action、oracle recovery
action 和 teacher privileged state。privileged state 只允许作为训练标签和审计，
不应进入最终在线输入。

## 6. 已经证明与尚未证明

### 已经证明

1. 可以产生肉眼清楚的“抓起后掉落”，并在掉落后才检测。
2. failure detector 可以只使用 Stage-B predicted-mask motion。
3. recovery target xyz 可以由 predicted mask + depth 估计。
4. continuous recovery actions 可以由 learned geometry residual controller 执行。
5. recovery phase transitions 可以由 learned monotonic predictor 执行。
6. 上述 learned action + learned phase 在未见 episode11 上端到端成功。
7. regrasp verification 可以用视觉时序模型替代 privileged bowl height。
8. 视觉 verifier + eef-only settle 已在线完成一次最终任务成功。

### 尚未证明

1. 尚未完成视觉 verifier 的全新初始状态 final test。
2. 尚未去除 `VISUAL SETTLE` 手工过渡。
3. 尚未去除 phase-based gripper sign safety override。
4. 当前 recovery controller/phase/verifier 是小型 MLP，不是最终 SlotSSM temporal
   recovery policy。
5. 尚未证明跨任务、跨目标或跨 failure type 泛化。
6. 只重点验证 lost grasp，未系统验证空抓、错目标和错误放置。
7. 尚未证明 Failure-Aware 系统能在 benchmark 上提高总体成功率且不损伤原成功
   轨迹。

## 7. 当前暴露的主要失败

### 7.1 名义策略限制可测试状态

- episode12、episode14：OpenVLA 名义阶段没有抓起碗，visual-lift 注入未触发；
- 因此无法用这些 episode 测试恢复或 verifier。

### 7.2 Phase distribution shift

- episode13：成功触发 lost grasp 和 failure detector；
- learned phase 长期停留在 DESCEND；
- 没有进入 CLOSE/VERIFY。

这说明下一阶段的主要瓶颈已经从 verifier 转移到 phase coverage 和名义策略成功率。

### 7.3 恢复出口分布

视觉确认需要观察持续抓稳，而 OpenVLA 对交还状态高度敏感。当前用手工 settle
修复，但最终应将 `verify-confirmed → nominal handoff` 纳入学习动作数据。

## 8. 与用户提出的两阶段方案的距离

用户预想可概括为：

```text
阶段一：失败理解 / 状态识别 / 恢复规划
阶段二：条件化恢复控制 / 执行 / 验证 / 重规划
```

目前已经有可运行的模块化原型：

- Stage-B mask tracker + failure detector：承担部分阶段一；
- learned phase predictor：承担粗粒度恢复计划；
- learned residual controller：承担阶段二动作；
- visual verifier：承担恢复结果验证。

但这些模块尚未由统一 temporal SlotSSM state 串联，也没有失败类型 token、恢复历史
memory 和多轮重规划。因此当前更准确的定位是：

> 用户两阶段方案的单任务、单失败类型 feasibility prototype 已成立，但统一模型、
> 多失败恢复和 benchmark 级验证尚未完成。

## 9. 下一步优先级

### P0：修复 phase coverage

1. 将 episode13 的在线状态作为 DAgger-style hard state；
2. 用 oracle phase 对相同状态采 teacher trace；
3. 重训 phase predictor；
4. 要求 episode13 或新的未见 episode 能进入 CLOSE/VERIFY；
5. 再执行 visual verifier final test。

### P1：学习恢复出口

1. 将 `VISUAL SETTLE` transition 写入 recovery dataset；
2. 增加 `VERIFY_CONFIRMED/HANDOFF` phase 或 mode；
3. 训练 residual controller 学会从确认高度回到 nominal handoff 分布；
4. 删除手工 settle controller。

### P2：删除 gripper override

1. 增加 close/open 的 hard negative 和边界数据；
2. 单独审计 gripper sign false action；
3. 通过在线安全 gate 后删除 phase-based sign override。

### P3：回归真正两阶段 SlotSSM

在上述低维模块建立可靠 teacher/data contract 后：

1. 用 SlotSSM temporal state 替换独立 phase/verifier MLP；
2. 显式建模 failure type、phase、confidence 和 recovery history；
3. 支持验证失败后的再次规划，而非单次恢复；
4. 扩展到 missed grasp、wrong target、wrong placement；
5. 做跨 episode、跨 task 和无恢复损伤 benchmark。

## 10. 关键文件与产物

代码：

```text
experiments/robot/libero/run_single_scene_recovery.py
scripts/train_recovery_residual_smoke.py
scripts/train_recovery_phase_predictor.py
scripts/train_visual_regrasp_verifier.py
scripts/audit_recovery_visual_3d.py
scripts/build_recovery_expert_manifest.py
```

最值得查看的视频：

```text
# learned actions + learned phase，未见 episode11
/data2/lty/naive-e-slotssm-runs/recovery-learned-phase-final-test-v1/
task9-episode11.mp4

# visual verifier 替换 bowl-height oracle，episode10 online smoke
/data2/lty/naive-e-slotssm-runs/recovery-visual-verifier-online-smoke-v4-settle/
task9-episode10.mp4
```

核心 checkpoint：

```text
/data2/lty/naive-e-slotssm-runs/recovery-residual-smoke-v6-geometry/
recovery-residual-mlp.safetensors

/data2/lty/naive-e-slotssm-runs/recovery-phase-predictor-v2-dagger/
recovery-phase-predictor.safetensors

/data2/lty/naive-e-slotssm-runs/visual-regrasp-verifier-v10-temporal/
visual-regrasp-verifier.safetensors
```

## 11. 一句话结论

目前已经从“oracle 手工状态机重抓”推进到“视觉失败检测 + 视觉三维定位 + 学习恢复
动作 + 学习 phase + 视觉抓稳验证”的在线闭环，并获得成功 MP4；剩余最关键工作是
增强 phase distribution coverage、学习 handoff/settle，并最终把这些模块统一进用户
提出的两阶段 temporal SlotSSM。
