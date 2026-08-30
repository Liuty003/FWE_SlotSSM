# LIBERO-Plus / LIBERO-10 统一恢复实验计划

最后更新：2026-08-28

本文件是接下来 LIBERO-10 工作的执行清单。方法总说明见
[`LIBERO_PLUS_10030_RECOVERY_METHOD.md`](./LIBERO_PLUS_10030_RECOVERY_METHOD.md)，历史资产见
[`ARTIFACT_INDEX.md`](./ARTIFACT_INDEX.md)。若历史记录与本文件的正式评测口径冲突，以本文件为准。

## 1. 目标与范围

在冻结 OpenVLA-OFT 的前提下，为 LIBERO-Plus 官方 `libero_10` suite 的全部 2,519 个实例加入
native-input-only V4.4 recovery，并与同一批官方初始状态上的 OFT baseline 公平比较。

冻结输入与评测约束：

- 在线只使用 OFT 原本拥有的双视角 RGB、语言、proprio、OFT hidden/nominal action；
- 在线禁止 depth、segmentation、对象/fixture 真值、contact、BDDL truth 和 MuJoCo state；
- privileged 信息只允许用于离线 teacher、标签和辅助监督；
- 正常状态必须 exact bypass 到冻结 OFT；
- 所有正式 rollout 使用官方 initial state、episode 0、`wait_steps=10` 和官方最大步数；
- 每条正式 rollout 必须保存 `report.json` 和 MP4；
- gate、resume 和 recovery 不得读取 task ID、扰动类别或 difficulty；
- 最终报告必须同时给出 repaired、regressed、unchanged-success 和 unchanged-failure。

## 2. 模型组织方式

第一阶段采用 **10 个语义任务各一个 checkpoint**，而不是一个模型硬学全部任务，也不是每个扰动或每个
task ID 一个模型。

```text
LIBERO-10 官方实例
        ↓ 仅离线按任务语义分组
10 个语义任务族
        ↓ 每族共享同一结构与训练脚本
family-specific V4.4 checkpoint
        ↓ 在线不读取 task ID / 扰动标签
该语义任务在七类官方扰动下统一评测
```

这样做的原因：恢复 action 强烈依赖具体物体、容器和子目标，但 ownership 结构、数据契约和训练流程可以
完全共享。完成 10 个独立模型后，再做跨任务共享 checkpoint 是后续扩展，不阻塞第一版 2,519 条结果。

## 3. 十个语义任务族

官方 manifest 中的 2,519 条应当无遗漏、无重叠地映射到以下十族：

| family slug | 语义 | 官方实例数 |
|---|---|---:|
| `stove-moka` | 打开炉灶并放上 moka pot | 281 |
| `mug-microwave` | 黄白杯放入微波炉并关门 | 278 |
| `two-moka-stove` | 两个 moka pot 都放到炉灶上 | 233 |
| `black-bowl-drawer` | 黑碗放入底层抽屉并关抽屉 | 304 |
| `two-mugs-two-plates` | 两个杯子分别放到左右盘子 | 287 |
| `mug-pudding` | 白杯放盘子、pudding 放盘子右侧 | 283 |
| `cream-cheese-butter-basket` | cream cheese 与 butter 放入篮子 | 210 |
| `soup-tomato-basket` | alphabet soup 与 tomato sauce 放入篮子 | 212 |
| `soup-cream-cheese-basket` | alphabet soup 与 cream cheese 放入篮子 | 194 |
| `book-caddy` | 书放入 caddy 后部隔间 | 237 |
| 合计 |  | **2,519** |

任务族映射只用于离线选择对应 checkpoint；正式论文应明确这是 per-semantic-task recovery 设置。模型内部不能
再用具体 task ID 选择动作轨迹或 recovery horizon。

## 4. 已有资产与不可混用口径

完整 OFT baseline 已存在：

- 根目录：`/data2/lty/naive-e-slotssm-runs/libero-plus-global-v1/full-audit-v1/libero_10`
- 完整度：2,519/2,519；
- OFT 成功：1,960/2,519（77.8%）；
- OFT 失败：559/2,519。

历史上已经开发过三个语义族的 Robot Initial States 子集，共 120 条：

- Black Bowl 44 条：混合 learned gate + deterministic grasp FSM 为 31/44；
- Mug + Pudding 42 条：单一 V4.3 checkpoint 严格结果 31/42；
- Book 34 条：纯 V4.3 为 16/34，29/34 是 task-ID selector，不能作为统一神经恢复成绩。

这些历史 teacher、失败 fork、特征和 checkpoint 可以作为候选初始化或 teacher 资产，但不能与新协议结果直接
拼接。所有十族最终都必须用相同 V4.4 native-input 数据契约重新训练并重新评测。

## 5. 统一流水线

### 阶段 A：冻结任务族与 baseline 清单

1. 从 `official-manifest.json` 自动生成十个 `family-manifest.json`；
2. 验证十族并集恰好 2,519，交集为空；
3. 对齐 full-audit，生成每族成功/失败数、扰动与 difficulty 分布；
4. 生成 suite 级 `family-index.json`，记录输入 manifest 的 SHA256。

完成标准：2,519 条全部有唯一 family、metadata、steps、rollout.mp4 和 dense bundle；成功数合计 1,960，失败数
合计 559。

### 阶段 B：自动失败分类与失败边界

对每族运行统一分类器和 onset locator：

1. 用 BDDL progress、物体运动、夹爪、contact 和 fixture joint 做离线失败分类；
2. 自动输出高/中/低置信以及候选 recovery skill；
3. 定位最后正常状态、首次失败证据和可恢复 fork；
4. 每族按 failure type × perturbation × difficulty 抽取少量 MP4 人工审计，而不是逐条看 559 个视频；
5. 人工修订只写入独立 override manifest，禁止修改原始 rollout。

完成标准：所有 559 条失败均有分类；高/中置信条目有可重放 onset；低置信条目被保留为 unresolved，不伪造
teacher 标签。

### 阶段 C：teacher 与 handoff 认证

1. 从官方初态重放 saved executed-action prefix，恢复 controller 与 RNG 状态；
2. 按失败类型运行 privileged teacher 技能；
3. 只保留 simulator-verified 成功 recovery；
4. 从 teacher 轨迹多个候选点 fresh-replan 冻结 OFT；
5. OFT 能继续完成任务的最早点成为 certified handoff；
6. teacher 失败、错误归还与近成功分支作为 resume/verifier 负样本。

优先复用历史 teacher 逻辑，但每个来源必须在新 full-audit 初态和统一协议下重新验证。禁止把 task-specific action
trace 当作最终在线 policy。

### 阶段 D：native-input 特征与训练

每个 family 构造：

- `normal-all`：该族所有 OFT 成功轨迹；
- `failed-prefix`：失败边界前后 ownership 样本；
- `recovery-all`：成功 teacher 与 DAgger action；
- `resume-boundaries`：certified handoff 正样本及 pre-handoff/failed-handoff 负样本。

每族训练一个 V4.4 dual-ownership checkpoint：

- shared slot representation；
- enter-risk gate；
- direct closed-loop recovery action head；
- independent resume-value head；
- phase/contact/geometry 可作 training-only auxiliary，不直接支配在线动作；
- gate 训练按实际错误接管代价做 hard-negative/calibration，不再只看 50:50 balanced loss；
- 默认连续两次 gate 确认后才接管，resume 连续三次确认后归还。

### 阶段 E：分层闭环验证

每个 family 按固定顺序推进：

1. **4 条 smoke**：2 条预期修复 + 2 条 baseline success protection；
2. **targeted matrix**：全部 baseline failure + gate-hard normal；
3. **完整 family**：该族所有官方实例统一重跑；
4. 只有完整 family 结果才进入主表，targeted 结果只作开发诊断。

若 targeted 阶段净提升不为正，停止该 checkpoint 的全族评测，优先修 gate/action，而不是继续浪费 GPU。

### 阶段 F：完整 2,519 条汇总

最终汇总必须报告：

- OFT 与 recovery 的总成功率；
- 每个语义族成功率；
- 七类扰动与五级 difficulty；
- repaired / regressed / unchanged-success / unchanged-failure；
- gate 激活率、handoff 率、错误接管率和最大时长 fallback；
- 每个 failure type 的 teacher 覆盖率与实际修复率；
- 2,519 个报告与 MP4 完整性。

## 6. 执行优先级

1. 先完成十族 manifest 和 baseline 统计；
2. 优先推进已有 teacher 资产的 `black-bowl-drawer`、`mug-pudding`、`book-caddy`，但使用新统一协议；
3. 随后推进结构相近、teacher 可复用的 basket 三族；
4. 最后推进 articulation 较多的 stove、microwave 与双 moka 任务；
5. 每完成一族立即冻结 checkpoint、完整评测和 artifact index，不等十族全部完成后再整理。

## 7. 统一资产目录

新主线根目录固定为：

```text
/data2/lty/naive-e-slotssm-runs/libero-plus-global-v1/libero-10-recovery-v1
```

目录约定：

```text
family-index.json
families/<family>/family-manifest.json
families/<family>/baseline-summary.json
families/<family>/failure-classification-v1.json
families/<family>/failure-onsets-v1/
families/<family>/teachers-v1/
families/<family>/handoff-certified-v1.json
families/<family>/features-v1/
families/<family>/model-v1/
families/<family>/smoke-v1/
families/<family>/targeted-v1/
families/<family>/full-eval-v1/
suite-summary.json
```

任何新结果只有在报告、配置、checkpoint、MP4 路径和输入数据 provenance 都写入该目录及
`ARTIFACT_INDEX.md` 后，才视为完成。

## 8. 当前执行状态

- [x] 2,519 条 OFT baseline 全部采集并验证；
- [x] 历史三个语义族资产已归档；
- [x] 冻结本执行计划；
- [x] 生成十族唯一映射和 baseline summary；
- [x] 自动分类 559 条失败并定位 onset（548 条自动定位，11 条稀疏人工审计）；
- [ ] 构造统一 teacher/handoff 数据（34 条跨族/失败类型 exact-fork pilot 正在审计）；
- [ ] 训练十个 V4.4 checkpoint；
- [ ] 完成十族 full family eval；
- [ ] 汇总完整 2,519 条正式结果。

### 8.1 teacher pilot 审计结论（2026-08-28）

第一轮 29 条非 microwave relational teacher 暴露出：当前瓶颈首先是恢复边界、子目标路由和物体 affordance，
不能把所有失败交给固定中心点 top-grasp。

- 34 条 pilot 中有 5 条在 fork 时 selected predicate 已为真：task401、task1989、task1967、task548、
  task578。这些禁止作为 recovery action 正样本；应改为 post-regression fork，或作为 gate 不接管 / resume
  样本。
- task544 在几乎没有动作证据的第 9 步选择 pudding，但任务同时要求先后处理 mug 与 pudding。多目标路由必须
  从同族成功示范或实际 approach/contact 证据推断当前子目标，不能随意选第一个未完成谓词。
- task2048 与 task306 在罐头掉落后才分叉。placement failure 应定位到错误释放或首次失去稳定接触之前；倒地
  状态只作为负样本，除非 teacher 明确支持 fallen-object side grasp。
- 黑碗需要 rim grasp；小盒/布丁需要由几何高度确定下降终点；斜书优先保持 fork 时可行夹爪姿态；双物体放置
  必须在目标 region 内做 occupancy-aware free-space placement。
- predicate destination 可以是对象名（如 plate_1），运动目标必须解析为对应 `*_default_site`，但官方验证谓词
  保持原对象名不变。

下一轮 teacher 的强制前置检查：

1. selected predicate 在 fork 必须为 false；
2. fork 必须早于不可逆掉落/倾倒，并有明确 target evidence；
3. 多目标任务必须有可审计的 subgoal-order 依据；
4. grasp profile 必须由物体资产几何、当前姿态或同族成功示范生成；
5. 已占用的 destination region 必须先求空闲放置点；
6. 只有静置后 predicate 仍为 true 的轨迹才进入 recovery action 训练集。
