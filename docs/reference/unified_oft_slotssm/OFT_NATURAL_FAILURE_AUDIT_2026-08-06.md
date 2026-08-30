# 官方 OpenVLA-OFT 自然失败补充审计（2026-08-06）

运行设置：官方 `openvla-7b-oft-finetuned-libero-10`、LIBERO-10、相同 initial
state、`open_loop_steps=8`，无故障注入。产物根目录为
`/data2/lty/naive-e-slotssm-runs/openvla-oft-natural-audit-v2/`；每个 episode 保存
主/腕 RGB、`steps.jsonl`（planned/executed action）、metadata 和 MP4。

## 结果

| Task | Episode | 结果 | 控制步 |
|---|---:|---|---:|
| 7 | 4 | failure | 520 |
| 7 | 5--9 | success | 227--257 |
| 8 | 0 | failure | 520 |
| 8 | 1 | failure | 520 |
| 8 | 2--7 | success | 351--445 |
| 8 | 8 | failure | 520 |
| 8 | 9 | failure | 520 |

因此，补充的 Task7 ep4--9 为 5/6 成功；结合已知 ep0--3 成功，Task7 当前为
9/10。Task8 为 6/10 成功、4/10 失败（ep0、1、8、9），是当前最优的主任务候选。

## 初步视频复核与下一步

Task8 的四个失败回合均在 520 步耗尽前未完成；采样帧显示其主要停留在 moka pot
接近/抓取阶段，尚未能仅凭稀疏采样帧区分空抓、目标选择错误或接近停滞。它们是有价值的
自然失败源，但不是已经证实的可恢复案例。

下一步应在 ep0、1、8、9 的多个可观察 fork 点采集 teacher branches，分别测量：

1. 是否存在短程重新接近/重新抓取后可完成的 branch；
2. 哪些时刻是可检测的持续无进展或抓取关系破坏；
3. 同一失败类型在不同 episode 是否可由共同的视觉/SlotSSM 状态识别。

只有其中存在多个可恢复的独立 failure source，Task8 才升级为正式恢复主任务；否则保留
为 failure-detection hard-negative 集，并继续扩大更难的自然失败 benchmark。
