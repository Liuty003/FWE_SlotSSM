    # 基于 Embodied-SlotSSM 的对象中心失败检测与通用恢复策略

## 1. 目标

本项目以 **Embodied-SlotSSM** 为基础，构建一套面向机器人操作失败的闭环检测与恢复框架，重点处理：抓取未成功、运输中物体滑落、目标物体被意外推动、放置位置小范围偏移。

系统使用持续对象 slot 表示场景，训练单标量风险评分器判断当前执行是否可能失败；检测到高风险后，切换到一个 **Slot 条件化的通用恢复策略**，生成重新抓取候选，再利用同一风险模型选择低风险候选并短视野执行。

\[
\text{对象建模}
\rightarrow
\text{失败检测}
\rightarrow
\text{恢复策略生成候选}
\rightarrow
\text{候选风险评分}
\rightarrow
\text{短视野闭环执行}
\]

---

## 2. 总体结构

系统由四部分组成：

1. **Embodied-SlotSSM 主策略**：负责正常任务执行；
2. **对象中心风险评分器**：输出失败概率 \(p_{\mathrm{fail}}\)；
3. **通用恢复策略 \(\pi_{\mathrm{rec}}\)**：负责漏抓或掉落后的重新抓取；
4. **动作条件化 Slot Dynamics**：预测候选动作对应的未来对象状态。

正常模式：

\[
A_t^{\mathrm{task}}
=
\pi_{\mathrm{task}}(I_t,g,q_t,M_t)
\]

其中 \(I_t\) 为 RGB 图像，\(g\) 为任务指令，\(q_t\) 为机器人内部状态，\(M_t=\{m_t^1,\ldots,m_t^K\}\) 为 SlotSSM 持续对象状态。

风险评分器：

\[
p_{\mathrm{fail},t}
=
R_\phi(M_t,E_A(A_t),E_q(q_t),D_t)
\]

当

\[
p_{\mathrm{fail},t}>\tau_{\mathrm{enter}}
\]

时，中断当前 action chunk，进入恢复模式。

---

## 3. 持续对象表示

当前图像经过视觉编码器：

\[
F_t=E_{\mathrm{vision}}(I_t)
\]

Slot Attention 使用上一帧 slot 初始化：

\[
S_t=\operatorname{SlotAttention}(F_t,S_{t-1})
\]

每个对象 slot 沿时间更新：

\[
m_t^i=\operatorname{SlotSSM}(m_{t-1}^i,s_t^i)
\]

得到：

\[
M_t=\{m_t^1,\ldots,m_t^K\}
\]

动作模型保留全部 slot，通过交叉注意力读取：

\[
H_t'=H_t+\gamma\operatorname{CrossAttn}(Q=H_t,K=M_t,V=M_t)
\]

Slot 在三个阶段中的作用：

| 阶段 | Slot 的作用 |
|---|---|
| 正常操作 | 提供目标物体、夹爪、容器和干扰物体的持续表示 |
| 失败检测 | 表达对象历史、运动关系和状态偏差 |
| 失败恢复 | 高亮异常目标对象，条件化恢复动作生成和候选风险评估 |

---

## 4. 对象中心风险评分器

风险评分器只输出：

\[
p_{\mathrm{fail},t}\in[0,1]
\]

不预测失败类型、失败对象或恢复模式。

训练使用成功/失败轨迹的弱标签：

\[
y_t=
\begin{cases}
0,&\text{成功轨迹}\\
1,&\text{失败轨迹}
\end{cases}
\]

损失：

\[
\mathcal L_{\mathrm{fail}}
=
\operatorname{BCE}(p_{\mathrm{fail},t},y_t)
\]

风险模型内部保留 slot attention：

\[
\rho_t=\{\rho_t^1,\ldots,\rho_t^K\}
\]

\(\rho_t\) 不作为监督输出，只用于恢复阶段高亮异常对象。

---

## 5. 动作条件化 Slot Dynamics

训练轻量预测模型评估动作后果：

\[
\widehat M_{t+1}^{(k)}
=
D_\psi(M_t,A_t^{(k)},q_t)
\]

第一版只预测下一时刻或短窗口末端对象状态。

预测损失：

\[
\mathcal L_{\mathrm{dyn}}
=
\frac{1}{K}
\sum_{i=1}^{K}
\left[1-\cos(\widehat m_{t+1}^i,m_{t+1}^i)\right]
\]

对象预测偏差：

\[
d_{t+1}^i
=
1-\cos(\widehat m_{t+1}^i,m_{t+1}^i)
\]

\[
D_{t+1}=[d_{t+1}^1,\ldots,d_{t+1}^K]
\]

例如夹爪上升但目标物体 slot 未同步移动时，目标 slot 偏差应明显增大。

---

## 6. 通用恢复策略

采用一个统一的 **Slot-Conditioned Regrasp Policy**：

\[
A_t^{\mathrm{rec}}
=
\pi_{\mathrm{rec}}(I_t,q_t,\widetilde M_t,\texttt{<RECOVER>})
\]

该策略只负责：

\[
\text{重新定位}
\rightarrow
\text{重新接近}
\rightarrow
\text{重新抓取}
\rightarrow
\text{稳定抬升}
\]

恢复完成后切回原 Embodied-SlotSSM 主策略。

### 6.1 模型形式

共享：

- 视觉编码器；
- Slot Attention；
- SlotSSM；
- Relation Encoder；
- 动作解码器主体。

只训练：

- Recovery LoRA 或 Adapter；
- 恢复 slot adapter；
- 可选抓取关系辅助头。

\[
\pi_{\mathrm{task}}=\pi_{\theta,\eta_{\mathrm{task}}}
\]

\[
\pi_{\mathrm{rec}}=\pi_{\theta,\eta_{\mathrm{rec}}}
\]

其中 \(\theta\) 冻结，\(\eta_{\mathrm{rec}}\) 为恢复专用参数。

### 6.2 恢复 Slot 高亮

结合任务相关性 \(\alpha_t^i\) 与风险贡献 \(\rho_t^i\)：

\[
\beta_t^i
=
\operatorname{softmax}_i(\alpha_t^i+\lambda_\rho\rho_t^i)
\]

构造恢复输入：

\[
\widetilde m_t^i
=
m_t^i+
\lambda_{\mathrm{rec}}\beta_t^iW_{\mathrm{rec}}m_t^i
\]

恢复策略仍读取全部 slot，但重点关注与任务相关且状态异常的对象。

---

## 7. 恢复数据生成：仿真扰动 + 脚本化恢复

不建立新的公开数据集，只在现有 LIBERO / LIBERO-Mem 环境中自动生成恢复轨迹。

### 7.1 数据来源

- Embodied-SlotSSM 对应公开训练数据；
- LIBERO 或 LIBERO-Mem 成功示范；
- 当前基础策略在公开环境中生成的成功与失败 rollout。

### 7.2 自动构造失败状态

从成功轨迹中定位抓取和运输阶段。

**漏抓：**

- 末端位置偏移；
- 目标物体小范围位移；
- 夹爪闭合时刻扰动；
- 接近高度扰动。

**物体掉落：**

- 强制打开夹爪；
- 解除抓取约束；
- 将目标物体释放到夹爪下方附近；
- 对物体位置和姿态加入小扰动。

得到失败状态：

\[
s_t^{\mathrm{fail}}
\]

### 7.3 脚本化恢复轨迹

利用仿真环境中的目标物体位姿生成：

1. 移动到物体上方；
2. 对齐夹爪；
3. 向下接近；
4. 闭合夹爪；
5. 向上抬升；
6. 保持若干步确认稳定抓取。

得到：

\[
\tau_{\mathrm{rec}}
=
\{(I_t,q_t,M_t,A_t^{\mathrm{rec}})\}_{t=1}^{T}
\]

### 7.4 自动过滤

只保留满足：

\[
\|\Delta p_{\mathrm{obj}}-\Delta p_{\mathrm{gripper}}\|<\epsilon
\]

且：

\[
z_{\mathrm{obj}}>z_{\mathrm{table}}+\delta
\]

的轨迹，即目标物体与夹爪同步运动并被稳定抬起。

每条恢复轨迹保存：

```text
episode_id
task_instruction
rgb
robot_state
slots
slot_memory
target_slot_attention
recovery_action
next_slots
recovery_success
```

---

## 8. 恢复策略训练

恢复策略使用行为克隆训练。

连续动作损失：

\[
\mathcal L_{\mathrm{BC}}
=
\sum_t\|\widehat a_t-a_t\|_1
\]

可选抓取关系标签由环境自动生成：

\[
y_t^{\mathrm{grasp}}
=
\mathbb I
\left[
\|\Delta p_{\mathrm{obj}}-\Delta p_{\mathrm{gripper}}\|<\epsilon
\right]
\]

辅助损失：

\[
\mathcal L_{\mathrm{grasp}}
=
\operatorname{BCE}(\widehat y_t^{\mathrm{grasp}},y_t^{\mathrm{grasp}})
\]

总损失：

\[
\mathcal L_{\mathrm{rec}}
=
\mathcal L_{\mathrm{BC}}+
\lambda_g\mathcal L_{\mathrm{grasp}}
\]

第一版可只使用 \(\mathcal L_{\mathrm{BC}}\)。

---

## 9. 候选恢复动作与统一风险评分

恢复策略生成 \(K_c\) 个候选：

\[
\mathcal A_t^{\mathrm{rec}}
=
\{A_t^{(1)},\ldots,A_t^{(K_c)}\}
\]

建议：

\[
K_c=3
\]

候选通过不同采样噪声、接近方向或抓取点扰动获得。

对每个候选预测未来对象状态：

\[
\widehat M_{t+1}^{(k)}
=
D_\psi(M_t,A_t^{(k)},q_t)
\]

使用同一个风险评分器：

\[
r_t^{(k)}
=
R_\phi(\widehat M_{t+1}^{(k)},E_A(A_t^{(k)}),q_t)
\]

选择：

\[
k^*=\arg\min_k r_t^{(k)}
\]

只执行候选动作前一到两步：

\[
A_{t,\mathrm{exec}}
=
A_t^{(k^*)}[0:H_{\mathrm{rec}}]
\]

其中：

\[
H_{\mathrm{rec}}=1\sim2
\]

随后重新观测、更新 slot 和重新评分。

---

## 10. 模式切换

### 进入恢复模式

当：

\[
p_{\mathrm{fail},t}>\tau_{\mathrm{enter}}
\]

且满足抓取关系异常，例如：

\[
\text{gripper moving up}
\land
\text{target slot not moving up}
\]

或：

\[
\text{target previously followed gripper}
\land
\text{target now separates}
\]

调用通用恢复策略。

### 退出恢复模式

连续 \(N_{\mathrm{safe}}\) 次满足：

\[
p_{\mathrm{fail},t}<\tau_{\mathrm{exit}}
\]

并且：

\[
\|\Delta p_{\mathrm{obj}}-\Delta p_{\mathrm{gripper}}\|<\epsilon
\]

以及：

\[
z_{\mathrm{obj}}>z_{\mathrm{table}}+\delta
\]

然后切回主策略。

初始参数建议：

```yaml
tau_enter: 0.6
tau_exit: 0.4
safe_steps: 3
num_candidates: 3
recovery_horizon: 1
```

---

## 11. 分阶段训练方案

### 阶段一：加载或复现 Embodied-SlotSSM

确认 Slot Attention、SlotSSM、Relation Encoder、正常动作生成及各特征接口。

### 阶段二：训练动作条件化 Slot Dynamics

冻结主干，训练：

\[
D_\psi(M_t,A_t,q_t)
\]

损失使用 \(\mathcal L_{\mathrm{dyn}}\)。

### 阶段三：训练风险评分器

冻结主干和 Slot Dynamics，训练：

\[
R_\phi(M_t,A_t,q_t,D_t)
\]

损失使用 \(\mathcal L_{\mathrm{fail}}\)。

### 阶段四：生成恢复数据

在公开仿真环境中自动注入漏抓和掉落，并使用脚本控制器生成恢复轨迹。

### 阶段五：训练通用 Recovery LoRA

冻结共享主干，只训练恢复 LoRA / Adapter。

### 阶段六：闭环联合评测

部署：

\[
\text{主策略}
+
\text{风险检测}
+
\text{恢复策略}
+
\text{候选评分}
\]

第一版不进行全部模块端到端联合训练。

---

## 12. 实验对比

### 基础策略

- OpenVLA；
- 原始 Embodied-SlotSSM；
- 完整方法。

### 失败检测

- SAFE 风格 VLA hidden only；
- SlotSSM only；
- SlotSSM + action；
- SlotSSM + action + prediction deviation。

### 恢复

- 无恢复；
- Direct Retry；
- Short-Horizon Retry；
- 原 VLA 多候选随机选择；
- 通用 Recovery Policy 单候选；
- Recovery Policy 多候选随机选择；
- Recovery Policy + VLA hidden 风险评分；
- **Recovery Policy + SlotSSM 预测风险评分**。

外部恢复方法优先对比：

- ProbeAct；
- FAR；
- 资源允许时增加 FailSafe 或 FLARE。

主要指标：

- 正常任务成功率；
- 失败检测 AUROC / F1；
- 从失败状态开始的恢复成功率；
- 最终任务成功率；
- 重新抓取成功率；
- 再次掉落率；
- 相同失败重复率；
- 平均恢复步数；
- 推理时间与 VLA 调用次数。

---

## 13. 最小可行版本

第一版只处理：

> 漏抓或物体掉落到桌面后重新抓取。

必须完成：

\[
\boxed{
\text{Embodied-SlotSSM}
+
\text{单标量风险评分器}
+
\text{自动恢复数据生成}
+
\text{通用 Recovery LoRA}
+
\text{三候选风险选择}
+
\text{短视野闭环}
}
\]

暂不处理：

- 物体掉到不可达区域；
- 物体翻倒后需要扶正；
- 完全新的操作技能；
- 多种 Reset Skill；
- 失败类型分类；
- 文本恢复规划。

论文主线可概括为：

> 利用 SlotSSM 构建对象中心风险状态，检测当前执行失败；通过同一对象表示条件化通用恢复策略，并在预测的对象未来中选择低风险恢复动作。
