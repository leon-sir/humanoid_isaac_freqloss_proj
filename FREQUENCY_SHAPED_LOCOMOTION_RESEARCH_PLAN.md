# 面向 Isaac Lab 的足式机器人频域正则化研究计划

> 暂定题目：**Frequency-Shaped Reinforcement Learning for Legged Locomotion**  
> 平台：Isaac Lab Manager-Based RL Environment + RSL-RL PPO  
> 定位：偏代码实现；阶段 A、B 详细规划，阶段 C、D 暂只定义目的、效果与候选损失。

## 1. 核心思想

传统 locomotion reward 通过逐时刻的关节位置、速度、加速度、动作变化率和镜像误差约束运动。本研究拟保留速度命令追踪与必要安全约束，将一部分 regulation reward 改写为频域目标：

> 不再逐时刻规定机器人“应该是什么姿势”，而是规定不同时间尺度上的运动能量、跨关节相位关系，以及机器人硬件允许的频率范围。

初期把归一化关节运动分为三档：

1. **零频（DC）**：静态工作点或时间平均姿态；
2. **运动频带（locomotion band）**：步频、主要谐波和有效周期运动；
3. **高频（high-frequency band）**：关节抖动、控制噪声与接触激发的快速振动。

阶段 B 再利用复数谱的幅值与相位构造允许时间偏移的频域对称约束。

---

## 2. 架构决策：阶段 A、B 的频域项应作为环境 reward

### 2.1 推荐结论

针对**实际关节轨迹**计算的频域 penalty 应在 Isaac Lab 环境侧转换成 reward：

\[
r_t=r_t^{\mathrm{task}}+r_t^{\mathrm{safety}}-\lambda_f L_{f,t},
\]

RSL-RL 保持标准 PPO：

\[
L_{\mathrm{PPO}}
=L_{\mathrm{clip}}+c_vL_{\mathrm{value}}-c_e\mathcal H.
\]

第一阶段不修改 `PPO.update()`，也不把实际关节角 FFT 直接加入 PPO optimizer loss。

### 2.2 原因

PPO 是 model-free policy gradient。环境 reward 不需要对 FFT、机器人动力学或策略参数可微；频域 penalty 只需形成标量 reward，再经 return/advantage 影响策略更新。

若把已经发生的关节轨迹频谱直接放入 PPO loss：

1. storage 中的关节角对当前 actor 参数是常量；
2. 没有可微动力学时，梯度不能从关节频谱回到策略；
3. RSL-RL 前馈 PPO 会把时间与环境维展平后随机组成 mini-batch，时间邻接关系丢失；
4. FFT 窗口可能比 `num_steps_per_env` 更长，被 PPO update 边界截断；
5. episode reset 可能位于 rollout 中间，需要另外处理分段、mask 和跨 rollout 历史。

当前 RSL-RL storage 使用 `[num_steps_per_env, num_envs, ...]` 保存 rollout，前馈更新时展平并随机采样；见 [RSL-RL RolloutStorage](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/storage/rollout_storage.py)。

### 2.3 适合加入 PPO loss 的特殊情况

策略等变辅助项不经过环境动力学，可以进入 PPO loss：

\[
L_{\mathrm{policy\text{-}mirror}}
=\left\|\pi_\theta(M_o o_t)-M_a\pi_\theta(o_t)\right\|_2^2.
\]

RSL-RL 已提供 symmetry data augmentation 与 mirror loss，应把它作为阶段 B 的对照方法，而不是与“实际闭环轨迹的频域对称 reward”混为一谈。见 [RSL-RL symmetry extension](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/extensions/symmetry.py)。

理论上还可在 storage 中对当前 actor 的 action mean 序列做 FFT 并反传，但这要求 sequence mini-batch，而且约束的是策略输出而非机器人实际运动，暂不采用。

---

## 3. 频谱历史应维护在 Manager-Based Env 内

### 3.1 推荐位置

在任务 `mdp` 子包中实现有状态 `JointFrequencyAnalyzer`，由继承 `ManagerTermBase` 的 reward term 持有或共享。它负责：

- 为每个并行环境维护独立关节历史；
- 每个 control step 写入归一化关节角；
- 每隔 `fft_update_interval` 批量调用 `torch.fft.rfft`；
- 缓存复谱、功率谱、DC 和频带能量；
- 同一步多个 reward term 访问时只更新一次；
- 对 reset 环境独立清空历史与有效长度。

Isaac Lab `RewardManager` 支持 callable class reward term，并在 reset 时调用其 `reset(env_ids)`，适合维护跨时间状态：[RewardManager 源码](https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/managers/reward_manager.html)。

### 3.2 环境缓冲的优势

环境环形缓冲：

- 可以跨多次 PPO update；
- 窗口长度独立于 rollout horizon；
- reset 与每个并行环境的 episode 生命周期一致；
- 不需要改 RSL-RL；
- 更换学习算法后仍可复用。

例如

\[
N_{\mathrm{env}}=4096,\quad N_w=128,\quad N_j=12,
\]

float32 历史缓冲约占

\[
4096\times128\times12\times4\ \mathrm{bytes}
\approx25.2\ \mathrm{MiB}.
\]

完整 complex64 单边频谱还会增加约 25 MiB，因此只缓存最近一次 FFT，并尽早聚合成实数频带特征。

### 3.3 历史奖励与可观测性

滑动窗口奖励依赖

\[
r_t=r(s_t,a_t,q_{t-N_w+1:t}),
\]

可把频谱缓冲视为增广环境状态。初期 actor 不必观察完整频谱；可选地给 critic 添加少量摘要：

\[
o_t^{\mathrm{critic}}
=[o_t,\bar z_t,E_t^{\mathrm{loc}},E_t^{\mathrm{high}}],
\]

减少 value prediction 的历史不确定性。该项作为阶段 A 消融，不是首版前置条件。

---

## 4. 简要相关工作

稳定平地行走的关节轨迹通常近似周期信号，可由步态基频与若干谐波表示；接触切换和快速落足引入高阶谐波。加速、转向、扰动和崎岖地形使信号非平稳，因此全窗口 FFT 适合阶段 A 的平地恒速验证，STFT/Welch 适合后续扩展。

- Solo12 工作使用关节角 FFT 对学得步态做后验频率分析：[Controlling the Solo12 quadruped robot with deep reinforcement learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC10366154/)。
- Cassie periodic reward composition 用周期性力/速度代价规定多种步态，但不直接优化关节 FFT：[Sim-to-Real Learning of All Common Bipedal Gaits](https://arxiv.org/abs/2011.01387)。
- CPG-RL 用振荡器的频率、幅值和相位组织四足运动：[CPG-RL](https://arxiv.org/abs/2211.00458)。
- Frequency-Aware MPC 用 frequency-shaped cost 匹配 ANYmal 执行器与接触带宽，是阶段 C、D 连接经典控制的重要先例：[Frequency-Aware MPC](https://arxiv.org/abs/1809.04539)。

本项目的区别是：不依赖完整参考轨迹或显式 CPG，而在实际闭环关节轨迹上施加 command-conditioned 多频带能量分配与复谱对称约束。

---

# 阶段 A：零频—运动频带—高频整体框架

## 5. 研究问题与最小 reward

研究问题：有限的三档频域指标能否替代传统逐时刻 regulation reward，并实现：

1. 无命令时回到合理工作点并稳定站立；
2. 有命令时形成能量集中的周期运动；
3. 降低关节和动作高频抖动；
4. 不依赖完整参考轨迹而保持速度追踪。

首版不删除全部传统 reward：

\[
r_t=
r_t^{\mathrm{lin\text{-}vel}}
+r_t^{\mathrm{yaw}}
+r_t^{\mathrm{alive}}
-r_t^{\mathrm{unsafe}}
-\lambda_fL_{f,t}.
\]

`unsafe` 至少覆盖 joint-limit barrier、非法接触、明显超限的 torque/velocity、跌倒 termination，必要时保留很弱的 base height/orientation 安全项。

优先尝试用频域项替代：

- joint position deviation regulation；
- joint velocity/acceleration penalty；
- action-rate penalty；
- 逐时刻左右对称 penalty。

## 6. 关节归一化与限位

### 6.1 机械范围归一化

\[
z_j=2\frac{q_j-q_j^{\min}}{q_j^{\max}-q_j^{\min}}-1.
\]

它便于跨关节比较频谱，但 \(z_j=0\) 只是机械范围中点，不一定是合理站姿。

### 6.2 推荐 nominal-centered normalization

\[
z_j=\frac{q_j-q_j^{\mathrm{nom}}}{s_j},\qquad
s_j=\min(q_j^{\mathrm{nom}}-q_j^{\min},
q_j^{\max}-q_j^{\mathrm{nom}}).
\]

此时 \(z_j=0\) 对应 nominal posture，\(|z_j|=1\) 对应距 nominal 最近的机械限位。

频域指标不能保证瞬时不越界，仍需：

\[
L_{\mathrm{limit}}
=\sum_j\operatorname{softplus}\!\left(\beta(|z_j|-\rho)\right)/\beta,
\qquad \rho\in[0.8,0.9].
\]

频域负责窗口尺度的运动结构，barrier 负责每个时刻的机械安全。

## 7. 采样、窗口与频率轴

策略控制周期：

\[
\Delta t=\texttt{sim.dt}\times\texttt{decimation},\qquad
f_s=\frac1{\Delta t}.
\]

窗口长度 \(N_w\) 对应：

\[
\Delta f=\frac{f_s}{N_w},\qquad f_{\mathrm{Nyq}}=\frac{f_s}{2}.
\]

启动建议：

- 保持 baseline 控制频率不变；
- `window_size=128`；若 \(f_s=50\) Hz，窗口 2.56 s，\(\Delta f\approx0.39\) Hz；
- `fft_update_interval=2` 或 4；
- `torch.fft.rfft` + `torch.fft.rfftfreq`；
- 全部在仿真 GPU 上批量执行。

窗口应至少覆盖 2--4 个预期 gait cycle；消融 \(N_w\in\{64,128,256\}\)。

## 8. DC 与 AC 分离

不要把 Hann-window FFT 的首个 bin 直接当作 DC。先计算原始时间均值：

\[
\mu_j(t)=\frac1{N_w}\sum_{n=0}^{N_w-1}z_j(t-n),
\]

再去均值：

\[
\tilde z_j(t-n)=z_j(t-n)-\mu_j(t),
\]

对 AC 信号加 Hann window：

\[
Z_j(k)=\operatorname{RFFT}\{w(n)\tilde z_j(t-n)\}.
\]

功率谱必须做单边谱和窗函数能量校正，使改变窗口长度时 reward 量级尽量稳定，并用 Parseval 测试验证。

## 9. 三档频域指标

### 9.1 DC：command-conditioned static operating point

命令强度：

\[
c_t=\sqrt{(v_x^{\mathrm{cmd}})^2+(v_y^{\mathrm{cmd}})^2+
\kappa(\omega_z^{\mathrm{cmd}})^2}.
\]

站立门控：

\[
g_{\mathrm{stand}}(c_t)=
\exp\left(-\frac{c_t^2}{\sigma_c^2}\right).
\]

DC 工作点：

\[
L_{\mathrm{DC}}
=g_{\mathrm{stand}}\sum_j
(\mu_j-z_j^{\mathrm{stand},*})^2.
\]

首版令 \(z_j^{\mathrm{stand},*}=0\)，即 nominal posture。站立时还应抑制 AC：

\[
L_{\mathrm{stand\text{-}AC}}
=g_{\mathrm{stand}}\sum_j\sum_{k>0}P_j(k).
\]

### 9.2 运动频带：能量集中而非无条件增大

\[
\mathcal B_{\mathrm{loc}}
=\{f:f_{\mathrm{loc}}^{\min}\le f\le f_{\mathrm{loc}}^{\max}\}.
\]

初始给宽频带，例如 0.5--4 Hz，之后由具体机器人 baseline PSD 校准。

\[
E_j^{\mathrm{loc}}
=\sum_{f_k\in\mathcal B_{\mathrm{loc}}}P_j(k),\qquad
E_j^{\mathrm{AC}}=\sum_{k>0}P_j(k),
\]

\[
R_j^{\mathrm{loc}}
=\frac{E_j^{\mathrm{loc}}}{E_j^{\mathrm{AC}}+\varepsilon}.
\]

令 \(g_{\mathrm{move}}=1-g_{\mathrm{stand}}\)：

\[
L_{\mathrm{loc}}
=g_{\mathrm{move}}\sum_j(1-R_j^{\mathrm{loc}}).
\]

仅奖励绝对低频能量会诱导无意义摆腿，故首版使用“能量集中率”，由速度追踪要求机器人真正前进。若出现运动幅值过小/过大，再启用：

\[
L_{\mathrm{loc\text{-}range}}
=g_{\mathrm{move}}\sum_j\left[
\operatorname{ReLU}(E_j^{\min}-E_j^{\mathrm{loc}})^2+
\operatorname{ReLU}(E_j^{\mathrm{loc}}-E_j^{\max})^2
\right].
\]

### 9.3 高频：抑制关节振动

\[
\mathcal B_{\mathrm{high}}=\{f:f\ge f_{\mathrm{high}}\},
\]

\[
L_{\mathrm{high}}
=\sum_j\sum_{f_k\in\mathcal B_{\mathrm{high}}}
W_{\mathrm{high}}(f_k)P_j(k).
\]

首版 \(W_{\mathrm{high}}=1\)。阶段 D 再由真机执行器带宽和结构共振设计权重。

### 9.4 阶段 A 总频域项

\[
L_f^A=
\lambda_{\mathrm{DC}}L_{\mathrm{DC}}+
\lambda_{\mathrm{stand}}L_{\mathrm{stand\text{-}AC}}+
\lambda_{\mathrm{loc}}L_{\mathrm{loc}}+
\lambda_{\mathrm{high}}L_{\mathrm{high}}.
\]

Isaac Lab `RewardManager` 会把 term 输出乘配置 weight 与环境 step time；内部不要再次乘 `dt`。见 [RewardManager 文档](https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/managers/reward_manager.html)。

## 10. 计划代码结构

    loco_f_reg/
    ├─ source/loco_f_reg/loco_f_reg/
    │  └─ tasks/manager_based/frequency_locomotion/
    │     ├─ __init__.py
    │     ├─ frequency_locomotion_env_cfg.py
    │     ├─ frequency_locomotion_flat_env_cfg.py
    │     ├─ agents/rsl_rl_ppo_cfg.py
    │     └─ mdp/
    │        ├─ __init__.py
    │        ├─ frequency_analyzer.py
    │        ├─ frequency_rewards.py
    │        ├─ frequency_observations.py
    │        ├─ rewards.py
    │        └─ terminations.py
    ├─ scripts/
    │  ├─ train.py
    │  ├─ play.py
    │  ├─ export_frequency_rollout.py
    │  └─ plot_frequency_report.py
    ├─ tests/
    │  ├─ test_frequency_analyzer.py
    │  ├─ test_frequency_rewards.py
    │  ├─ test_frequency_reset.py
    │  └─ test_frequency_symmetry.py
    └─ configs/experiments/
       ├─ baseline.yaml
       ├─ stage_a_dc.yaml
       ├─ stage_a_three_band.yaml
       └─ stage_b_spectral_symmetry.yaml

具体路径按最终 Isaac Lab external template 调整，但 analyzer、reward、配置、离线分析与测试应分离。

## 11. `JointFrequencyAnalyzer` 详细设计

### 11.1 配置

    @configclass
    class JointFrequencyAnalyzerCfg:
        window_size: int = 128
        fft_update_interval: int = 4
        window_type: str = "hann"
        normalization: str = "nominal_centered"
        locomotion_band_hz: tuple[float, float] = (0.5, 4.0)
        high_band_hz: tuple[float, float | None] = (8.0, None)
        eps: float = 1.0e-8
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")

频带只是启动参数，必须由机器人、控制频率和 baseline PSD 修订。

### 11.2 GPU 张量

    history          [num_envs, window_size, num_joints] float32
    valid_count      [num_envs]                           int32/int64
    write_index      scalar                               int64
    mean             [num_envs, num_joints]               float32
    spectrum         [num_envs, num_freqs, num_joints]    complex64
    power            [num_envs, num_freqs, num_joints]    float32
    band_energy      [num_envs, num_bands, num_joints]    float32
    last_update_step scalar                                int64

并行环境同步 step，可共用一个 ring-buffer 写指针；`valid_count` 必须按环境维护，因为 reset 时刻不同。

### 11.3 更新流程

    class JointFrequencyAnalyzer:
        def update(self, env):
            step = env.common_step_counter

            # 同一步只采样一次
            if self.last_sample_step != step:
                q = robot.data.joint_pos[:, joint_ids]
                z = normalize_joint_pos(q)
                self.history[:, self.write_index, :] = z
                self.valid_count.add_(1).clamp_max_(self.window_size)
                self.write_index = (self.write_index + 1) % self.window_size
                self.last_sample_step = step

            # 非 FFT step 返回缓存
            if step % self.fft_update_interval != 0:
                return self.cached_features

            ordered = reorder_ring_buffer(self.history, self.write_index)
            mean = ordered.mean(dim=1, keepdim=True)
            ac = (ordered - mean) * self.hann_window
            spectrum = torch.fft.rfft(ac, dim=1)
            power = one_sided_window_corrected_power(spectrum)
            self.cached_features = aggregate_bands(mean, spectrum, power)
            self.last_fft_step = step
            return self.cached_features

实现时要防止多个 reward term 在同一步重复写 buffer 或重复 FFT，使用 `env.common_step_counter` 作为缓存键。

### 11.4 多 term 共享 analyzer

推荐在 `env._joint_frequency_analyzers[key]` 注册共享 analyzer：

- 第一个 term 懒初始化；
- 后续 term 用同一 key 获取；
- `update_once(step)` 保证只计算一次；
- 各 callable term 的 `reset(env_ids)` 调同一 analyzer reset；操作必须幂等。

这样 RewardManager 可分别记录 `dc_posture`、`locomotion_band` 和 `high_frequency`。最早原型也可用单一复合 reward，但正式实验前应拆分日志。不建议每个 term 保存一份历史并重复 FFT。

### 11.5 reset 与 warm-up

    def reset(self, env_ids):
        self.history[env_ids] = 0.0
        self.valid_count[env_ids] = 0
        self.cached_ready[env_ids] = False

窗口未满时不要对零填充历史发放正式频域 reward：

\[
m_e=\mathbb I[N_e^{\mathrm{valid}}\ge N_w],\qquad
L_{f,e}\leftarrow m_eL_{f,e}.
\]

可后续测试半窗到全窗的线性 warm-up。注意终止前 reward 先计算、随后 RewardManager reset，因此最后一个 transition 仍能得到旧 episode 的合法频域 reward，reset 后历史被清空。

### 11.6 FFT 更新间隔

每 \(K\) 步更新一次，其他 step 保持最近值：

\[
L_{f,t}=L_{f,K\lfloor t/K\rfloor}.
\]

这比只在 FFT step 发放脉冲 reward 更平滑，也不需要额外乘 \(K\)。消融 \(K\in\{1,2,4,8\}\)，记录 steps/s。

## 12. Manager-Based 配置映射

Isaac Lab 推荐用 `RewardTermCfg` 配置自定义 callable term，任务函数放到专属 `mdp` 包：[Manager-Based RL tutorial](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_rl_env.html)。

    @configclass
    class FrequencyRewardsCfg:
        track_lin_vel_xy = RewTerm(
            func=mdp.track_lin_vel_xy_exp,
            weight=1.0,
            params={"command_name": "base_velocity", "std": 0.5},
        )

        dc_posture = RewTerm(
            func=mdp.JointDcPostureReward,
            weight=-1.0,
            params={"analyzer_cfg": JOINT_FREQ_CFG,
                    "command_name": "base_velocity"},
        )

        locomotion_band = RewTerm(
            func=mdp.JointLocomotionBandReward,
            weight=-0.2,
            params={"analyzer_cfg": JOINT_FREQ_CFG,
                    "command_name": "base_velocity"},
        )

        joint_high_frequency = RewTerm(
            func=mdp.JointHighFrequencyReward,
            weight=-0.05,
            params={"analyzer_cfg": JOINT_FREQ_CFG},
        )

        joint_limit_barrier = RewTerm(
            func=mdp.normalized_joint_limit_barrier,
            weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "margin": 0.85},
        )

具体 API 以锁定的 Isaac Lab 版本为准。RSL-RL 使用 `RslRlVecEnvWrapper` 包装 Manager-Based 环境，PPO 配置保持标准接口：[Isaac Lab RSL-RL API](https://isaac-sim.github.io/IsaacLab/main/source/api/lab_rl/isaaclab_rl.html)。

## 13. 阶段 A 测试

训练前完成：

1. 常值输入：AC 能量接近零，mean 正确；
2. bin-centered 正弦：谱峰落在正确频率；
3. off-bin 正弦：Hann 的泄漏低于矩形窗；
4. 幅值加倍：功率约四倍；
5. Parseval：时域与频域 AC 能量一致；
6. batch isolation：环境之间不串数据；
7. reset leakage：单个环境 reset 后不继承旧 episode；
8. warm-up mask：窗口未满时 reward 为零；
9. ring-order：跨写指针边界后频谱正确；
10. GPU benchmark：不同 `num_envs/window_size/update_interval` 的 steps/s 与显存。

## 14. 阶段 A 实验矩阵与验收

| 编号 | 方法 | 目的 |
|---|---|---|
| A0 | 原始 Isaac Lab locomotion reward | 标准基线 |
| A1 | 速度追踪 + 最小安全项 | 检验欠约束程度 |
| A2 | A1 + DC | 无命令站立 |
| A3 | A2 + locomotion band | 周期运动组织 |
| A4 | A3 + high-frequency | 完整三频带 |
| A5 | A1 + action-rate penalty | 传统平滑对照 |
| A6 | A4 + critic spectral summary | 历史可观测性消融 |

每种设置使用多个随机种子。记录：

- linear/yaw tracking error；
- fall/success rate；
- normalized joint-limit margin；
- joint/action/velocity/acceleration PSD；
- 三档 band energy；
- base acceleration RMS；
- mechanical CoT；
- sample efficiency、wall-clock throughput；
- dominant frequency--commanded velocity curve。

阶段 A 验收：

1. 训练稳定，无 NaN、reset 污染和显存泄漏；
2. 相近速度追踪下，高频关节能量低于 A1；
3. 无命令时 DC 偏差和 AC 能量下降；
4. 运动能量集中于可解释频带；
5. 相比 action-rate penalty，能够直接控制指定频带。

---

# 阶段 B：复数频谱对称性

## 15. 研究目标

不要求左右关节逐时刻相同，而约束：

\[
z_i(t)\approx s_{ij}z_j(t-\Delta_{ij}),
\]

频域为：

\[
Z_i(f)\approx s_{ij}Z_j(f)e^{-\mathrm{i}2\pi f\Delta_{ij}},
\]

其中 \(s_{ij}\in\{-1,+1\}\) 处理坐标方向，\(\Delta_{ij}\) 表示允许的时间延迟。

## 16. 关节配对配置

    @configclass
    class SpectralSymmetryCfg:
        joint_pairs: list[tuple[str, str]]
        mirror_signs: list[float]
        phase_mode: str = "fixed"  # fixed | best_shift | coherence
        target_phase_rad: list[float] | None = None
        active_band_hz: tuple[float, float] = (0.5, 4.0)
        amplitude_eps: float = 1.0e-6
        phase_energy_floor: float = 1.0e-4

不能按 joint name 顺序隐式猜测映射。启动时解析固定 IDs 并断言：

- pair 均存在；
- pair、sign、phase 数量一致；
- joint ordering 与 action/observation 一致；
- 镜像变换两次后返回原映射。

## 17. 幅值谱对称

\[
L_{\mathrm{amp}}^{ij}
=
\frac{\sum_{k\in\mathcal B_{\mathrm{loc}}}w_{ij}(k)
[\log(|Z_i(k)|+\varepsilon)-\log(|Z_j(k)|+\varepsilon)]^2}
{\sum_{k\in\mathcal B_{\mathrm{loc}}}w_{ij}(k)+\varepsilon},
\]

推荐

\[
w_{ij}(k)=\sqrt{P_i(k)P_j(k)},
\]

让低能量、相位不可靠的 bin 自动降权。幅值相同不代表时域对称，必须再加相位项。

## 18. 固定目标复相位

跨谱：

\[
C_{ij}(k)=Z_i(k)Z_j^*(k).
\]

不直接计算有 \(-\pi/\pi\) 跳变的 angle difference，而用单位复数：

\[
\hat C_{ij}(k)=\frac{C_{ij}(k)}{|C_{ij}(k)|+\varepsilon},
\]

目标为

\[
p_{ij}^*(k)=s_{ij}e^{-\mathrm{i}\phi_{ij}^*(k)},
\]

相位 loss：

\[
L_{\mathrm{phase}}^{ij}
=
\frac{\sum_{k\in\mathcal B_{\mathrm{loc}}}w_{ij}(k)
|\hat C_{ij}(k)-p_{ij}^*(k)|^2}
{\sum_{k\in\mathcal B_{\mathrm{loc}}}w_{ij}(k)+\varepsilon}.
\]

固定相位适合指定 trot、pace 或 bound，但引入 gait prior。

## 19. 不预设 gait 的 best-shift symmetry

在候选延迟集合 \(\mathcal D\) 上求：

\[
L_{\mathrm{shift}}^{ij}
=
\min_{\Delta\in\mathcal D}
\sum_{k\in\mathcal B_{\mathrm{loc}}}w_{ij}(k)
\left|
\frac{Z_i(k)}{|Z_i(k)|+\varepsilon}
-s_{ij}\frac{Z_j(k)}{|Z_j(k)|+\varepsilon}
e^{-\mathrm{i}2\pi f_k\Delta}
\right|^2.
\]

这是 reward，不要求 `min` 可微。预计算：

    phase_table[num_delays, num_active_freqs] complex64

然后广播计算每个环境、pair 和 delay 的误差。所有频率必须共享同一 \(\Delta\)，否则失去真实时间延迟含义。

建议先做两步：

1. `B-fixed`：固定相位，验证实现可生成预期 gait；
2. `B-free`：best-shift，不固定 gait，观察是否涌现稳定相位结构。

## 20. coherence 的注意事项

单个 FFT 窗口下

\[
\frac{|Z_iZ_j^*|^2}{|Z_i|^2|Z_j|^2}
\]

在非零 bin 上理论上恒为 1，不能作为有意义的 magnitude-squared coherence。真正 coherence 需要 Welch/STFT 多段平均：

\[
\gamma_{ij}^2(f)=
\frac{|\sum_m Z_{i,m}(f)Z_{j,m}^*(f)|^2}
{(\sum_m|Z_{i,m}(f)|^2)(\sum_m|Z_{j,m}(f)|^2)+\varepsilon}.
\]

阶段 B 最小实现采用 amplitude similarity + complex phase/best-shift；Welch coherence 作为增强实验。

## 21. 阶段 B 总损失与代码

\[
L_f^B=L_f^A+
\lambda_{\mathrm{amp}}\sum_{(i,j)\in\mathcal P}L_{\mathrm{amp}}^{ij}
+\lambda_{\mathrm{phase}}\sum_{(i,j)\in\mathcal P}
L_{\mathrm{phase/shift}}^{ij}.
\]

新增接口：

    frequency_analyzer.py
      ├─ get_complex_spectrum()
      ├─ get_pair_amplitude_error(pair_ids)
      ├─ get_pair_phase_error(pair_ids, target_phase)
      ├─ get_best_shift_error(pair_ids, delay_grid)
      └─ get_welch_coherence(pair_ids)  # 可选

    frequency_rewards.py
      ├─ JointSpectralAmplitudeSymmetryReward
      ├─ JointSpectralPhaseSymmetryReward
      └─ JointBestShiftSymmetryReward

    frequency_observations.py
      └─ spectral_summary_for_critic  # 可选

复数保持 complex64，最终 reward 聚合后转 float32；不把完整裸复谱放入 actor observation。

## 22. 阶段 B 测试

1. 同频同相：amplitude 和零相位 loss 近零；
2. 同频反相：目标相位 \(\pi\) 时近零；
3. 已知延迟：best-shift 恢复对应 delay；
4. 幅值相同但频率不同：不得误判对称；
5. 相位跨 \(-\pi/\pi\)：复数 loss 连续；
6. 低能量 bin：不产生 NaN/巨大 penalty；
7. `mirror_sign=-1`：坐标翻转正确；
8. 不同并行环境独立求 shift；
9. reset 后不继承旧 pair spectrum；
10. Welch 版本中独立噪声 coherence 低、固定相位信号高。

## 23. 阶段 B 实验与验收

| 编号 | 方法 |
|---|---|
| B0 | 阶段 A 最优模型 |
| B1 | 时域逐时刻 mirror reward |
| B2 | RSL-RL mirror loss |
| B3 | 仅幅值谱对称 |
| B4 | 幅值 + 固定复相位 |
| B5 | 幅值 + best-shift（主方法） |
| B6 | B5 + Welch coherence |

评价：

- 对应关节 PSD 距离；
- 目标频带相位误差；
- best-shift 跨 episode 方差；
- 时域状态分布不对称程度；
- foot-contact sequence/gait classification；
- tracking、fall rate、CoT；
- 扰动后恢复稳定相位结构的时间。

验收：

1. 减少跨关节谱差异；
2. 相位项比仅幅值项改善接触节律；
3. 相比逐时刻 mirror reward，允许合理时间偏移；
4. best-shift 不需要完整参考轨迹；
5. 必须实测时域状态分布对称性，不能只凭谱相似宣称完全替代等变网络或数据增强。

---

# 阶段 C：扩展到足端、机身和接触力

> 暂不规划代码，只定义目的、效果和 loss。

## 24. 目的与预期效果

研究关节运动如何传递到 task/contact space：

- 降低 base roll/pitch 与垂向加速度振动；
- 减少落足冲击和接触力高频尖峰；
- 避免“关节周期运动但接触节律错误”的 reward hacking；
- 改善相机、IMU、雷达和机械臂基座稳定性；
- 建立 joint--base--contact 跨谱关系。

## 25. 候选损失

机身振动：

\[
L_{\mathrm{base}}
=\sum_{f\in\mathcal B_{\mathrm{undesired}}}
[w_z|A_z(f)|^2+w_r|\Omega_{\mathrm{roll}}(f)|^2+
w_p|\Omega_{\mathrm{pitch}}(f)|^2].
\]

接触力高频：

\[
L_{\mathrm{contact\text{-}HF}}
=\sum_\ell\sum_{f\ge f_c}W_F(f)|F_{\ell,z}(f)|^2.
\]

关节—接触跨谱：

\[
L_{q\text{-}c}
=\sum_{(j,\ell)}\sum_{f\in\mathcal B_{\mathrm{loc}}}
w_f|\hat C_{q_jc_\ell}(f)-C_{j\ell}^*(f)|^2.
\]

力矩—速度交叉谱：

\[
S_{\tau\omega,j}(f)=T_j(f)\Omega_j^*(f),
\]

\[
L_{\mathrm{power}}
=\sum_j\sum_f W_P(f)
\max(0,\operatorname{Re}S_{\tau\omega,j}(f)).
\]

频域足端阻抗：

\[
Z_\ell(f)=\frac{F_\ell(f)}{V_\ell(f)+\varepsilon},
\qquad
L_Z=\sum_\ell\sum_f w_f|Z_\ell(f)-Z_\ell^*(f)|^2.
\]

必要的 COM 周期运动不能被无差别抑制；需给 gait band 保留通带。

---

# 阶段 D：经典控制、经验闭环频响与鲁棒控制

> 暂不规划代码，只定义目的、效果和 loss。

## 26. 研究目的

阶段 A--C 主要塑造输出频谱。阶段 D 引入已知命令、外力、地形扰动或噪声，估计扰动到误差、状态与控制输入的经验频率响应，并借鉴 sensitivity shaping、\(H_2\)、\(H_\infty\) 和执行器带宽设计。

普通自主 locomotion 的输出 PSD 不能替代频率响应，因为没有区分输入、系统和输出。

## 27. 经验频率响应

已知输入 \(d(t)\)、输出 \(y(t)\)：

\[
\hat G_{dy}(f)=\frac{S_{dy}(f)}{S_{dd}(f)+\varepsilon},
\qquad S_{dy}(f)=D^*(f)Y(f).
\]

输入可为 base multisine/chirp force、joint disturbance torque、band-limited terrain、velocity command modulation、observation noise 或 control delay sweep。

## 28. mixed sensitivity

局部或经验闭环：

\[
S=(I+PK)^{-1},\qquad T=PK(I+PK)^{-1}.
\]

候选目标：

\[
L_{\mathrm{mixed}}=
\sum_k\big(
\|W_S\hat S\|_F^2+
\|W_T\hat T\|_F^2+
\|W_U\widehat{KS}\|_F^2
\big).
\]

期望：

- 低频 \(S\) 小：跟踪与慢扰动抑制；
- locomotion band 保留控制能力；
- 高频 \(T\) 小：噪声与未建模动力学不过度放大；
- 高频 \(KS\) 小：限制控制动作和电机带宽需求。

## 29. 经验 \(H_2\) 与 \(H_\infty\)

平均频带响应：

\[
L_{H_2}=\sum_k w_k\|\hat G(\omega_k)\|_F^2.
\]

最坏频率/方向：

\[
L_{H_\infty}
=\max_k\bar\sigma[W(\omega_k)\hat G(\omega_k)].
\]

平滑近似：

\[
L_{\mathrm{soft}\text{-}H_\infty}
=\frac1\alpha\log\sum_k
\exp\{\alpha\bar\sigma^2[W_k\hat G_k]\}.
\]

## 30. 真机执行器频率权重

通过扫频/辨识得到：

\[
G_j^{\mathrm{act}}(f)
=|G_j^{\mathrm{act}}(f)|e^{\mathrm{i}\phi_j(f)}.
\]

设置：

\[
L_{\mathrm{actuator}}
=\sum_j\sum_f W_j^{\mathrm{act}}(f)|A_j(f)|^2,
\]

在幅频衰减、相位滞后快速增长和结构共振附近提高权重。目标是 hardware-aware bandwidth allocation，而不是统一压制所有高频。

## 31. 鲁棒性的谨慎表述

locomotion 是非线性、接触切换、近似周期时变系统。不能只凭 PSD 声称获得经典 LTI gain/phase margin 或 \(H_\infty\) 保证。

阶段 D 优先报告：

- empirical sensitivity peak；
- disturbance-to-output gain；
- command tracking bandwidth；
- delay robustness；
- gait-phase-conditioned FRF；
- 多 operating point 局部结果。

只有建立周期轨道线性化、Floquet 或 harmonic transfer function 模型后，才讨论更严格稳定裕度。

---

## 32. 里程碑

### M0：基线与离线频谱

- 跑通 Manager-Based velocity tracking baseline；
- 导出 joint/action/base/contact rollout；
- 建立 PSD、dominant frequency 和 band-energy 报告；
- 用 baseline 数据确定频带。

### M1：阶段 A 最小实现

- GPU ring buffer、normalization、reset、warm-up、cache；
- DC、locomotion-band、high-frequency reward；
- 单元测试与吞吐 benchmark。

### M2：阶段 A 消融

- 与原始 reward、最小 reward、action-rate 比较；
- window、FFT interval、band、weight 消融；
- 决定是否需要 critic spectral summary。

### M3：阶段 B

- joint pair/sign；
- amplitude symmetry；
- fixed phase；
- best-shift；
- 对比 time-domain mirror 与 RSL-RL mirror loss。

### M4：阶段 C

- base、foot、contact force；
- joint--contact cross-spectrum；
- power/impedance。

### M5：阶段 D

- multisine/chirp；
- empirical FRF；
- \(H_2/H_\infty\)-style objective；
- 执行器扫频与硬件权重；
- delay、actuator dynamics、sim-to-real。

## 33. 最先执行的代码任务

1. 锁定 Isaac Lab、Isaac Sim、RSL-RL、PyTorch 版本；
2. 选择一个 Manager-Based 平地速度跟踪任务；
3. 记录 `sim.dt`、`decimation`、control frequency、episode duration、`num_steps_per_env`；
4. 先写 rollout exporter 和离线 PSD 报告；
5. 实现/测试 nominal-centered normalization 与 limit barrier；
6. 实现 ring buffer、per-env reset、valid count；
7. 实现 Hann、RFFT、单边 PSD、Parseval 测试；
8. 实现三个阶段 A reward term 与分项日志；
9. 小规模 16/64 env 正确性测试；
10. 4096 env 吞吐与显存 benchmark；
11. 完成 A0--A5；
12. 冻结阶段 A 最优配置；
13. 实现 pair/sign 映射、幅值和复相位 loss；
14. 实现 best-shift 和阶段 B 单元测试；
15. 完成 B0--B5 后决定是否上 Welch coherence。

## 34. 关键风险

| 风险 | 表现 | 预案 |
|---|---|---|
| 窗口奖励延迟 | 学习慢、value loss 大 | curriculum、critic spectral summary、短窗起步 |
| reset 污染 | 新 episode 继承旧 gait | per-env reset + valid mask |
| 频谱泄漏 | 能量散到相邻 bin | Hann；后续 Welch/STFT |
| 低频 reward hacking | 不前进只摆腿 | 集中率而非绝对正奖励；保留 velocity tracking |
| 高频抑制导致僵硬 | 落足与抗扰变慢 | locomotion/disturbance band 留通带 |
| 幅值对称但 gait 错 | 接触不同步 | 加复相位/best-shift并检查 contact |
| coherence 误用 | 单窗结果恒接近 1 | 必须多段平均 |
| FFT 开销 | steps/s 下降 | update interval、共享缓存、少存 complex tensor |
| 过度鲁棒性声称 | 只有 PSD 无 FRF | 阶段 D 注入已知扰动，区分经验与理论 |

## 35. 预期贡献分层

- **阶段 A**：reference-free command-conditioned 三频带 locomotion regulation；
- **阶段 B**：complex cross-spectral symmetry，允许共享时间延迟而非逐帧镜像；
- **阶段 C**：joint--base--contact 多物理量频域协调；
- **阶段 D**：经验闭环 loop shaping、\(H_2/H_\infty\)-style 指标与 hardware-aware frequency weighting。

总体路线从“可运行、可验证的频谱 reward”逐步推进到“具有经典控制含义的闭环频率整形”，避免第一阶段同时承担复杂接触信号、系统辨识和鲁棒理论的全部风险。

## 36. 工程排查总结：连续楼梯的 collection time 与地形 contact offset

### 36.1 现象与诊断（2026-09-08）

在 4096 个环境的感知 locomotion Base 任务中，开启 pure stair 跨行高度累计以保持楼梯连续后，collection time 明显变长。相同 iteration 50–199 区间，连续楼梯的平均 collection time 为 **2.860 s**，关闭高度累计后为 **1.478 s**，金字塔楼梯为 **1.531 s**；三组 learning time 均约 1.12 s，碰撞栈均为 `2**29`。因此不能把差异归因于 PPO 更新、频域 reward 或碰撞栈预留容量本身。

进一步使用相同 checkpoint 分项计时：高度累计使每控制步的物理仿真耗时由约 **13.67 ms 增至 63.53 ms**，深度相机耗时基本不变。整体 Z 平移没有改善；去除内部面并细分外表面的 surface 网格没有收益；拆成 261 个碰撞网格反而明显恶化。GPU 时间线最终定位到 **`convexTrimeshNarrowphase`（凸形碰撞体与三角网格的窄阶段碰撞处理）**，而不是相机射线查询。机器人后端 contact offsets 在两组中一致，为 1.4–3 mm，rest offsets 均为 0；地形偏移量则未显式设置，USD 返回自动值标记 `-inf`，它不是实际接触距离。

### 36.2 有效干预及正式修复

保持连续楼梯、boxes、单碰撞网格、机器人参数和 `2**29` 碰撞栈不变，只修改地形偏移量，得到以下四组对照：

| 地形设置 | 总耗时（ms／控制步） | 窄阶段 kernel 累计耗时（ms） | PhysX 报告的碰撞栈需求（MiB） |
|---|---:|---:|---:|
| contact/rest 均自动 | 133.50 | 861.68 | 288.69 |
| 仅 contact offset = 0.02 m | 64.80 | 32.40 | 2.05 |
| 仅 rest offset = 0 m | 131.01 | 848.37 | 288.69 |
| contact = 0.02 m，rest = 0 m | 62.42 | 32.40 | 2.05 |

这里是预热 48 个控制步后采集的 **24 个控制步／96 个物理步**；kernel 时间为整个采集区间的累计值，不能与每步墙钟时间直接相加。PhysX 资源需求为末步接口读数，不是整个 rollout 的平均实际接触数量。原始数据见 [offset GPU trace](test/output/offset_gpu_trace_20260908_161201/)。

**实测确认起作用的是显式 contact offset，而非 rest offset。** 正式感知 Base／FreqReward 任务已通过 `ContactOffsetTerrainImporterCfg` 设置 `contact_offset=0.02`、`rest_offset=0.0`，在仿真初始化前写入地形碰撞 prim，并保存到 `env.yaml`；保留连续楼梯、boxes 和单网格，不改机器人。碰撞栈暂保留 `2**29`：本次小需求读数不足以保证所有难度、摔倒状态和长期训练都适合降低容量。

### 36.3 为什么 contact offset 会影响 PhysX：定义与机制推测

PhysX 在两个形状的距离小于双方 contact offsets 之和时开始生成接触信息：

\[
d_{\mathrm{contact}}=c_{\mathrm{robot}}+c_{\mathrm{terrain}},
\qquad
d_{\mathrm{rest}}=r_{\mathrm{robot}}+r_{\mathrm{terrain}}.
\]

其中 \(c\) 是 contact offset，\(r\) 是 rest offset。前者控制提前生成接触的距离范围，后者控制静止接触的目标间距；**contact offset = 2 cm 不等于把地形抬高 2 cm，也不意味着机器人必然悬空 2 cm**。例如机器人形状取 2 mm、地形取 20 mm，则接触生成距离约为 22 mm。较大的接触范围可能生成更多接触信息，增加计算开销；过小则可能影响离散步长下的稳定性。参见 [PhysX Advanced Collision Detection](https://nvidia-omniverse.github.io/PhysX/physx/5.1.2/docs/AdvancedCollisionDetection.html)。

**与结果一致、但尚未完全证实的解释**是：累计高度改变了整张地形的几何范围，可能影响导入层对自动 contact offset 的解析；偏大的有效接触范围使更多三角形进入窄阶段候选／接触处理，增加临时栈和接触资源需求，并进一步加重求解负担。显式设置 0.02 m 则限制了这一范围。这也解释了“大碰撞栈需求与慢同时出现”更可能是同一碰撞负担的两个表现，而不是“预留显存越多就必然越慢”。

但目前**没有直接读取静态地形的自动后端偏移量，也没有验证它与包围盒尺寸的计算公式**，因此不能断言自动值具体多大、必然按高度跨度缩放，或候选三角形数量增加了多少。已证实的是干预效果及主要耗时 kernel，自动值解析的内部因果链仍属于假设。

### 36.4 对后续频域实验的约束

后续 reward 消融应固定地形接触参数、物理步长及碰撞配置，并与旧日志区分：修改 contact offset 会改变接触动力学和关节／接触力频谱，不能把这类变化误记为 frequency reward 的效果。当前短测试和小规模启动验证尚不能代替长期训练、较高地形难度、接触稳定性及步态质量验证；应在无 profiler 的正常训练中复核吞吐，并检查滑移、抖动和穿透后，再决定是否降低碰撞栈容量。
