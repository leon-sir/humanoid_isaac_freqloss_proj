# 任务：在 Isaac Lab Manager-Based RL 环境中实现频率整形关节奖励

你是一名熟悉 Isaac Lab、PyTorch、数字信号处理、足式机器人控制与 PPO/RSL-RL 的高级机器人学习工程师。请在当前 Isaac Lab 项目中实现一套可配置、GPU 批处理、可测试的关节频域奖励。不要只给设计建议；需要完成代码、配置接入、测试、最小运行验证和简短文档。

在开始编码前，必须先阅读当前仓库的 `AGENTS.md`（若存在）、环境配置、action term、reward 配置、机器人 articulation 配置、训练脚本和版本锁定文件，并阅读本项目的 `FREQUENCY_SHAPED_LOCOMOTION_RESEARCH_PLAN.md`。先报告实际使用的 Isaac Lab、Isaac Sim、RSL-RL 和 PyTorch 版本以及目标 Manager-Based task 路径。所有 API 必须以本地安装/锁定版本为准，不得凭印象套用其他版本接口。

## 1. 目标与作用边界

实现以下三个可独立配置、独立记录日志的正值 penalty term，由 `RewardTermCfg` 使用负权重加入总 reward：

1. `joint_dc_posture`：约束窗口平均关节姿态接近 default/nominal joint position；
2. `joint_fampc_spectral_shape`：借鉴 Frequency-Aware MPC（FAMPC）的 $\alpha,\beta$ 频率权重，使高频关节运动比低频运动承担更大的代价；
3. `joint_spectral_phase_symmetry`：约束显式配置的左右关节对在有效运动频带内满足指定镜像符号及相位/时间延迟关系。

这些量针对仿真产生的实际关节轨迹，属于环境 reward。不要修改 PPO surrogate loss，也不要声称 FFT 梯度穿过 Isaac/PhysX 动力学直接反传。正确路径是

$$
L_f\longrightarrow r_t\longrightarrow \text{return/advantage}
\longrightarrow L_{\mathrm{PPO}}\longrightarrow \theta.
$$

它们在控制意义上可以称为 penalty/loss，但代码上必须作为 Manager-Based Env 的 reward terms 接入标准 PPO。

## 2. 开发前必须核实的官方规范

至少核实以下事实并在实现说明中给出所用版本对应的源码位置：

- `ManagerBasedRLEnv.step()` 在完成 `decimation` 次 physics step 后才调用 `RewardManager.compute(dt=self.step_dt)`；因此普通 reward term 默认只能以环境/策略步频率采样，而不是 physics 频率采样。
- `RewardManager` 会把 term 返回值乘以配置的 `weight` 和环境 `step_dt`；term 内不得再次乘 `dt`。
- 有状态 reward term 应继承 `ManagerTermBase` 并实现 `__call__()` 与 `reset(env_ids)`；RewardManager reset 时会调用 class term 的 reset。
- reward term 必须返回设备上形状为 `[num_envs]` 的有限值张量。

官方参考：

- Manager-Based RL 教程：<https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_rl_env.html>

不要修改 Isaac Lab 核心包源码。实现应位于该 task 的 `mdp` 子包及 task 配置中。

## 3. 必须遵守的原理修正

以下修正属于非协商要求：

### 3.1 DC 不是原始关节角 FFT 的 0 号 bin 直接趋零

若关节 default position 不为零，令原始 $q$ 的 DC 趋零会把关节拉向坐标零点而不是 default posture。必须先定义归一化偏差(如果现在action term使用relative joint pos则无视)

$$
z_j(t)=\frac{q_j(t)-q_j^{\mathrm{default}}}{s_j},
$$

再用未加窗数据的时间均值

$$
\mu_j(t)=\frac{1}{N}\sum_{n=0}^{N-1}z_j(t-n)
$$

构造 DC penalty。`default_joint_pos`、soft joint limits 和 joint ordering 必须从当前 articulation/action 配置中解析，不能手写隐式顺序。尺度 $s_j$ 应显式配置；若选择由 soft joint limits 生成 nominal-centered scale，必须校验其为正并记录计算规则。

移动时合理的平均关节姿态可能偏离 default，因此提供：

- `dc_gate_mode="stand"`：默认，仅在速度命令接近零时启用；
- `dc_gate_mode="always"`：仅作为明确的实验选项。

若任务有 `base_velocity` command，可定义站立门控 

$$
g_{\mathrm{stand}}= boolean(v_{cmd}<0.2m/s)
.
$$

### 3.2 FAMPC 并不是“奖励低频能量、惩罚高频能量”两个独立权重

Grandia 等的 FAMPC 使用

$$
\widetilde R(\omega)=
\left|\frac{1+\beta j\omega}{1+\alpha j\omega}\right|^2R,
\qquad \beta>\alpha>0,
$$

它令低频权重趋于 1，高频权重趋于 $(\beta/\alpha)^2$，本质是“所有输入都付代价，但高频更贵”。原论文通过状态增广实现连续时间滤波；本任务可以在有限窗口 FFT bin 上离散采样同一权重，但必须明确这是借鉴/离散近似，不得声称完全复现 FAMPC 的 MPC 状态增广或鲁棒性结论。

原论文：<https://arxiv.org/abs/1809.04539>，DOI: 10.1109/LRA.2019.2895882。

仅最小化加权总能量可能得到“完全不动”。低频运动由速度追踪/任务 reward 产生；频率项负责把已有 AC 能量推向可行频带。默认实现应支持归一化频谱整形，而不是无条件给低频绝对正奖励。

### 3.3 补零不提高真实频率分辨率

$$
\Delta f=\frac{f_s}{N}=\frac{1}{T_w}.
$$

FFT zero-padding 只插值频谱显示，不能增加由观测时长决定的真实分辨率。首版 reward 默认 `n_fft=N`，不要补零。尤其禁止在 episode reset 后用零填充未完成的历史并立即计算正式 reward，否则会制造阶跃、泄漏和错误 DC。窗口未满时使用 `valid_count` mask，penalty 返回零；可选 warm-up 必须单独配置并测试。

### 3.4 “左右关节固定相差 $\pi$”不是通用对称定义

相位目标必须显式配置：

- 左右 joint pair；
- 坐标镜像符号 $s_{ij}\in\{-1,+1\}$；
- 相位模式及目标；
- 有效频带；
- 低幅值相位屏蔽阈值。

若真实关系是时间延迟

$$
z_i(t)\approx s_{ij}z_j(t-\Delta_{ij}),
$$

则频率相关目标应为

$$
Z_i(f)\approx s_{ij}e^{-j2\pi f\Delta_{ij}}Z_j(f),
$$

而不是对所有谐波都强制同一个相位差。不要根据 joint name 自动猜测 pair、sign、gait 或相位；配置缺失时应给出清晰错误或保持该 term 禁用。

## 4. 采样与共享分析器架构

实现一个共享的有状态 `JointFrequencyAnalyzer`，由三个 `ManagerTermBase` callable reward term 访问。推荐放置：

```text
<task>/mdp/frequency_analyzer.py
<task>/mdp/frequency_rewards.py
<task>/mdp/__init__.py
<task>/<task>_env_cfg.py
tests/test_frequency_analyzer.py
tests/test_frequency_rewards.py
tests/test_frequency_reset.py
tests/test_frequency_phase.py
```

不得为三个 reward term 各复制一份历史和 FFT。实现一个 task-local 的 env analyzer registry，例如按 `analyzer_key` 在 env 上持有共享实例；不要修改 Isaac Lab core。所有 wrapper term 每次调用都执行 `analyzer.update_once(env)`，由 `common_step_counter` 去重，不能依赖“第一个 reward term 一定启用”，因为 `weight==0` 的 term 会被 RewardManager 跳过。

`reset(env_ids)` 必须幂等；三个 wrapper 依次调用同一 analyzer reset 不得出错。

### 4.1 默认采样频率

首版按环境控制步采样：

$$
\Delta t=\texttt{env.step\_dt}
=\texttt{sim.dt}\times\texttt{decimation},
\qquad f_s=1/\Delta t.
$$

如果当前配置为 50 Hz 策略、200 Hz physics，则首版 reward 的 $f_s=50$ Hz、Nyquist 为 25 Hz。不能因为 physics 为 200 Hz 就在普通 reward term 中声称获得了 100 Hz 频谱。

如果后续确实需要 200 Hz 子步采样，请作为单独阶段实现：必须在 physics decimation loop 内记录每个 substep，说明为何需要自定义 env hook/recorder，并单独测试；不要在本首版中侵入式改写基类。

### 4.2 窗口

配置使用秒而不是只使用样本数：

```python
window_duration_s: float = 2.0
window_size = round(window_duration_s / env.step_dt)
actual_window_duration_s = window_size * env.step_dt
```

默认 2.0 s，至少 1.0 s，并在启动时打印/记录：

- `sample_rate_hz`；
- `window_size`；
- `actual_window_duration_s`；
- `frequency_resolution_hz=sample_rate_hz/window_size`；
- `nyquist_hz=sample_rate_hz/2`。

同时警告：窗口还应覆盖至少 2 个目标 gait cycle。若 locomotion band 下限为 0.5 Hz，2 s 只覆盖 1 个周期，因此必须保留 1、2、4 s 的消融配置，不能宣称 2 s 对所有步态都足够。

### 4.3 GPU ring buffer

至少维护：

```text
history       [num_envs, window_size, num_joints] float32
valid_count   [num_envs] int
write_index   scalar int
last_sample_step scalar int
last_fft_step scalar int
cached_mean   [num_envs, num_joints] float32
cached_spectrum [num_envs, num_freqs, num_joints] complex64（若相位项需要）
cached_power  [num_envs, num_freqs, num_joints] float32
cached_ready  [num_envs] bool
```

所有计算留在 `env.device`，禁止在训练 step 中调用 NumPy、CPU 循环或逐环境 Python FFT。并行环境可共享全局写指针，但 `valid_count` 和 reset 必须按 env 维护。单个 env reset 时不要重置全局写指针；只清该 env 的历史、有效长度和缓存状态。

FFT 可每 `fft_update_interval` 个控制步批量更新一次，其余步返回最近缓存。无论几个 term 在同一步调用，只允许采样一次、FFT 一次。窗口未满的 env 返回零 penalty，不得继承旧 episode 数据。

## 5. 频谱计算与归一化

对满窗后的每个关节：

1. 从 ring buffer 按时间顺序恢复窗口；
2. 用未加窗数据计算 $\mu_j$；
3. 计算 AC：$x_j[n]=z_j[n]-\mu_j$；
4. 对 AC 使用 Hann window；测试中也支持 rectangular window；
5. 使用 `torch.fft.rfft(..., dim=time_dim, norm="ortho")`；
6. 用 `torch.fft.rfftfreq(N, d=env.step_dt)` 生成 Hz 频率轴；
7. 构造经过单边谱与窗能量校正的非负功率贡献 $P_j[k]$，使

$$
\sum_kP_j[k]
\approx
\frac{\sum_n w[n]^2x_j[n]^2}{\sum_n w[n]^2}.
$$

对实信号的单边谱，除 DC 以及偶数 $N$ 时 Nyquist bin 外，内部 bin 能量乘 2。DC term 使用未加窗均值单独计算；AC 频谱聚合时排除 $k=0$。所有分母加 `eps`，最终输出必须经过 `torch.isfinite` 测试，但不要用无声 `nan_to_num` 掩盖实现错误。

## 6. 三个 reward 的数学定义

所有 term 返回正 penalty `[num_envs]`，由 config 使用负 weight。

### 6.1 DC posture penalty

默认：

$$
L_{\mathrm{DC},e}
=g_{\mathrm{stand},e}
\frac{1}{N_j}\sum_j\mu_{e,j}^2.
$$

提供可选站立 AC 抑制但默认关闭，避免把第四项混入主要消融：

$$
L_{\mathrm{stand\text{-}AC},e}
=g_{\mathrm{stand},e}\frac{1}{N_j}\sum_{j,k>0}P_{e,j}[k].
$$

### 6.2 FAMPC-style spectral shaping penalty

频率轴使用 Hz，因此令 $\omega_k=2\pi f_k$：

$$
W_{\alpha\beta}(f_k)
=\frac{1+(2\pi f_k\beta)^2}
{1+(2\pi f_k\alpha)^2},
\qquad\beta>\alpha>0.
$$

支持两种等价配置方式，但内部只能保留一个明确真值来源：

1. 直接配置 `alpha_s`, `beta_s`；
2. 推荐配置 corner frequencies：

$$
f_z=\frac{1}{2\pi\beta},
\qquad
f_p=\frac{1}{2\pi\alpha},
\qquad 0<f_z<f_p<f_{\mathrm{Nyq}}.
$$

启动时必须校验参数。建议 `f_p <= 0.8*f_nyquist`，否则发出清晰 warning，说明可观测频带内没有充分看到高频平台。对于 50 Hz 采样，原论文示例 $\alpha=0.01$ s、$\beta=0.1$ s 对应约 15.9 Hz 与 1.59 Hz 两个 corner；它在数学上可采样，但会从约 1.6 Hz 开始提高代价，未必适合本机器人的 0.5--4 Hz locomotion band，所以不要未经校准照搬。

至少实现以下模式，默认使用 `normalized_excess`：

```text
absolute:
    L = mean_j sum_k W_ab(f_k) P_j(k)

excess:
    L = mean_j sum_k (W_ab(f_k) - 1) P_j(k)

normalized_excess:
    Wn = (W_ab - 1) / max(W_ab(f_reachable_max) - 1, eps)
    L = g_move * mean_j [sum_k Wn(k) P_j(k) / (sum_k P_j(k) + eps)]
```

`normalized_excess` 只惩罚 AC 能量位于较高频率的比例，避免不同关节幅值主导。它不能单独防止完全静止，必须保留 linear/yaw velocity tracking、alive 与安全项。若要限制低频运动幅值，另提供默认关闭的 `locomotion_energy_min/max` soft range；不要直接无上限奖励低频绝对能量。

支持 per-joint corner/weight，但首版可先全关节共享；接口不得阻碍后续每关节硬件带宽配置。

### 6.3 Left-right complex phase symmetry penalty

配置必须包含明确 joint names：

```python
joint_pairs: list[tuple[str, str]]
mirror_signs: list[float]       # 每项只能为 -1 或 +1
phase_mode: str                 # "fixed_phase" | "time_delay"
target_phase_rad: list[float] | None
target_delay_s: list[float] | None
active_band_hz: tuple[float, float]
phase_energy_floor: float
```

启动时解析固定 joint IDs 并校验 pair、sign、target 长度。对每对 $(i,j)$：

$$
C_{ij}(k)=Z_i(k)Z_j^*(k),
\qquad
\widehat C_{ij}(k)=\frac{C_{ij}(k)}{|C_{ij}(k)|+\varepsilon}.
$$

目标单位复数：

```text
fixed_phase:
    p*(k) = s_ij * exp(j * target_phase_rad_ij)

time_delay:
    p*(k) = s_ij * exp(-j * 2*pi*f_k*target_delay_s_ij)
```

请固定并在注释/测试中验证上述 cross-spectrum 方向和正负号约定。使用连续、无 $-\pi/\pi$ 跳变的损失，例如

$$
\ell_{ij}(k)=1-\operatorname{Re}
\{\widehat C_{ij}(k)p_{ij}^{*}(k)^*\}.
$$

相位仅在 `active_band_hz` 内且两侧具有足够能量时有效。推荐权重

$$
w_{ij}(k)=\sqrt{P_i(k)P_j(k)}
$$

并使用相对/绝对 energy floor mask。归一化聚合：

$$
L_{\mathrm{phase},e}
=g_{\mathrm{move},e}
\frac{\sum_{(i,j),k}m_{ij}(k)w_{ij}(k)\ell_{ij}(k)}
{\sum_{(i,j),k}m_{ij}(k)w_{ij}(k)+\varepsilon}.
$$

若无有效 bin，返回零，不得 NaN。相位 penalty 本身不保证左右幅值相同，也可能在两侧都无运动时失效；必须记录 paired amplitude ratio/PSD-distance 作为 metric，并保留任务 reward。可把 amplitude symmetry 作为默认关闭的后续扩展，但不要悄悄混入当前相位消融。

## 7. 可选的 mimic/reference-spectrum 对照（完成核心三项后再做）

若仓库已有重定向后的参考运动数据，实现一个独立、默认禁用的 `reference_spectrum_mimic` 对照：

1. 离线读取重定向关节运动；
2. 严格按训练时 joint ordering、default-centering、scales、采样率、窗口、window function 和 PSD normalization 处理；
3. 保存带元数据的 `.pt` 频率表，包括 joint names、sample rate、window duration、frequency bins、normalization 和信号单位；
4. 训练时检查元数据完全匹配，不允许静默插值错误 joint/order；
5. 默认比较 `log(P+eps)` 或归一化 PSD，以避免单纯幅值支配；
6. 只有参考窗口与当前窗口具有可靠时间/步态相位对齐时才允许逐 bin 复谱比较。否则只比较 magnitude/PSD，并用跨关节相位关系单独约束，因为绝对 FFT phase 依赖窗口时间原点；
7. 若速度命令会改变步频，参考表应 command-conditioned 或按 gait phase/cycle 归一化，不能用单一固定频谱表冒充所有速度下的目标。

若当前没有参考数据或重定向工具，只实现清晰接口、数据 schema 和测试夹具，不要伪造参考表。

## 8. 通过 Parseval 初始化 reward 权重

只能对二次型 L2 指标做严格 Parseval 换算。对 exponential、L1、clipped reward 不得声称等价。

在 rectangular window、`norm="ortho"`、完整单边能量校正下建立测试：

$$
\frac1N\sum_n z_j[n]^2
=\mu_j^2+\sum_{k>0}P_j[k]
$$

达到 float32 合理误差。由此：

- 原 `joint_pos_l2` 可用 DC + 全 AC 能量近似对应；
- 若直接采集 `joint_vel` 并做同样 PSD，其全谱能量可对应 `joint_vel_l2`；
- 一阶离散 action-rate penalty 的精确频率权重与

$$
4\sin^2(\pi f/f_s)
$$

成正比；若时域差分除以 `dt`，再乘 $f_s^2$；
- 二阶差分对应上述权重的平方。

运行 baseline rollout，记录旧时域 penalty 与新频域 raw penalty 的均值、中位数、P90 和每秒累计贡献。初始频域 config weight 应使其在 baseline 上与待替代项具有相近数量级，然后再训练微调。不要在 term 内乘 `dt`，因为 RewardManager 已经处理。

提供一个简短校准脚本或测试工具，输出推荐的初始外部 weight，但不要自动覆盖用户配置。

## 9. 配置接口建议

根据当前项目风格用 `@configclass` 实现等价配置，至少包括：

```python
class JointFrequencyAnalyzerCfg:
    analyzer_key: str
    asset_cfg: SceneEntityCfg
    window_duration_s: float = 2.0
    fft_update_interval: int = 1
    window_type: str = "hann"
    normalization: str = "nominal_centered"
    joint_scales: dict[str, float] | None = None
    eps: float = 1.0e-8

    # command gates
    command_name: str | None = "base_velocity"
    stand_command_std: float = ...
    yaw_scale: float = ...

    # FAMPC-style shaping
    shaping_mode: str = "normalized_excess"
    low_corner_hz: float = ...
    high_corner_hz: float = ...
    alpha_s: float | None = None
    beta_s: float | None = None

    # phase
    joint_pairs: list[tuple[str, str]] = ...
    mirror_signs: list[float] = ...
    phase_mode: str = "time_delay"
    target_phase_rad: list[float] | None = None
    target_delay_s: list[float] | None = None
    active_band_hz: tuple[float, float] = (0.5, 4.0)
    phase_energy_floor: float = ...
```

不要把尚未辨识的机器人特定数值伪装成最终参数。给出清晰的“启动默认值”和注释，说明必须由 baseline PSD、控制频率和真机执行器频响校准。所有频带必须检查不超过 Nyquist。

三个 `RewardTermCfg` 必须分别出现在环境 reward 配置中，以便 Isaac Lab 自动记录 episodic contribution。函数返回正 penalty，配置 weight 为负；不得 double-negative。

## 10. 测试要求

不得只依靠训练是否运行。至少完成以下自动测试：

1. default 常值：DC≈0、AC≈0；
2. default 加常偏置：DC 等于归一化均值平方；
3. bin-centered 正弦：峰值频率正确；
4. off-bin 正弦：Hann 泄漏低于 rectangular；
5. 幅值翻倍：绝对功率约四倍；
6. Parseval：rectangular 下时域 mean-square 与单边谱能量一致；
7. FAMPC 权重：DC 为 1、单调不减、高频极限接近 $(\beta/\alpha)^2$；
8. corner 参数：$f_z=1/(2\pi\beta)$、$f_p=1/(2\pi\alpha)$ 及 Nyquist 校验正确；
9. 高频正弦的 shaping penalty 大于同幅低频正弦；
10. 同相、反相、已知时间延迟的 pair phase loss 与目标一致；
11. 跨 $-\pi/\pi$ 时复数 phase loss 连续；
12. 低能量/零信号相位不产生 NaN；
13. 不同并行 env 的历史与 reward 相互隔离；
14. 单 env reset 后不继承旧 episode 频谱；
15. warm-up 未满窗返回零；
16. ring buffer wrap 后时间顺序与 FFT 正确；
17. 多个 reward term 同一步只采样一次、FFT 一次；
18. reward 输出 shape、dtype、device 正确且全部 finite；
19. 16/64 env 最小 smoke test 可运行；
20. 在目标并行规模上报告 analyzer 显存和 steps/s 开销。

测试纯数学部分应尽量不依赖启动完整 Isaac Sim；环境集成另做最小 smoke test。

## 11. 实验配置与验收

至少提供以下可切换实验配置或覆盖项：

```text
A0 baseline 原 reward
A1 baseline + DC
A2 baseline + FAMPC spectral shaping
A3 baseline + DC + shaping
A4 A3 + phase symmetry
A5 baseline + 原 action-rate/joint-velocity smoothness 对照
A6 可选 reference-spectrum mimic
```

保留速度追踪、alive、joint limit、非法接触、torque/velocity limit 和 termination 等任务/安全项。首版不要一次性删除全部传统 regulation reward。

训练和评估至少记录：

- 三个 frequency raw penalties 及加权 reward contribution；
- DC mean error；
- AC total energy；
- 频率整形 weighted energy 与 high-frequency ratio；
- dominant frequency；
- pair phase error、有效 phase bin 数、paired amplitude metric；
- velocity tracking、fall rate、joint limit、torque、action rate；
- steps/s 与显存。

验收标准：代码不修改 Isaac Lab core；reset 无污染；无 NaN；频率轴与控制采样率一致；三个 term 可独立开关；测试通过；小规模训练/rollout 可运行；文档明确区分 FAMPC 原方法、FFT reward 近似以及 PPO advantage 路径。

## 12. 工作方式与最终交付

按以下顺序执行并持续推进到可验证状态：

1. 检查仓库、版本和目标环境；
2. 写一页以内的实现映射，指出将修改的文件；
3. 实现 analyzer、三个 reward term 与配置；
4. 添加测试；
5. 运行数学测试和环境 smoke test；
6. 修复失败；
7. 运行小规模性能检查；
8. 给出最终修改摘要、参数表、测试命令与结果。

如果当前 task、joint pair、mirror sign 或目标 gait/time delay 无法从仓库可靠确定，不要猜测机器人特定值。先完成通用实现和验证夹具，把 phase term 默认禁用，并在最终结果中准确列出需要用户提供的最小配置。其他可以从代码和官方文档查明的问题不要反问用户。

