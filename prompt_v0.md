# 任务：在 Isaac Lab Manager-Based 环境中实现 v0 关节角频域奖励

你是一名熟悉 Isaac Lab、PyTorch、数字信号处理和足式机器人强化学习的工程师。请在当前项目的 Manager-Based RL 环境中实现一版尽可能简单、可配置、GPU 并行的关节角频域奖励。

v0 只分析实际关节角 `joint_pos`。不要分析或约束 action、$q_{des}$、关节速度、电机力矩，也不要实现专家频谱模仿。不要修改 PPO loss 或 Isaac Lab 核心代码。

## 1. 开始实现前

先检查当前项目中的：

- 目标 Manager-Based 环境和 reward 配置；
- 机器人 articulation、joint 顺序和 `default_joint_pos`；
- `sim.dt`、`decimation` 和实际环境步长 `env.step_dt`；
- 速度命令在 `CommandManager` 中的名称和排列；
- 项目现有 `mdp` 目录和 reward term 的写法。

以本地安装版本的 Isaac Lab API 为准。不要修改 Isaac Lab package 源码。

## 2. 关节角预处理

对每个关节使用

$$
z_j(t)=\frac{q_j(t)-q_j^{default}}{s_j},
$$

其中：

- $q_j$ 是实际关节角；
- $q_j^{default}$ 从 articulation data 中读取；
- $s_j>0$ 是人工配置的关节尺度，单位为 rad；
- joint ID 必须由 joint name 解析，不能依赖手写的隐式顺序。

`joint_scales` 必须允许用户逐关节修改。scale 的目的只是让不同关节的频谱能量处于相近数量级，不是让关节跟踪专家能量。

根据 `./spectrum_analysis/cmd2`、`cmd3`、`cmd4` 的稳态时域范围，可以先使用以下启动值：

```python
joint_scales = {
    "left_hip_pitch_joint": 0.55,
    "right_hip_pitch_joint": 0.55,
    "waist_yaw_joint": 0.75,
    "left_hip_roll_joint": 0.10,
    "right_hip_roll_joint": 0.10,
    "left_hip_yaw_joint": 0.05,
    "right_hip_yaw_joint": 0.05,
    "left_knee_joint": 0.85,
    "right_knee_joint": 0.85,
    "left_ankle_pitch_joint": 0.55,
    "right_ankle_pitch_joint": 0.55,
    "left_ankle_roll_joint": 0.25,
    "right_ankle_roll_joint": 0.25,
}
```

这些只是初值。左右同类关节优先使用相同 scale。对于固定、锁定或几乎不运动的关节，不要根据极小的时域波动设置极小 scale；应将其从基频、左右频率和相位 reward 的 joint 列表中排除。

## 3. 共享频谱分析器

在 task 的 `mdp` 目录中实现一个共享的 `JointFrequencyAnalyzer`，供所有频域 reward term 使用。不要让每个 reward 各自保存一份历史或重复执行 FFT。

### 3.1 采样与窗口

按照环境步采样实际关节角：

$$
f_s=\frac{1}{env.step\_dt}.
$$

当前系统若为 50 Hz 控制频率、200 Hz physics 频率，则 v0 的采样率是 50 Hz，Nyquist 频率是 25 Hz。

默认配置：

```python
window_duration_s = 2.0
window_size = round(window_duration_s / env.step_dt)
fft_update_interval = 5
window_type = "hann"
```

在 50 Hz 下，2 s 窗口包含 100 点，真实频率分辨率为 0.5 Hz。v0 不补零，使用 `n_fft=window_size`。

每个环境步都更新 GPU ring buffer，但默认每 5 个控制步才重新计算一次 FFT，其余步骤使用缓存结果，以减小对并行仿真的影响。`fft_update_interval` 必须可配置。

窗口未填满时，所有频域 penalty 返回 0。环境 reset 时，只清除对应 `env_ids` 的历史、有效长度和缓存。

### 3.2 FFT

对恢复为正确时间顺序的窗口，先计算未加窗均值：

$$
\mu_j=\frac{1}{N}\sum_{n=0}^{N-1}z_j[n].
$$

然后去均值：

$$
x_j[n]=z_j[n]-\mu_j.
$$

对 $x_j$ 乘 Hann window，使用：

```python
X = torch.fft.rfft(window * x, dim=time_dim, norm="ortho")
freq = torch.fft.rfftfreq(window_size, d=env.step_dt)
P = abs(X) ** 2
```

构造单边功率谱时，除 DC 和 Nyquist bin 外，其余 bin 的功率乘 2。频谱 reward 排除 DC bin；DC reward 直接使用未加窗的 $\mu_j$。

全部历史、FFT 和 reward 计算必须留在 `env.device`，训练循环中不要使用 NumPy、CPU 拷贝或逐环境 Python 循环。

## 4. 实现五个 reward term

所有函数返回形状为 `[num_envs]` 的非负 penalty，再由 `RewardTermCfg` 配置负权重。term 内不要再次乘 `env.step_dt`。

定义速度命令大小：

$$
c=\sqrt{v_x^2+v_y^2+(k_\omega\omega_z)^2}.
$$

默认 `stand_command_threshold=0.1`。

### 4.1 `joint_dc_posture`

当 $c\leq0.1$ 时，认为是 stand-still 指令，约束窗口平均姿态接近 default：

$$
L_{DC}^{stand}=\frac{1}{J}\sum_j\mu_j^2.
$$

当 $c>0.1$ 时，不要求均值严格等于 0，但不允许 DC 偏置过大：

$$
L_{DC}^{move}=\frac{1}{J}\sum_j
\max\left(0,|\mu_j|-d_j^{move}\right)^2.
$$

`moving_dc_limit` 允许配置为一个标量或逐关节字典。归一化后的启动值可以先取 `1.0`，之后由用户调整。不要从专家数据生成移动姿态目标。

### 4.2 `joint_spectral_high_frequency`

该项只要求低频代价小、高频代价大。使用简单的 alpha-beta 单调频率权重：

$$
A(f)=\frac{1+(2\pi f\beta)^2}{1+(2\pi f\alpha)^2},
\qquad \beta>\alpha>0,
$$

并归一化为

$$
W(f)=\frac{A(f)-1}{A(f_{Nyq})-1+\epsilon}.
$$

频谱 penalty 为

$$
L_{HF}=\frac{1}{J}\sum_j
\frac{\sum_{k>0}W(f_k)P_j(k)}
{\sum_{k>0}P_j(k)+\epsilon}.
$$

这样只约束能量分布，不追踪任何频段的指定绝对能量。可先使用 `alpha_s=0.01`、`beta_s=0.03` 作为启动值，但必须允许配置。

### 4.3 `joint_fundamental_concentration`

只在允许的基频范围内寻找机器人基频。启动配置：

```python
fundamental_search_band_hz = (1.0, 4.0)
fundamental_half_width_hz = 0.5
fundamental_joint_names = [
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
]
```

先聚合这些关节的归一化功率谱：

$$
P_{robot}(k)=\sum_{j\in\mathcal J_f}P_j(k).
$$

在 $[f_{min},f_{max}]$ 内寻找最大功率 bin，得到 $f_0$。定义主频带

$$
B_0=\{f_k:|f_k-f_0|\leq b_0\}.
$$

主频集中率为

$$
\rho_0=\frac{\sum_{f_k\in B_0}P_{robot}(k)}
{\sum_{k>0}P_{robot}(k)+\epsilon},
$$

移动时的 penalty 为

$$
L_{fund}=1-\rho_0.
$$

站立指令下该项返回 0。若总 AC 能量低于 `spectrum_energy_floor`，该项也返回 0。不要设置某个频带必须达到某个绝对能量；速度跟踪 reward 负责避免机器人通过完全不动获得好结果。

因为 $f_0$ 只能从配置的搜索区间中选择，而分母包含全部非零频率能量，所以过低或过高频率占主导时，$\rho_0$ 会降低并受到惩罚。

### 4.4 `joint_left_right_fundamental_match`

对显式配置的左右关节 pair，分别在同一个 `fundamental_search_band_hz` 内寻找峰值频率 $f_L$ 和 $f_R$：

$$
L_{pair-f}=\frac{1}{N_p}\sum_{(L,R)}
\left(\frac{f_L-f_R}{f_{max}-f_{min}}\right)^2.
$$

启动 pair：

```python
frequency_joint_pairs = [
    ("left_hip_pitch_joint", "right_hip_pitch_joint"),
    ("left_knee_joint", "right_knee_joint"),
    ("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
]
```

只有左右两侧在搜索频带内都具有足够能量时才计算该 pair；否则忽略，避免低能量噪声产生随机基频。站立指令下返回 0。

### 4.5 `joint_left_right_phase`

该项只在机器人主频 $f_0$ 附近的小频带 $B_0$ 内计算，并默认使用较小 reward weight。

若左右交替对应半个 gait period 的时间延迟，则

$$
\Delta=\frac{1}{2f_0}.
$$

对 pair $(L,R)$ 定义 cross spectrum：

$$
C_{LR}(k)=X_L(k)X_R^*(k),
$$

目标复数关系为

$$
p_{LR}(k)=s_{LR}\exp(-j2\pi f_k\Delta),
$$

其中 `mirror_sign` $s_{LR}$ 由用户对每个 pair 配置为 $+1$ 或 $-1$。使用连续相位损失：

$$
\ell_{LR}(k)=1-\operatorname{Re}\left[
\frac{C_{LR}(k)}{|C_{LR}(k)|+\epsilon}p_{LR}^*(k)
\right].
$$

在 $B_0$ 内按 $\sqrt{P_L(k)P_R(k)}$ 加权平均。低能量 pair 或无有效 bin 时返回 0，不得产生 NaN。

该项的 `RewardTermCfg.weight` 初始绝对值建议设为 `joint_fundamental_concentration` 权重绝对值的约 0.1 倍。不要把整个频谱固定为相差 $\pi$。

## 5. Isaac Lab 接入方式

建议文件：

```text
<task>/mdp/frequency_rewards.py
<task>/mdp/__init__.py
<task>/<task>_env_cfg.py
```

实现一个共享 analyzer，并为五个 penalty 提供五个可独立配置的 `ManagerTermBase` wrapper。可以在 env 上按 `analyzer_key` 保存 analyzer 实例。

每个 wrapper 调用 `analyzer.update_once(env)`；analyzer 根据 `env.common_step_counter` 保证同一个环境步只采样一次、最多执行一次 FFT。不要依赖 reward term 的调用顺序。

需要维护的最小状态：

```text
history      [num_envs, window_size, num_joints]
valid_count  [num_envs]
write_index
last_sample_step
last_fft_step
cached_mean
cached_spectrum
cached_power
```

`reset(env_ids)` 必须只重置指定环境。所有 reward 输出必须位于 `env.device`，shape 为 `[num_envs]`，且为有限值。

## 6. 最小配置项

至少提供：

```python
analyzer_key
asset_cfg
joint_scales
window_duration_s = 2.0
fft_update_interval = 5
stand_command_threshold = 0.1
moving_dc_limit
alpha_s = 0.01
beta_s = 0.03
fundamental_search_band_hz = (1.0, 4.0)
fundamental_half_width_hz = 0.5
fundamental_joint_names
frequency_joint_pairs
phase_joint_pairs
mirror_signs
spectrum_energy_floor
eps = 1.0e-8
```

五个 reward term 返回正 penalty，在环境配置中分别使用负 weight。相位项初始权重较小。

## 7. 交付要求

直接完成代码和环境配置接入，并在最终回复中说明：

- 修改了哪些文件；
- 五个 reward term 的配置位置和默认参数；
- 实际采样率、窗口点数、频率分辨率和 Nyquist 频率；
- 哪些关节被排除在基频或相位计算之外；
- 用户之后应优先调整哪些 scale、频带和 reward weight。

v0 不实现专家模仿、不追踪指定频段的绝对能量、不增加测试文件、不设计消融实验，也不修改 PPO 或 Isaac Lab 核心代码。
