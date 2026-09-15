

## 切换参考数据

在当前版本配置 `ymboy_21dof_envcfg_freq_mimic_v1.py` 顶部修改：

```python
REFERENCE_MOTION = "Neutral_walk_forward_002__A057"
# 切换003，只修改上面一行：
# REFERENCE_MOTION = "Neutral_walk_forward_003__A057"
```

默认读取 `datasets/motions/<名称>/<名称>_periodic_filtered.csv` 及其同名JSON报告。
也可直接修改 `REFERENCE_MOTION_FILE`，指定该目录下的相对CSV路径或绝对CSV路径。
报告提供精确主频和采样率；加载器从CSV拟合全部关节的DC/cos/sin（rad），
并将根轨迹速度转换到机器人坐标系求平均。无需手动生成训练参考JSON，也无需改奖励文件。
仅在配置加载时处理数据，不在训练step中读取CSV。
配置实例保存 `reference_motion_file` 和 `reference_motion` 快照到env.yaml，奖励读取该快照。
第二阶段速度中心也读取同一快照；保留用户当前第一阶段速度范围和其他指令设置。

默认的 `_periodic_filtered.csv` 应由 `scripts/collector/filter_mocap_fundamental.py`
使用 `--symmetry` 生成。该选项会在保留各关节DC和主频幅值尺度的前提下：

- 将每一对左右同名关节在参考主频处的能量调整为左右均值；
- 将左右腿的 `hip_pitch -> knee`、`knee -> ankle_pitch` 链内相位延迟分别取圆周均值；
- 将左右手臂的 `shoulder_pitch -> elbow` 链内相位延迟取圆周均值。

它只对左右链的内部延迟作对称修正，不会把左右同名关节本身强制为180度反相。
训练中的核心左右反相仍由 `bilateral_phase_match_core` 单独约束。修改或切换原始
周期数据后，应重新运行滤波脚本；训练配置不会在加载CSV时再次修改参考动作。

例如重新生成默认002参考：

```bash
python scripts/collector/filter_mocap_fundamental.py \
  --csv source/humanoid_isaac_freq/humanoid_isaac_freq/datasets/motions/Neutral_walk_forward_002__A057/Neutral_walk_forward_002__A057_periodic.csv \
  --spectrum scripts/output/spectrum_mocap_analysis/2026-09-10_10-49-03_453816_Neutral_walk_forward_002__A057_periodic/runner_joint_power_scale.csv \
  --fps 120 \
  --output source/humanoid_isaac_freq/humanoid_isaac_freq/datasets/motions/Neutral_walk_forward_002__A057/Neutral_walk_forward_002__A057_periodic_filtered.csv \
  --symmetry --overwrite
```

## Resume 阶段标记

### V1 最新实验覆盖项

V1的 `spectral_fundamental_energy_match_core` 现使用 `ReferenceBandEnergyReward`，
权重 **+2**（不再是下文历史表中的-4幅值L2）。`CORE_ENERGY_BAND_HZ=(0.5,1.5)`，
`CORE_REFERENCE_ENERGY_RATIO=0.25`：对参考正弦按训练窗口去均值、Hann归一化，
取64个起始相位的平均频带能量，乘0.25作为各关节最低门槛（下限1e-4）。
得分为逐关节 `clamp(E_band / E_target, 0, 1)` 的均值；command term的
`is_standing_env=True` 或窗口未满时返回0。不按速度缩放，其他频域项尚未添加standing门控。
此修改应用于V1两阶段；other幅值项与相位/非谐波约束仍沿用原来的参考频率逻辑，
因此不是所有频域约束都已放宽到整个频段。V0不变。

V1 root reset 改用 `reset_root_state_uniform_body_frame`，初始机身系纵向速度
范围为 `(0.4,0.8)` m/s，其余pose/velocity范围不变。线速度和角速度均按新采样
姿态旋转到世界系；standing env也会获得该初速度。当前TimeReward对照类继承V1，
因此也继承这个reset设置。用户已有速度指令配置不改动。

主频幅值和DC在两个阶段都拆为核心4关节与其他17关节，核心项使用更强权重。
左右相位已拆为核心2组和其他8组；当前只启用hip/shoulder pitch核心项，
其他8组配置保留为注释。链内参考相位项负责hip pitch到knee、
knee到ankle pitch，以及shoulder pitch到elbow的时序。
腰部参与主频幅值、DC和非谐波能量，但没有左右相位配对。
异侧肩髋始终定义2组配对，reward权重为+0.25，阶段切换不改变其范围。
每组左右相位固定要求180度反相，不继承参考动作中的轻微左右不对称。

共享配置 `ymboy_21dof_envcfg_freq_mimic_common.py` 提供模块变量 `firstStage`，与 DreamLab 一致：
训练不加 `--resume` 为1，加 `--resume` 为0。两个train入口均在导入任务配置之前
将CLI标志写入 `RSL_RL_RESUME`。可在配置类或 `__post_init__` 中用
`if firstStage: ... else: ...` 设置多阶段MDP。V1第一阶段关闭chain phase，resume后打开；
两阶段都保留other的弱幅值/DC约束。`mimic_training_stage` 写入env.yaml便于核对。
V0的奖励配置及默认非谐波公式保持不变。

```bash
python scripts/rsl_rl/train.py \
  --task FreqLab-Velocity-Flat-YMBOY21DOF-FreqMimic-V1 --headless

# Stage 2: latest V1 run/checkpoint; add --load_run / --checkpoint to select explicitly
python scripts/rsl_rl/train.py \
  --task FreqLab-Velocity-Flat-YMBOY21DOF-FreqMimic-V1 --headless --resume
```

该标记在模块导入时读取，并非根据checkpoint是否存在自动推断。CLI `--resume` 是
此功能的入口；不要用导入后的Hydra覆盖来选择阶段。Play入口未作改动：按第二阶段
MDP播放时，在play命令前显式设置 `RSL_RL_RESUME=1`。

## Task + Time 对照任务

新增 `FreqLab-Velocity-Flat-YMBOY21DOF-TimeReward`，仅组合
`YMBOYTaskRewardsCfg()` 和 `YMBOY21DOFTimeOnlyRewardsCfg()`，不包含频域奖励。
继承频域任务的其他环境设置：机器人、指令范围、域随机化、termination、观测和FFT更新均一致。

两个任务的 `experiment_name` 均为 `flat_21dof_freq_mimic`；
频域任务 `run_name=freq_mimic_scaffold`，对照任务 `run_name=time_reward_baseline`。

```bash
# Train baseline
python scripts/rsl_rl/train.py \
  --task FreqLab-Velocity-Flat-YMBOY21DOF-TimeReward --headless

# Play baseline: replace the checkpoint path with an actual baseline run
python scripts/rsl_rl/play.py \
  --task FreqLab-Velocity-Flat-YMBOY21DOF-TimeReward --num_envs 32 \
  --checkpoint logs/rsl_rl/flat_21dof_freq_mimic/<run-directory>_time_reward_baseline/model_<iteration>.pt

# Compare both tasks in the same TensorBoard log directory
tensorboard --logdir logs/rsl_rl/flat_21dof_freq_mimic
```

任务：`FreqLab-Velocity-Flat-YMBOY21DOF-FreqMimic`。
频域模仿目标来自指定 filtered CSV及其自然 periodic 源数据。尚未验证新版训练收敛或步态质量。

## MDP版本

- `FreqLab-Velocity-Flat-YMBOY21DOF-FreqMimic-V0`：冻结复现
  `2026-09-12_16-20-01_freq_mimic_scaffold`。使用2秒频谱窗口、覆盖全部10组
  左右同名关节的单个 bilateral reward，不含链内相位reward；参考主频拟合系数也冻结为
  该次run写入 `env.yaml` 的快照，不受后来重新生成的对称CSV影响。
- `FreqLab-Velocity-Flat-YMBOY21DOF-FreqMimic-V1`：两阶段均使用2秒窗口，
  bilateral只启用核心hip/shoulder pitch，resume后增加参考链内相位reward。
- 无版本后缀的 `FreqLab-Velocity-Flat-YMBOY21DOF-FreqMimic` 保留兼容性，当前指向v1。

对应环境类分别为 `YMBOY21DOFFrequencyMimicEnvCfg_v0` 和
`YMBOY21DOFFrequencyMimicEnvCfg_v1`；versioned runner的run name分别为
`freq_mimic_v0` 和 `freq_mimic_v1`。

## 文件结构

```text
mimic/
  __init__.py                         # Gym 注册
  agents/rsl_rl_ppo_cfg.py            # 原生 RSL-RL PPO，actor/critic 均 LSTM
  env/manager_based_rl_env.py         # 独立复制的 YM 环境基础状态
  env/manager_based_rl_freq_mimic_env.py  # 主动采样/FFT/reset
  mdp/freq_rewards.py                # 频域奖励、自然谐波投影与共享参考统计
  reference_motion.py               # 从所选动捕CSV构建训练参考快照
  mdp/reference_motion.json          # 历史归档；默认不再读取
  ymboy_21dof_envcfg_freq_mimic_common.py # scene、观测、action、task/time reward等共享配置
  ymboy_21dof_envcfg_freq_mimic_v0.py  # 冻结的v0参考、frequency reward和2秒窗口EnvCfg
  ymboy_21dof_envcfg_freq_mimic_v1.py  # 两阶段v1参考、frequency reward和2秒窗口EnvCfg
```

V0和V1分别继承common中的 `YMBOY21DOFFrequencyMimicEnvCfgBase`，
该公共类继承 `LocomotionVelocityEnvCfg`；无版本兼容类继承V1。
不从 `config/ymboy_dash` 导入配置或环境；task/time 配置独立复制，并扩展全局关节项到21DOF。
共用项目级 MDP 函数、频谱分析器、terrain 和 robot asset。

奖励用 `compose_reward_cfgs(task, time, frequency)` 拼接：

- `YMBOYTaskRewardsCfg`：速度追踪、躯干、末端、终止和站立项。
- `YMBOY21DOFTimeOnlyRewardsCfg`：复制12DOF time组的权重/公式，全局项覆盖21关节；踝部和 hip-yaw 局部项保持原范围。
- `YMBOY21DOFFrequencyOnlyRewardsCfg_v0/v1`：各版本对应的分组主频幅值、分组DC、左右相位、异侧手腿相位和非谐波能量。

## 第一版参考与奖励

默认参考为002的 `periodic_filtered.csv`，不依赖collector频谱输出目录。
拟合系数、源文件hash、主频和速度保存在env.yaml的参考快照中。
002的f0为0.9375 Hz，平均机身前向速度约0.94893 m/s。
当前第一阶段vx范围为(0, 1)，第二阶段为(参考速度-0.5, 参考速度+0.1)；
vy为0，yaw范围为(-1, 1)，以主env cfg的设置为准。
不会随指令改变参考步频或幅值。

| reward | 权重 | 默认范围 |
|---|---:|---|
| fundamental_energy_match_core (v2) | -4.0 | 双侧hip/shoulder pitch；精确f0幅值L2误差 |
| fundamental_energy_match_other (v2) | -0.5 | 其他17关节；精确f0幅值L2误差 |
| reference_dc_match_core | -2.0 | 双侧hip/shoulder pitch；平方DC误差 |
| reference_dc_match_other | -0.5 | 其他17关节；平方DC误差 |
| bilateral_phase_match_core | +0.5 | hip/shoulder pitch左右关节对固定180度相位差 |
| bilateral_phase_match_other | +0.1 | 其他8组；当前禁用 |
| cross_limb_phase_match | +0.25 | 左肩/右髋同相、右肩/左髋同相 |
| kinematic_chain_phase_match | +0.5 | 仅第二阶段；四肢链内参考f0相位延迟 |
| non_harmonic_energy_core | -0.2 | 允许DC、f0、2f0，残差容差0.01 |
| non_harmonic_energy_other | -0.2 | 允许DC和1至5阶谐波，保留自然残差容差 |

主频幅值与DC始终覆盖全部21关节，核心4关节使用更强权重，其他17关节为-0.5。
bilateral core/other分别覆盖2组和8组左右同名关节，避免其他关节稀释核心相位得分。
异侧手腿配对保持显式两组，不推断其他身体部位的配对。
可编辑 `CORE_FREQUENCY_JOINTS`、`CORE_BILATERAL_PHASE_PAIRS`、
`OTHER_BILATERAL_PHASE_PAIRS`、`CROSS_LIMB_PHASE_PAIRS`。

主频v2在精确参考f0处对窗口做DC+cos+sin最小二乘拟合，以复系数模长作为幅值A，
惩罚为 `mean(((A-A_ref)/(A_ref+0.1))^2)`。它不依赖最近FFT bin，也不会像旧指数得分
那样在偏差较大时饱和到0；函数返回非负loss，因此使用负权重。旧版函数仍保留但配置不再调用。
自然 periodic 数据仍加载参考主频 `f0` 的前5阶信息。V1非谐波项新增
`harmonic_count`：core设2，other设5；仅允许DC和这些整数倍频率的最小二乘子空间，
不是允许整个连续频带。Core仅使用固定0.01残差容差，other使用自然5阶拟合残差+0.01。
被排除的高阶谐波不能重新加入容差。V0不传新参数，保持原5阶计算路径。
这里的f0始终来自所选参考报告，不从当前策略频谱重新寻找。
DC项复用 `JointDcPosturePenalty` 的平方误差，目标为参考绝对DC减机器人默认角后除以joint scale。
`moving_dc_limit=0.0`，不再使用指数得分或隐式短窗口均值容差；站立时也追参考DC。
通用项新增 `target_dc`（标量或按精确关节名映射，归一化单位），默认0保持原任务行为。
相位通过窗口内DC+cos/sin最小二乘拟合获得精确f0复系数，而不是最近FFT bin。
左右项为 `(1-cos(delta_phase))/2`，在180度取1；异侧手腿同相项为
`(1+cos(delta_phase))/2`，在0度取1。幅值小于0.02的对得0分。
链内相位项沿左右四肢顺序链，在精确参考f0处追踪hip→knee、
knee→ankle pitch及shoulder→elbow的参考相位差，共6组有向链边。
非谐波超额残差除以自然参考所选前K阶谐波能量+0.02后按关节平均；旧 `OutOfBandEnergy`
保留供复现实验，但新配置不再调用，避免惩罚自然接触谐波。

参考统计使用实际训练dt、window、scales和默认关节姿态，对64个参考起始相位计算同样的Hann FFT。
因此没有将15秒collector峰值直接当作2秒训练目标。完整窗口前所有频域项返回0。
2秒窗口包含100个50 Hz控制步，FFT bin间隔为0.5 Hz；精确f0拟合不要求参考频率落在bin上，
但有限窗口仍会影响相邻频率的区分能力。此次不新增包络或频率硬门控；相位沿用幅值门限。

这些是初始权重。time组原有回默认姿态、yaw/ankle整定可能与参考DC或摆幅竞争，
目前未擅自移除；训练时需对照分项reward调整。单步得分正确不保证能学出物理可行步态。

全局 `joint_pos_penalty` 仅在命令和机身平面速度都低于阈值时追机器人默认姿态，
并乘 `stand_still_scale=5`；有运动命令或机身尚未停止时返回0。运动DC完全由两个
reference DC频域项追踪，从而不再以瞬时姿态正则压制参考摆幅。

采用21DOF training URDF，包括对齐动捕零位的 elbow。上肢动作也参与策略输出。
上肢 actuator 参数沿用现有21DOF asset 的暂定配置，不代表经过硬件验证。
平地 Perlin terrain、域随机化、非对称观测沿用 dash 规范；主 height scan 禁用，
保留 base-height raycaster 给高度奖励。没有 gait phase 观测或步态规划器。

## 频谱缓存（即使 frequency reward 为空也运行）

环境在每个控制步物理更新后、termination/reward 计算前主动调用 analyzer，
避免 reward 为空时没有 FFT。当前采样50 Hz，窗口2秒（100点），每5控制步更新FFT。
参数在 env cfg 的 `joint_frequency_analyzer` 中，scales 位于 `FREQUENCY_JOINT_SCALES`。
腿部 scales 与 collector 相同；腰0.25 rad、手臂0.5 rad。归一化为
`(q-q_default)/scale`，窗口去均值、对称 Hann、正交 rFFT 和单边功率。
完整窗口前 `cached_ready=False`；异步 reset 清空对应环境历史，不能混入上一episode。

后续 reward 可访问 `env.frequency_analyzer` 的 `freq`、`cached_mean`、
`cached_spectrum`、`cached_power`、`cached_ready`、`window_energy`。
也可用共享 analyzer registry 的 `mimic_joints` key。不要为每个 reward 单独计算 FFT。
环境 step 循环基于本机 Isaac Lab 实现复制，升级 Isaac Lab 时需同步检查该循环。

## Train

在项目根目录、原生 RSL-RL 所在的 Isaac Lab conda 环境执行：

```bash
python scripts/rsl_rl/train.py \
  --task FreqLab-Velocity-Flat-YMBOY21DOF-FreqMimic-V0 --headless

python scripts/rsl_rl/train.py \
  --task FreqLab-Velocity-Flat-YMBOY21DOF-FreqMimic-V1 --headless

```


```bash
python scripts/rsl_rl/train.py \
  --task FreqLab-Velocity-Flat-YMBOY21DOF-TimeReward --headless && \

# FreqLab-Velocity-Flat-YMBOY21DOF-FreqMimic-V0 还行，但是走的有点快，主频没对上
python scripts/rsl_rl/train.py \
  --task FreqLab-Velocity-Flat-YMBOY21DOF-FreqMimic-V0  --headless --resume \
  --load_run 2026-09-11_11-49-18_freq_mimic_scaffold \
  --checkpoint model_5999.pt

python scripts/rsl_rl/train.py \
  --task FreqLab-Velocity-Flat-YMBOY21DOF-FreqMimic-V1  --headless --resume \
  --load_run 2026-09-11_11-49-18_freq_mimic_scaffold \
  --checkpoint model_5999.pt
```

## Play

```bash
python scripts/rsl_rl/play.py \
  --task FreqLab-Velocity-Flat-YMBOY21DOF-FreqMimic-V1 --num_envs 32 \
  --checkpoint logs/rsl_rl/flat_21dof_freq_mimic/2026-09-13_14-44-26_freq_mimic_v1/model_4000.pt
```

21DOF模型与12DOF策略维度不同，不能直接加载12DOF checkpoint。

## TensorBoard

```bash
tensorboard --logdir logs/rsl_rl/flat_21dof_freq_mimic
```


# YMBOY 21DOF Frequency Mimic（训练架构）

## V2

V2组合4秒窗口、`(0.75,1.0) Hz`低门槛能量奖励和频率拟合惩罚（−1），开启固定参考f0的chain match（+0.1），保留20步too-short和弱非core约束，禁用额外踝限位项；V0/V1及`v1_tests.py`不变。

```bash
python scripts/rsl_rl/train.py \
  --task FreqLab-Velocity-Flat-YMBOY21DOF-FreqMimic-V2 --headless --resume \
  --load_run 2026-09-14_17-50-57_freq_mimic_v1_frequency_fit_both \
  --checkpoint model_11998.pt
```

推荐以上Both checkpoint，也可显式指定原V1策略；`run_name=freq_mimic_v2`，experiment不变。
只加`--resume`默认匹配`.*_freq_mimic_v1`，加载Both或继续V2时请显式指定`--load_run`。
V2无论是否resume都启用上述奖励；resume仍保留Base的参考速度范围切换。

两种swap termination统一使用 `swap_time_threshold`（控制步数）：too-short默认20步（0.4秒），too-long当前200步（4秒），以50 Hz控制频率换算；too-short防抖参数保留为函数默认值，可按需覆盖。

FrequencyFit/NarrowBand新增 `swap_time_too_short`：髋pitch大小关系带0.02 rad滞回交替，连续两次完整间隔小于0.4秒终止；首次交替只启动计时，standing/零指令禁用，reset清空状态，不修改原V1或too-long项。该指标不是脚接触事件，可能在频域窗口填满前终止快步态，训练时需监测该终止率。

`V1-FrequencyFit` 保留2秒窗口并增加core时域拟合频率范围惩罚，`V1-NarrowBand` 使用4秒窗口和 `(0.75,1.0) Hz` 能量频带，两者均从V1策略resume、保留速度切换但关闭chain及额外踝限位整定。
play 用"scripts/collector/collect_runner_data_analysis_scale.py"脚本

```bash
python scripts/rsl_rl/train.py \
  --task FreqLab-Velocity-Flat-YMBOY21DOF-FreqMimic-V1  --headless

python scripts/rsl_rl/train.py \
  --task FreqLab-Velocity-Flat-YMBOY21DOF-FreqMimic-V1-FrequencyFit --headless --resume \
  --load_run 2026-09-13_20-05-11_freq_mimic_v1 --checkpoint model_5999.pt && \
python scripts/rsl_rl/train.py \
  --task FreqLab-Velocity-Flat-YMBOY21DOF-FreqMimic-V1-NarrowBand --headless --resume \
  --load_run 2026-09-13_20-05-11_freq_mimic_v1 --checkpoint model_5999.pt



```
