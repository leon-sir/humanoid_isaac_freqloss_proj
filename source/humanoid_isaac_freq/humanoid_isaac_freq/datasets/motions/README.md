# 动捕数据处理与频域分析

## 一键执行完整流程

先激活本项目的 Isaac Lab conda 环境，在项目根目录执行：

```bash
bash source/humanoid_isaac_freq/humanoid_isaac_freq/datasets/motions/process_and_analyze.bash Neutral_walk_forward_002__A057
```

替换末尾动作名即可处理其他数据；省略动作名默认使用002。动作名不带 `.csv`。
依次执行：周期拼接 → 原动作录像与频谱 → 仅主频滤波及对称化 → 滤波后录像与频谱。
固定参数为120 Hz源数据、30秒周期化、50 Hz采集、5秒warmup、15秒分析、5 Hz频谱上限。

无界面快速运行示例：

```bash
bash source/humanoid_isaac_freq/humanoid_isaac_freq/datasets/motions/process_and_analyze.bash Neutral_walk_forward_003__A057 --headless --fast
```

脚本自动定位项目根目录，使用当前环境的 `python`。每次重新生成 periodic 和 filtered
数据，直接覆盖对应 CSV/JSON，不再复用或添加文件名时间戳；原始动捕 CSV 不修改。
两次 collector 仍生成新的时间戳分析目录，保留历史视频与频谱。
独立调用两个处理脚本时默认仍防止覆盖，可添加 `--overwrite`；一键 Bash 已自动添加。
自动选择本次原动作分析生成的频谱，
若检测到同一动作并发导出造成歧义则停止。任一步失败停止后续步骤。
录像启用 `--fast_exit` 规避退出卡顿；其他参数会传给两次 collector，建议仅追加
`--headless --fast`，不要覆盖输入路径或采样参数。

以下命令均在**项目根目录**执行，而不是本目录。目前包含两个独立步骤：

1. 在原始动作中寻找最适合作为平稳周期的片段。
2. 在 Isaac Sim 中回放动作，并导出时域和频域分析结果。

支持通过第一步的 `--duration 30` 将筛选周期闭合并拼接成30秒 CSV，再用第二步播放。第二步本身不自动循环输入。

## 快速开始：拼接30秒动作并录制频谱分析视频

```bash
# 1. 筛选周期、平滑闭合并拼接为30秒（输入为原始动捕数据）
python scripts/collector/process_mocap_data.py --duration 30

# 2. 播放生成的 periodic 数据：5秒 warmup + 15秒频谱分析，并保存视频
python scripts/collector/collect_mocap_data_analysis_scale.py \
  --csv source/humanoid_isaac_freq/humanoid_isaac_freq/datasets/motions/Neutral_walk_forward_002__A057/Neutral_walk_forward_002__A057_periodic.csv \
  --source_fps 120 --sample_fps 50 \
  --warmup_steps 250 --num_frames 750 --max_frequency_hz 25 --video
```

`--duration 30` 是 **process_mocap_data.py** 的参数，不是播放脚本的参数。
若 `_periodic.csv` 已生成，可直接运行第二步；第一步不会覆盖已有文件。

## 仅保留关节主频的对照数据

滤波脚本可追加 `--symmetry` 开启左右主频能量和链内相位延迟对称化；默认关闭，
`--no-symmetry` 显式关闭。一键 `process_and_analyze.bash` 默认启用该选项。
对全部10组左右同名关节应用，腰部不变。保留各侧 DC 与相位，将幅值调整为
`A_new = sqrt((A_left^2 + A_right^2) / 2)`，即取能量均值而非幅值均值。
当前左右 scales 相同，因此归一化周期能量也相同；有限 Hann 窗的单个谱峰未必完全相等。
若一侧原幅值为零、相位无定义，采用另一侧的反相并在报告中提示。

相位对称化以各侧核心pitch关节为锚点，将左右链条对应延迟取圆周均值：

```text
hip_pitch      → knee
knee           → ankle_pitch
shoulder_pitch → elbow
```

随后在保持锚点相位、远端关节幅值和全部DC不变的前提下，分别旋转左右远端关节的
主频系数，使两侧链内延迟完全相同。JSON中的 `symmetry_chain_phase_delays` 保存修改前
左右延迟和采用的目标延迟。roll/yaw不属于这三组矢状面链内关系。

开启时默认输出 `_filtered_symmetric.csv`，也可以用 `--output` 指定其他名称。
该操作不会强制核心左右关节严格反相；核心左右相位仍来自原动作。它保证的是左右链条
具有相同的内部延迟，训练中的bilateral core reward再单独要求核心关节反相。

完成周期化和原频谱分析后，可生成 DC + 单个公共步频的关节轨迹：

```bash
python scripts/collector/filter_mocap_fundamental.py \
  --csv source/humanoid_isaac_freq/humanoid_isaac_freq/datasets/motions/Neutral_walk_forward_002__A057/Neutral_walk_forward_002__A057_periodic.csv \
  --spectrum scripts/output/spectrum_mocap_analysis/2026-09-09_23-16-29_701085_Neutral_walk_forward_002__A057_periodic/runner_joint_power_scale.csv \
  --symmetry
```

默认输出同目录 `_periodic_filtered.csv` 和 JSON 报告，已有文件不覆盖。
准确步频来自 periodic JSON 的周期帧数，而非频谱最大峰（最大峰可能是谐波）；
无 periodic JSON 时需显式给 `--frequency <Hz>`。频谱输入用于逐关节峰值诊断，
绝对角度和相位由原始时域信号拟合：`q = c + a*cos(2*pi*f0*t) + b*sin(2*pi*f0*t)`。
有周期报告时用一个完整周期拟合，保留周期均值，去掉二次及以上谐波。
不能直接对现有 Hann power CSV 做逆 FFT：它不是完整原始复频谱，且已经去均值、加窗。

仅修改21个关节列，根部轨迹、单位、采样率和时长保持不变；根部仍可能包含谐波。
这不是接触约束下的动作优化，膝踝形状改变可能产生滑步、穿地或限位问题。
滤波后的频谱仍会因有限窗口和 Hann 窗展宽；这不代表二次谐波没有删除。
播放时使用原 collector，将 `--csv` 换成 `_periodic_filtered.csv` 即可。

## 数据格式与目录

每段动作单独放在同名子目录，例如：

```text
motions/Neutral_walk_forward_003__A057/
  Neutral_walk_forward_003__A057.csv
  Neutral_walk_forward_003__A057_periodic.csv
  Neutral_walk_forward_003__A057_periodic.json
```

003–006 的处理与录像命令（已生成的 periodic 文件不要重复覆盖）：

```bash
for id in 003 004 005 006; do
  motion="source/humanoid_isaac_freq/humanoid_isaac_freq/datasets/motions/Neutral_walk_forward_${id}__A057/Neutral_walk_forward_${id}__A057"
  python scripts/collector/process_mocap_data.py --csv "${motion}.csv" \
    --fps 120 --min_period 0.7 --max_period 1.6 --trim 0.5 --duration 30 || break
  python scripts/collector/collect_mocap_data_analysis_scale.py --csv "${motion}_periodic.csv" \
    --source_fps 120 --sample_fps 50 --warmup_steps 250 --num_frames 750 \
    --max_frequency_hz 5 --video --headless --fast --fast_exit || break
done
```

若 periodic 数据已存在，直接运行循环中的 collector 命令即可。

`--fast_exit` 在导出和视频资源清理完成后使用 Isaac Sim 的快速关闭接口，
绕过本机可能卡住的 Replicator 停止流程；默认不启用。不改变采样或视频内容。

- 五段原始数据：`Neutral_walk_forward_002__A057.csv` 至 `006`。
- 标称采样率：120 Hz；`Frame` 是连续帧编号，不是秒。
- 根部位移 `root_translateX/Y/Z`：cm。
- 根部旋转 `root_rotateX/Y/Z`：XYZ Euler，degree。
- 21 个关节角 `*_joint_dof`：degree，按列名映射到机器人关节。
- 构成：12 个腿足关节、1 个 waist yaw、8 个手臂关节。

原始片段只有约 8.3–11 秒，不能直接满足 5 秒 warmup 加 15 秒采集。










## 1. 寻找平稳周期片段

脚本：`scripts/collector/process_mocap_data.py`。只依赖 NumPy、SciPy，不需要启动 Isaac Sim。

```bash
python scripts/collector/process_mocap_data.py
```

默认处理本目录的 `Neutral_walk_forward_002__A057.csv`。指定输入、周期搜索范围和输出：

```bash
python scripts/collector/process_mocap_data.py \
  --csv source/humanoid_isaac_freq/humanoid_isaac_freq/datasets/motions/Neutral_walk_forward_002__A057/Neutral_walk_forward_002__A057.csv \
  --fps 120 --min_period 0.7 --max_period 1.6 --trim 0.5 \
  --output source/humanoid_isaac_freq/humanoid_isaac_freq/datasets/motions/Neutral_walk_forward_002__A057/Neutral_walk_forward_002__A057_stable_cycle_v2.csv
```

默认跳过原动作首尾各 0.5 秒，在 0.7–1.6 秒范围内搜索一个**完整 stride 周期**，不是左右脚交换的半周期。对两个连续、等长候选周期进行比较：

- 双侧 hip pitch、knee、ankle pitch 的波形重复性。
- 全部 21 关节的首尾姿态与速度差。
- 根部首尾旋转、高度与平移速度差；不惩罚正常的前进位移。
- 设置最低运动幅度，避免优先选到静止片段。

评分越低越好。评分阶段使用轻度平滑和归一化，但**导出数据不平滑、不修改关节角、不修改根部轨迹**，只把 Frame 重新编号。该方法是关节状态重复性的启发式筛选，不是足接触检测或动力学稳定性认证。

### 输出和报告

在输入 CSV 同目录默认生成：

- `<原文件名>_stable_cycle.csv`：选中的第一个周期，保留原始单位和列格式。
- `<原文件名>_stable_cycle.json`：源文件、原始帧区间、评分、首尾误差、根部位移及候选排名。

终端显示前五个候选、选中时间范围、周期/频率、首尾误差及输出路径。已有输出不会被覆盖；重新处理时请用 `--output` 指定新文件名。

CSV 采用 `[start, end)`，不重复保存下一个周期的起始帧；匹配边界行另存于 JSON 的 `endpoint_row`。

当前默认原始数据的筛选结果是 `[2.933, 4.000)` 秒，共 128 帧，周期约 1.067 秒，频率约 0.938 Hz。它仍存在首尾残差，不能视为已经无缝闭合。

## 2. Isaac Sim 回放与频域分析

脚本：`scripts/collector/collect_mocap_data_analysis_scale.py`。在可运行本项目的 Isaac Lab conda 环境中执行。

对已生成的 stable cycle，使用 **零 warmup + 1 秒采集**，同时录制视频：

```bash
python scripts/collector/collect_mocap_data_analysis_scale.py \
  --csv source/humanoid_isaac_freq/humanoid_isaac_freq/datasets/motions/Neutral_walk_forward_002__A057/Neutral_walk_forward_002__A057_stable_cycle.csv \
  --source_fps 120 --sample_fps 50 \
  --warmup_steps 0 --num_frames 50 --max_frequency_hz 25 --video
```

指定其他原始动作：

```bash
python scripts/collector/collect_mocap_data_analysis_scale.py \
  --csv source/humanoid_isaac_freq/humanoid_isaac_freq/datasets/motions/Neutral_walk_forward_003__A057/Neutral_walk_forward_003__A057.csv \
  --source_fps 120 --sample_fps 50 \
  --warmup_steps 250 --num_frames 150 --max_frequency_hz 25
```

加 `--headless --fast` 可进行无界面快速分析，也支持同时开启 `--video`。GUI 默认按实时速度播放，镜头保持世界系方向并随根部平移。

视频默认关闭，`--video` 开启、`--no-video` 关闭；开启时自动启用 camera rendering。
MP4 保存到本次分析输出目录，文件名为 `runner_joint_angles_scale_rollout.mp4`，
分辨率 1280×720、帧率等于 `sample_fps`，包括 warmup 和采集阶段。
编码使用 imageio/imageio-ffmpeg。这里只是单次回放录制，**不会自动循环 stable cycle**。

### 时间参数

| 参数 | 脚本默认值 | 含义 |
|---|---:|---|
| `--source_fps` | 120 | 原 CSV 采样率 |
| `--sample_fps` | 50 | 回放及分析采样率 |
| `--warmup_steps` | 250 | 跳过前 5 秒，不纳入分析 |
| `--num_frames` | 750 | 分析 15 秒；当前短原始数据须改小 |
| `--max_frequency_hz` | 25 | 频谱图和频谱 CSV 的频率上限 |

首个采样点在 `1 / sample_fps` 秒。输入必须覆盖 warmup 和采集所需时长，否则脚本报错，不循环、不补零。150 帧、50 Hz 对应 3 秒窗口，FFT bin 间隔为 `50 / 150 = 0.333 Hz`。

**不要将 128 帧的 stable_cycle CSV 直接搭配默认 warmup 使用。** 它只有约 1.067 秒；当前脚本不会将其重复成 15 秒动作。若只想查看该短片段，可以执行：

```bash
python scripts/collector/collect_mocap_data_analysis_scale.py \
  --csv source/humanoid_isaac_freq/humanoid_isaac_freq/datasets/motions/Neutral_walk_forward_002__A057/Neutral_walk_forward_002__A057_stable_cycle.csv \
  --warmup_steps 0 --num_frames 50 --video
```

这个示例只分析 1 秒，频率分辨率只有 1 Hz，不适合精细比较步频。

### 回放模型与零位

- Robot cfg：`source/humanoid_isaac_freq/humanoid_isaac_freq/assets/ymbot_boy_21dof.py`。
- 模型：`data/Robots/ymboy-12dof/ymboy_21dof_training.urdf`。
- 腿足使用本项目 12DOF 的关节及子 link 定义；上肢来自原 21DOF 模型。
- training URDF 左右 elbow 的 `origin rpy="0 -1.5 0"` 对齐重定向 MJCF 零位。因此 CSV elbow 角度直接使用，**不要额外减 1.5 rad**。
- 使用根部和关节状态赋值回放，不推进动力学、不做 PD 跟踪，也不裁剪超限角度。分析仿真器回读的关节角；超限帧数写入 metadata。

### 归一化和频谱

处理方法与策略 collector `scripts/collector/collect_runner_data_analysis_scale.py` 一致：

1. CSV degree 转为 rad，根部 cm 转为 m。
2. 关节/平移线性插值至 50 Hz，根部旋转用 quaternion SLERP。当前没有抗混叠滤波，高于 25 Hz 的源信号可能混叠。
3. `z = (q - q_default) / scale`。
4. 减去 z 的窗口均值，乘对称 Hann 窗。
5. `rfft(norm="ortho")` 后取模平方，单边谱内部频点乘 2。

腿部默认姿态和 scales 与策略 collector 相同。腰、手臂默认姿态为 0；暂定 scale 分别为 0.25 rad、0.5 rad。所有实际值写入 metadata。

频谱图展示的是**归一化关节信号的原始单边 Hann FFT power**，不是 PSD，也没有再除以窗能量。summary 另外提供 `hann_normalized_ac_energy`，即非 DC 总 power 除以 Hann 窗平方和。不同窗口长度下，不宜直接比较原始谱峰高度。

FFT 定义位于 `scripts/collector/mocap_spectrum_utils.py`；与策略 collector 的一致性测试：

```bash
python test/test_mocap_spectrum.py
```

### 分析输出

保存到项目根目录下：

```text
scripts/output/spectrum_mocap_analysis/<日期时间>_<输入CSV文件名>/
  runner_joint_angles_rad.csv
  runner_joint_angles_scale.csv
  runner_joint_angles_scale.png
  runner_joint_power_scale.csv
  runner_joint_power_scale.png
  runner_joint_power_scale_summary.csv
  rollout_metadata.json
  runner_joint_angles_scale_rollout.mp4  # 开启 --video 时生成
```

metadata 记录输入路径、采样率、源时间区间、窗口参数、模型、默认姿态、scales 和超限情况。比较不同动作之前应先核对这些设置。

## 周期闭合与拼接（推荐用于长窗口频谱分析）

从原始 CSV 重新筛选并拼接（不要输入只有一个周期的 stable_cycle）：

```bash
python scripts/collector/process_mocap_data.py \
  --csv source/humanoid_isaac_freq/humanoid_isaac_freq/datasets/motions/Neutral_walk_forward_002__A057/Neutral_walk_forward_002__A057.csv \
  --fps 120 \
  --min_period 0.7 --max_period 1.6 \
  --trim 0.5 \
  --duration 30
```

默认生成同目录下 `Neutral_walk_forward_002__A057_periodic.csv` 和 `.json`，
保留原文件及 stable_cycle 文件。30秒包含3601帧（含30秒终点），采样率仍为120 Hz。
如文件已存在，请使用 `--output` 指定新文件名。

播放拼接数据、录视频，并恢复5秒 warmup + 15秒分析：

```bash
python scripts/collector/collect_mocap_data_analysis_scale.py \
  --csv source/humanoid_isaac_freq/humanoid_isaac_freq/datasets/motions/Neutral_walk_forward_002__A057/Neutral_walk_forward_002__A057_periodic.csv \
  --source_fps 120 --sample_fps 50 \
  --warmup_steps 250 --num_frames 750 --max_frequency_hz 5 --video 

 # max_frequency_hz 5Hz，导出的图会更加清晰点
```


拼接采用覆盖整个周期的三次多项式修正，使连续插值曲线的首尾位置和速度匹配（C1），
不改变周期长度。根部 XY 每周期累计原水平位移；Z 和姿态周期重复，适用于直行。
根部旋转在相对初始姿态的 rotation-vector 坐标中闭合，再转换回 CSV Euler 角。
JSON 和终端报告包含修正幅度及闭合残差。默认动作的最大关节修正约6.05度，
因此输出已不是原始动作，应检查回放和频谱。该处理不保证加速度连续或足接触约束。

处理后仍需检查足接触、滑步、关节限位和动态可行性；数学闭合不代表物理稳定。

原始片段和周期化数据分别保存。重复一个周期能构造理想周期参考，但不会增加独立动作信息，也不能代替原始动作的稳定性评估。




# Example Command:

```
# 周期拼合
python scripts/collector/process_mocap_data.py \
  --csv source/humanoid_isaac_freq/humanoid_isaac_freq/datasets/motions/Neutral_walk_forward_002__A057/Neutral_walk_forward_002__A057.csv \
  --fps 120 \
  --min_period 0.7 --max_period 1.6 \
  --trim 0.5 \
  --duration 30

# 播放与频域分析
python scripts/collector/collect_mocap_data_analysis_scale.py \
  --csv source/humanoid_isaac_freq/humanoid_isaac_freq/datasets/motions/Neutral_walk_forward_002__A057/Neutral_walk_forward_002__A057_periodic.csv \
  --source_fps 120 --sample_fps 50 \
  --warmup_steps 250 --num_frames 750 --max_frequency_hz 5 --video 

# 滤波
python scripts/collector/filter_mocap_fundamental.py \
  --csv source/humanoid_isaac_freq/humanoid_isaac_freq/datasets/motions/Neutral_walk_forward_002__A057/Neutral_walk_forward_002__A057_periodic.csv \
  --spectrum scripts/output/spectrum_mocap_analysis/2026-09-09_23-16-29_701085_Neutral_walk_forward_002__A057_periodic/runner_joint_power_scale.csv \
  --output source/humanoid_isaac_freq/humanoid_isaac_freq/datasets/motions/Neutral_walk_forward_002__A057/Neutral_walk_forward_002__A057_periodic_filtered.csv


# 滤波后验证播放与频域分析
python scripts/collector/collect_mocap_data_analysis_scale.py \
  --csv source/humanoid_isaac_freq/humanoid_isaac_freq/datasets/motions/Neutral_walk_forward_002__A057/Neutral_walk_forward_002__A057_periodic_filtered.csv \
  --source_fps 120 --sample_fps 50 \
  --warmup_steps 250 --num_frames 750 --max_frequency_hz 5 --video 

```
