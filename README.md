# Humanoid Isaac Frequency Locomotion

This project develops frequency-based MDP reward designs for 12-DoF YMBOY locomotion on mildly uneven Perlin-generated flat terrain. It targets Isaac Lab 2.2.3 and uses the native RSL-RL PPO implementation.

## Task

The project registers two corresponding manager-based tasks:

```text
FreqLab-Velocity-Flat-YMBOY12DOF-TimeReward
FreqLab-Velocity-Flat-YMBOY12DOF-FreqReward
```

`TimeReward` is the time-domain joint-regularization baseline. `FreqReward`
keeps the same task, deployable observation spaces, and time-domain
regularization while adding shared-window joint-position frequency terms.

The actor receives deployable proprioceptive observations. The critic is asymmetric and additionally receives privileged base velocity, foot wrench, foot pose, and foot velocity observations. No gait-phase state or auxiliary reconstruction observation groups are used.

The terrain is generated with `PerlinPlaneTerrainCfg`, uses `noise_scale=[0.0, 0.02]`, and has terrain-level curriculum disabled.

## Installation

Activate the conda environment containing Isaac Lab 2.2.3, then install this extension in editable mode:

```bash
python -m pip install -e source/humanoid_isaac_freq
```

Verify task registration:

```bash
python scripts/list_envs.py
```

## Training

Run native RSL-RL PPO headlessly:

```bash
python scripts/rsl_rl/train.py \
  --task FreqLab-Velocity-Flat-YMBOY12DOF-TimeReward \
  --headless
```

Play a trained policy:

```bash
python scripts/rsl_rl/play.py \
  --task FreqLab-Velocity-Flat-YMBOY12DOF-TimeReward \
  --num_envs 32
```

Train the frequency-reward task:

```bash
python scripts/rsl_rl/train.py \
  --task FreqLab-Velocity-Flat-YMBOY12DOF-FreqReward \
  --headless

python scripts/rsl_rl/play.py \
  --task FreqLab-Velocity-Flat-YMBOY12DOF-FreqReward \
  --num_envs=32 \
  --video --video_length=400 \
  --checkpoint=logs/rsl_rl/flat_12dof_freq_reward/2026-08-12_16-36-14_freq_rewards/model_2999.pt

tensorboard --logdir=logs/rsl_rl/flat_12dof_freq_reward
```

### Rough-terrain depth-perception task

The perception baseline and frequency-reward variant use the same
`rsl_rl_dream` CNN-LSTM PPO runner and write runs to
`logs/rsl_rl/rough_12dof_perception_rnn/` with distinct run-name suffixes.

Both perception tasks explicitly configure terrain `contact_offset=0.02` m and
`rest_offset=0.0` m in `scene.terrain` (serialized in `params/env.yaml`). The
terrain importer applies these before simulation initialization; robot offsets
are unchanged. Continuous pure stairs use the original `boxes` mesh and a single
terrain collider. The collision stack remains `2**29` pending long-run validation.
Existing train/play commands need no extra flags; leave `TERRAIN_ABLATION` unset
for normal training. Explicit contact offsets avoid the measured automatic-offset
narrowphase overhead; changing contact offsets can also change policy trajectories.
See [PhysX contact tuning](https://nvidia-omniverse.github.io/PhysX/physx/5.1.2/docs/AdvancedCollisionDetection.html).

```bash
RSL_RL_DREAM_ENABLE_CUDNN_RNN=1 python scripts/rsl_rl_dream/train.py \
  --task=DreamLab-Velocity-Rough-YMBOY12DOF-Perception-RNN-Base-v0 \
  --headless \
  --max_iterations=3000

RSL_RL_DREAM_ENABLE_CUDNN_RNN=1 python scripts/rsl_rl_dream/train.py \
  --task=DreamLab-Velocity-Rough-YMBOY12DOF-Perception-RNN-FreqReward-v0 \
  --headless \
  --export_network_structure
```

Training exports the actor network diagram once before PPO updates, by default:
`<run-directory>/exported/EncoderRNNModel_network_structure.png` for the perception
CNN-LSTM actor. This is a structure diagram, not an exported inference policy.
Use `--no-export_network_structure` to skip, `--network_structure_format svg`
(or `pdf`) for vector output, `--network_structure_direction LR` for horizontal
layout, and `--network_structure_dpi 300` for higher-resolution PNG.
Export traces a separate model copy using one observation on CPU. Failures emit
a warning and do not stop training. Dependencies are `torchview`, Python
`graphviz`, and the Graphviz `dot` executable; install missing Python packages
in the training environment with `pip install torchview graphviz`.

Play a selected baseline or frequency-reward run without exporting the policy:

```bash
python scripts/rsl_rl_dream/play/play.py \
  --task=DreamLab-Velocity-Rough-YMBOY12DOF-Perception-RNN-Base-v0 \
  --num_envs=32 \
  --no-export_policy \
  --load_run=<base-run-directory>

python scripts/rsl_rl_dream/play/play.py \
  --task=DreamLab-Velocity-Rough-YMBOY12DOF-Perception-RNN-FreqReward-v0 \
  --num_envs=32 \
  --no-export_policy \
  --load_run=<freq-reward-run-directory>
```

Inspect training metrics:

```bash
tensorboard --logdir=logs/rsl_rl/rough_12dof_perception_rnn
```

### Frequency-reward ablations

Each task below removes exactly one frequency reward term from
`FreqLab-Velocity-Flat-YMBOY12DOF-FreqReward`. Run all seven ablations
sequentially (the chain stops if any training run fails):

```bash
python scripts/rsl_rl/train.py \
  --task FreqLab-Velocity-Flat-YMBOY12DOF-FreqReward-Ablation-NoDcDefaultPosture \
  --headless && \
python scripts/rsl_rl/train.py \
  --task FreqLab-Velocity-Flat-YMBOY12DOF-FreqReward-Ablation-NoBandEnergyEncourage \
  --headless && \
python scripts/rsl_rl/train.py \
  --task FreqLab-Velocity-Flat-YMBOY12DOF-FreqReward-Ablation-NoHighFrequencyEncourage \
  --headless && \
python scripts/rsl_rl/train.py \
  --task FreqLab-Velocity-Flat-YMBOY12DOF-FreqReward-Ablation-NoEnergyDisencourage \
  --headless && \
python scripts/rsl_rl/train.py \
  --task FreqLab-Velocity-Flat-YMBOY12DOF-FreqReward-Ablation-NoFundamentalConcentration \
  --headless && \
python scripts/rsl_rl/train.py \
  --task FreqLab-Velocity-Flat-YMBOY12DOF-FreqReward-Ablation-NoLeftRightFrequencyEnergyMatch \
  --headless && \
python scripts/rsl_rl/train.py \
  --task FreqLab-Velocity-Flat-YMBOY12DOF-FreqReward-Ablation-NoLeftRightPhase \
  --headless
```

Training outputs are written under `logs/rsl_rl/flat_12dof_time_reward/`,
`logs/rsl_rl/flat_12dof_freq_reward/`, and
`logs/rsl_rl/flat_12dof_freq_ablation_1/`.

## Configuration layout

### Retargeted mocap replay and spectrum analysis (21 DOF)

```bash
python scripts/collector/collect_mocap_data_analysis_scale.py
# Another clip:
python scripts/collector/collect_mocap_data_analysis_scale.py --csv <motion.csv>
```

Run in the Isaac Lab conda environment. Defaults: 120 Hz source CSV,
50 Hz playback/sampling, 250 warmup samples (5 s), then 750 samples (15 s),
25 Hz plot limit. Playback stops after that window. Use `--headless --fast`
for non-interactive analysis. All 21 joint angles are assigned absolutely,
without clipping or PD tracking. Root translation is converted cm to m,
Euler/joint angles degrees to radians. Linear resampling has no anti-alias
filter; source content above 25 Hz can alias at the default output rate.

Results (radian/normalized CSV, spectrum CSV/plots, summary and metadata) go to
`scripts/output/spectrum_mocap_analysis/<datetime>_<motion-name>/`.
FFT normalization matches the policy collector; the plotted power is raw
single-sided Hann-window FFT power, while the summary also reports power
divided by Hann-window energy. Leg scales/defaults are unchanged; provisional
waist/arm scales are 0.25/0.5 rad, with zero upper-body default posture.

`assets/ymbot_boy_21dof.py` uses `ymboy_21dof_training.urdf`: the supplied
21-DOF upper body with the existing 12-DOF leg joints and child links.
The original URDF is retained. Upper-body meshes come from soma-retargeter's
`assets/robots/ymbot_e/meshes`. Arm actuator tuning follows AMPMINIMAL and
is not a claim of hardware-safe motor limits. Joint-limit violations during
replay are retained and counted in metadata.

CPU normalization regression: `python test/test_mocap_spectrum.py`.

- `ymboy_12dof_envcfg_base.py` contains the robot task, observations, actions, domain randomization, terminations, and task rewards.
- `ymboy_12dof_envcfg_time_rewards.py` adds the baseline time-domain joint and action regularization.
- `ymboy_12dof_envcfg_freq_rewards.py` configures joint scales, bilateral mappings, and frequency shaping terms.
- `mdp/freq_rewards.py` implements the shared GPU ring buffer, cached FFT, and frequency reward terms.
