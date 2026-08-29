# Humanoid Isaac Frequency Locomotion

This project develops frequency-based MDP reward designs for 12-DoF YMBOY locomotion on mildly uneven Perlin-generated flat terrain. It targets Isaac Lab 2.2.3 and uses the native RSL-RL PPO implementation.

## Task

The project registers two corresponding manager-based tasks:

```text
FreqLab-Velocity-Flat-YMBOY12DOF-NoPhase
FreqLab-Velocity-Flat-YMBOY12DOF-FreqReward
```

`NoPhase` is the time-domain regularization baseline. `FreqReward` keeps the
same task and deployable observation spaces while replacing that regularization
with shared-window joint-position frequency penalties.

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
  --task FreqLab-Velocity-Flat-YMBOY12DOF-NoPhase \
  --headless
```
Play a trained policy:

```bash
python scripts/rsl_rl/play.py \
  --task FreqLab-Velocity-Flat-YMBOY12DOF-NoPhase \
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



Training outputs are written under `logs/rsl_rl/flat_12dof_freq_no_phase/`,
`logs/rsl_rl/flat_12dof_freq_reward/`, and
`logs/rsl_rl/flat_12dof_freq_ablation_1/`.

## Configuration layout

- `ymboy_12dof_envcfg_base.py` contains the robot task, observations, actions, domain randomization, terminations, and task rewards.
- `ymboy_12dof_envcfg_time_rewards.py` adds the baseline time-domain joint and action regularization.
- `ymboy_12dof_envcfg_freq_rewards.py` configures joint scales, bilateral mappings, and frequency shaping terms.
- `mdp/freq_rewards.py` implements the shared GPU ring buffer, cached FFT, and frequency reward terms.
