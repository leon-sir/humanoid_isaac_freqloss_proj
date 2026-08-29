# Copyright (c) 2026 Zheng Pan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRNNModelCfg,
)


@configclass
class YMBOY12DOFFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Native RSL-RL PPO with asymmetric actor and critic observations."""

    seed = 42
    num_steps_per_env = 24
    max_iterations = 6000
    save_interval = 500
    experiment_name = "flat_12dof_freq_no_phase"
    run_name = "time_rewards"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    actor = RslRlRNNModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    critic = RslRlRNNModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class YMBOY12DOFFreqRewardPPORunnerCfg(YMBOY12DOFFlatPPORunnerCfg):
    """Recurrent PPO runner with a separate experiment namespace for frequency rewards."""

    experiment_name = "flat_12dof_freq_reward"
    run_name = "freq_rewards"
    max_iterations = 3000


@configclass
class YMBOY12DOFFreqAblation1PPORunnerCfg(YMBOY12DOFFreqRewardPPORunnerCfg):
    """Common runner settings for the first single-term frequency-reward ablation."""

    experiment_name = "flat_12dof_freq_ablation_1"


@configclass
class YMBOY12DOFFreqAblationNoDcDefaultPosturePPORunnerCfg(YMBOY12DOFFreqAblation1PPORunnerCfg):
    run_name = "no_dc_default_posture"


@configclass
class YMBOY12DOFFreqAblationNoBandEnergyEncouragePPORunnerCfg(YMBOY12DOFFreqAblation1PPORunnerCfg):
    run_name = "no_band_energy_encourage"


@configclass
class YMBOY12DOFFreqAblationNoHighFrequencyEncouragePPORunnerCfg(YMBOY12DOFFreqAblation1PPORunnerCfg):
    run_name = "no_high_frequency_encourage"


@configclass
class YMBOY12DOFFreqAblationNoEnergyDisencouragePPORunnerCfg(YMBOY12DOFFreqAblation1PPORunnerCfg):
    run_name = "no_energy_disencourage"


@configclass
class YMBOY12DOFFreqAblationNoFundamentalConcentrationPPORunnerCfg(YMBOY12DOFFreqAblation1PPORunnerCfg):
    run_name = "no_fundamental_concentration"


@configclass
class YMBOY12DOFFreqAblationNoLeftRightFrequencyEnergyMatchPPORunnerCfg(YMBOY12DOFFreqAblation1PPORunnerCfg):
    run_name = "no_left_right_frequency_energy_match"


@configclass
class YMBOY12DOFFreqAblationNoLeftRightPhasePPORunnerCfg(YMBOY12DOFFreqAblation1PPORunnerCfg):
    run_name = "no_left_right_phase"
