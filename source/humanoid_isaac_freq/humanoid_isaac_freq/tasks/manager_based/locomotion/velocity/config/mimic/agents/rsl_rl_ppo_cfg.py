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
class YMBOY21DOFMimicPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Native RSL-RL PPO with asymmetric actor and critic observations."""

    seed = 42
    num_steps_per_env = 24
    max_iterations = 6000
    save_interval = 500
    experiment_name = "flat_21dof_freq_mimic"
    run_name = "freq_mimic_scaffold"
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
class YMBOY21DOFTimeRewardsMimicPPORunnerCfg(YMBOY21DOFMimicPPORunnerCfg):
    """Same experiment and PPO settings, separate baseline run label."""

    run_name = "time_reward_baseline"


@configclass
class YMBOY21DOFMimicV0PPORunnerCfg(YMBOY21DOFMimicPPORunnerCfg):
    """Runner label for the frozen 2-second-window v0 MDP."""

    run_name = "freq_mimic_v0"


@configclass
class YMBOY21DOFMimicV1PPORunnerCfg(YMBOY21DOFMimicPPORunnerCfg):
    """Two-stage V1 runner; automatic resume stays within V1 runs."""

    run_name = "freq_mimic_v1"
    load_run = ".*_freq_mimic_v1"


@configclass
class YMBOY21DOFMimicFrequencyFitPPORunnerCfg(YMBOY21DOFMimicV1PPORunnerCfg):
    run_name = "freq_mimic_v1_frequency_fit"


@configclass
class YMBOY21DOFMimicNarrowBandPPORunnerCfg(YMBOY21DOFMimicV1PPORunnerCfg):
    run_name = "freq_mimic_v1_narrow_band"


@configclass
class YMBOY21DOFMimicV2PPORunnerCfg(YMBOY21DOFMimicV1PPORunnerCfg):
    """Same network/experiment; default resume selection remains original V1 runs."""

    run_name = "freq_mimic_v2"
