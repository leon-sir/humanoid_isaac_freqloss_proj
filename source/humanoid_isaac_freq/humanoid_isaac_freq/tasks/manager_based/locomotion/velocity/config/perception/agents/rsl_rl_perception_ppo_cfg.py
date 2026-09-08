"""Dream RSL-RL CNN-LSTM PPO configuration for the 12-DoF perception task."""

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class RslRlCNNEncoderCfg(RslRlCNNModelCfg.CNNCfg):
    """One named CNN encoder accepted by rsl_rl_dream's EncoderRNNModel."""

    class_name: str = "CNN"
    component_names: list[str] = MISSING


@configclass
class RslRlEncoderRNNModelCfg(RslRlMLPModelCfg):
    """Configuration bridge for the EncoderRNNModel supplied by rsl_rl_dream."""

    class_name: str = "EncoderRNNModel"
    encoder_configs: object = MISSING
    rnn_type: str = MISSING
    rnn_hidden_dim: int = MISSING
    rnn_num_layers: int = MISSING


@configclass
class DepthEncoderConv2dCfg(RslRlCNNEncoderCfg):
    """Encode the source-equivalent 18-by-32 current depth frame."""

    component_names = ["depth_image"]
    output_channels = [4]
    kernel_size = [3]
    stride = [1]
    dilation = [1]
    padding = "zeros"
    norm = "none"
    activation = "relu"
    max_pool = [True]
    global_pool = "none"
    flatten = True


@configclass
class EncoderConfigs:
    depth_encoder: DepthEncoderConv2dCfg = DepthEncoderConv2dCfg()


@configclass
class YMBOY12DOFPerceptionRNNBasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO/CNN/LSTM settings for the perception baseline."""

    seed = 42
    num_steps_per_env = 24
    max_iterations = 6000
    save_interval = 500
    experiment_name = "rough_12dof_perception_rnn"
    run_name = "RNN_CNN_Depth-Base"
    empirical_normalization = False
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    actor = RslRlEncoderRNNModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="scalar",
        ),
        encoder_configs=EncoderConfigs(),
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=None,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class YMBOY12DOFPerceptionRNNFreqRewardPPORunnerCfg(
    YMBOY12DOFPerceptionRNNBasePPORunnerCfg
):
    """Use a distinct run suffix for the frequency-reward task."""

    run_name = "RNN_CNN_Depth-FreqReward"
