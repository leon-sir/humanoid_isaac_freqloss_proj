# Copyright (c) 2026 Zheng Pan
# SPDX-License-Identifier: Apache-2.0

"""Frequency-domain joint regularization for the 12-DoF YMBOY task."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import humanoid_isaac_freq.tasks.manager_based.locomotion.velocity.mdp as mdp

from .ymboy_12dof_envcfg_base import (
    ACTUATED_JOINT_NAMES,
    YMBOY12DOFEnvCfgBase,
    YMBOYTaskRewardsCfg,
    compose_reward_cfgs,
)
from .ymboy_12dof_envcfg_time_rewards import YMBOY12DOFTimeOnlyRewardsCfg

# The analyzer resolves these names against the articulation; dictionary order is
# explicit configuration, not an implicit assumption about articulation ordering.
JOINT_FREQUENCY_SCALES = {
    "left_hip_pitch_joint": 0.55,
    "right_hip_pitch_joint": 0.55,
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

DISCOURAGED_FREQUENCY_JOINTS = [
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]
HIP_PITCH_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
]

ONLY_HIP_PITCH = True
if ONLY_HIP_PITCH:
    ENCOURAGED_FREQUENCY_JOINTS = list(HIP_PITCH_JOINT_NAMES)
    FREQUENCY_JOINT_PAIRS = [
        ("left_hip_pitch_joint", "right_hip_pitch_joint"),
    ]
else:
    ENCOURAGED_FREQUENCY_JOINTS = [
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
    ]
    FREQUENCY_JOINT_PAIRS = [
        ("left_hip_pitch_joint", "right_hip_pitch_joint"),
        ("left_knee_joint", "right_knee_joint"),
        ("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
    ]

PHASE_JOINT_PAIRS = list(FREQUENCY_JOINT_PAIRS)
MIRROR_SIGNS = [1.0] * len(FREQUENCY_JOINT_PAIRS)

JOINT_FREQUENCY_ANALYZER_CFG = mdp.JointFrequencyAnalyzerCfg(
    analyzer_key="ymboy_12dof_joint_pos",
    asset_cfg=SceneEntityCfg("robot"),
    joint_scales=JOINT_FREQUENCY_SCALES,
    window_duration_s=2.0,
    fft_update_interval=5,
    window_type="hann",
    eps=1.0e-8,
)

COMMAND_PARAMS = {
    "command_name": "base_velocity",
    "k_omega": 0.5,
    "stand_command_threshold": 0.1,
}
FUNDAMENTAL_PARAMS = {
    **COMMAND_PARAMS,
    "fundamental_search_band_hz": (0.5, 1.5),
    "spectrum_energy_floor": 1.0e-4,
}
GAIT_ENERGY_BAND_HZ = (0.5, 1.5)    # (0.5, 3.0)


@configclass
class YMBOY12DOFFrequencyOnlyRewardsCfg:
    """Joint-position frequency shaping added to the shared task rewards."""

    spectral_dc_default_posture = RewTerm(
        func=mdp.JointDcPosturePenalty,
        weight=-2,    # -0.5
        params={
            "analyzer_cfg": JOINT_FREQUENCY_ANALYZER_CFG,
            **COMMAND_PARAMS,
            "joint_names": ACTUATED_JOINT_NAMES,
            "moving_dc_limit": 0.0,  # 允许的均值偏差
        },
    )
    spectral_left_right_dc_match = RewTerm(
        func=mdp.JointLeftRightDcMatchPenalty,
        weight=-0.5,
        params={
            "analyzer_cfg": JOINT_FREQUENCY_ANALYZER_CFG,
            **COMMAND_PARAMS,
            "dc_joint_pairs": FREQUENCY_JOINT_PAIRS,
            "mirror_signs": MIRROR_SIGNS,
        },
    )
    # joint_spectral_dc_default_posture_hip_pitch = RewTerm(
    #     func=mdp.JointDcPosturePenalty,
    #     weight=-2,    # -0.5
    #     params={
    #         "analyzer_cfg": JOINT_FREQUENCY_ANALYZER_CFG,
    #         **COMMAND_PARAMS,
    #         "joint_names": HIP_PITCH_JOINT_NAMES,
    #         "moving_dc_limit": 0.0,  # 允许的均值偏差
    #     },
    # )
    spectral_band_energy_encourage = RewTerm(
        func=mdp.JointSpectralBandEnergyReward,
        weight=2,
        params={
            "analyzer_cfg": JOINT_FREQUENCY_ANALYZER_CFG,
            **COMMAND_PARAMS,
            "joint_names": ENCOURAGED_FREQUENCY_JOINTS,
            "energy_band_hz": GAIT_ENERGY_BAND_HZ,
            # cmd1 expert: 0.075--0.125 after Hann-energy normalization.
            # Unlike raw FFT-bin power, this target is window-length independent.
            "target_band_energy": 0.08,
        },
    )
    # joint_spectral_band_energy_encourage_hip_pitch = RewTerm(
    #         func=mdp.JointSpectralBandEnergyReward,
    #         weight=2,
    #         params={
    #             "analyzer_cfg": JOINT_FREQUENCY_ANALYZER_CFG,
    #             **COMMAND_PARAMS,
    #             "joint_names": HIP_PITCH_JOINT_NAMES,
    #             "energy_band_hz": GAIT_ENERGY_BAND_HZ,
    #             # cmd1 expert: 0.075--0.125 after Hann-energy normalization.
    #             # Unlike raw FFT-bin power, this target is window-length independent.
    #             "target_band_energy": 0.08,
    #         },
    #     )
    spectral_high_frequency_encourage = RewTerm(
        func=mdp.JointSpectralHighFrequencyPenalty,
        weight=-1e-5,  # -0.1
        params={
            "analyzer_cfg": JOINT_FREQUENCY_ANALYZER_CFG,
            "joint_names": ENCOURAGED_FREQUENCY_JOINTS,
            "alpha_s": 0.01,  # 0.1592 * 1/alpha_s =  15.92 Hz
            "beta_s": 0.05,  # 0.1592 * 1 / beta_s =   3 Hz
        },
    )

    spectral_energy_disencourage = RewTerm(
        func=mdp.JointSpectralEnergyPenalty,
        weight=-0.5,
        params={
            "analyzer_cfg": JOINT_FREQUENCY_ANALYZER_CFG,
            "joint_names": DISCOURAGED_FREQUENCY_JOINTS,
        },
    )
    spectral_gait_fundamental_frequency_concentration = RewTerm(
        func=mdp.JointFundamentalConcentrationPenalty,
        weight=-1,  # -0.2
        params={
            "analyzer_cfg": JOINT_FREQUENCY_ANALYZER_CFG,
            **FUNDAMENTAL_PARAMS,
            "fundamental_half_width_hz": 0.5,
            "fundamental_joint_names": ENCOURAGED_FREQUENCY_JOINTS,
        },
    )
    spectral_left_right_frequency_energy_match = RewTerm(
        func=mdp.JointLeftRightFrequencyEnergyMatchPenalty,
        weight=-0.1, # -0.1
        params={
            "analyzer_cfg": JOINT_FREQUENCY_ANALYZER_CFG,
            **FUNDAMENTAL_PARAMS,
            "frequency_joint_pairs": FREQUENCY_JOINT_PAIRS,
            "energy_match_weight": 1.0,
        },
    )
    spectral_left_right_phase = RewTerm(
        func=mdp.JointLeftRightPhasePenalty,
        weight=-2,   # -0.02
        params={
            "analyzer_cfg": JOINT_FREQUENCY_ANALYZER_CFG,
            **FUNDAMENTAL_PARAMS,
            "fundamental_half_width_hz": 0.5,
            "fundamental_joint_names": ENCOURAGED_FREQUENCY_JOINTS,
            "phase_joint_pairs": PHASE_JOINT_PAIRS,
            "mirror_signs": MIRROR_SIGNS,
        },
    )


@configclass
class YMBOY12DOFFrequencyRewardsEnvCfg(YMBOY12DOFEnvCfgBase):
    rewards: YMBOYTaskRewardsCfg = compose_reward_cfgs(
        YMBOYTaskRewardsCfg(),
        YMBOY12DOFFrequencyOnlyRewardsCfg(),
        YMBOY12DOFTimeOnlyRewardsCfg(),
    )

    def __post_init__(self):
        super().__post_init__()
        # Match the deployable TimeReward baseline observation space.
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.disable_zero_weight_rewards()


@configclass
class YMBOY12DOFFrequencyRewardsNoDcDefaultPostureEnvCfg(YMBOY12DOFFrequencyRewardsEnvCfg):
    """Frequency-reward ablation without the DC default-posture penalty."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.spectral_dc_default_posture = None


@configclass
class YMBOY12DOFFrequencyRewardsNoBandEnergyEncourageEnvCfg(YMBOY12DOFFrequencyRewardsEnvCfg):
    """Frequency-reward ablation without encouraged-joint band energy."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.spectral_band_energy_encourage = None


@configclass
class YMBOY12DOFFrequencyRewardsNoHighFrequencyEncourageEnvCfg(YMBOY12DOFFrequencyRewardsEnvCfg):
    """Frequency-reward ablation without encouraged-joint high-frequency regularization."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.spectral_high_frequency_encourage = None


@configclass
class YMBOY12DOFFrequencyRewardsNoEnergyDisencourageEnvCfg(YMBOY12DOFFrequencyRewardsEnvCfg):
    """Frequency-reward ablation without discouraged-joint spectral-energy regularization."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.spectral_energy_disencourage = None


@configclass
class YMBOY12DOFFrequencyRewardsNoFundamentalConcentrationEnvCfg(YMBOY12DOFFrequencyRewardsEnvCfg):
    """Frequency-reward ablation without gait fundamental-frequency concentration."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.spectral_gait_fundamental_frequency_concentration = None


@configclass
class YMBOY12DOFFrequencyRewardsNoLeftRightFrequencyEnergyMatchEnvCfg(YMBOY12DOFFrequencyRewardsEnvCfg):
    """Frequency-reward ablation without left/right frequency and energy matching."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.spectral_left_right_frequency_energy_match = None


@configclass
class YMBOY12DOFFrequencyRewardsNoLeftRightPhaseEnvCfg(YMBOY12DOFFrequencyRewardsEnvCfg):
    """Frequency-reward ablation without left/right phase matching."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.spectral_left_right_phase = None
