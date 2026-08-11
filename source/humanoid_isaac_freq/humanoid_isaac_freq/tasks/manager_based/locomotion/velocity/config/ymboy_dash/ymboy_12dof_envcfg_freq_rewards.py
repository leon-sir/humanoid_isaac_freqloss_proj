# Copyright (c) 2026 Zheng Pan
# SPDX-License-Identifier: Apache-2.0

"""Frequency-domain joint regularization for the 12-DoF YMBOY task."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import humanoid_isaac_freq.tasks.manager_based.locomotion.velocity.mdp as mdp

from .ymboy_12dof_envcfg_base import (
    YMBOY12DOFEnvCfgBase,
    YMBOYTaskRewardsCfg,
)

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

ENCOURAGED_FREQUENCY_JOINTS = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
]

DISCOURAGED_FREQUENCY_JOINTS = [
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]

FREQUENCY_JOINT_PAIRS = [
    ("left_hip_pitch_joint", "right_hip_pitch_joint"),
    ("left_knee_joint", "right_knee_joint"),
    ("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
]

PHASE_JOINT_PAIRS = list(FREQUENCY_JOINT_PAIRS)
MIRROR_SIGNS = [1.0, 1.0, 1.0]

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
    "fundamental_search_band_hz": (0.5, 2.0),
    "spectrum_energy_floor": 1.0e-4,
}


@configclass
class YMBOY12DOFFrequencyRewardsCfg(YMBOYTaskRewardsCfg):
    """Task rewards plus six joint-position frequency penalties."""
    action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.01)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    joint_dc_posture = RewTerm(
        func=mdp.JointDcPosturePenalty,
        weight=-0.5,
        params={
            "analyzer_cfg": JOINT_FREQUENCY_ANALYZER_CFG,
            **COMMAND_PARAMS,
            "moving_dc_limit": 1.0,
        },
    )
    joint_spectral_high_frequency_encourage = RewTerm(
        func=mdp.JointSpectralHighFrequencyPenalty,
        weight=-0.1,
        params={
            "analyzer_cfg": JOINT_FREQUENCY_ANALYZER_CFG,
            "joint_names": ENCOURAGED_FREQUENCY_JOINTS,
            "alpha_s": 0.01,     # 0.1592 * 1/alpha_s =  15.92 Hz
            "beta_s": 0.05,     # 0.1592 * 1 / beta_s =   3 Hz
        },
    )
    joint_spectral_energy_disencourage = RewTerm(
        func=mdp.JointSpectralEnergyPenalty,
        weight=-0.1,
        params={
            "analyzer_cfg": JOINT_FREQUENCY_ANALYZER_CFG,
            "joint_names": DISCOURAGED_FREQUENCY_JOINTS,
        },
    )
    joint_gait_fundamental_frequency_concentration = RewTerm(
        func=mdp.JointFundamentalConcentrationPenalty,
        weight=-0.2,
        params={
            "analyzer_cfg": JOINT_FREQUENCY_ANALYZER_CFG,
            **FUNDAMENTAL_PARAMS,
            "fundamental_half_width_hz": 0.5,
            "fundamental_joint_names": ENCOURAGED_FREQUENCY_JOINTS,
        },
    )
    joint_left_right_fundamental_match = RewTerm(
        func=mdp.JointLeftRightFundamentalMatchPenalty,
        weight=-0.1,
        params={
            "analyzer_cfg": JOINT_FREQUENCY_ANALYZER_CFG,
            **FUNDAMENTAL_PARAMS,
            "frequency_joint_pairs": FREQUENCY_JOINT_PAIRS,
        },
    )
    joint_left_right_phase = RewTerm(
        func=mdp.JointLeftRightPhasePenalty,
        weight=-0.02,
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
    rewards: YMBOY12DOFFrequencyRewardsCfg = YMBOY12DOFFrequencyRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Match the deployable NoPhase baseline observation space.
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.disable_zero_weight_rewards()
