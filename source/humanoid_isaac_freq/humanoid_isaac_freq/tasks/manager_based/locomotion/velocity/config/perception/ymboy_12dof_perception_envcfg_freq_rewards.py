"""Frequency-reward variant of the self-contained 12-DoF perception task."""

from copy import deepcopy

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
import humanoid_isaac_freq.tasks.manager_based.locomotion.velocity.config.perception.mdp.freq_rewards as freq_mdp
from .ymboy_12dof_perception_envcfg_base import (
    ACTUATED_JOINT_NAMES,
    YMBOY12DOFPerceptionEnvCfgBase,
    YMBOY12DOFPerceptionRewardsCfg,
)


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

ONLY_HIP_PITCH = True
if ONLY_HIP_PITCH:
    ENCOURAGED_FREQUENCY_JOINTS = [
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
    ]
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

FREQUENCY_JOINT_MIRROR_SIGNS = [1.0] * len(FREQUENCY_JOINT_PAIRS)

JOINT_FREQUENCY_ANALYZER_CFG = freq_mdp.JointFrequencyAnalyzerCfg(
    analyzer_key="ymboy_12dof_perception_joint_pos",
    asset_cfg=SceneEntityCfg("robot"),
    joint_scales=JOINT_FREQUENCY_SCALES,
    window_duration_s=2.0,
    fft_update_interval=5,
    window_type="hann",
    eps=1.0e-8,
)

FREQ_WEIGHT_RATE = 1.0

@configclass
class YMBOY12DOFPerceptionFrequencyOnlyRewardsCfg:
    """The exact frequency-domain reward group used by the source task."""


    spectral_dc_default_posture = RewTerm(
        func=freq_mdp.JointDcPosturePenalty,
        weight=-1.0*FREQ_WEIGHT_RATE,
        params={
            "analyzer_cfg": JOINT_FREQUENCY_ANALYZER_CFG,
            "command_name": "base_velocity",
            "k_omega": 0.5,
            "stand_command_threshold": 0.1,
            "joint_names": ACTUATED_JOINT_NAMES,
            "moving_dc_limit": 0.0,
        },
    )
    spectral_left_right_dc_match = RewTerm(
        func=freq_mdp.JointLeftRightDcMatchPenalty,
        weight=-0.5*FREQ_WEIGHT_RATE,
        params={
            "analyzer_cfg": JOINT_FREQUENCY_ANALYZER_CFG,
            "command_name": "base_velocity",
            "k_omega": 0.5,
            "stand_command_threshold": 0.1,
            "dc_joint_pairs": FREQUENCY_JOINT_PAIRS,
            "mirror_signs": FREQUENCY_JOINT_MIRROR_SIGNS,
        },
    )
    spectral_band_energy_encourage = RewTerm(
        func=freq_mdp.JointSpectralBandEnergyReward,
        weight=1.0*FREQ_WEIGHT_RATE,
        params={
            "analyzer_cfg": JOINT_FREQUENCY_ANALYZER_CFG,
            "command_name": "base_velocity",
            "k_omega": 0.5,
            "stand_command_threshold": 0.1,
            "joint_names": ENCOURAGED_FREQUENCY_JOINTS,
            "energy_band_hz": (0.5, 1.5),
            "target_band_energy": 0.08,
        },
    )
    spectral_high_frequency_encourage = RewTerm(
        func=freq_mdp.JointSpectralHighFrequencyPenalty,
        weight=-1.0e-5*FREQ_WEIGHT_RATE,
        params={
            "analyzer_cfg": JOINT_FREQUENCY_ANALYZER_CFG,
            "joint_names": ENCOURAGED_FREQUENCY_JOINTS,
            "alpha_s": 0.01,
            "beta_s": 0.05,
        },
    )
    spectral_energy_disencourage = RewTerm(
        func=freq_mdp.JointSpectralEnergyPenalty,
        weight=-0.5*FREQ_WEIGHT_RATE,
        params={
            "analyzer_cfg": JOINT_FREQUENCY_ANALYZER_CFG,
            "joint_names": DISCOURAGED_FREQUENCY_JOINTS,
        },
    )
    spectral_fundamental_frequency_concentration = RewTerm(
        func=freq_mdp.JointFundamentalConcentrationPenalty,
        weight=-1.0*FREQ_WEIGHT_RATE,
        params={
            "analyzer_cfg": JOINT_FREQUENCY_ANALYZER_CFG,
            "command_name": "base_velocity",
            "k_omega": 0.5,
            "stand_command_threshold": 0.1,
            "fundamental_search_band_hz": (0.5, 1.5),
            "spectrum_energy_floor": 1.0e-4,
            "fundamental_half_width_hz": 0.5,
            "fundamental_joint_names": ENCOURAGED_FREQUENCY_JOINTS,
        },
    )
    spectral_left_right_frequency_energy_match = RewTerm(
        func=freq_mdp.JointLeftRightFrequencyEnergyMatchPenalty,
        weight=-0.5*FREQ_WEIGHT_RATE,
        params={
            "analyzer_cfg": JOINT_FREQUENCY_ANALYZER_CFG,
            "command_name": "base_velocity",
            "k_omega": 0.5,
            "stand_command_threshold": 0.1,
            "fundamental_search_band_hz": (0.5, 1.5),
            "spectrum_energy_floor": 1.0e-4,
            "frequency_joint_pairs": FREQUENCY_JOINT_PAIRS,
            "energy_match_weight": 1.0,
        },
    )
    spectral_left_right_phase = RewTerm(
        func=freq_mdp.JointLeftRightPhasePenalty,
        weight=-0.1*FREQ_WEIGHT_RATE,
        params={
            "analyzer_cfg": JOINT_FREQUENCY_ANALYZER_CFG,
            "command_name": "base_velocity",
            "k_omega": 0.5,
            "stand_command_threshold": 0.1,
            "fundamental_search_band_hz": (0.5, 1.5),
            "spectrum_energy_floor": 1.0e-4,
            "fundamental_half_width_hz": 0.5,
            "fundamental_joint_names": ENCOURAGED_FREQUENCY_JOINTS,
            "phase_joint_pairs": FREQUENCY_JOINT_PAIRS,
            "mirror_signs": FREQUENCY_JOINT_MIRROR_SIGNS,
        },
    )


def compose_reward_cfgs(base_cfg, *additional_cfgs):
    """Compose configclass instances while preserving their declared field order."""

    result = deepcopy(base_cfg)
    for additional_cfg in additional_cfgs:
        for name, value in vars(additional_cfg).items():
            if not name.startswith("_"):
                setattr(result, name, deepcopy(value))
    return result


@configclass
class YMBOY12DOFPerceptionFrequencyRewardsEnvCfg(YMBOY12DOFPerceptionEnvCfgBase):
    """Source-equivalent perception task: base MDP plus frequency rewards."""

    rewards: YMBOY12DOFPerceptionRewardsCfg = compose_reward_cfgs(
        YMBOY12DOFPerceptionRewardsCfg(),
        YMBOY12DOFPerceptionFrequencyOnlyRewardsCfg(),
    )

    def __post_init__(self):
        super().__post_init__()
