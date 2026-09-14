# Copyright (c) 2024-2026 Zheng Pan
# SPDX-License-Identifier: Apache-2.0

"""Current v1 21-DOF frequency-mimic environment configuration."""

from copy import deepcopy
from pathlib import Path

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from humanoid_isaac_freq.tasks.manager_based.locomotion.velocity.mdp.freq_rewards import (
    JointFrequencyAnalyzerCfg,
)

from . import mdp as mimic_mdp
from humanoid_isaac_freq.tasks.manager_based.locomotion.velocity.mdp.events import reset_root_state_uniform_body_frame
from .reference_motion import load_reference_motion
from .ymboy_21dof_envcfg_freq_mimic_common import (
    ACTUATED_JOINT_NAMES,
    ALL_FREQUENCY_JOINTS,
    CORE_FREQUENCY_JOINTS,
    CROSS_LIMB_PHASE_PAIRS,
    FREQUENCY_JOINT_SCALES,
    OTHER_FREQUENCY_JOINTS,
    YMBOY21DOFFrequencyMimicEnvCfgBase,
    YMBOY21DOFTimeOnlyRewardsCfg,
    YMBOYTaskRewardsCfg,
    compose_reward_cfgs,
    firstStage,
)

# Change this one name to switch all v1 reference targets (for example 002 -> 003).
REFERENCE_MOTION = "Neutral_walk_forward_002__A057"
REFERENCE_MOTION_FILE = f"{REFERENCE_MOTION}/{REFERENCE_MOTION}_periodic_filtered.csv"
REFERENCE_HARMONIC_COUNT = 5
CORE_ENERGY_BAND_HZ = (0.5, 1.5)
CORE_REFERENCE_ENERGY_RATIO = 0.25

_reference_path = Path(REFERENCE_MOTION_FILE).expanduser()
if not _reference_path.is_absolute():
    _reference_path = Path(__file__).resolve().parents[6] / "datasets" / "motions" / _reference_path
REFERENCE = load_reference_motion(_reference_path, harmonic_count=REFERENCE_HARMONIC_COUNT)

# Directed sequential edges within the two leg and two arm chains.
KINEMATIC_CHAIN_PHASE_PAIRS = [
    ("left_hip_pitch_joint", "left_knee_joint"),
    ("left_knee_joint", "left_ankle_pitch_joint"),
    ("right_hip_pitch_joint", "right_knee_joint"),
    ("right_knee_joint", "right_ankle_pitch_joint"),
    ("left_shoulder_pitch_joint", "left_elbow_joint"),
    ("right_shoulder_pitch_joint", "right_elbow_joint"),
]
CORE_BILATERAL_PHASE_PAIRS = [
    (name, "right_" + name[5:])
    for name in CORE_FREQUENCY_JOINTS
    if name.startswith("left_")
]
OTHER_BILATERAL_PHASE_PAIRS = [
    (name, "right_" + name[5:])
    for name in OTHER_FREQUENCY_JOINTS
    if name.startswith("left_")
]


@configclass
class YMBOY21DOFFrequencyOnlyRewardsCfg_v1:
    """Current frequency MDP with core bilateral and limb-chain phase terms."""

    spectral_fundamental_energy_match_core = RewTerm(
        func=mimic_mdp.ReferenceBandEnergyReward,
        weight=2.0,
        params={"joint_names": CORE_FREQUENCY_JOINTS,
                "energy_band_hz": CORE_ENERGY_BAND_HZ,
                "reference_energy_ratio": CORE_REFERENCE_ENERGY_RATIO,
                "command_name": "base_velocity", "energy_floor": 1.e-4},
    )
    spectral_fundamental_energy_match_other = RewTerm(
        func=mimic_mdp.FundamentalEnergyMatch_v2,
        weight=-0.5,
        params={"joint_names": OTHER_FREQUENCY_JOINTS, "amplitude_floor": 0.1},
    )
    spectral_reference_dc_match_core = RewTerm(
        func=mimic_mdp.ReferenceDcMatch,
        weight=-2.0,
        params={"joint_names": CORE_FREQUENCY_JOINTS, "moving_dc_limit": 0.0},
    )
    spectral_reference_dc_match_other = RewTerm(
        func=mimic_mdp.ReferenceDcMatch,
        weight=-0.5,
        params={"joint_names": OTHER_FREQUENCY_JOINTS, "moving_dc_limit": 0.0},
    )
    spectral_bilateral_phase_match_core = RewTerm(
        func=mimic_mdp.BilateralPhaseMatch,
        weight=0.5,
        params={"joint_pairs": CORE_BILATERAL_PHASE_PAIRS, "amplitude_floor": 0.02},
    )
    # Kept available for ablations; chain phase currently replaces this weak group.
    # spectral_bilateral_phase_match_other = RewTerm(
    #     func=mimic_mdp.BilateralPhaseMatch,
    #     weight=0.1,
    #     params={"joint_pairs": OTHER_BILATERAL_PHASE_PAIRS, "amplitude_floor": 0.02},
    # )
    spectral_cross_limb_phase_match = RewTerm(
        func=mimic_mdp.CrossLimbPhaseMatch,
        weight=0.25,
        params={"joint_pairs": CROSS_LIMB_PHASE_PAIRS, "amplitude_floor": 0.02},
    )
    # Stage 1 retains weak other-joint amplitude/DC regulation. Only detailed
    # chain timing is deferred until --resume selects stage 2.
    spectral_kinematic_chain_phase_match = None
    if not firstStage:
        spectral_kinematic_chain_phase_match = RewTerm(
            func=mimic_mdp.KinematicChainPhaseMatch,
            weight=0.5,
            params={"joint_pairs": KINEMATIC_CHAIN_PHASE_PAIRS, "amplitude_floor": 0.02},
        )
    spectral_non_harmonic_energy_core = RewTerm(
        func=mimic_mdp.NonHarmonicEnergyPenalty,
        weight=-0.2,
        params={"joint_names": CORE_FREQUENCY_JOINTS, "energy_floor": 0.02,
                "allowance": 0.01, "harmonic_count": 2, "use_reference_residual": False},
    )
    spectral_non_harmonic_energy_other = RewTerm(
        func=mimic_mdp.NonHarmonicEnergyPenalty,
        weight=-0.2,
        params={"joint_names": OTHER_FREQUENCY_JOINTS, "energy_floor": 0.02,
                "allowance": 0.01, "harmonic_count": 5, "use_reference_residual": True},
    )


@configclass
class YMBOY21DOFFrequencyMimicEnvCfg_v1(YMBOY21DOFFrequencyMimicEnvCfgBase):
    """Two-second core rhythm training, with chain timing added on resume."""

    mimic_training_stage: int = 1 if firstStage else 2

    reference_motion_file: str = str(_reference_path.resolve())
    reference_motion: dict = deepcopy(REFERENCE)
    mimic_band_half_width_hz: float = 0.5
    mimic_harmonic_count: int = REFERENCE_HARMONIC_COUNT
    rewards: YMBOYTaskRewardsCfg = compose_reward_cfgs(
        YMBOYTaskRewardsCfg(),
        YMBOY21DOFTimeOnlyRewardsCfg(),
        YMBOY21DOFFrequencyOnlyRewardsCfg_v1(),
    )
    joint_frequency_analyzer: JointFrequencyAnalyzerCfg = JointFrequencyAnalyzerCfg(
        analyzer_key="mimic_joints",
        asset_cfg=SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True),
        joint_scales=FREQUENCY_JOINT_SCALES,
        window_duration_s=2.0,
        fft_update_interval=5,
    )


    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_reset_base.func = reset_root_state_uniform_body_frame
        self.events.randomize_reset_base.params["velocity_range"]["x"] = (0.4, 0.8)


@configclass
class YMBOY21DOFFrequencyMimicEnvCfg(YMBOY21DOFFrequencyMimicEnvCfg_v1):
    """Backward-compatible alias for v1; prefer the explicit versioned class."""

    pass


@configclass
class YMBOY21DOFTimeRewardsMimicEnvCfg(YMBOY21DOFFrequencyMimicEnvCfg_v1):
    """Matched baseline: same environment and FFT, only task + time rewards."""

    rewards: YMBOYTaskRewardsCfg = compose_reward_cfgs(
        YMBOYTaskRewardsCfg(), YMBOY21DOFTimeOnlyRewardsCfg()
    )
