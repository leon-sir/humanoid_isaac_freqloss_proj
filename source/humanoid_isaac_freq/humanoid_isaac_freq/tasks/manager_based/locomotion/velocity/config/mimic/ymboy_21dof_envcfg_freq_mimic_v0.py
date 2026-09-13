# Copyright (c) 2024-2026 Zheng Pan
# SPDX-License-Identifier: Apache-2.0

"""Frozen v0 frequency-mimic MDP from the 2026-09-12 16:20 run."""

from copy import deepcopy
from pathlib import Path

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from humanoid_isaac_freq.tasks.manager_based.locomotion.velocity.mdp.freq_rewards import (
    JointFrequencyAnalyzerCfg,
)

from . import mdp as mimic_mdp
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
)

REFERENCE_MOTION = "Neutral_walk_forward_002__A057"
REFERENCE_MOTION_FILE = f"{REFERENCE_MOTION}/{REFERENCE_MOTION}_periodic_filtered.csv"
REFERENCE_HARMONIC_COUNT = 5

_reference_path = Path(REFERENCE_MOTION_FILE).expanduser()
if not _reference_path.is_absolute():
    _reference_path = Path(__file__).resolve().parents[6] / "datasets" / "motions" / _reference_path

# Natural-harmonic statistics are unchanged from the archived run.  The exact-f0
# DC/cos/sin fit was changed later by symmetry processing and is frozen below.
REFERENCE_V0 = load_reference_motion(_reference_path, harmonic_count=REFERENCE_HARMONIC_COUNT)
REFERENCE_V0["source_sha256"] = "d7fcf6f158acd709ba0169429b609c7b4b5f0f74cf6337172024c73721f0ce44"
REFERENCE_V0["fit_max_error_rad"] = 9.9298347322474e-13
REFERENCE_V0["joints"] = {
    "left_hip_pitch_joint": [-0.15760644366809118, -0.12984203482409182, -0.31976803559027045],
    "left_hip_roll_joint": [0.0674485242485703, 0.175715268632939, 0.06588716051667912],
    "left_hip_yaw_joint": [0.32983564797010834, 0.04066051339873193, 0.08076565871707134],
    "left_knee_joint": [0.5250385175379095, 0.3067744750784678, -0.1714526930171342],
    "left_ankle_pitch_joint": [-0.22124974673680042, 0.12454662217179671, 0.03770302065037545],
    "left_ankle_roll_joint": [-0.006730647570343051, -0.12375384212979396, -0.03691305569601516],
    "right_hip_pitch_joint": [-0.20514119566734504, 0.1411255440706834, 0.28040456749882914],
    "right_hip_roll_joint": [0.01761782233172045, 0.16550549369959672, 0.012822759434861379],
    "right_hip_yaw_joint": [-0.04860344387791409, 0.09098340306129826, 0.09337591579967545],
    "right_knee_joint": [0.5302091389095105, -0.3220857294339152, 0.2530033994188577],
    "right_ankle_pitch_joint": [-0.15385517173330057, -0.08505998035463339, -0.04417987095185428],
    "right_ankle_roll_joint": [0.0359475127179261, -0.07564646960128459, -0.030528122069901364],
    "waist_yaw_joint": [0.0102935085471973, 0.05138896769249185, 0.21734148648037777],
    "left_shoulder_pitch_joint": [0.06480777070526827, 0.08170272928580008, 0.3554142551539558],
    "left_shoulder_roll_joint": [0.47478953585301487, 0.005998675042939978, -0.05428458553275102],
    "left_shoulder_yaw_joint": [-0.4747930234186191, -0.10531385075392943, -0.13458277407761482],
    "left_elbow_joint": [0.8925963807140729, 0.1758071872167656, 0.16936968097376737],
    "right_shoulder_pitch_joint": [0.024639009144292696, -0.12444296560429574, -0.3826031359954478],
    "right_shoulder_roll_joint": [-0.5159729055198368, 0.00988260708413306, -0.04965704518450599],
    "right_shoulder_yaw_joint": [0.6293184949565758, -0.08980524384791405, -0.09401365815023613],
    "right_elbow_joint": [0.7549470055271379, -0.14034796245237152, -0.22440738037103194],
}

BILATERAL_PHASE_PAIRS = [
    (name, "right_" + name[5:])
    for name in ALL_FREQUENCY_JOINTS
    if name.startswith("left_")
]


@configclass
class YMBOY21DOFFrequencyOnlyRewardsCfg_v0:
    """Frequency terms exactly matching the archived v0 environment YAML."""

    spectral_fundamental_energy_match_core = RewTerm(
        func=mimic_mdp.FundamentalEnergyMatch_v2,
        weight=-4.0,
        params={"joint_names": CORE_FREQUENCY_JOINTS, "amplitude_floor": 0.1},
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
    spectral_bilateral_phase_match = RewTerm(
        func=mimic_mdp.BilateralPhaseMatch,
        weight=0.5,
        params={"joint_pairs": BILATERAL_PHASE_PAIRS, "amplitude_floor": 0.02},
    )
    spectral_cross_limb_phase_match = RewTerm(
        func=mimic_mdp.CrossLimbPhaseMatch,
        weight=0.25,
        params={"joint_pairs": CROSS_LIMB_PHASE_PAIRS, "amplitude_floor": 0.02},
    )
    spectral_non_harmonic_energy = RewTerm(
        func=mimic_mdp.NonHarmonicEnergyPenalty,
        weight=-0.2,
        params={"joint_names": ALL_FREQUENCY_JOINTS, "energy_floor": 0.02, "allowance": 0.01},
    )


@configclass
class YMBOY21DOFFrequencyMimicEnvCfg_v0(YMBOY21DOFFrequencyMimicEnvCfgBase):
    """Reproduce the MDP and reference targets of the 2026-09-12 16:20 run."""

    reference_motion_file: str = str(_reference_path.resolve())
    reference_motion: dict = deepcopy(REFERENCE_V0)
    mimic_band_half_width_hz: float = 0.5
    mimic_harmonic_count: int = REFERENCE_HARMONIC_COUNT
    rewards: YMBOYTaskRewardsCfg = compose_reward_cfgs(
        YMBOYTaskRewardsCfg(),
        YMBOY21DOFTimeOnlyRewardsCfg(),
        YMBOY21DOFFrequencyOnlyRewardsCfg_v0(),
    )
    joint_frequency_analyzer: JointFrequencyAnalyzerCfg = JointFrequencyAnalyzerCfg(
        analyzer_key="mimic_joints",
        asset_cfg=SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True),
        joint_scales=FREQUENCY_JOINT_SCALES,
        window_duration_s=2.0,
        fft_update_interval=5,
    )
