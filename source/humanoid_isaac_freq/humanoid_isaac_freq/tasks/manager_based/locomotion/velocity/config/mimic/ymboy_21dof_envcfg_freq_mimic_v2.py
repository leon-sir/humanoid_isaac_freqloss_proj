"""V2: four-second narrow-band + frequency-fit continuation with chain matching."""

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

from .mdp.freq_rewards import KinematicChainPhaseMatch
from .mdp.frequency_fit import CoreFrequencyRangePenalty
from .mdp.terminations import SwapTimeTooShort
from .ymboy_21dof_envcfg_freq_mimic_common import CORE_FREQUENCY_JOINTS
from .ymboy_21dof_envcfg_freq_mimic_v1 import (
    YMBOY21DOFFrequencyMimicEnvCfg_v1,
    KINEMATIC_CHAIN_PHASE_PAIRS,
)


@configclass
class YMBOY21DOFFrequencyMimicEnvCfg_v2(YMBOY21DOFFrequencyMimicEnvCfg_v1):
    """Resume a V1 policy; keep its weak non-core regulation and command staging."""

    def __post_init__(self):
        super().__post_init__()
        self.joint_frequency_analyzer.window_duration_s = 4.0
        self.rewards.spectral_fundamental_energy_match_core.params['energy_band_hz'] = (0.75, 1.0)
        self.rewards.ankle_joint_pos_limits = None
        self.terminations.swap_time_too_short = DoneTerm(
            func=SwapTimeTooShort, params={'swap_time_threshold': 20},
        )
        self.rewards.spectral_core_frequency_range = RewTerm(
            func=CoreFrequencyRangePenalty, weight=-1.0,
            params={
                'joint_names': CORE_FREQUENCY_JOINTS, 'command_name': 'base_velocity',
                'search_band_hz': (0.4, 2.5), 'search_step_hz': 0.025,
                'allowed_band_hz': (0.8, 1.2), 'sigma_hz': 0.5,
                'min_reference_energy_ratio': 0.05, 'energy_floor': 1.e-4,
            },
        )
        # Fixed reference f0 (002: 0.9375 Hz), NOT the policy's fitted core rhythm.
        # Alternative experiment: measure chain phase at the estimated core frequency.
        self.rewards.spectral_kinematic_chain_phase_match = RewTerm(
            func=KinematicChainPhaseMatch, weight=0.1,
            params={'joint_pairs': KINEMATIC_CHAIN_PHASE_PAIRS, 'amplitude_floor': 0.02},
        )
