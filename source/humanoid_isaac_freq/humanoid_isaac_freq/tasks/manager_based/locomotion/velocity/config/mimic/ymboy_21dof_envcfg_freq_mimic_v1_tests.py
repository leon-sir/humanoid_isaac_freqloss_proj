"""Controlled V1 continuation experiments: only command staging is retained."""

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

from .mdp.frequency_fit import CoreFrequencyRangePenalty
from .mdp.terminations import SwapTimeTooShort
from .ymboy_21dof_envcfg_freq_mimic_common import CORE_FREQUENCY_JOINTS
from .ymboy_21dof_envcfg_freq_mimic_v1 import YMBOY21DOFFrequencyMimicEnvCfg_v1


@configclass
class YMBOY21DOFMimicContinuationEnvCfg(YMBOY21DOFFrequencyMimicEnvCfg_v1):
    def __post_init__(self):
        super().__post_init__()
        # Keep first-stage joint regulation, even when --resume loads a checkpoint.
        self.rewards.spectral_kinematic_chain_phase_match = None
        self.rewards.ankle_joint_pos_limits = None
        self.terminations.swap_time_too_short = DoneTerm(
            func=SwapTimeTooShort,
            params={'swap_time_threshold': 20},  # control steps: 20 * 0.02 = 0.4 s
        )


@configclass
class YMBOY21DOFMimicFrequencyFitEnvCfg(YMBOY21DOFMimicContinuationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.joint_frequency_analyzer.window_duration_s = 2.0
        self.rewards.spectral_core_frequency_range = RewTerm(
            func=CoreFrequencyRangePenalty, weight=-1,
            params={
                'joint_names': CORE_FREQUENCY_JOINTS, 'command_name': 'base_velocity',
                'search_band_hz': (0.4, 2.5), 'search_step_hz': 0.025,
                'allowed_band_hz': (0.8, 1.2), 'sigma_hz': 0.5,
                'min_reference_energy_ratio': 0.05, 'energy_floor': 1.e-4,
            },
        )


@configclass
class YMBOY21DOFMimicNarrowBandEnvCfg(YMBOY21DOFMimicContinuationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.joint_frequency_analyzer.window_duration_s = 4.0
        self.rewards.spectral_fundamental_energy_match_core.params['energy_band_hz'] = (0.75, 1.0)
