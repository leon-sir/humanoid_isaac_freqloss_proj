# Copyright (c) 2026 Zheng Pan
# SPDX-License-Identifier: Apache-2.0

"""Gym registrations for single-term frequency-reward ablations."""

import gymnasium as gym

from . import agents, env

_ABLATION_TASKS = {
    "FreqLab-Velocity-Flat-YMBOY12DOF-FreqReward-Ablation-NoDcDefaultPosture": (
        "YMBOY12DOFFrequencyRewardsNoDcDefaultPostureEnvCfg",
        "YMBOY12DOFFreqAblationNoDcDefaultPosturePPORunnerCfg",
    ),
    "FreqLab-Velocity-Flat-YMBOY12DOF-FreqReward-Ablation-NoBandEnergyEncourage": (
        "YMBOY12DOFFrequencyRewardsNoBandEnergyEncourageEnvCfg",
        "YMBOY12DOFFreqAblationNoBandEnergyEncouragePPORunnerCfg",
    ),
    "FreqLab-Velocity-Flat-YMBOY12DOF-FreqReward-Ablation-NoHighFrequencyEncourage": (
        "YMBOY12DOFFrequencyRewardsNoHighFrequencyEncourageEnvCfg",
        "YMBOY12DOFFreqAblationNoHighFrequencyEncouragePPORunnerCfg",
    ),
    "FreqLab-Velocity-Flat-YMBOY12DOF-FreqReward-Ablation-NoEnergyDisencourage": (
        "YMBOY12DOFFrequencyRewardsNoEnergyDisencourageEnvCfg",
        "YMBOY12DOFFreqAblationNoEnergyDisencouragePPORunnerCfg",
    ),
    "FreqLab-Velocity-Flat-YMBOY12DOF-FreqReward-Ablation-NoFundamentalConcentration": (
        "YMBOY12DOFFrequencyRewardsNoFundamentalConcentrationEnvCfg",
        "YMBOY12DOFFreqAblationNoFundamentalConcentrationPPORunnerCfg",
    ),
    "FreqLab-Velocity-Flat-YMBOY12DOF-FreqReward-Ablation-NoLeftRightFrequencyEnergyMatch": (
        "YMBOY12DOFFrequencyRewardsNoLeftRightFrequencyEnergyMatchEnvCfg",
        "YMBOY12DOFFreqAblationNoLeftRightFrequencyEnergyMatchPPORunnerCfg",
    ),
    "FreqLab-Velocity-Flat-YMBOY12DOF-FreqReward-Ablation-NoLeftRightPhase": (
        "YMBOY12DOFFrequencyRewardsNoLeftRightPhaseEnvCfg",
        "YMBOY12DOFFreqAblationNoLeftRightPhasePPORunnerCfg",
    ),
}


for task_id, (env_cfg_class_name, runner_cfg_class_name) in _ABLATION_TASKS.items():
    gym.register(
        id=task_id,
        entry_point=f"{env.__name__}.manager_based_rl_freq_env:ManagerBasedRLYMFreqEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__package__}.ymboy_12dof_envcfg_freq_rewards:{env_cfg_class_name}"
            ),
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:{runner_cfg_class_name}"
            ),
        },
    )
