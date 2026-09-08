# Copyright (c) 2026 Zheng Pan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents, env

gym.register(
    id="FreqLab-Velocity-Flat-YMBOY12DOF-TimeReward",
    entry_point=f"{env.__name__}.manager_based_rl_env:ManagerBasedRLYMEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ymboy_12dof_envcfg_time_rewards:YMBOY12DOFTimeRewardsEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:YMBOY12DOFFlatPPORunnerCfg",
    },
)


gym.register(
    id="FreqLab-Velocity-Flat-YMBOY12DOF-FreqReward",
    entry_point=f"{env.__name__}.manager_based_rl_freq_env:ManagerBasedRLYMFreqEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ymboy_12dof_envcfg_freq_rewards:YMBOY12DOFFrequencyRewardsEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:YMBOY12DOFFreqRewardPPORunnerCfg"
        ),
    },
)


# Register the single-frequency-term ablation tasks after the full task.
from . import ablation_1_register  # noqa: E402, F401
