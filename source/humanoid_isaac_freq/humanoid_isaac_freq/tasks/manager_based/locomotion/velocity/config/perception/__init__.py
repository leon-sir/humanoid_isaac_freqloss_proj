"""Gym registrations for the 12-DoF depth-perception tasks."""

import gymnasium as gym

from . import agents, envs


BASE_TASK_ID = "DreamLab-Velocity-Rough-YMBOY12DOF-Perception-RNN-Base-v0"
FREQ_REWARD_TASK_ID = "DreamLab-Velocity-Rough-YMBOY12DOF-Perception-RNN-FreqReward-v0"

gym.register(
    id=BASE_TASK_ID,
    entry_point=f"{envs.__name__}.manager_based_rl_env:ManagerBasedRLYMEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ymboy_12dof_perception_envcfg_base:"
            "YMBOY12DOFPerceptionEnvCfgBase"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_perception_ppo_cfg:"
            "YMBOY12DOFPerceptionRNNBasePPORunnerCfg"
        ),
    },
)

gym.register(
    id=FREQ_REWARD_TASK_ID,
    entry_point=f"{envs.__name__}.manager_based_rl_freq_env:ManagerBasedRLYMFreqEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ymboy_12dof_perception_envcfg_freq_rewards:"
            "YMBOY12DOFPerceptionFrequencyRewardsEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_perception_ppo_cfg:"
            "YMBOY12DOFPerceptionRNNFreqRewardPPORunnerCfg"
        ),
    },
)

__all__ = ["BASE_TASK_ID", "FREQ_REWARD_TASK_ID"]
