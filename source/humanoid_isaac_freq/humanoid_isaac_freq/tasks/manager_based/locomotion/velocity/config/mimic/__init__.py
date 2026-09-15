"""21-DOF frequency-mimic scaffold registration."""
import gymnasium as gym

gym.register(
    id="FreqLab-Velocity-Flat-YMBOY21DOF-FreqMimic-V2",
    entry_point=f"{__name__}.env.manager_based_rl_freq_mimic_env:ManagerBasedRLYMMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ymboy_21dof_envcfg_freq_mimic_v2:YMBOY21DOFFrequencyMimicEnvCfg_v2",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:YMBOY21DOFMimicV2PPORunnerCfg",
    },
)

for suffix, config_name in (("FrequencyFit", "FrequencyFit"), ("NarrowBand", "NarrowBand")):
    gym.register(
        id=f"FreqLab-Velocity-Flat-YMBOY21DOF-FreqMimic-V1-{suffix}",
        entry_point=f"{__name__}.env.manager_based_rl_freq_mimic_env:ManagerBasedRLYMMimicEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ymboy_21dof_envcfg_freq_mimic_v1_tests:YMBOY21DOFMimic{config_name}EnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:YMBOY21DOFMimic{config_name}PPORunnerCfg",
        },
    )

gym.register(
    id="FreqLab-Velocity-Flat-YMBOY21DOF-FreqMimic-V0",
    entry_point=f"{__name__}.env.manager_based_rl_freq_mimic_env:ManagerBasedRLYMMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ymboy_21dof_envcfg_freq_mimic_v0:YMBOY21DOFFrequencyMimicEnvCfg_v0",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:YMBOY21DOFMimicV0PPORunnerCfg",
    },
)

gym.register(
    id="FreqLab-Velocity-Flat-YMBOY21DOF-FreqMimic-V1",
    entry_point=f"{__name__}.env.manager_based_rl_freq_mimic_env:ManagerBasedRLYMMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ymboy_21dof_envcfg_freq_mimic_v1:YMBOY21DOFFrequencyMimicEnvCfg_v1",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:YMBOY21DOFMimicV1PPORunnerCfg",
    },
)

gym.register(
    id="FreqLab-Velocity-Flat-YMBOY21DOF-TimeReward",
    entry_point=f"{__name__}.env.manager_based_rl_freq_mimic_env:ManagerBasedRLYMMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ymboy_21dof_envcfg_freq_mimic_v1:YMBOY21DOFTimeRewardsMimicEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:YMBOY21DOFTimeRewardsMimicPPORunnerCfg",
    },
)
