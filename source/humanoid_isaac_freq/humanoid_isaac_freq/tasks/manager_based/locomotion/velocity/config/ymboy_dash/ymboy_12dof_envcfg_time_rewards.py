# Copyright (c) 2024-2026 Zheng Pan
# SPDX-License-Identifier: Apache-2.0

"""Time-domain joint regularization for the 12-DoF YMBOY task."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

import humanoid_isaac_freq.tasks.manager_based.locomotion.velocity.mdp as mdp

from .ymboy_12dof_envcfg_base import (
    ACTUATED_JOINT_NAMES,
    ANKLE_JOINT_NAMES,
    ROBOT_FOOT_LINKS,
    YAW_JOINT_NAMES,
    VelocitySceneCfg,
    YMBOY12DOFEnvCfgBase,
    YMBOYTaskRewardsCfg,
    compose_reward_cfgs,
)


@configclass
class YMBOYTimeRewardsSceneCfg(VelocitySceneCfg):
    """Scene with small foot-local scanners used by ``feet_at_plane``."""

    left_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.04, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.12, size=[0.12, 0.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=0.02,
    )
    right_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.04, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.12, size=[0.12, 0.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=0.02,
    )


@configclass
class YMBOY12DOFTimeOnlyRewardsCfg:
    """Time-domain joint regulation added to the shared task rewards."""

    action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.01)
    ankle_joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ANKLE_JOINT_NAMES)},
    )
    ankle_joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2e-4,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ANKLE_JOINT_NAMES, preserve_order=True)
        },
    )
    ankle_joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=ANKLE_JOINT_NAMES, preserve_order=True),
            "stand_still_scale": 1.0,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )
    ankle_action_rate_l2 = RewTerm(
        func=mdp.policy_action_rate_l2,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ANKLE_JOINT_NAMES, preserve_order=True)
        },
    )
    # feet_at_plane = RewTerm(
    #     func=mdp.feet_at_plane,
    #     weight=-1.0,
    #     params={
    #         "contact_sensor_cfg": SceneEntityCfg(
    #             "contact_forces", body_names=ROBOT_FOOT_LINKS, preserve_order=True
    #         ),
    #         "left_height_scanner_cfg": SceneEntityCfg("left_height_scanner"),
    #         "right_height_scanner_cfg": SceneEntityCfg("right_height_scanner"),
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", body_names=ROBOT_FOOT_LINKS, preserve_order=True
    #         ),
    #         "height_offset": 0.04,
    #     },
    # )
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    # non_hip_pitch_action_rate_l2 = RewTerm(
    #     func=mdp.policy_action_rate_l2,
    #     weight=-0.02,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", joint_names=NON_HIP_PITCH_JOINT_NAMES, preserve_order=True
    #         )
    #     },
    # )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty,
        weight=-0.1,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_power = RewTerm(
        func=mdp.joint_power,
        weight=-1e-5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True
            )
        },
    )
    yaw_joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty_zero,
        weight=-5.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=YAW_JOINT_NAMES, preserve_order=True),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )


@configclass
class YMBOY12DOFTimeRewardsEnvCfg(YMBOY12DOFEnvCfgBase):
    scene: YMBOYTimeRewardsSceneCfg = YMBOYTimeRewardsSceneCfg(num_envs=4096, env_spacing=2.5)
    rewards: YMBOYTaskRewardsCfg = compose_reward_cfgs(
        YMBOYTaskRewardsCfg(),
        YMBOY12DOFTimeOnlyRewardsCfg(),
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.left_height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.right_height_scanner.update_period = self.decimation * self.sim.dt

        # The deployable time-reward policy does not consume terrain scans.  Keep the
        # scanner definition in the base so future frequency configs can enable it.
        if self.__class__.__name__ == "YMBOY12DOFTimeRewardsEnvCfg":
            self.scene.left_height_scanner = None
            self.scene.right_height_scanner = None
            self.scene.height_scanner = None
            self.observations.policy.height_scan = None
            self.observations.critic.height_scan = None
            self.disable_zero_weight_rewards()
