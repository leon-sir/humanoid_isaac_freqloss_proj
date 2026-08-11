# Copyright (c) 2024-2026 Zheng Pan
# SPDX-License-Identifier: Apache-2.0

"""Common 12-DoF YMBOY task configuration.

This module intentionally contains no periodic timing state or algorithm-specific
auxiliary observation groups.  Reward subclasses can add either time-domain
or frequency-domain joint regularization without changing the task definition.
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from humanoid_isaac_freq.assets.ymbot_boy_12dof import YMBOT_BOY_12DOF_CFG
import humanoid_isaac_freq.tasks.manager_based.locomotion.velocity.mdp as mdp
from humanoid_isaac_freq.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityEnvCfg,
    VelocitySceneCfg,
)


ACTUATED_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]
ANKLE_JOINT_NAMES = [
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]
HIP_PITCH_AND_KNEE_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
]
NON_HIP_PITCH_JOINT_NAMES = [
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
]
YAW_JOINT_NAMES = ["left_hip_yaw_joint", "right_hip_yaw_joint"]
ROBOT_BASE_LINK = "base_link"
ROBOT_FOOT_LINKS = ["left_ankle_roll_link", "right_ankle_roll_link"]
MAX_VEL = 1.0


@configclass
class YMBOYCommandsCfg:
    base_velocity = mdp.UniformThresholdVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=0.5,
        heading_command=True,
        heading_control_stiffness=1.0,
        debug_vis=True,
        ranges=mdp.UniformThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-3.14, 3.14),
        ),
    )


@configclass
class YMBOYActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ACTUATED_JOINT_NAMES,
        scale=0.5,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )


@configclass
class YMBOYObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0 / MAX_VEL,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True)
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True)
            },
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=1.0 / MAX_VEL)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0 / MAX_VEL,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True)
            },
            clip=(-100.0, 100.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True)
            },
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))
        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_FOOT_LINKS)},
            clip=(-100.0, 100.0),
            scale=0.01,
        )
        feet_pos_b = ObsTerm(
            func=mdp.feet_pos_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_FOOT_LINKS)},
            clip=(-100.0, 100.0),
        )
        feet_lin_vel_b = ObsTerm(
            func=mdp.feet_lin_vel_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_FOOT_LINKS)},
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class YMBOYTaskRewardsCfg:
    """Task-level rewards shared by time- and frequency-domain designs."""

    force_too_large = RewTerm(
        func=mdp.force_too_large,
        weight=0.0,  # -1e-2
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=ROBOT_FOOT_LINKS),
            "threshold": 500.0,
            "max_reward": 400.0,
        },
    )
    knee_distance_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_knee_link", "right_knee_link"]),
            "threshold": 0.15,
        },
    )
    feet_distance_y_exp = RewTerm(
        func=mdp.feet_distance_y_square_biped,
        weight=-0.0, # -100
        params={
            "std": math.sqrt(0.25),
            "asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_FOOT_LINKS),
            "stance_width": 0.29,
        },
    )
    feet_distance_too_near = RewTerm(
        func=mdp.feet_distance_y_too_near_humanoid,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_FOOT_LINKS),
            "threshold": 0.1,
        },
    )
    flat_orientation_l2_feet = RewTerm(
        func=mdp.flat_orientation_l2_feet,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_FOOT_LINKS)},
    )
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=ROBOT_FOOT_LINKS),
            "asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_FOOT_LINKS),
        },
    )
    feet_yaw_slide = RewTerm(
        func=mdp.feet_yaw_slide,
        weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=ROBOT_FOOT_LINKS),
            "asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_FOOT_LINKS),
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=ROBOT_FOOT_LINKS)},
    )
    base_vel_z_l2 = RewTerm(
        func=mdp.base_vel_z_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": MAX_VEL / 2.0},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    upward = RewTerm(func=mdp.upward, weight=0.5, params={"std": math.sqrt(0.5)})
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-10.0)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_BASE_LINK),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "target_height": 0.65,
        },
    )


@configclass
class YMBOYEventCfg:
    # randomize_rigid_body_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.6, 1.0),
    #         "dynamic_friction_range": (0.6, 0.8),
    #         "restitution_range": (0.0, 0.2),
    #         "num_buckets": 64,
    #     },
    # )
    randomize_rigid_body_inertia = EventTerm(
        func=mdp.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "inertia_distribution_params": (0.5, 1.5),
            "operation": "scale",
        },
    )
    randomize_com_positions = EventTerm(
        func=mdp.randomize_com_positions,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_BASE_LINK),
            "com_distribution_params": (-0.1, 0.1),
            "operation": "add",
        },
    )
    randomize_joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "armature_distribution_params": [0.5, 2.0],
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    randomize_reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.5, 0.5), "velocity_range": (-2.5, 2.5)},
    )
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.5, 2.0),
            "damping_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": [0.1, 10.0],
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    randomize_rigid_feet_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_FOOT_LINKS),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.6),
            "restitution_range": (0.05, 0.5),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    randomize_rigid_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    randomize_reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (0.0, 0.0),
            },
        },
    )
    randomize_push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "velocity_range": {
                "x": (-0.5, -0.5),
                "y": (-0.5, 0.5),
                "yaw": (-0.75, 0.75),
                "pitch": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
            }
        },
    )


@configclass
class YMBOYTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )
    # heading_too_large = DoneTerm(
    #     func=mdp.heading_error_too_large,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity", "threshold": 1.5},
    # )
    swap_time_too_long = DoneTerm(func=mdp.swap_time_too_long, params={"swap_time_threshold": 200})
    root_height_below_minimum = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.4})
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[".*_hip_yaw_link", ROBOT_BASE_LINK]
            ),
            "threshold": 1.0,
        },
    )
    commanded_no_progress = DoneTerm(
        func=mdp.CommandedNoProgressTermination,
        params={
            "command_name": "base_velocity",
            "window_time": 2.0,
            "min_expected_distance": 0.25,
            "progress_ratio": 0.5,
            "command_threshold": 0.2,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class YMBOYCurriculumCfg:
    terrain_levels: CurrTerm | None = None


@configclass
class YMBOY12DOFEnvCfgBase(LocomotionVelocityEnvCfg):
    scene: VelocitySceneCfg = VelocitySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: YMBOYObservationsCfg = YMBOYObservationsCfg()
    actions: YMBOYActionsCfg = YMBOYActionsCfg()
    commands: YMBOYCommandsCfg = YMBOYCommandsCfg()
    rewards: YMBOYTaskRewardsCfg = YMBOYTaskRewardsCfg()
    terminations: YMBOYTerminationsCfg = YMBOYTerminationsCfg()
    events: YMBOYEventCfg = YMBOYEventCfg()
    curriculum: YMBOYCurriculumCfg = YMBOYCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = YMBOT_BOY_12DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.commands.base_velocity.rel_standing_envs = 0.1


        
