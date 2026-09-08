"""Self-contained base configuration for 12-DoF YMBOY depth locomotion."""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from humanoid_isaac_freq.sensors import NoisyRayCasterCameraCfg
from humanoid_isaac_freq.utils.noise import (
    CropAndResizeCfg,
    DepthNormalizationCfg,
    GaussianBlurNoiseCfg,
)
from humanoid_isaac_freq.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityEnvCfg,
)

from . import mdp
from .mdp.base_height_anomaly_recorder import BaseHeightAnomalyRecorderManagerCfg
from .robot_cfg import YMBOT_BOY_URDF_12DOF_NOARM_CFG
from .terrain_cfg import PERCEPTION_ROUGH_TERRAINS_CFG


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
ROBOT_BASE_LINK = "base_link"
ROBOT_FOOT_LINKS = ["left_ankle_roll_link", "right_ankle_roll_link"]
MAX_VEL = 1.0
ROBOT_FOOT_LINK = ["left_ankle_roll_link", "right_ankle_roll_link"]
DEPTH_RAW_HEIGHT = 36
DEPTH_RAW_WIDTH = 64
DEPTH_HEIGHT = 18
DEPTH_WIDTH = 32


@configclass
class YMBOY12DOFPerceptionSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = YMBOT_BOY_URDF_12DOF_NOARM_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    future_robot: ArticulationCfg | None = None
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=PERCEPTION_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            compliant_contact_stiffness=1e6,
            compliant_contact_damping=1e5,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.4, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.8, 0.5]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    height_scanner_base = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=(
                f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/"
                "kloofendal_43d_clear_puresky_4k.hdr"
            ),
        ),
    )
    camera = NoisyRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=PinholeCameraPatternCfg(
            focal_length=1.0,
            horizontal_aperture=2 * math.tan(math.radians(89.51) / 2),
            vertical_aperture=2 * math.tan(math.radians(58.29) / 2),
            width=DEPTH_RAW_WIDTH,
            height=DEPTH_RAW_HEIGHT,
        ),
        debug_vis=False,
        attach_yaw_only=False,
        data_types=["distance_to_image_plane"],
        update_period=0.02,
        depth_clipping_behavior="max",
        offset=NoisyRayCasterCameraCfg.OffsetCfg(
            pos=(0.0807988662332928, 0.01, 0.3638029937970051),
            rot=(
                0.8869950945007488,
                0.0052839859131982,
                0.4617404201981222,
                -0.0027506689619906,
            ),
            convention="world",
        ),
        noise_pipeline={
            "crop_and_resize": CropAndResizeCfg(crop_region=(18, 0, 16, 16)),
            "gaussian_blur": GaussianBlurNoiseCfg(kernel_size=3, sigma=1.0),
            "depth_normalization": DepthNormalizationCfg(
                depth_range=(0.0, 2.5), normalize=True, output_range=(0.0, 1.0)
            ),
        },
        data_histories={},
    )


@configclass
class YMBOY12DOFPerceptionObservationsCfg:
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
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True
                )
            },
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        gait_phase = None
        height_scan = None

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=0.25)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True
                )
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True
                )
            },
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        gait_phase = None
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
            scale=1.0,
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
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class DepthImageCfg(ObsGroup):
        depth_image = ObsTerm(
            func=mdp.visualizable_image,
            params={
                "data_type": "distance_to_image_plane_noised",
                "sensor_cfg": SceneEntityCfg("camera"),
                "debug_vis": False,
            },
            noise=None,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    depth_image: DepthImageCfg = DepthImageCfg()


@configclass
class YMBOY12DOFPerceptionActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ACTUATED_JOINT_NAMES,
        scale=0.5,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )


@configclass
class YMBOY12DOFPerceptionCommandsCfg:
    base_velocity = mdp.UniformThresholdVelocityCommandCfg(
        asset_name="robot",
        heading_asset_cfg=None,
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=1,
        debug_vis=True,
        lane_keeping_command=True,
        rel_lane_keeping_envs=1.0,
        lane_keeping_heading_target=0.0,
        lane_heading_gain=1.2,
        lane_lateral_gain=0.8,
        ranges=mdp.UniformThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-0.0, 0.0),
        ),
    )


@configclass
class YMBOY12DOFPerceptionRewardsCfg:
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-200)
    lin_vel_z_l2 = None
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5)
    flat_orientation_l2_feet = RewTerm(
        func=mdp.flat_orientation_l2_feet,
        weight=-2.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_FOOT_LINK),
        },
    )
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_BASE_LINK),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "target_height": 0.65,
        },
    )
    body_lin_acc_l2 = None
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_vel_l2 = None
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_vel_limits = None
    joint_power = RewTerm(
        func=mdp.joint_power,
        weight=-1e-5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True
            )
        },
    )
    stand_still_without_cmd = RewTerm(
        func=mdp.stand_still_without_cmd_v2,
        weight=-2,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "use_zeros_pos": True,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True
            ),
        },
    )
    joint_pos_penalty = None
    wheel_vel_penalty = None
    joint_mirror = None
    applied_torque_limits = None
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.01)
    undesired_contacts = None
    contact_forces = None
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=5.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=5.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    feet_air_time = None
    feet_gait = None
    feet_contact = None
    feet_contact_without_cmd = None
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=ROBOT_FOOT_LINKS)},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=ROBOT_FOOT_LINKS),
            "asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_FOOT_LINKS),
        },
    )
    feet_height = None
    feet_height_body = None
    feet_distance_y_exp = RewTerm(
        func=mdp.feet_distance_y_square_biped,
        weight=-100,
        params={
            "std": 0.5,
            "asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_FOOT_LINKS),
            "stance_width": 0.25,
        },
    )
    upward = RewTerm(func=mdp.upward, weight=1, params={"std": math.sqrt(0.5)})
    heading_error_l2 = None
    force_too_large = RewTerm(
        func=mdp.force_too_large,
        weight=-2e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=ROBOT_FOOT_LINKS),
            "threshold": 500.0,
            "max_reward": 400,
        },
    )
    knee_distance_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-10,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_knee_link", "right_knee_link"]
            ),
            "threshold": 0.15,
        },
    )
    feet_distance_too_near = RewTerm(
        func=mdp.feet_distance_y_too_near_humanoid,
        weight=-10,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_FOOT_LINKS),
            "threshold": 0.1,
        },
    )
    gait_flight_frc_penalty = None
    gait_stance_spd_penalty = None
    gait_stance_force_reward = None
    gait_flight_speed_reward = None
    flat_orientation_l2_feet = None
    joint_pos_zero_penalty_knee = None
    joint_pos_zero_penalty_yaw_joint = RewTerm(
        func=mdp.joint_pos_penalty_zero,
        weight=-5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["left_hip_yaw_joint", "right_hip_yaw_joint"],
                preserve_order=True,
            ),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )
    is_alive = None
    base_vel_z_l2 = RewTerm(
        func=mdp.base_vel_z_l2,
        weight=-1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    encourage_action_rate_l2_ = None
    encourage_joint_torques_l2_ = None
    encourage_joint_pos_default_penalty_ = None
    encourage_joint_acc_l2_ = None
    disencourage_action_rate_l2_ = None
    disencourage_joint_torques_l2_ = None
    disencourage_joint_pos_zero_penalty_ = None
    disencourage_joint_acc_l2_ = None
    joint_pos_penalty_ = RewTerm(
        func=mdp.joint_pos_penalty,
        weight=-0.1,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )
    ankle_action_ = RewTerm(
        func=mdp.policy_action_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ANKLE_JOINT_NAMES)},
    )
    feet_yaw_slide = RewTerm(
        func=mdp.feet_yaw_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=ROBOT_FOOT_LINKS),
            "asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_FOOT_LINKS),
        },
    )
    adaptive_feet_height_body = None


@configclass
class YMBOY12DOFPerceptionEventsCfg:
    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.6, 0.8),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
        },
    )
    randomize_rigid_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_BASE_LINK),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
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
    randomize_reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (0, 0),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    randomize_push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (0, 0)}
        },
    )
    randomize_rigid_feet_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_FOOT_LINKS),
            "static_friction_range": (0.7, 1.2),
            "dynamic_friction_range": (0.5, 0.8),
            "restitution_range": (0.0, 0.005),
            "num_buckets": 64,
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
    randomize_each_rigid_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    randomize_push_robot_disturber = None


@configclass
class YMBOY12DOFPerceptionTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[".*_hip_yaw_link", ROBOT_BASE_LINK]
            ),
            "threshold": 1.0,
        },
    )
    heading_too_large = DoneTerm(
        func=mdp.heading_error_too_large,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "threshold": 1.5,
        },
    )
    lateralerror_too_large = DoneTerm(
        func=mdp.lateral_error_too_large,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": 3.5,
        },
        time_out=True,
    )
    swap_time_too_long = DoneTerm(
        func=mdp.swap_time_too_long, params={"swap_time_threshold": 200}
    )
    root_height_below_minimum = None
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
class YMBOY12DOFPerceptionCurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    fine_tunning = None
    check_base_height = CurrTerm(
        func=mdp.check_base_height,
        params={"sensor_cfg": SceneEntityCfg("height_scanner_base")},
    )
    command_levels = None


@configclass
class YMBOY12DOFPerceptionEnvCfgBase(LocomotionVelocityEnvCfg):
    """12-DoF perception baseline without frequency-domain reward terms."""

    # Retained as inert source-compatible metadata so serialized env configs
    # stay structurally identical. No gait-phase observation or reward uses it.
    gait_base_phase_period: float = 1.0
    gait_phase_period_variation: float = 0.3
    gait_phase_offset: float = 0.5
    gait_phase_kappa: float = 0.04

    scene: YMBOY12DOFPerceptionSceneCfg = YMBOY12DOFPerceptionSceneCfg(
        num_envs=4096, env_spacing=2.5
    )
    observations: YMBOY12DOFPerceptionObservationsCfg = (
        YMBOY12DOFPerceptionObservationsCfg()
    )
    actions: YMBOY12DOFPerceptionActionsCfg = YMBOY12DOFPerceptionActionsCfg()
    commands: YMBOY12DOFPerceptionCommandsCfg = YMBOY12DOFPerceptionCommandsCfg()
    rewards: YMBOY12DOFPerceptionRewardsCfg = YMBOY12DOFPerceptionRewardsCfg()
    terminations: YMBOY12DOFPerceptionTerminationsCfg = (
        YMBOY12DOFPerceptionTerminationsCfg()
    )
    events: YMBOY12DOFPerceptionEventsCfg = YMBOY12DOFPerceptionEventsCfg()
    curriculum: YMBOY12DOFPerceptionCurriculumCfg = YMBOY12DOFPerceptionCurriculumCfg()
    # only for run time (training) debug
    # recorders: BaseHeightAnomalyRecorderManagerCfg = BaseHeightAnomalyRecorderManagerCfg()

    def __post_init__(self):
        super().__post_init__()
        self.wait_for_textures = True
        self.sim.physx.gpu_collision_stack_size = 2**29
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.terrain_generator.difficulty_range = (0.0, 1.0)
        self.scene.terrain.max_init_terrain_level = 5
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.height_scanner_base.update_period = 0.0
        self.scene.camera.update_period = self.decimation * self.sim.dt

        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 1.5)

        if os.environ.get("RSL_RL_PLAY", "0") == "1":
            self.scene.terrain.terrain_generator.num_rows = 3
            self.scene.terrain.terrain_generator.num_cols = 3
