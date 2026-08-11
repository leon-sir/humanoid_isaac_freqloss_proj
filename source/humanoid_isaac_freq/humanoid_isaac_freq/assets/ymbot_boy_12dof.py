# Copyright (c) 2024-2026 Zheng Pan
# SPDX-License-Identifier: Apache-2.0

"""Articulation configuration for the arm-less 12-DoF YMBOY."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from . import HUMANOID_ISAAC_FREQ_DATA_DIR


ARMATURE_EC_A8112_P1_18 = 150e-6 * 18**2
ARMATURE_EC_A6408_P2_25 = 62e-6 * 25**2
ARMATURE_EC_A10020_P1_12 = 485e-6 * 12**2
ARMATURE_EC_A4310_P2_36 = 18e-6 * 36**2


YMBOT_BOY_12DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(HUMANOID_ISAAC_FREQ_DATA_DIR / "Robots/ymboy-12dof/ymboy_12dof_no_arm.urdf"),
        fix_base=False,
        make_instanceable=True,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        merge_fixed_joints=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            retain_accelerations=False,
            enable_gyroscopic_forces=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.7),
        joint_pos={
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            "left_hip_roll_joint": 0.1,
            "left_ankle_roll_joint": -0.1,
            "right_hip_roll_joint": -0.1,
            "right_ankle_roll_joint": 0.1,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim=94.0,
            velocity_limit_sim=15.0,
            stiffness={
                ".*_hip_pitch_joint": 150.0,
                ".*_hip_roll_joint": 60.0,
                ".*_hip_yaw_joint": 100.0,
                ".*_knee_joint": 150.0,
            },
            damping={
                ".*_hip_pitch_joint": 15.0,
                ".*_hip_roll_joint": 10.0,
                ".*_hip_yaw_joint": 10.0,
                ".*_knee_joint": 15.0,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_EC_A8112_P1_18,
                ".*_hip_roll_joint": ARMATURE_EC_A8112_P1_18,
                ".*_hip_yaw_joint": ARMATURE_EC_A6408_P2_25,
                ".*_knee_joint": ARMATURE_EC_A10020_P1_12,
            },
            friction=2e-3,
            dynamic_friction=1e-3,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=72.0,
            velocity_limit_sim=16.0,
            stiffness={".*_ankle_pitch_joint": 20.0, ".*_ankle_roll_joint": 20.0},
            damping={".*_ankle_pitch_joint": 2.0, ".*_ankle_roll_joint": 2.0},
            armature={
                ".*_ankle_pitch_joint": ARMATURE_EC_A4310_P2_36,
                ".*_ankle_roll_joint": ARMATURE_EC_A4310_P2_36,
            },
            friction=2e-3,
            dynamic_friction=1e-3,
        ),
    },
)
