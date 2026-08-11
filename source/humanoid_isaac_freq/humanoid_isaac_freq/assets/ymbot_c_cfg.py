# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for FFTAI robots.

The following configurations are available:

* :obj:`FFTAI_GR1T1_CFG`: FFTAI GR1T1 humanoid robot

Reference: https://github.com/FFTAI
"""

import math
import os

import isaaclab.sim as sim_utils

from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Isaac Lab 官方明确规定：
# 显式执行器的 effort_limit_sim=None：默认设为 1.0e9，避免求解器再次裁剪；
# 隐式执行器的 effort_limit_sim=None：使用 USD 关节上限；
# 显式执行器的 effort_limit=None：使用 USD 关节上限

# =========================================================================
# YMBOT_C Lower Body Dynamics Constants & Control Limits (Dynamic Values)
# Derived from hardware motor specifications.
# =========================================================================

# =========================================================================
# Phase 1: 全局系统动力学常量 (频带解耦与物理边界)
# =========================================================================
# 任务频带划分: 支撑与维稳关节目标带宽 (髋、膝) 
FREQ_SUPPORT_HZ = 10.0 
FREQ_SUPPORT_RAD = FREQ_SUPPORT_HZ * 2.0 * math.pi

# 任务频带划分: 所有负责摆动的关节
FREQ_SWING_HZ = 6.5
FREQ_SWING_RAD = FREQ_SWING_HZ * 2.0 * math.pi

# 任务频带划分: 摆动与末端关节目标带宽 (肩、肘、踝、腕)
FREQ_FEET_HZ   = 3.0
FREQ_FEET_RAD   = FREQ_FEET_HZ * 2.0 * math.pi

DAMPING_RATIO = 2.0
MAX_HARDWARE_KP = 500.0 * 0.7  # 严格的工程安全边界: 驱动器 500 极限刚度 * 0.7 安全边际

GLOBAL_EFFORT_SCALE = 1.0  
GLOBAL_VEL_SCALE    = 1.0  


# EC-A10020-P1-12
armature_EC_A10020_P1_12 = 0.000485 * 12 ** 2
effort_EC_A10020_P1_12 = 150.0 * GLOBAL_EFFORT_SCALE
velocity_EC_A10020_P1_12 = 140.0 * 2.0 * math.pi / 60.0 * GLOBAL_VEL_SCALE

# EC-A8112-P1-18
armature_EC_A8112_P1_18 = 0.000150 * 18 ** 2
effort_EC_A8112_P1_18 = 90.0 * GLOBAL_EFFORT_SCALE
velocity_EC_A8112_P1_18 = 157.0 * 2.0 * math.pi / 60.0 * GLOBAL_VEL_SCALE

# EC-A10020-P2-24
armature_EC_A10020_P2_24 = 0.000477 * 24 ** 2
effort_EC_A10020_P2_24 = 330.0 * GLOBAL_EFFORT_SCALE
velocity_EC_A10020_P2_24 = 123.0 * 2.0 * math.pi / 60.0 * GLOBAL_VEL_SCALE

# EC-A6408-P2-25
armature_EC_A6408_P2_25 = 0.000062 * 25 ** 2
effort_EC_A6408_P2_25 = 60.0 * GLOBAL_EFFORT_SCALE  # 这里严格按照电机手册编写，后续关节处进行手动修改
velocity_EC_A6408_P2_25 = 149.0 * 2.0 * math.pi / 60.0 * GLOBAL_VEL_SCALE

# EC-A4310-P2-36
armature_EC_A4310_P2_36 = 0.000019 * 36 ** 2
effort_EC_A4310_P2_36 = 36.0 * GLOBAL_EFFORT_SCALE
velocity_EC_A4310_P2_36 = 89.0 * 2.0 * math.pi / 60.0 * GLOBAL_VEL_SCALE


def compute_safe_motor_stiffness(
    motor_armature: float,
    target_frequency_rad: float,
    max_motor_stiffness: float = MAX_HARDWARE_KP
):
    """计算单个电机对应的安全刚度。"""
    motor_stiffness_raw = motor_armature * target_frequency_rad ** 2
    return min(motor_stiffness_raw, max_motor_stiffness)


def compute_safe_motor_damping(
    motor_armature: float,
    target_frequency_rad: float,
    damping_ratio: float = DAMPING_RATIO,
    max_motor_stiffness: float = MAX_HARDWARE_KP
):
    """根据单个电机裁剪后的刚度计算阻尼。"""
    motor_stiffness = compute_safe_motor_stiffness(motor_armature, target_frequency_rad, max_motor_stiffness)

    # 1.按照原始公式计算，需要保证转子惯量在连杆端的等效惯量是合理的
    # return 2.0 * damping_ratio * math.sqrt(final_stiffness * armature)

    # 2.直接利用kp和kd之间的关系来计算
    return 2.0 * damping_ratio * motor_stiffness / target_frequency_rad


# =========================================================================
# Phase 4: 执行器与仿真参数总装 (Configuration)
# =========================================================================
YMBOT_C_23DOF_ANALYTICAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(CURRENT_DIR, "usd", "ymbot_c_23dof_convex.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_pitch_joint": -0.12,
            ".*knee_joint": 0.30,
            ".*_ankle_pitch_joint": -0.18,
            ".*_elbow_joint": 1.0,
            "left_shoulder_roll_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
       actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                ".*waist_yaw_joint",
            ],
            effort_limit={
                ".*_hip_yaw_joint": effort_EC_A8112_P1_18,
                ".*_hip_roll_joint": effort_EC_A10020_P1_12,
                ".*_hip_pitch_joint": effort_EC_A10020_P2_24,
                ".*_knee_joint": effort_EC_A10020_P2_24,
                ".*waist_yaw_joint": effort_EC_A10020_P1_12,
            },
            velocity_limit={
                ".*_hip_yaw_joint": velocity_EC_A8112_P1_18,
                ".*_hip_roll_joint": velocity_EC_A10020_P1_12,
                ".*_hip_pitch_joint": velocity_EC_A10020_P2_24,
                ".*_knee_joint": velocity_EC_A10020_P2_24,
                ".*waist_yaw_joint": velocity_EC_A10020_P1_12,
            },
            stiffness={
                ".*_hip_yaw_joint": compute_safe_motor_stiffness(armature_EC_A8112_P1_18, FREQ_SUPPORT_RAD),
                ".*_hip_roll_joint": compute_safe_motor_stiffness(armature_EC_A10020_P1_12, FREQ_SUPPORT_RAD),
                ".*_hip_pitch_joint": compute_safe_motor_stiffness(armature_EC_A10020_P2_24, FREQ_SUPPORT_RAD),
                ".*_knee_joint": compute_safe_motor_stiffness(armature_EC_A10020_P2_24, FREQ_SUPPORT_RAD),
                ".*waist_yaw_joint": compute_safe_motor_stiffness(armature_EC_A10020_P1_12, FREQ_SWING_RAD),
            },
            damping={
                ".*_hip_yaw_joint": compute_safe_motor_damping(armature_EC_A8112_P1_18, FREQ_SUPPORT_RAD),
                ".*_hip_roll_joint": compute_safe_motor_damping(armature_EC_A10020_P1_12, FREQ_SUPPORT_RAD),
                ".*_hip_pitch_joint": compute_safe_motor_damping(armature_EC_A10020_P2_24, FREQ_SUPPORT_RAD),
                ".*_knee_joint": compute_safe_motor_damping(armature_EC_A10020_P2_24, FREQ_SUPPORT_RAD),
                ".*waist_yaw_joint": compute_safe_motor_damping(armature_EC_A10020_P1_12, FREQ_SUPPORT_RAD),
            },
            armature={
                ".*_hip_yaw_joint": armature_EC_A8112_P1_18,
                ".*_hip_roll_joint": armature_EC_A10020_P1_12,
                ".*_hip_pitch_joint": armature_EC_A10020_P2_24,
                ".*_knee_joint": armature_EC_A10020_P2_24,
                ".*waist_yaw_joint": armature_EC_A10020_P1_12,
            },
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit={
                ".*_ankle_pitch_joint": 75.0,
                ".*_ankle_roll_joint": 25.0,
            },
            velocity_limit={
                ".*_ankle_pitch_joint": velocity_EC_A6408_P2_25,
                ".*_ankle_roll_joint": velocity_EC_A6408_P2_25,
            },
            stiffness={
                ".*_ankle_pitch_joint": 2.0 * compute_safe_motor_stiffness(armature_EC_A6408_P2_25, FREQ_FEET_RAD),
                ".*_ankle_roll_joint": 2.0 * compute_safe_motor_stiffness(armature_EC_A6408_P2_25, FREQ_FEET_RAD),
            },
            damping={
                ".*_ankle_pitch_joint": 2.0 * compute_safe_motor_damping(armature_EC_A6408_P2_25, FREQ_FEET_RAD),
                ".*_ankle_roll_joint": 2.0 * compute_safe_motor_damping(armature_EC_A6408_P2_25, FREQ_FEET_RAD),
            },
            armature={
                ".*_ankle_pitch_joint": 2.0 * armature_EC_A6408_P2_25,
                ".*_ankle_roll_joint": 2.0 * armature_EC_A6408_P2_25,
            },
        ),
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit={
                ".*_shoulder_pitch_joint": effort_EC_A6408_P2_25,
                ".*_shoulder_roll_joint": effort_EC_A4310_P2_36,
                ".*_shoulder_yaw_joint": effort_EC_A4310_P2_36,
                ".*_elbow_joint": effort_EC_A6408_P2_25,
                ".*_wrist_yaw_joint": effort_EC_A4310_P2_36,
            },
            velocity_limit={
                ".*_shoulder_pitch_joint": velocity_EC_A6408_P2_25,
                ".*_shoulder_roll_joint": velocity_EC_A4310_P2_36,
                ".*_shoulder_yaw_joint": velocity_EC_A4310_P2_36,
                ".*_elbow_joint": velocity_EC_A6408_P2_25,
                ".*_wrist_yaw_joint": velocity_EC_A4310_P2_36,
            },
            stiffness={
                ".*_shoulder_pitch_joint": compute_safe_motor_stiffness(armature_EC_A6408_P2_25, FREQ_SWING_RAD),
                ".*_shoulder_roll_joint": compute_safe_motor_stiffness(armature_EC_A4310_P2_36, FREQ_SWING_RAD),
                ".*_shoulder_yaw_joint": compute_safe_motor_stiffness(armature_EC_A4310_P2_36, FREQ_SWING_RAD),
                ".*_elbow_joint": compute_safe_motor_stiffness(armature_EC_A6408_P2_25, FREQ_SWING_RAD),
                ".*_wrist_yaw_joint": compute_safe_motor_stiffness(armature_EC_A4310_P2_36, FREQ_SWING_RAD),
            },
            damping={
                ".*_shoulder_pitch_joint": compute_safe_motor_damping(armature_EC_A6408_P2_25, FREQ_SWING_RAD),
                ".*_shoulder_roll_joint": compute_safe_motor_damping(armature_EC_A4310_P2_36, FREQ_SWING_RAD),
                ".*_shoulder_yaw_joint": compute_safe_motor_damping(armature_EC_A4310_P2_36, FREQ_SWING_RAD),
                ".*_elbow_joint": compute_safe_motor_damping(armature_EC_A6408_P2_25, FREQ_SWING_RAD),
                ".*_wrist_yaw_joint": compute_safe_motor_damping(armature_EC_A4310_P2_36, FREQ_SWING_RAD),
            },
            armature={
                ".*_shoulder_pitch_joint": armature_EC_A6408_P2_25,
                ".*_shoulder_roll_joint": armature_EC_A4310_P2_36,
                ".*_shoulder_yaw_joint": armature_EC_A4310_P2_36,
                ".*_elbow_joint": armature_EC_A6408_P2_25,
                ".*_wrist_yaw_joint": armature_EC_A4310_P2_36,
            },
        ),
    }
)

# =========================================================================
# 根据执行器力矩和刚度构建动作缩放矩阵
# =========================================================================
YMBOT_C_23DOF_ANALYTICAL_ACTION_SCALE = {}

for actuator_name, actuator_cfg in YMBOT_C_23DOF_ANALYTICAL_CFG.actuators.items():
    if not isinstance(actuator_cfg.effort_limit, dict):
        raise TypeError(f"执行器 {actuator_name!r} 的 effort_limit 必须是字典。")

    if not isinstance(actuator_cfg.stiffness, dict):
        raise TypeError(f"执行器 {actuator_name!r} 的 stiffness 必须是字典。")

    for joint_name in actuator_cfg.joint_names_expr:
        if joint_name not in actuator_cfg.effort_limit:
            raise KeyError(f"执行器 {actuator_name!r} 缺少关节 {joint_name!r} 的 effort_limit。")

        if joint_name not in actuator_cfg.stiffness:
            raise KeyError(f"执行器 {actuator_name!r} 缺少关节 {joint_name!r} 的 stiffness。")

        joint_effort_limit = actuator_cfg.effort_limit[joint_name]
        joint_stiffness = actuator_cfg.stiffness[joint_name]

        if joint_stiffness <= 0.0:
            raise ValueError(f"关节 {joint_name!r} 的 stiffness 必须大于0。")

        YMBOT_C_23DOF_ANALYTICAL_ACTION_SCALE[joint_name] = 0.33 * joint_effort_limit / joint_stiffness


# ========================================================================================
# 打印
# ========================================================================================
def print_analytical_configuration(robot_cfg, action_scale_cfg):
    """打印最终写入 actuator 和 ActionsCfg 的参数。"""
    table_width = 132

    print("\n" + "=" * table_width)
    print("YMBOT-C 最终执行器与 Action Scale 配置")
    print("=" * table_width)

    table_header = (
        f"{'GROUP':<8s} | {'JOINT EXPRESSION':<28s} | {'ARMATURE':>10s} | {'KP':>10s} | "
        f"{'KD':>10s} | {'EFFORT':>10s} | {'VELOCITY':>10s} | {'ACTION SCALE':>12s}"
    )
    print(table_header)
    print("-" * table_width)

    for actuator_name, actuator_cfg in robot_cfg.actuators.items():
        for joint_name in actuator_cfg.joint_names_expr:
            joint_armature = actuator_cfg.armature[joint_name]
            joint_stiffness = actuator_cfg.stiffness[joint_name]
            joint_damping = actuator_cfg.damping[joint_name]
            joint_effort_limit = actuator_cfg.effort_limit[joint_name]
            joint_velocity_limit = actuator_cfg.velocity_limit[joint_name]
            joint_action_scale = action_scale_cfg[joint_name]

            debug_line = (
                f"{actuator_name:<8s} | {joint_name:<28s} | {joint_armature:>10.5f} | "
                f"{joint_stiffness:>10.2f} | {joint_damping:>10.2f} | {joint_effort_limit:>10.2f} | "
                f"{joint_velocity_limit:>10.2f} | {joint_action_scale:>12.4f}"
            )
            print(debug_line)

    print("=" * table_width)

    # 方便在  clean_ckpt.py 中使用
    print("\nACTION_SCALE_DICT = {")
    for joint_name, action_scale in action_scale_cfg.items():
        print(f"    {joint_name!r}: {action_scale:.4f},")
    print("}\n")


print_analytical_configuration(
    YMBOT_C_23DOF_ANALYTICAL_CFG,
    YMBOT_C_23DOF_ANALYTICAL_ACTION_SCALE
)














# =========================================================================
# 工程整定法（废弃）
# =========================================================================
YMBOT_C_23DOF_EMPIRICAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(CURRENT_DIR, "usd", "ymbot_c_23dof_convex.usd"), # 直接指向编译好的 USD
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, # 依然开启，因为你的 USD 里已经配置好了精确的过滤名单
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_pitch_joint": -0.12,
            ".*knee_joint": 0.30,
            ".*_ankle_pitch_joint": -0.18,
            ".*_elbow_joint": 1.0,
            "left_shoulder_roll_joint": 0.2,
            # "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            # "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                ".*waist_yaw_joint",
            ],
            effort_limit={
                # 公式: 峰值力矩 (Peak Torque) * 安全缩放系数
                ".*_hip_roll_joint":  150.0 * 0.9,
                ".*_hip_yaw_joint":   90.0 * 0.9,
                ".*_hip_pitch_joint": 330.0 * 0.9,
                ".*_knee_joint":      330.0 * 0.9,
                ".*waist_yaw_joint":  150.0 * 0.9,
            },
            velocity_limit={
                # 公式: 峰值RPM (Peak RPM) * 转换常数(2*pi/60) * 速度预留缩放系数
                ".*_hip_roll_joint":  140.0 * (2.0 * math.pi / 60.0) * 0.9, # 14.65333338 * 0.9
                ".*_hip_yaw_joint":   157.0 * (2.0 * math.pi / 60.0) * 0.9, # 16.432666719 * 0.9
                ".*_hip_pitch_joint": 123.0 * (2.0 * math.pi / 60.0) * 0.9, # 12.874000041 * 0.9
                ".*_knee_joint":      123.0 * (2.0 * math.pi / 60.0) * 0.9, # 12.874000041 * 0.9
                ".*waist_yaw_joint":  140.0 * (2.0 * math.pi / 60.0) * 0.9, # 14.65333338 * 0.9
            },
            stiffness={
                ".*_hip_roll_joint": 200.0,
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_pitch_joint": 350.0,
                ".*_knee_joint": 350.0,    
                ".*waist_yaw_joint": 300.0,  
            },
            damping={
                ".*_hip_roll_joint": 10.0,
                ".*_hip_yaw_joint": 7.5,
                ".*_hip_pitch_joint": 17.5,
                ".*_knee_joint": 17.5,
                ".*waist_yaw_joint": 15.0,
            },
            armature={
                ".*_hip_yaw_joint": armature_EC_A8112_P1_18,
                ".*_hip_roll_joint": armature_EC_A10020_P1_12,
                ".*_hip_pitch_joint": armature_EC_A10020_P2_24,
                ".*_knee_joint": armature_EC_A10020_P2_24,
                ".*waist_yaw_joint": armature_EC_A10020_P1_12,
            },
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit={
                # 这里放开能力上限，但是奖励函数部分一定要限制，由于存在耦合关系，两个关节的能力无法同时到达上限
                ".*_ankle_pitch_joint": 60.0 * 2.0 * 0.9, # 120 * 0.9
                ".*_ankle_roll_joint":  60.0 * 2.0 * 0.9, # 120 * 0.9
                # ".*_ankle_pitch_joint": 40.0,
                # ".*_ankle_roll_joint":  40.0,
            },
            velocity_limit={
                # 力矩和惯量叠加，速度无需叠加
                ".*_ankle_pitch_joint": 149.0 * (2.0 * math.pi / 60.0) * 0.9, # 15.595333383 * 0.9
                ".*_ankle_roll_joint":  149.0 * (2.0 * math.pi / 60.0) * 0.9, # 15.595333383 * 0.9
            },
            stiffness={
                ".*_ankle_pitch_joint": 60.0,
                ".*_ankle_roll_joint": 60.0,
            },
            damping={
                ".*_ankle_pitch_joint": 3.0,
                ".*_ankle_roll_joint": 3.0,
            },
            armature={
                ".*_ankle_pitch_joint": 2.0 * armature_EC_A6408_P2_25,
                ".*_ankle_roll_joint": 2.0 * armature_EC_A6408_P2_25,
            },
        ),
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit={
                # 峰值力矩 * 安全缩放系数
                ".*_shoulder_pitch_joint": 60.0 * 0.9,
                ".*_shoulder_roll_joint":  36.0 * 0.9,
                ".*_shoulder_yaw_joint":   36.0 * 0.9,
                ".*_elbow_joint":          60.0 * 0.9,
                ".*_wrist_yaw_joint":      36.0 * 0.9,
            },
            velocity_limit={
                # 峰值RPM * 转换常数(2*pi/60) * 速度预留缩放系数
                ".*_shoulder_pitch_joint": 149.0 * (2.0 * math.pi / 60.0) * 0.9, # 15.595333383
                ".*_shoulder_roll_joint":  89.0 * (2.0 * math.pi / 60.0) * 0.9, # 9.315333363
                ".*_shoulder_yaw_joint":   89.0 * (2.0 * math.pi / 60.0) * 0.9, # 9.315333363
                ".*_elbow_joint":          149.0 * (2.0 * math.pi / 60.0) * 0.9, # 15.595333383
                ".*_wrist_yaw_joint":      89.0 * (2.0 * math.pi / 60.0) * 0.9, # 9.315333363
            },
            stiffness={
                ".*_shoulder_pitch_joint": 50.0,
                ".*_shoulder_roll_joint": 50.0,
                ".*_shoulder_yaw_joint": 50.0,
                ".*_elbow_joint": 50.0,
                ".*_wrist_yaw_joint": 50.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 2.5,
                ".*_shoulder_roll_joint": 2.5,
                ".*_shoulder_yaw_joint": 2.5,
                ".*_elbow_joint": 2.5,
                ".*_wrist_yaw_joint": 2.5,
            },
            armature={
                ".*_shoulder_pitch_joint": armature_EC_A6408_P2_25,
                ".*_shoulder_roll_joint": armature_EC_A4310_P2_36,
                ".*_shoulder_yaw_joint": armature_EC_A4310_P2_36,
                ".*_elbow_joint": armature_EC_A6408_P2_25,
                ".*_wrist_yaw_joint": armature_EC_A4310_P2_36,
            },
        ),
    },
)