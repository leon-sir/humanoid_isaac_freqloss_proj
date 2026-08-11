# @Author: Zheng Pan
# @Emial: pan_zheng@nwpu.edu.cn
# Copyright (c) 2025 Zheng Pan
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING


import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    # from humanoid_isaac_dash.tasks.locomotion.velocity.config.humanoid.ymboy_flip.env.manager_based_rl_env import MyManagerBasedRLYMEnv


class ActionSmoothnessPenalty(ManagerTermBase):
    """Penalize the second finite difference of policy actions."""

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.prev_prev_action = torch.zeros_like(env.action_manager.action)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.prev_prev_action[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        penalty = torch.sum(
            torch.square(env.action_manager.action - 2.0 * env.action_manager.prev_action + self.prev_prev_action),
            dim=1,
        )
        self.prev_prev_action.copy_(env.action_manager.prev_action)
        return penalty


# def feet_air_time(
#     env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
# ) -> torch.Tensor:
#     """Reward long steps taken by the feet using L2-kernel.

#     This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
#     that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
#     the time for which the feet are in the air.

#     If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
#     """
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     # compute the reward
#     first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
#     last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
#     reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
#     # no reward for zero command
#     reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
#     return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def track_torso_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tilt_threshold: float = 0.7
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]

    # vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    # lin_vel_error = torch.sum(
    #     torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    # )
    vel_torso_yaw = math_utils.quat_apply_inverse(
        yaw_quat(asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]),
        asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :3]
    ).squeeze(dim=1)
    # env.torso_lin_vel_b = vel_torso_yaw
    
    lin_vel_torso_error = env.command_manager.get_command(command_name)[:, 0] - vel_torso_yaw[:, 0]
    # env.lin_vel_torso_error = lin_vel_torso_error

    lin_vel_error = torch.square(lin_vel_torso_error)

    # if not hasattr(env, "lin_vel_torso_error"):
    #     raise ValueError("lin_vel_torso_error do not exists, please update rl_env files or," \
    #                      " comment the follower code statements")

    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, tilt_threshold) / tilt_threshold
    return reward


def track_lin_vel_x_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = \
        torch.square(env.command_manager.get_command(command_name)[:, 0] - asset.data.root_lin_vel_b[:, 0])

    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_lin_vel_y_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = \
        torch.square(env.command_manager.get_command(command_name)[:, 1] - asset.data.root_lin_vel_b[:, 1])

    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def track_torso_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tilt_threshold: float = 0.7
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # asset_cfg.body_ids
    # ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    # asset.data.root_ang_vel_w
    torso_ang_vel_z = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, 2].squeeze(-1)
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - torso_ang_vel_z
    )
    # env.torso_ang_vel_z = torso_ang_vel_z
    # if not hasattr(env, "ang_vel_error"):
    #     raise ValueError("ang_vel_error do not exists, please update rl_env files or," \
    #     " comment the follower code statements")
    # env.ang_vel_error = ang_vel_error
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, tilt_threshold) / tilt_threshold
    return reward

def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward


def stand_still_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one when no command."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(diff_angle), dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def stand_still_without_cmd_v2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    velocity_threshold: float = 0.2,
    use_zeros_pos: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one when no command."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    # compute out of limits constraints
    if use_zeros_pos:
        diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - 0
    else:
        diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    # diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - 0
    reward = torch.sum(torch.abs(diff_angle), dim=1)
    condition_small_command = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    # condition_small_vel = body_vel < velocity_threshold
    # reward *= torch.logical_and(condition_small_command, condition_small_vel)
    reward *= condition_small_command
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_pos_penalty_zero(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - 0), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    # print(asset_cfg.joint_ids)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_pos_penalty_zero_fly(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    height_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - 0), dim=1
    )
    reward = torch.where(
        asset.data.root_pos_w[:, 2] > height_threshold,
        running_reward,
        0,
    )
    # print(asset_cfg.joint_ids)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "mirror_joints_cache") or env.mirror_joints_cache is None:
        env.mirror_joints_cache = [
            asset.find_joints(joint_name) for joint_pair in mirror_joints for joint_name in joint_pair
        ]
    # compute out of limits constraints
    diff1 = torch.sum(
        torch.square(
            asset.data.joint_pos[:, env.mirror_joints_cache[0][0]]
            - asset.data.joint_pos[:, env.mirror_joints_cache[1][0]]
        ),
        dim=-1,
    )
    diff2 = torch.sum(
        torch.square(
            asset.data.joint_pos[:, env.mirror_joints_cache[2][0]]
            - asset.data.joint_pos[:, env.mirror_joints_cache[3][0]]
        ),
        dim=-1,
    )
    reward = 0.5 * (diff1 + diff2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    mode_time: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Reward longer feet air and contact time."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    t_max = torch.max(current_air_time, current_contact_time)
    t_min = torch.clip(t_max, max=mode_time)
    stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)
    # cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1).unsqueeze(dim=1).expand(-1, 4)
    # body_vel = torch.linalg.norm(asset.data.root_com_lin_vel_b[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 4)
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1).unsqueeze(dim=1).expand(-1, 2)
    body_vel = torch.linalg.norm(asset.data.root_com_lin_vel_b[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 2)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        torch.where(t_max < mode_time, t_min, 0),
        stance_cmd_reward,
    )
    gravity_coeff = torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    gravity_coeff = gravity_coeff.unsqueeze(1)  # 增加第二维度 (num_envs, 1)
    # 应用重力系数（广播到腿的数量维度）
    reward *= gravity_coeff
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return torch.sum(reward, dim=1)


def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(contact, dim=-1).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # env.scene.
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 5 * forces_z, dim=1).float()
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_y_exp_biped(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    footsteps_in_body_frame = torch.zeros(env.num_envs, 2, 3, device=env.device)
    for i in range(2):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    if 0:   # using absoluate values
        desired_ys = torch.cat(
            [stance_width_tensor/2, stance_width_tensor/2], dim=1
        )
        stance_diff = torch.square(desired_ys - torch.abs(footsteps_in_body_frame[:, :, 1]))
    else:
        desired_ys = torch.cat(
            [stance_width_tensor/2, -stance_width_tensor/2], dim=1
        )
        stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    
    # print(f"foot y_distance in body frame-left:{torch.mean(footsteps_in_body_frame[:, 0, 1])}")
    # print(f"foot y_distance in body frame-right:{torch.mean(footsteps_in_body_frame[:, 1, 1])}")

    reward = torch.exp(torch.sum(stance_diff, dim=1) / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_y_square_biped(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    footsteps_in_body_frame = torch.zeros(env.num_envs, 2, 3, device=env.device)
    for i in range(2):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    if 0:   # using absoluate values
        desired_ys = torch.cat(
            [stance_width_tensor/2, stance_width_tensor/2], dim=1
        )
        stance_diff = torch.square(desired_ys - torch.abs(footsteps_in_body_frame[:, :, 1]))
    else:
        desired_ys = torch.cat(
            [stance_width_tensor/2, -stance_width_tensor/2], dim=1
        )
        stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    
    # print(f"foot y_distance in body frame-left:{torch.mean(footsteps_in_body_frame[:, 0, 1])}")
    # print(f"foot y_distance in body frame-right:{torch.mean(footsteps_in_body_frame[:, 1, 1])}")
    # reward = stance_diff
    reward = torch.sum(stance_diff, dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def knee_distance_y_square_biped(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    footsteps_in_body_frame = torch.zeros(env.num_envs, 2, 3, device=env.device)
    for i in range(2):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    if 0:   # using absoluate values
        desired_ys = torch.cat(
            [stance_width_tensor/2, stance_width_tensor/2], dim=1
        )
        stance_diff = torch.square(desired_ys - torch.abs(footsteps_in_body_frame[:, :, 1]))
    else:
        desired_ys = torch.cat(
            [stance_width_tensor/2, -stance_width_tensor/2], dim=1
        )
        stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # print(f"knee y_distance in body frame-left:{torch.mean(footsteps_in_body_frame[:, 0, 1])}")
    # print(f"knee y_distance in body frame-right:{torch.mean(footsteps_in_body_frame[:, 1, 1])}")
    # reward = stance_diff
    reward = torch.sum(stance_diff, dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height_v2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    expect_air = ~in_contact
    asset: RigidObject = env.scene[asset_cfg.name]

    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_z_target_error = torch.where(expect_air, foot_z_target_error, 0)
    
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(torch.exp(-foot_z_target_error / 0.1**2) * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    # print(torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide_v2(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_yaw_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize yaw rotation of feet while they are in contact with the ground."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    foot_yaw_ang_vel = torch.abs(asset.data.body_ang_vel_w[:, asset_cfg.body_ids, 2])
    reward = torch.sum(foot_yaw_ang_vel * contacts, dim=1)
    return reward


# def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     return torch.sum(diff, dim=1)


# def smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     diff = diff * (env.action_manager.prev_prev_action[:, :] != 0)  # ignore second step
#     return torch.sum(diff, dim=1)


def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    standing_reward = torch.sum(joint_vel, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward


def upward(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward
    # upward_error = torch.square(asset.data.projected_gravity_b[:, 2] - (-1))
    # return torch.exp(-upward_error / std**2)


def base_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_w[:, 2])
    return reward


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            # print(torch.mean(ray_hits, dim=1))
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def base_height_below_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= (asset.data.root_pos_w[:, 2] < adjusted_target_height).float()  # Only penalize if below target height
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def root_orientation_l2(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        target_roll: float = 0.0,
        target_pitch: float = 0.0,
) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    is_standing_env = env.command_manager.get_term("base_velocity").command[:, 0] <= 0.1
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    target_pitch_tensor = torch.where(
        is_standing_env,
        torch.tensor(0, device=env.device, dtype=torch.float32),
        torch.tensor(target_pitch, device=env.device, dtype=torch.float32)
    )
    target_roll_tensor = torch.tensor(target_roll, device=env.device, dtype=torch.float32)
    target_pitch_tensor = torch.tensor(target_pitch, device=env.device, dtype=torch.float32)

    expected_gravity_x = torch.sin(target_pitch_tensor) * torch.cos(target_roll_tensor)
    expected_gravity_y = -torch.sin(target_roll_tensor) * torch.cos(target_pitch_tensor)
    expected_gravity_z = -torch.cos(target_pitch_tensor) * torch.cos(target_roll_tensor)
    
    expected_proj_gravity = torch.stack([
        expected_gravity_x,
        expected_gravity_y,
        expected_gravity_z
    ], dim=0).unsqueeze(0).expand(env.num_envs, -1)

    reward = torch.sum(
        torch.square(expected_proj_gravity - asset.data.projected_gravity_b),
        dim=1
    )
    
    # print(f"asset.data.root_pos_w[:, 2]={asset.data.root_pos_w[:, 2]}")
    # reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def bend_knee_and_hip_pitch_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    
    assert len(asset_cfg.joint_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.square(torch.sum(asset.data.joint_pos[:, asset_cfg.joint_ids], dim=1))
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def flat_orientation_l2_feet(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    feet_orientation = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    gravity_tensor = torch.tensor([0, 0, -1.0], device=env.device)
    gravity_batch = gravity_tensor.unsqueeze(0).repeat(env.num_envs, 2, 1)
    feet_projected_gravity = torch.zeros(env.num_envs, 2, 3, device=env.device)
    feet_projected_gravity = math_utils.quat_apply_inverse(feet_orientation, gravity_batch)
    # extract the used quantities (to enable type-hinting)
    
    reward = torch.sum(torch.square(feet_projected_gravity[:, :, :2]), dim=(1,2))
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_orientation_contact(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize non-flat feet orientation only when the feet are in contact."""
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    feet_orientation = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    gravity_tensor = torch.tensor([0, 0, -1.0], device=env.device)
    gravity_batch = gravity_tensor.unsqueeze(0).repeat(env.num_envs, len(asset_cfg.body_ids), 1)
    feet_projected_gravity = math_utils.quat_apply_inverse(feet_orientation, gravity_batch)

    net_contact_forces = contact_sensor.data.net_forces_w_history
    in_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1.0

    reward = torch.norm(feet_projected_gravity[:, :, :2], dim=-1) * in_contact.float()
    reward = torch.sum(reward, dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_at_plane(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    left_height_scanner_cfg: SceneEntityCfg,
    right_height_scanner_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_offset: float = 0.035,
) -> torch.Tensor:
    """Penalize contacting feet that are higher than the local terrain plane."""
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]

    net_contact_forces = contact_sensor.data.net_forces_w_history
    in_contact = torch.max(torch.norm(net_contact_forces[:, :, contact_sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1.0

    left_sensor: RayCaster = env.scene[left_height_scanner_cfg.name]
    left_ground_height = left_sensor.data.ray_hits_w[..., 2]
    left_ground_height = torch.where(torch.isinf(left_ground_height), 0.0, left_ground_height)

    right_sensor: RayCaster = env.scene[right_height_scanner_cfg.name]
    right_ground_height = right_sensor.data.ray_hits_w[..., 2]
    right_ground_height = torch.where(torch.isinf(right_ground_height), 0.0, right_ground_height)

    left_height = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    right_height = asset.data.body_pos_w[:, asset_cfg.body_ids[1], 2]

    left_penalty = (
        torch.clamp(left_height.unsqueeze(-1) - left_ground_height - height_offset, min=0.0, max=0.3)
        * in_contact[:, 0:1]
    )
    right_penalty = (
        torch.clamp(right_height.unsqueeze(-1) - right_ground_height - height_offset, min=0.0, max=0.3)
        * in_contact[:, 1:2]
    )
    return torch.sum(left_penalty, dim=-1) + torch.sum(right_penalty, dim=-1)


def flat_orientation_l2_body(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    torso_orientation = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    gravity_tensor = torch.tensor([0, 0, -1.0], device=env.device)
    gravity_batch = gravity_tensor.unsqueeze(0).repeat(env.num_envs, len(asset_cfg.body_ids), 1)
    torso_projected_gravity = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    torso_projected_gravity = math_utils.quat_apply_inverse(torso_orientation, gravity_batch)
    # extract the used quantities (to enable type-hinting)
    
    reward = torch.sum(torch.square(torso_projected_gravity[:, :, :2]), dim=(1,2))
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def swithc_mode(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    # air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    # contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    ## 自己和自己比
    # reward = torch.sum(torch.square(last_air_time-last_contact_time), dim=1) 
    ## 计算双腿差异
    delta_contact = last_contact_time[:, 0] - last_contact_time[:, 1]  # 左腿接触时间 - 右腿接触时间
    delta_air = last_air_time[:, 0] - last_air_time[:, 1]              # 左腿悬空时间 - 右腿悬空时间
    reward = torch.square(delta_contact) + torch.square(delta_air)
    return reward


# 比较左右腿行走时最大关节角度差异
def max_joint_pos_diff(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "mirror_joints_pos_cache") or env.mirror_joints_pos_cache is None:
        env.mirror_joints_pos_cache = [
            asset.data.joint_pos[:, asset_cfg.joint_ids]
        ]
    return 0


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def fly(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5

def body_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def body_force(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def feet_too_near_humanoid(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: RigidObject = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_distance_y_too_near_humanoid(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: RigidObject = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, 1]
    distance = torch.abs(feet_pos[:, 0] - feet_pos[:, 1])
    return (threshold - distance).clamp(min=0)


def hip_roll_action(
        env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
        ) -> torch.Tensor:
    """Penalize hip roll joint actions."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(env.action_manager.action[:, asset_cfg.joint_ids]), dim=1)


def hip_yaw_action(
        env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
        ) -> torch.Tensor:
    """Penalize hip yaw joint actions."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(env.action_manager.action[:, asset_cfg.joint_ids]), dim=1)


def ankle_torque(
        env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize ankle joint actions."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def policy_action_l2(
        env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize hip roll joint actions.
    Only works when preserve order = True and specify the joint order same as asset_cfg"""

    # asset: Articulation = env.scene[asset_cfg.name]
    # ids = env.action_manager.get_term("joint_pos")._asset.joint_names
    return torch.sum(torch.square(env.action_manager.action[:, asset_cfg.joint_ids]), dim=1)


def policy_action_rate_l2(
        env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action[:, asset_cfg.joint_ids] - env.action_manager.prev_action[:, asset_cfg.joint_ids]), dim=1)


"""
for back-flipping task
"""

def body_orientation_back_flip(
        env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), one_cycle_time: float = 0.8, std: float=0.25
        ) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    # golbal_time = (env.episode_length_buf * env.step_dt).unsqueeze(1)
    golbal_time = (env.episode_length_buf * env.step_dt)
    x_proj = torch.sin(2*torch.pi/one_cycle_time*golbal_time + torch.pi)
    z_proj = torch.cos(golbal_time + torch.pi)
    reward = torch.exp(-(torch.square(x_proj-asset.data.projected_gravity_b[:, 0])+torch.square(z_proj-asset.data.projected_gravity_b[:, 2])) / std**2)
    # proj_x = 
    # reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    return reward



# def flat_orientation_quat_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """Penalize non-flat base orientation using L2 squared kernel.

#     Hope the quaterion's z axis close to 1.
#     """
#     # extract the used quantities (to enable type-hinting)
#     asset: RigidObject = env.scene[asset_cfg.name]
#     # actual means z_up-1
#     z_up = -2*(asset.data.body_quat_w[:,asset_cfg.body_ids,1]**2+asset.data.body_quat_w[:,asset_cfg.body_ids,2]**2)
#     z_up_deviation = torch.sum(torch.square(z_up), dim=1)
#     # allow a threhold of 0.2 rad
#     return torch.where(z_up_deviation < 0.04, torch.zeros_like(z_up_deviation), z_up_deviation) 


def flip_base_height_l2(
    env: MyManagerBasedRLYMEnv,
    target_height: float = 0.6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    com_pos = torch.tensor([0, 0, 0.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    jump_height = asset.data.root_pos_w[:, 2] + math_utils.quat_apply_inverse(asset.data.root_link_quat_w, com_pos)[: ,2]
    sit_target_height = 0.4 
    reward  = env.flip_state_buff[:, 0, 0]*(-torch.abs(asset.data.root_pos_w[:, 2] - target_height))
    reward += env.flip_state_buff[:, 1, 0]*(-torch.abs(asset.data.root_pos_w[:, 2] - sit_target_height))
    reward += env.flip_state_buff[:, 2, 0]*(jump_height <= 1.8)*(jump_height) # as high as possible. Note 1.8 means: not too heigh 
    reward += env.flip_state_buff[:, 3, 0]*(jump_height <= 1.8)*(jump_height)
    reward += env.flip_state_buff[:, 4, 0]*(-torch.abs(asset.data.root_pos_w[:, 2] - target_height))
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def flip_body_balance(
    env: MyManagerBasedRLYMEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """ body balance
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    # world_z = torch.tensor([0, 0, 1], device=env.device)
    world_z = torch.tensor([0, 0, 1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    body_z = math_utils.quat_apply_inverse(asset.data.root_link_quat_w, world_z)

    reward =  env.flip_state_buff[:, 0, 0]*(-torch.arccos(torch.clamp(body_z[:, 2], -1.0, 1.0)))
    reward += env.flip_state_buff[:, 1, 0]*(-torch.arccos(torch.clamp(body_z[:, 2], -1.0, 1.0)))
    reward += env.flip_state_buff[:, 2, 0]*(-torch.abs(torch.arccos(torch.clamp(body_z[:, 1], -1.0, 1.0)) - torch.pi/2.0))
    reward += env.flip_state_buff[:, 3, 0]*(-torch.abs(torch.arccos(torch.clamp(body_z[:, 1], -1.0, 1.0)) - torch.pi/2.0))
    reward += env.flip_state_buff[:, 4, 0]*(-torch.arccos(torch.clamp(body_z[:, 2], -1.0, 1.0)))

    return reward


def flip_max_pitch_vel(
    env: MyManagerBasedRLYMEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # pitch vel
    asset: RigidObject = env.scene[asset_cfg.name]
    # base_lin_vels = torch_utils.quat_rotate_inverse(self.base_quaternions, self.base_lin_vels)
    # base_ang_vels = torch_utils.quat_rotate_inverse(self.base_quaternions, self.base_ang_vels)
    
    base_vel_penalties = torch.square(asset.data.root_link_lin_vel_b[:, 0]) + torch.square(asset.data.root_link_lin_vel_b[:, 1]) + torch.square(asset.data.root_link_lin_vel_b[:, 2])
    reward  = env.flip_state_buff[:, 0, 0]*(-base_vel_penalties)
    reward += env.flip_state_buff[:, 1, 0]*(-base_vel_penalties)
    reward += env.flip_state_buff[:, 2, 0]*(1.0 - env.is_one_turn_buf)*(-asset.data.root_link_lin_vel_b[:, 1])  # why？
    reward += env.flip_state_buff[:, 3, 0]*(1.0 - env.is_one_turn_buf)*(-asset.data.root_link_lin_vel_b[:, 1])
    reward += env.flip_state_buff[:, 4, 0]*(-base_vel_penalties)

    return reward

def flip_posture_l2(
    env: MyManagerBasedRLYMEnv,
    hip_pitch_name: str,
    knee_pitch_name: str,
    ankle_pitch_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    
    asset: Articulation = env.scene[asset_cfg.name]
    
    # asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward  = env.flip_state_buff[:, 0, 0]*(-torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] - 0.0).mean(dim=-1))

    # sit_joint_pos = torch.zeros_like(asset.data.joint_pos[:, asset_cfg.joint_ids], device=env.device)
    # sit_joint_pos[:, [0, 1]] = -0.1 
    # sit_joint_pos[:, [6, 7]] = 0.3 

    sit_joint_pos = torch.zeros(len(asset_cfg.joint_ids), device=env.device)

    asset.find_joints(hip_pitch_name, preserve_order=True)[0]
    sit_joint_pos[asset.find_joints(hip_pitch_name, preserve_order=True)[0]] = -1.4
    sit_joint_pos[asset.find_joints(knee_pitch_name, preserve_order=True)[0]] = 1.8
    sit_joint_pos[asset.find_joints(ankle_pitch_name, preserve_order=True)[0]] = -1

 
    reward += env.flip_state_buff[:, 1, 0]*(-torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] - sit_joint_pos).mean(dim=-1))

    # reward += env.flip_state_buff[:, 2, 0]*(-torch.square(self.dof_positions[:, 10:] - self.default_dof_positions[:, 10:]).mean(dim=-1))
    # reward += env.flip_state_buff[:, 3, 0]*(-torch.square(self.dof_positions[:, 10:] - self.default_dof_positions[:, 10:]).mean(dim=-1))
    reward += env.flip_state_buff[:, 2, 0]*0
    reward += env.flip_state_buff[:, 3, 0]*0
    # reward += env.flip_state_buff[:, 4, 0]*(-torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]).mean(dim=-1))
    reward += env.flip_state_buff[:, 4, 0]*(-torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] - 0.0).mean(dim=-1))

    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def flip_flat_orientation_l2(env: MyManagerBasedRLYMEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    reward = env.flip_state_buff[:, 0, 0]*torch.square(1 - asset.data.projected_gravity_b[:, 2])
    # reward += env.flip_state_buff[:, 1, 0]*torch.square(1.5 - asset.data.projected_gravity_b[:, 2]/2)
    SQRT2_OVER_2 = torch.sqrt(torch.tensor(2.0)) / 2.0
    reward += env.flip_state_buff[:, 1, 0]*(4 - torch.square(asset.data.projected_gravity_b[:, 2] + 0.5) - torch.square(asset.data.projected_gravity_b[:, 0]-torch.sqrt(torch.tensor(3.0)) / 2.0))
    reward += env.flip_state_buff[:, 2, 0]*4
    reward += env.flip_state_buff[:, 3, 0]*4
    reward += env.flip_state_buff[:, 4, 0]*torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def flip_stage_conversion(env: MyManagerBasedRLYMEnv)-> torch.Tensor:

    reward = env.flip_state_buff[:, 0, 0]*1
    reward += env.flip_state_buff[:, 1, 0]*1
    reward += env.flip_state_buff[:, 2, 0]*10
    reward += env.flip_state_buff[:, 3, 0]*1
    reward += env.flip_state_buff[:, 4, 0]*1
    return reward


def flip_has_one_turn(env: MyManagerBasedRLYMEnv)-> torch.Tensor:
    return env.is_one_turn_buf


def flip_has_jumped(env: MyManagerBasedRLYMEnv)-> torch.Tensor:
    eps = 1e-6
    
    # return torch.mean(episode_sums[env_ids])
    return env.flip_state_buff[:, 2, 1] > eps


""" 
for stairs climbing task
"""

def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str,
                              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize tracking orientation error."""
    asset: RigidObject = env.scene[asset_cfg.name]
    # command = env.command_manager.get_command(command_name)
    command = env.command_manager.get_term(command_name)
    heading_b = asset.data.heading_w - command.heading_target
    # self.robot.data.heading_w[env_ids]
    # print(command1.heading_target)
    return heading_b.abs()


def heading_command_error_square(env: ManagerBasedRLEnv, command_name: str,
                              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize tracking orientation error."""
    asset: RigidObject = env.scene[asset_cfg.name]
    # command = env.command_manager.get_command(command_name)
    command = env.command_manager.get_term(command_name)
    heading_b = asset.data.heading_w - command.heading_target
    # self.robot.data.heading_w[env_ids]
    # print(command1.heading_target)
    reward = torch.square(heading_b)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def heading_command_error_l2(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize root heading error to the heading command."""
    asset: RigidObject = env.scene["robot"]
    command = env.command_manager.get_term(command_name)
    heading_env = command.is_heading_env
    if hasattr(command, "is_lane_keeping_env"):
        heading_env = torch.logical_and(heading_env, ~command.is_lane_keeping_env)
    heading_error = math_utils.wrap_to_pi(asset.data.heading_w - command.heading_target)
    reward = torch.square(heading_error)
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    reward *= heading_env.float()
    return reward


def force_too_large(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 500,
    max_reward: float = 400,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward
