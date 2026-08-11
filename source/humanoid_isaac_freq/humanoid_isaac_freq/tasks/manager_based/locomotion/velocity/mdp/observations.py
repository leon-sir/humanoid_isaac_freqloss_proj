# Copyright (c) 2025 Zheng Pan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    # from humanoid_isaac_dash.tasks.locomotion.velocity.config.humanoid.ymboy_flip.env.manager_based_rl_env import MyManagerBasedRLYMEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def feet_contact_state(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_state = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
    return contact_state.float()


def feet_pos_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Feet/body positions in the robot body frame.

    The returned tensor is flattened as ``(num_envs, num_bodies * 3)`` following ``asset_cfg.body_ids`` order.
    Each selected body position is first translated by the root position and then rotated into the root body frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_pos_rel_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w.unsqueeze(1)
    foot_pos_b = torch.zeros_like(foot_pos_rel_w)
    for body_idx in range(len(asset_cfg.body_ids)):
        foot_pos_b[:, body_idx, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, foot_pos_rel_w[:, body_idx, :]
        )
    return foot_pos_b.reshape(env.num_envs, -1)


def feet_lin_vel_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Feet/body linear velocities in the robot body frame.

    The returned tensor is flattened as ``(num_envs, num_bodies * 3)`` following ``asset_cfg.body_ids`` order.
    Each selected body velocity is first made relative to the root velocity and then rotated into the root body frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_vel_rel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w.unsqueeze(1)
    foot_vel_b = torch.zeros_like(foot_vel_rel_w)
    for body_idx in range(len(asset_cfg.body_ids)):
        foot_vel_b[:, body_idx, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, foot_vel_rel_w[:, body_idx, :]
        )
    return foot_vel_b.reshape(env.num_envs, -1)


def current_stage(env: MyManagerBasedRLYMEnv) -> torch.Tensor:
    if not hasattr(env, "flip_state_buff") or env.flip_state_buff is None:
        env.flip_state_buff=torch.zeros((env.num_envs, 5, 2), device=env.device, dtype=torch.float32)
        print(f"flip_state_buff is None and be created= {env.flip_state_buff}")
    return env.flip_state_buff[:, :, 0]
