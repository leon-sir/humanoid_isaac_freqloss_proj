# @Author: Zheng Pan
# @Emial: pan_zheng@nwpu.edu.cn
# Copyright (c) 2025 Zheng Pan
# All rights reserved.

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def heading_error_too_large(
        env: ManagerBasedRLEnv, command_name: str, threshold: float = 1.0,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the heading error too large.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    command = env.command_manager.get_term(command_name)
    if asset_cfg.body_ids == slice(None):
        heading_w = asset.data.heading_w
    else:
        if len(asset_cfg.body_ids) != 1:
            raise ValueError(f"asset_cfg for heading_error_too_large must select exactly one body, got {asset_cfg.body_ids}.")
        quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :]
        qw = quat[:, 0]
        qx = quat[:, 1]
        qy = quat[:, 2]
        qz = quat[:, 3]
        heading_w = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    heading_b = math_utils.wrap_to_pi(heading_w - command.heading_target)

    return torch.abs(heading_b)>=threshold

    # if env.scene.cfg.terrain.terrain_type == "plane":
    #     # we have infinite terrain because it is a plane
    #     return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # elif env.scene.cfg.terrain.terrain_type == "generator":
    #     # obtain the size of the sub-terrains
    #     terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
    #     grid_width, grid_length = terrain_gen_cfg.size
    #     n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
    #     border_width = terrain_gen_cfg.border_width
    #     # compute the size of the map
    #     map_width = n_rows * grid_width + 2 * border_width
    #     map_height = n_cols * grid_length + 2 * border_width

    #     # extract the used quantities (to enable type-hinting)
    #     asset: RigidObject = env.scene[asset_cfg.name]

    #     # check if the agent is out of bounds
    #     x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
    #     y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
    #     return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    # else:
    #     raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")


def feet_distance_y_too_near(
    env: ManagerBasedRLEnv, stance_width_threhold: float = 0.17, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
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
    stance_width_tensor = stance_width_threhold * torch.ones([env.num_envs, 1], device=env.device)

    return torch.abs(footsteps_in_body_frame[:, 0, 1] - footsteps_in_body_frame[:, 1, 1]) < stance_width_tensor[:, 0]


def feet_distance_too_near(
    env: ManagerBasedRLEnv, stance_width_threhold: float = 0.17, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2

    asset: RigidObject = env.scene[asset_cfg.name]

    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    # stance_width_tensor = stance_width_threhold * torch.ones([env.num_envs], device=env.device)

    return distance < stance_width_threhold


def swap_time_too_long(env: ManagerBasedRLEnv, swap_time_threshold: float = 50) -> torch.Tensor:
    """Terminate when the left/right hip-pitch ordering has not changed for too long."""
    required = ("has_not_sweep_time", "left_hip_pitch_id", "right_hip_pitch_id", "prev_left_greater")
    if any(not hasattr(env, name) for name in required):
        raise ValueError("swap-time state is not initialized; use ManagerBasedRLYMEnv")

    robot = env.scene.articulations["robot"]
    left_pos = robot.data.joint_pos[:, env.left_hip_pitch_id].squeeze(-1)
    right_pos = robot.data.joint_pos[:, env.right_hip_pitch_id].squeeze(-1)
    moving = env.command_manager.get_command("base_velocity")[:, 0] > 0.1
    current_left_greater = left_pos > right_pos
    relation_changed = current_left_greater != env.prev_left_greater

    env.has_not_sweep_time[relation_changed | ~moving] = 0
    env.has_not_sweep_time[~relation_changed & moving] += 1
    env.prev_left_greater.copy_(current_left_greater)
    return env.has_not_sweep_time > swap_time_threshold


class CommandedNoProgressTermination(ManagerTermBase):
    """Terminate when commanded expected travel does not produce enough actual XY progress."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.command_name: str = cfg.params["command_name"]
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.window_time: float = cfg.params["window_time"]
        self.min_expected_distance: float = cfg.params["min_expected_distance"]
        self.progress_ratio: float = cfg.params["progress_ratio"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.asset: RigidObject = env.scene[self.asset_cfg.name]
        self.window_elapsed = torch.zeros(env.num_envs, device=env.device)
        self.start_root_pos_xy = self.asset.data.root_pos_w[:, :2].clone()
        self.start_expected_displacement_xy = self._expected_displacement_xy().clone()

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.window_elapsed[env_ids] = 0.0
        self.start_root_pos_xy[env_ids] = self.asset.data.root_pos_w[env_ids, :2]
        self.start_expected_displacement_xy[env_ids] = self._expected_displacement_xy(env_ids)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        window_time: float,
        min_expected_distance: float,
        progress_ratio: float,
        command_threshold: float,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        command = env.command_manager.get_command(command_name)
        has_move_command = torch.norm(command[:, :2], dim=1) > command_threshold
        self.window_elapsed += env.step_dt

        expected_delta = torch.norm(self._expected_displacement_xy() - self.start_expected_displacement_xy, dim=1)
        actual_delta = torch.norm(self.asset.data.root_pos_w[:, :2] - self.start_root_pos_xy, dim=1)

        window_ready = self.window_elapsed >= window_time
        no_progress = actual_delta < progress_ratio * expected_delta
        terminate = window_ready & has_move_command & (expected_delta > min_expected_distance) & no_progress

        refresh = window_ready | ~has_move_command
        refresh_env_ids = refresh.nonzero(as_tuple=False).flatten()
        if len(refresh_env_ids) > 0:
            self.window_elapsed[refresh_env_ids] = 0.0
            self.start_root_pos_xy[refresh_env_ids] = self.asset.data.root_pos_w[refresh_env_ids, :2]
            self.start_expected_displacement_xy[refresh_env_ids] = self._expected_displacement_xy(refresh_env_ids)

        return terminate

    def _expected_displacement_xy(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        command_term = self._env.command_manager.get_term(self.command_name)
        if hasattr(command_term, "get_expected_displacement_xy"):
            return command_term.get_expected_displacement_xy(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)[env_ids]
        return torch.zeros((len(env_ids), 2), device=self._env.device)
