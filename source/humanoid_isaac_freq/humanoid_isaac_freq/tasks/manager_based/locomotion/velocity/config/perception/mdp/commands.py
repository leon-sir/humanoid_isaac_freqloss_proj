"""Lane-keeping velocity command used by the perception task."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Uniform velocity command with thresholding and lane-keeping yaw control."""

    cfg: UniformThresholdVelocityCommandCfg

    def __init__(self, cfg: UniformThresholdVelocityCommandCfg, env):
        super().__init__(cfg, env)
        self.is_lane_keeping_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.lane_center_y = torch.zeros(self.num_envs, device=self.device)
        self.accumulated_displacement_x = torch.zeros(self.num_envs, device=self.device)
        self.accumulated_displacement_y = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tracking_exp_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tracking_exp_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["command_forward_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["actual_forward_vel"] = torch.zeros(self.num_envs, device=self.device)
        self._heading_body_id: int | None = None
        if self.cfg.heading_asset_cfg is not None:
            self.cfg.heading_asset_cfg.resolve(env.scene)
            if self.cfg.heading_asset_cfg.name != self.cfg.asset_name:
                raise ValueError("heading_asset_cfg.name must match asset_name")
            body_ids = self.cfg.heading_asset_cfg.body_ids
            if isinstance(body_ids, slice) or len(body_ids) != 1:
                raise ValueError("heading_asset_cfg must select exactly one body")
            self._heading_body_id = int(body_ids[0])

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self.accumulated_displacement_x[env_ids] = 0.0
        self.accumulated_displacement_y[env_ids] = 0.0
        return extras

    def compute(self, dt: float):
        super().compute(dt)
        if hasattr(self._env, "reset_buf"):
            env_ids = (~self._env.reset_buf).nonzero(as_tuple=False).flatten()
        else:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) > 0:
            self._accumulate_expected_displacement(dt, env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        self.vel_command_b[env_ids, :2] *= (
            torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.1
        ).unsqueeze(1)
        if self.cfg.lane_keeping_command:
            samples = torch.empty(len(env_ids), device=self.device)
            self.is_lane_keeping_env[env_ids] = (
                samples.uniform_(0.0, 1.0) <= self.cfg.rel_lane_keeping_envs
            )
            self.heading_target[env_ids] = self.cfg.lane_keeping_heading_target
            self.lane_center_y[env_ids] = self._env.scene.env_origins[env_ids, 1]
        else:
            self.is_lane_keeping_env[env_ids] = False
            self.lane_center_y[env_ids] = self._env.scene.env_origins[env_ids, 1]

    def _update_command(self):
        if self.cfg.heading_command:
            env_ids = torch.logical_and(
                self.is_heading_env, ~self.is_lane_keeping_env
            ).nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self._heading_w()[env_ids])
                self.vel_command_b[env_ids, 2] = torch.clip(
                    self.cfg.heading_control_stiffness * error,
                    min=self.cfg.ranges.ang_vel_z[0],
                    max=self.cfg.ranges.ang_vel_z[1],
                )

        if self.cfg.lane_keeping_command:
            env_ids = self.is_lane_keeping_env.nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                heading_error = math_utils.wrap_to_pi(
                    self.heading_target[env_ids] - self._heading_w()[env_ids]
                )
                lateral_error = self.lane_center_y[env_ids] - self.robot.data.root_pos_w[env_ids, 1]
                yaw_command = (
                    self.cfg.lane_heading_gain * heading_error
                    + self.cfg.lane_lateral_gain * lateral_error
                )
                self.vel_command_b[env_ids, 2] = torch.clip(
                    yaw_command,
                    min=self.cfg.ranges.ang_vel_z[0],
                    max=self.cfg.ranges.ang_vel_z[1],
                )

        self.vel_command_b[self.is_standing_env.nonzero(as_tuple=False).flatten(), :] = 0.0

    def _update_metrics(self):
        super()._update_metrics()
        lin_error = torch.sum(
            torch.square(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1
        )
        self.metrics["tracking_exp_vel_xy"] += (
            torch.exp(-lin_error / self.cfg.lin_vel_metrics_std**2) / self._env.max_episode_length
        )
        yaw_error = torch.square(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        self.metrics["tracking_exp_vel_yaw"] += (
            torch.exp(-yaw_error / self.cfg.ang_vel_metrics_std**2) / self._env.max_episode_length
        )
        self.metrics["command_forward_vel"] += (
            torch.clamp(self.vel_command_b[:, 0], min=0.0) / self._env.max_episode_length
        )
        self.metrics["actual_forward_vel"] += (
            torch.clamp(self.robot.data.root_lin_vel_b[:, 0], min=0.0) / self._env.max_episode_length
        )

    def _heading_w(self) -> torch.Tensor:
        if self._heading_body_id is None:
            return self.robot.data.heading_w
        quat = self.robot.data.body_link_quat_w[:, self._heading_body_id, :]
        qw, qx, qy, qz = quat.unbind(dim=-1)
        return torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy.square() + qz.square()))

    def _accumulate_expected_displacement(self, dt: float, env_ids: torch.Tensor):
        _, _, yaw = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w[env_ids])
        vx, vy = self.vel_command_b[env_ids, 0], self.vel_command_b[env_ids, 1]
        self.accumulated_displacement_x[env_ids] += (vx * torch.cos(yaw) - vy * torch.sin(yaw)) * dt
        self.accumulated_displacement_y[env_ids] += (vx * torch.sin(yaw) + vy * torch.cos(yaw)) * dt

    def get_expected_travel_distance(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        return torch.sqrt(
            self.accumulated_displacement_x[env_ids].square()
            + self.accumulated_displacement_y[env_ids].square()
        )

    def get_expected_displacement_xy(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Return accumulated commanded displacement in the world XY frame."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        return torch.stack(
            (
                self.accumulated_displacement_x[env_ids],
                self.accumulated_displacement_y[env_ids],
            ),
            dim=-1,
        )


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    class_type: type = UniformThresholdVelocityCommand
    lane_keeping_command: bool = False
    rel_lane_keeping_envs: float = 0.0
    lane_keeping_heading_target: float = 0.0
    lane_heading_gain: float = 0.0
    lane_lateral_gain: float = 0.0
    heading_asset_cfg: SceneEntityCfg | None = None
    lin_vel_metrics_std: float = 0.5
    ang_vel_metrics_std: float = 0.5
