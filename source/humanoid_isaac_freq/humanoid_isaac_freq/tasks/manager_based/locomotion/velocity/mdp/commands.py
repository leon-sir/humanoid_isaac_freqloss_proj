# Copyright (c) 2025-2026 Zheng Pan
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import torch

from isaaclab.envs import mdp as lab_mdp
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils


class UniformThresholdVelocityCommand(lab_mdp.UniformVelocityCommand):
    """Uniform velocity command that snaps very small planar commands to zero."""

    cfg: "UniformThresholdVelocityCommandCfg"

    def __init__(self, cfg: "UniformThresholdVelocityCommandCfg", env):
        super().__init__(cfg, env)
        self.accumulated_displacement_xy = torch.zeros(self.num_envs, 2, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self.accumulated_displacement_xy[env_ids] = 0.0
        return extras

    def compute(self, dt: float):
        super().compute(dt)
        active = ~self._env.reset_buf if hasattr(self._env, "reset_buf") else None
        if active is None:
            active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        _, _, yaw = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w[active])
        command = self.vel_command_b[active, :2]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        self.accumulated_displacement_xy[active, 0] += (
            command[:, 0] * cos_yaw - command[:, 1] * sin_yaw
        ) * dt
        self.accumulated_displacement_xy[active, 1] += (
            command[:, 0] * sin_yaw + command[:, 1] * cos_yaw
        ) * dt

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        moving = torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.1
        self.vel_command_b[env_ids, :2] *= moving.unsqueeze(1)

    def get_expected_displacement_xy(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            return self.accumulated_displacement_xy
        return self.accumulated_displacement_xy[env_ids]


@configclass
class UniformThresholdVelocityCommandCfg(lab_mdp.UniformVelocityCommandCfg):
    class_type: type = UniformThresholdVelocityCommand
