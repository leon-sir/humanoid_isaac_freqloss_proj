# Copyright (c) 2026 Zheng Pan
# SPDX-License-Identifier: Apache-2.0

"""Small Isaac Lab 2.2.3-compatible environment extension for swap-time state."""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg


class ManagerBasedRLYMEnv(ManagerBasedRLEnv):
    """Manager-based environment with state used by ``swap_time_too_long``."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        self.has_not_sweep_time = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.prev_left_greater = torch.zeros(cfg.scene.num_envs, dtype=torch.bool, device=cfg.sim.device)
        self.left_hip_pitch_id = None
        self.right_hip_pitch_id = None
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def load_managers(self):
        robot = self.scene.articulations["robot"]
        self.left_hip_pitch_id = robot.find_joints("left_hip_pitch_joint")[0]
        self.right_hip_pitch_id = robot.find_joints("right_hip_pitch_joint")[0]
        super().load_managers()

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        self.has_not_sweep_time[env_ids] = 0
        robot = self.scene.articulations["robot"]
        left_pos = robot.data.joint_pos[env_ids, self.left_hip_pitch_id].squeeze(-1)
        right_pos = robot.data.joint_pos[env_ids, self.right_hip_pitch_id].squeeze(-1)
        self.prev_left_greater[env_ids] = left_pos > right_pos
