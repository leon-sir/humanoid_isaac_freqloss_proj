"""Time-domain reward terms local to the 21-DOF frequency-mimic task."""

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg


class JointPosPenaltyMimic(ManagerTermBase):
    """Track the default pose only when both command and body speed are small.

    Moving posture is left to the reference-DC and spectral terms, avoiding an
    instantaneous attraction to the DC center that would suppress the desired
    oscillation.  The returned standstill L2 norm uses a negative reward weight.
    """

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

    def __call__(
        self,
        env,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        stand_still_scale: float,
        velocity_threshold: float,
        command_threshold: float,
    ) -> torch.Tensor:
        del asset_cfg
        joint_pos = self.asset.data.joint_pos[:, self.asset_cfg.joint_ids]
        default_pos = self.asset.data.default_joint_pos[:, self.asset_cfg.joint_ids]
        default_error = torch.linalg.vector_norm(joint_pos - default_pos, dim=1)

        command_speed = torch.linalg.vector_norm(
            env.command_manager.get_command(command_name), dim=1
        )
        body_speed = torch.linalg.vector_norm(self.asset.data.root_lin_vel_b[:, :2], dim=1)
        moving = torch.logical_or(
            command_speed > command_threshold, body_speed > velocity_threshold
        )
        penalty = torch.where(moving, torch.zeros_like(default_error), stand_still_scale * default_error)
        upright_scale = torch.clamp(-self.asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return penalty * upright_scale
