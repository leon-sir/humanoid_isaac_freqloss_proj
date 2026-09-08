"""Perception-local reward implementations whose source baseline differs from the flat task."""

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster


def base_height_l2(
    env,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize height error using finite per-environment terrain ray hits."""
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits = sensor.data.ray_hits_w[..., 2]
        valid_hits = torch.isfinite(ray_hits) & (torch.abs(ray_hits) < 1.0e6)
        hit_count = valid_hits.sum(dim=1)
        terrain_height = (
            torch.where(valid_hits, ray_hits, torch.zeros_like(ray_hits)).sum(dim=1)
            / hit_count.clamp_min(1)
        )
        adjusted_target = torch.where(
            hit_count > 0,
            target_height + terrain_height,
            asset.data.root_pos_w[:, 2],
        )
    else:
        adjusted_target = target_height
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward
