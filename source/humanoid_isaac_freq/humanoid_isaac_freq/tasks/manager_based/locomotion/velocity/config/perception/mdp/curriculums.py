"""Perception-local curriculum diagnostics."""

from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter


def terrain_levels_vel(
    env, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> dict[str, torch.Tensor]:
    """Update terrain difficulty from traveled and commanded distance."""
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = (
        distance
        < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    )
    move_down &= ~move_up
    stay = ~(move_up | move_down)
    terrain.update_env_origins(env_ids, move_up, move_down)
    levels = terrain.terrain_levels.float()
    return {
        "terrain_levels": torch.mean(levels),
        "terrain_level_hist_8_9": torch.mean((levels >= 8).float()),
        "terrain_stay_rate": torch.mean(stay.float()),
    }


def check_base_height(
    env,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = SceneEntityCfg("height_scanner_base"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    base_height_w = asset.data.root_pos_w[env_ids, 2]
    if sensor_cfg is None or sensor_cfg.name not in env.scene.sensors:
        return torch.mean(base_height_w)
    ray_hits = env.scene[sensor_cfg.name].data.ray_hits_w[env_ids, ..., 2]
    valid_hits = torch.isfinite(ray_hits) & (torch.abs(ray_hits) < 1.0e6)
    hit_count = valid_hits.sum(dim=1)
    terrain_height = (
        torch.where(valid_hits, ray_hits, torch.zeros_like(ray_hits)).sum(dim=1)
        / hit_count.clamp_min(1)
    )
    relative_height = torch.where(hit_count > 0, base_height_w - terrain_height, base_height_w)
    return torch.mean(relative_height)
