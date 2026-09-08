"""Perception-local termination terms."""

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg


def lateral_error_too_large(
    env,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the robot leaves its sub-terrain centerline in world Y."""
    if threshold <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold}")

    asset: RigidObject = env.scene[asset_cfg.name]
    terrain_center_y_w = env.scene.env_origins[:, 1]
    lateral_error_w = asset.data.root_pos_w[:, 1] - terrain_center_y_w
    return lateral_error_w.abs() > threshold


def swap_time_too_long(env, swap_time_threshold: float = 50) -> torch.Tensor:
    required = ("has_not_sweep_time", "left_hip_pitch_id", "right_hip_pitch_id", "prev_left_greater")
    if any(not hasattr(env, name) for name in required):
        raise ValueError("swap-time state is not initialized; use the perception ManagerBasedRLYMEnv")

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
