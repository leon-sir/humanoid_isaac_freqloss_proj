# Copyright (c) 2024-2026 ZhengPan
# SPDX-License-Identifier: Apache-2.0

"""Custom trimesh terrain functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh

if TYPE_CHECKING:
    from . import mesh_terrains_cfg


def _interp_range(value_range: tuple[float, float], difficulty: float) -> float:
    return float(value_range[0] + difficulty * (value_range[1] - value_range[0]))


def _make_top_box(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    top_z: float,
    bottom_z: float,
) -> trimesh.Trimesh:
    height = max(top_z - bottom_z, 1.0e-4)
    dims = (max(x1 - x0, 1.0e-4), max(y1 - y0, 1.0e-4), height)
    center = ((x0 + x1) * 0.5, (y0 + y1) * 0.5, bottom_z + height * 0.5)
    return trimesh.creation.box(dims, trimesh.transformations.translation_matrix(center))


def _side_sign(higher_side: str, lane_name: str) -> float:
    if lane_name == higher_side:
        return 1.0
    return -1.0


def _single_side_stair_height_magnitudes(num_steps: int, step_height: float) -> list[float]:
    if num_steps <= 1:
        return [0.5 * step_height]
    return [
        min(step_index + 0.5, num_steps - step_index - 0.5) * step_height
        for step_index in range(num_steps)
    ]


def single_side_stairs_terrain(
    difficulty: float,
    cfg: mesh_terrains_cfg.MeshSingleSideStairsTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate y-direction stairs that are extruded along the walking direction.

    The tile extends from ``(0, 0)`` to ``cfg.size``. Heading 0 is assumed to walk along +x. The terrain is split at
    the y center line. For ``higher_side="left"``, the +y side rises from the center and then tapers back to zero
    near the tile edge, while the -y side mirrors it with negative heights.
    """
    if cfg.step_width <= 0.0:
        raise ValueError(f"step_width must be positive, got {cfg.step_width}.")
    if cfg.border_width < 0.0 or 2.0 * cfg.border_width >= min(cfg.size):
        raise ValueError(f"border_width must be in [0, {0.5 * min(cfg.size)}), got {cfg.border_width}.")
    if not 0.0 <= cfg.origin_x_ratio <= 1.0:
        raise ValueError(f"origin_x_ratio must be in [0, 1], got {cfg.origin_x_ratio}.")

    step_height = _interp_range(cfg.step_height_range, difficulty)
    if step_height < 0.0:
        raise ValueError(f"step_height_range must resolve to a non-negative value, got {step_height}.")

    x_min = cfg.border_width
    x_max = cfg.size[0] - cfg.border_width
    y_min = cfg.border_width
    y_max = cfg.size[1] - cfg.border_width
    y_mid = 0.5 * (y_min + y_max)
    inner_x = x_max - x_min
    half_stepped_y = 0.5 * (y_max - y_min)
    num_side_steps = max(1, int(np.ceil(half_stepped_y / cfg.step_width)))
    height_magnitudes = _single_side_stair_height_magnitudes(num_side_steps, step_height)

    lane_tops = []
    meshes_list: list[trimesh.Trimesh] = []
    for height_mag in height_magnitudes:
        right_top = _side_sign(cfg.higher_side, "right") * height_mag
        left_top = _side_sign(cfg.higher_side, "left") * height_mag
        lane_tops += [right_top, left_top]

    bottom_z = min(0.0, min(lane_tops)) - max(cfg.base_thickness, 1.0e-3)

    for step_index in range(num_side_steps):
        height_mag = height_magnitudes[step_index]
        right_top = _side_sign(cfg.higher_side, "right") * height_mag
        left_top = _side_sign(cfg.higher_side, "left") * height_mag

        right_y0 = max(y_mid - (step_index + 1) * cfg.step_width, y_min)
        right_y1 = y_mid - step_index * cfg.step_width
        left_y0 = y_mid + step_index * cfg.step_width
        left_y1 = min(y_mid + (step_index + 1) * cfg.step_width, y_max)

        meshes_list.append(_make_top_box(x_min, x_max, right_y0, right_y1, right_top, bottom_z))
        meshes_list.append(_make_top_box(x_min, x_max, left_y0, left_y1, left_top, bottom_z))

    if cfg.border_width > 0.0:
        border_top = min(lane_tops)
        meshes_list.append(_make_top_box(0.0, cfg.size[0], 0.0, y_min, border_top, bottom_z))
        meshes_list.append(_make_top_box(0.0, cfg.size[0], y_max, cfg.size[1], border_top, bottom_z))
        meshes_list.append(_make_top_box(0.0, x_min, y_min, y_max, border_top, bottom_z))
        meshes_list.append(_make_top_box(x_max, cfg.size[0], y_min, y_max, border_top, bottom_z))

    origin = np.array(
        [
            x_min + cfg.origin_x_ratio * inner_x,
            0.5 * cfg.size[1],
            0.0,
        ]
    )

    return meshes_list, origin


def pure_stairs_terrain(
    difficulty: float,
    cfg: mesh_terrains_cfg.MeshPureStairsTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate stairs that change height monotonically along x.

    A flat platform of length :attr:`cfg.platform_width` is centered at the
    sub-terrain origin. Flat borders surround the stair field on all four
    sides. For ``direction="up"`` the stair height increases toward +x;
    ``direction="down"`` reverses that height profile.
    """
    if cfg.step_width <= 0.0:
        raise ValueError(f"step_width must be positive, got {cfg.step_width}.")
    if cfg.platform_width <= 0.0 or cfg.platform_width + 2.0 * cfg.border_width > cfg.size[0]:
        raise ValueError(
            "platform_width must be positive and fit between the two x borders, "
            f"got platform_width={cfg.platform_width}, border_width={cfg.border_width}, size={cfg.size}."
        )
    if cfg.border_width < 0.0 or 2.0 * cfg.border_width >= min(cfg.size):
        raise ValueError(
            f"border_width must be in [0, {0.5 * min(cfg.size)}), got {cfg.border_width}."
        )
    if cfg.direction not in ("up", "down"):
        raise ValueError(f"direction must be 'up' or 'down', got {cfg.direction!r}.")

    step_height = _interp_range(cfg.step_height_range, difficulty)
    if step_height < 0.0:
        raise ValueError(f"step_height_range must resolve to a non-negative value, got {step_height}.")

    x_mid = 0.5 * cfg.size[0]
    platform_x0 = x_mid - 0.5 * cfg.platform_width
    platform_x1 = x_mid + 0.5 * cfg.platform_width
    direction_sign = 1.0 if cfg.direction == "up" else -1.0
    stair_x_min = cfg.border_width
    stair_x_max = cfg.size[0] - cfg.border_width
    stair_y_min = cfg.border_width
    stair_y_max = cfg.size[1] - cfg.border_width

    left_step_count = int(np.ceil((platform_x0 - stair_x_min) / cfg.step_width))
    right_step_count = int(np.ceil((stair_x_max - platform_x1) / cfg.step_width))
    left_border_top_z = -direction_sign * left_step_count * step_height
    right_border_top_z = direction_sign * right_step_count * step_height
    minimum_top_z = min(0.0, left_border_top_z, right_border_top_z)
    bottom_z = min(0.0, minimum_top_z) - max(cfg.base_thickness, 1.0e-3)

    meshes_list: list[trimesh.Trimesh] = [
        _make_top_box(platform_x0, platform_x1, stair_y_min, stair_y_max, 0.0, bottom_z)
    ]

    # Build the -x half outwards from the center platform. A clipped final
    # strip guarantees complete coverage when step_width does not divide the
    # available length exactly.
    for step_index in range(left_step_count):
        x1 = platform_x0 - step_index * cfg.step_width
        x0 = max(stair_x_min, x1 - cfg.step_width)
        top_z = -direction_sign * (step_index + 1) * step_height
        meshes_list.append(_make_top_box(x0, x1, stair_y_min, stair_y_max, top_z, bottom_z))

    # Build the +x half outwards from the center platform.
    for step_index in range(right_step_count):
        x0 = platform_x1 + step_index * cfg.step_width
        x1 = min(stair_x_max, x0 + cfg.step_width)
        top_z = direction_sign * (step_index + 1) * step_height
        meshes_list.append(_make_top_box(x0, x1, stair_y_min, stair_y_max, top_z, bottom_z))

    # Finish both x ends with flat platforms spanning the complete y axis.
    # Neighboring curriculum rows align these end heights in the generator,
    # yielding a continuous 2 * border_width platform across each row seam.
    if cfg.border_width > 0.0:
        meshes_list.append(_make_top_box(0.0, stair_x_min, 0.0, cfg.size[1], left_border_top_z, bottom_z))
        meshes_list.append(
            _make_top_box(stair_x_max, cfg.size[0], 0.0, cfg.size[1], right_border_top_z, bottom_z)
        )
        # Heading points along x, so these two flat strips are the robot's
        # left/right borders. Exclude the x-end platforms to avoid overlapping
        # collision meshes at the four corners.
        meshes_list.append(_make_top_box(stair_x_min, stair_x_max, 0.0, stair_y_min, 0.0, bottom_z))
        meshes_list.append(
            _make_top_box(stair_x_min, stair_x_max, stair_y_max, cfg.size[1], 0.0, bottom_z)
        )

    origin = np.array([x_mid, 0.5 * cfg.size[1], 0.0])
    return meshes_list, origin
