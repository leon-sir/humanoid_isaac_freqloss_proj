# Copyright (c) 2024-2026 ZhengPan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

from . import mesh_terrains


@configclass
class MeshSingleSideStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for y-direction stairs with persistent left/right height bias.

    Heading 0 is assumed to move along +x, and +y is treated as the robot's left side. The terrain is extruded along
    x and stepped along y, so the robot can keep walking forward while one side gets higher and the other gets lower.
    """

    function = mesh_terrains.single_side_stairs_terrain

    border_width: float = 0.0
    """The flat border width around the terrain in m."""

    step_height_range: tuple[float, float] = MISSING
    """The min/max height difference between adjacent lateral stair strips in m."""

    step_width: float = MISSING
    """The width of each lateral stair strip along y in m."""

    higher_side: Literal["left", "right"] = "left"
    """Which side gets higher away from the center. ``left`` corresponds to +y."""

    origin_x_ratio: float = 0.5
    """Terrain origin x-position as a fraction of inner terrain length."""

    base_thickness: float = 0.1
    """Minimum mesh thickness below the lowest top surface in m."""


@configclass
class MeshPureStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for bordered stairs stepped monotonically along x."""

    function = mesh_terrains.pure_stairs_terrain

    step_height_range: tuple[float, float] = MISSING
    """The min/max height difference between adjacent steps in m."""

    step_width: float = MISSING
    """The length of each step along x in m."""

    platform_width: float = 1.0
    """Length of the flat center platform along x in m."""

    border_width: float = 0.0
    """Width of the flat border around the stair field on all four sides in m."""

    direction: Literal["up", "down"] = "up"
    """Height direction while moving along +x from the center platform."""

    base_thickness: float = 0.1
    """Minimum mesh thickness below the lowest top surface in m."""
