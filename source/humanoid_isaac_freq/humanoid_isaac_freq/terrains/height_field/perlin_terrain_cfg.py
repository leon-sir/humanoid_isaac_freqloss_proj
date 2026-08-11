# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.utils import configclass

from . import perlin_terrain


@configclass
class PerlinPlaneTerrainCfg(HfTerrainBaseCfg):
    """Configuration for an InstinctLab-style Perlin height-field plane."""

    function = perlin_terrain.perlin_plane_terrain

    noise_scale: float | list[float] = 0.02
    noise_frequency: int = 20
    fractal_octaves: int = 2
    fractal_lacunarity: float = 2.0
    fractal_gain: float = 0.25
    centering: bool = True
