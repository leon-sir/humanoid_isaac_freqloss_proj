# Copyright (c) 2024-2026 ZhengPan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the terrain generator with global surface noise."""

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

from .terrain_generator_global_noise import TerrainGeneratorWithGlobalNoise


@configclass
class TerrainGeneratorWithGlobalNoiseCfg(TerrainGeneratorCfg):
    """Configuration for global noise and continuous pure-stair terrain generation."""

    class_type: type = TerrainGeneratorWithGlobalNoise
    """Terrain generator class instantiated by the terrain importer."""

    num_flat_start_rows: int = 0
    """Number of leading curriculum rows generated at exactly zero difficulty."""

    align_terminal_border_height: bool = False
    """Whether to align the positive-x border with each pure-stair column's final height.

    If enabled, the global border on the positive x side is divided by terrain
    column. Pure-stair columns use the height at the right edge of their final
    tile; other columns keep the standard ``z=0`` border surface. The negative x
    and lateral borders are unchanged. Defaults to False.
    """

    add_fractal_noise: bool = False
    """Whether to apply global multi-octave height noise. Defaults to False."""

    fractal_noise_frequency: float = 0.05
    """Base spatial frequency of the fractal noise in height-field cell units."""

    fractal_noise_octaves: int = 10
    """Number of frequency octaves used to construct the fractal noise."""

    fractal_noise_persistence: float = 0.5
    """Amplitude multiplier applied after every noise octave."""

    fractal_noise_lacunarity: float = 2.0
    """Frequency multiplier applied after every noise octave."""

    fractal_noise_amplitude: float = 20.0
    """Noise amplitude for general terrains, expressed in vertical-scale units."""

    height_field_stair_fractal_noise_amplitude: float = 20.0
    """Noise amplitude for height-field stairs, expressed in vertical-scale units."""

    mesh_stair_fractal_noise_amplitude: float = 20.0
    """Noise amplitude for mesh stairs, expressed in vertical-scale units."""
