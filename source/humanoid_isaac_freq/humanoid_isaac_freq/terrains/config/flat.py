# Copyright (c) 2026 Zheng Pan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.terrains import TerrainGeneratorCfg

from ..height_field import PerlinPlaneTerrainCfg


PERLIN_FLAT_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=3.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=1.0,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "perlin_flat": PerlinPlaneTerrainCfg(
            proportion=1.0,
            noise_scale=[0.05, 0.15],
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
        )
    },
)
