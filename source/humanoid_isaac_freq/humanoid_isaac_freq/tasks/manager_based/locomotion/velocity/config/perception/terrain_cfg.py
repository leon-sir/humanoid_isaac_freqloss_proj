"""Rough-terrain mixture used by the 12-DoF perception task."""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import FlatPatchSamplingCfg

from humanoid_isaac_freq.terrains.height_field.perlin_terrain_cfg import PerlinPlaneTerrainCfg
from humanoid_isaac_freq.terrains.terrain_generator_global_noise_cfg import (
    TerrainGeneratorWithGlobalNoiseCfg,
)
from humanoid_isaac_freq.terrains.trimesh.mesh_terrains_cfg import (
    MeshPureStairsTerrainCfg,
    MeshSingleSideStairsTerrainCfg,
)


PYRAMID_RATIO = 0.5
PURE_RATIO = 0.4


PERCEPTION_ROUGH_TERRAINS_CFG = TerrainGeneratorWithGlobalNoiseCfg(
    curriculum=True,
    size=(9.0, 9.0),
    border_width=30.0,
    align_terminal_border_height=True,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # "stairs_up_28": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=PYRAMID_RATIO / 3,
        #     step_height_range=(0.0, 0.25),
        #     step_width=0.28,
        #     platform_width=2.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "stairs_up_30": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=PYRAMID_RATIO / 3,
        #     step_height_range=(0.0, 0.25),
        #     step_width=0.30,
        #     platform_width=2.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "stairs_up_32": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=PYRAMID_RATIO / 5,
        #     step_height_range=(0.0, 0.23),
        #     step_width=0.32,
        #     platform_width=2.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "stairs_up_35": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=PYRAMID_RATIO / 5,
        #     step_height_range=(0.0, 0.23),
        #     step_width=0.35,
        #     platform_width=2.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "stairs_down_30": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=PYRAMID_RATIO / 2,
        #     step_height_range=(0.0, 0.23),
        #     step_width=0.30,
        #     platform_width=2.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        "pure_stair_up_30": MeshPureStairsTerrainCfg(
            mesh_mode="boxes",
            proportion=1/4,
            step_height_range=(0.05, 0.18),
            step_width=0.30,
            platform_width=2.0,
            border_width=1.0,
            direction="up",
        ),
        "pure_stair_up_40": MeshPureStairsTerrainCfg(
            mesh_mode="boxes",
            proportion=1/4,
            step_height_range=(0.05, 0.18),
            step_width=0.40,
            platform_width=2.0,
            border_width=1.0,
            direction="up",
        ),
        # "pure_stair_up_40": MeshPureStairsTerrainCfg(
        #     proportion=PURE_RATIO / 8 * 3,
        #     step_height_range=(0.0, 0.23),
        #     step_width=0.40,
        #     platform_width=2.0,
        #     border_width=1.0,
        #     direction="up",
        # ),
        "pure_stair_down_30": MeshPureStairsTerrainCfg(
            mesh_mode="boxes",
            proportion=1/4,
            step_height_range=(0.05, 0.18),
            step_width=0.30,
            platform_width=2.0,
            border_width=1.0,
            direction="down",
        ),
        # "pure_stair_down_40": MeshPureStairsTerrainCfg(
        #     proportion=PURE_RATIO / 8,
        #     step_height_range=(0.0, 0.23),
        #     step_width=0.40,
        #     platform_width=2.0,
        #     border_width=1.0,
        #     direction="down",
        # ),
        # "left_high_y_stairs": MeshSingleSideStairsTerrainCfg(
        #     proportion=0.0,
        #     step_height_range=(0.0, 0.20),
        #     step_width=0.3,
        #     higher_side="left",
        #     border_width=0.0,
        # ),
        # "right_high_y_stairs": MeshSingleSideStairsTerrainCfg(
        #     proportion=0.0,
        #     step_height_range=(0.0, 0.20),
        #     step_width=0.3,
        #     higher_side="right",
        #     border_width=0.0,
        # ),
        # "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.0),
        "perlin_rough": PerlinPlaneTerrainCfg(
            proportion=1/4,
            noise_scale=[0.0, 0.1],
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                )
            },
        ),
        # "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.0,
        #     slope_range=(0.0, 0.35),
        #     platform_width=2.0,
        #     inverted=False,
        # ),
        # "slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.0,
        #     slope_range=(0.0, 0.35),
        #     platform_width=2.0,
        #     border_width=0.25,
        # ),
    },
)
