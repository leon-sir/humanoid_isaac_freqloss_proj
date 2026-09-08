# Copyright (c) 2024-2026 ZhengPan
# SPDX-License-Identifier: Apache-2.0

"""Terrain generator with global surface noise and continuous pure-stair columns."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator import TerrainGenerator

from .trimesh.mesh_terrains_cfg import MeshPureStairsTerrainCfg

if TYPE_CHECKING:
    from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg

    from .terrain_generator_global_noise_cfg import TerrainGeneratorWithGlobalNoiseCfg


class TerrainGeneratorWithGlobalNoise(TerrainGenerator):
    """Generate terrains with optional global noise and continuous pure stairs.

    In addition to the standard :class:`isaaclab.terrains.TerrainGenerator`, this
    generator can apply a deterministic fractal perturbation to all sub-terrain
    surfaces. Consecutive :class:`MeshPureStairsTerrainCfg` tiles in a column are
    translated vertically so that the right edge of one tile meets the left edge
    of the next tile.

    When requested by the configuration, the outer border at the positive x end
    is split by terrain column. A pure-stair column's border segment is placed at
    the final stair height, while all other segments retain the standard height.
    This makes the visible and collidable ground continue beyond the final
    curriculum level without adding an overlapping plane at ``z=0``.

    Args:
        cfg: Configuration for the terrain generator.
        device: Device used for terrain-origin and flat-patch tensors.
    """

    cfg: TerrainGeneratorWithGlobalNoiseCfg

    def __init__(self, cfg: TerrainGeneratorWithGlobalNoiseCfg, device: str = "cpu"):
        self._mesh_stair_meshes: list[bool] = []
        self._height_field_stair_meshes: list[bool] = []
        self._pure_stair_edge_heights: dict[int, tuple[float, float]] = {}
        self._pure_stair_previous_right_z: dict[int, float] = {}
        super().__init__(cfg=cfg, device=device)
        if self.cfg.add_fractal_noise:
            self._add_fractal_noise_to_terrain_mesh()

    def _add_sub_terrain(
        self,
        mesh: trimesh.Trimesh,
        origin: np.ndarray,
        row: int,
        col: int,
        sub_terrain_cfg: SubTerrainBaseCfg,
    ):
        """Add a sub-terrain and maintain the height chain of pure-stair columns."""
        if isinstance(sub_terrain_cfg, MeshPureStairsTerrainCfg):
            left_top_z, right_top_z = self._pure_stair_edge_heights.pop(id(mesh))
            if row == 0 or col not in self._pure_stair_previous_right_z:
                # Anchor the first tile at z=0. Later tiles meet the preceding
                # tile exactly at their shared x boundary.
                z_offset = -left_top_z
            else:
                z_offset = self._pure_stair_previous_right_z[col] - left_top_z
            mesh.apply_translation((0.0, 0.0, z_offset))
            origin = origin.copy()
            origin[2] += z_offset
            self._pure_stair_previous_right_z[col] = right_top_z + z_offset
        else:
            # Random generation may change terrain type inside a column. Do not
            # propagate a pure-stair height chain through an unrelated tile.
            self._pure_stair_previous_right_z.pop(col, None)

        self._mesh_stair_meshes.append(
            isinstance(
                sub_terrain_cfg,
                (terrain_gen.MeshPyramidStairsTerrainCfg, MeshPureStairsTerrainCfg),
            )
        )
        self._height_field_stair_meshes.append(
            isinstance(
                sub_terrain_cfg,
                (terrain_gen.HfPyramidStairsTerrainCfg,),
            )
        )
        super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)

    def _get_terrain_mesh(
        self, difficulty: float, cfg: SubTerrainBaseCfg
    ) -> tuple[trimesh.Trimesh, np.ndarray]:
        """Generate a sub-terrain and record pure-stair edge heights."""
        mesh, origin = super()._get_terrain_mesh(difficulty, cfg)
        if isinstance(cfg, MeshPureStairsTerrainCfg):
            step_height = cfg.step_height_range[0] + difficulty * (
                cfg.step_height_range[1] - cfg.step_height_range[0]
            )
            half_stair_length = 0.5 * (cfg.size[0] - cfg.platform_width) - cfg.border_width
            step_count = int(np.ceil(half_stair_length / cfg.step_width))
            direction_sign = 1.0 if cfg.direction == "up" else -1.0
            left_top_z = -direction_sign * step_count * step_height
            right_top_z = direction_sign * step_count * step_height
            self._pure_stair_edge_heights[id(mesh)] = (left_top_z, right_top_z)
        return mesh, origin

    def _generate_curriculum_terrains(self):
        """Generate curriculum rows with an optional exactly-flat prefix.

        Isaac Lab normally adds a random offset inside every curriculum row.
        Consequently, row zero may still contain non-zero stairs even when a
        terrain's difficulty range starts at zero. The configured flat prefix
        removes that offset while preserving the standard behavior afterward.
        """
        if self.cfg.num_flat_start_rows <= 0:
            return super()._generate_curriculum_terrains()

        proportions = np.asarray(
            [sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()],
            dtype=np.float64,
        )
        proportions /= proportions.sum()
        sub_terrain_indices = np.asarray(
            [
                np.min(np.where(col / self.cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
                for col in range(self.cfg.num_cols)
            ],
            dtype=np.int32,
        )
        sub_terrain_cfgs = list(self.cfg.sub_terrains.values())

        lower, upper = self.cfg.difficulty_range
        for col in range(self.cfg.num_cols):
            for row in range(self.cfg.num_rows):
                if row < self.cfg.num_flat_start_rows:
                    difficulty = 0.0
                else:
                    difficulty = (row + self.np_rng.uniform()) / self.cfg.num_rows
                    difficulty = lower + (upper - lower) * difficulty
                mesh, origin = self._get_terrain_mesh(difficulty, sub_terrain_cfgs[sub_terrain_indices[col]])
                self._add_sub_terrain(
                    mesh,
                    origin,
                    row,
                    col,
                    sub_terrain_cfgs[sub_terrain_indices[col]],
                )

    def _add_terrain_border(self):
        """Add the outer border, optionally matching pure-stair terminal heights.

        The custom path preserves the standard border on the negative x and both
        y sides. The positive x side is divided into one segment per terrain
        column so each pure-stair segment can start at its final surface height.
        Each segment keeps the configured border thickness instead of spanning
        vertically between its surface and ``z=0``.
        """
        if not self.cfg.align_terminal_border_height:
            return super()._add_terrain_border()
        if self.cfg.border_width <= 0.0:
            return

        terrain_length = self.cfg.num_rows * self.cfg.size[0]
        terrain_width = self.cfg.num_cols * self.cfg.size[1]
        border_width = self.cfg.border_width

        border_meshes = [
            # Negative-x strip, including its two outer corners.
            self._make_border_box(
                x_bounds=(-border_width, 0.0),
                y_bounds=(-border_width, terrain_width + border_width),
                surface_z=0.0,
            ),
            # Lateral strips also cover the two positive-x outer corners.
            self._make_border_box(
                x_bounds=(0.0, terrain_length + border_width),
                y_bounds=(-border_width, 0.0),
                surface_z=0.0,
            ),
            self._make_border_box(
                x_bounds=(0.0, terrain_length + border_width),
                y_bounds=(terrain_width, terrain_width + border_width),
                surface_z=0.0,
            ),
        ]

        for col in range(self.cfg.num_cols):
            y_min = col * self.cfg.size[1]
            y_max = (col + 1) * self.cfg.size[1]
            surface_z = self._pure_stair_previous_right_z.get(col, 0.0)
            border_meshes.append(
                self._make_border_box(
                    x_bounds=(terrain_length, terrain_length + border_width),
                    y_bounds=(y_min, y_max),
                    surface_z=surface_z,
                )
            )

        self.terrain_meshes.append(trimesh.util.concatenate(border_meshes))

    def _make_border_box(
        self,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        surface_z: float,
    ) -> trimesh.Trimesh:
        """Create a border box aligned with a requested contact-surface height.

        Args:
            x_bounds: Minimum and maximum x coordinates of the box.
            y_bounds: Minimum and maximum y coordinates of the box.
            surface_z: Height corresponding to the standard border's ``z=0``
                contact surface.

        Returns:
            A box mesh with the same vertical convention as Isaac Lab's border.
        """
        x_min, x_max = x_bounds
        y_min, y_max = y_bounds
        transform = np.eye(4)
        transform[:3, -1] = (
            0.5 * (x_min + x_max),
            0.5 * (y_min + y_max),
            surface_z - 0.5 * self.cfg.border_height,
        )
        border = trimesh.creation.box(
            extents=(x_max - x_min, y_max - y_min, abs(self.cfg.border_height)),
            transform=transform,
        )
        # Match Isaac Lab's optimization for the default downward border: keep
        # only triangles close to the contact surface. This avoids unnecessary
        # vertical and bottom collision faces on the large outer extension.
        selector = ~(np.asarray(border.triangles)[:, :, 2] < surface_z - 0.1).any(1)
        border.update_faces(selector)
        return border

    def _add_fractal_noise_to_terrain_mesh(self):
        """Apply deterministic multi-octave height noise to sub-terrain surfaces."""
        vertices = self.terrain_mesh.vertices
        if len(vertices) == 0:
            return

        # Sample in height-field cell coordinates so frequency remains stable
        # when the terrain's physical size changes.
        coords = vertices[:, :2] / self.cfg.horizontal_scale
        noise = np.zeros(vertices.shape[0], dtype=np.float64)
        amplitude = 1.0
        frequency = self.cfg.fractal_noise_frequency
        norm = 0.0
        for octave in range(self.cfg.fractal_noise_octaves):
            phase = 1.61803398875 * (octave + 1)
            octave_noise = np.sin(2.0 * np.pi * frequency * coords[:, 0] + phase) * np.cos(
                2.0 * np.pi * frequency * coords[:, 1] + 0.5 * phase
            )
            noise += amplitude * octave_noise
            norm += amplitude
            amplitude *= self.cfg.fractal_noise_persistence
            frequency *= self.cfg.fractal_noise_lacunarity
        if norm > 0.0:
            noise /= norm

        # Perturb generated sub-terrains, including negative-height surfaces of
        # inverted HF stairs, while leaving the global border untouched.
        surface_mask = np.zeros(vertices.shape[0], dtype=bool)
        noise_amplitude = np.full(vertices.shape[0], self.cfg.fractal_noise_amplitude)
        vertex_offset = 0
        for mesh_index, mesh in enumerate(self.terrain_meshes):
            next_vertex_offset = vertex_offset + len(mesh.vertices)
            is_sub_terrain = mesh_index < len(self._mesh_stair_meshes)
            if is_sub_terrain:
                surface_mask[vertex_offset:next_vertex_offset] = True
            if is_sub_terrain and self._mesh_stair_meshes[mesh_index]:
                noise_amplitude[vertex_offset:next_vertex_offset] = self.cfg.mesh_stair_fractal_noise_amplitude
            elif is_sub_terrain and self._height_field_stair_meshes[mesh_index]:
                noise_amplitude[vertex_offset:next_vertex_offset] = (
                    self.cfg.height_field_stair_fractal_noise_amplitude
                )
            vertex_offset = next_vertex_offset

        vertices[surface_mask, 2] += (
            noise[surface_mask] * noise_amplitude[surface_mask] * self.cfg.vertical_scale
        )
        self.terrain_mesh.vertices = vertices
        self.terrain_mesh.fix_normals()
