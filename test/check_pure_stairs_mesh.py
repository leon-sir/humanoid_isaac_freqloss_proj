"""Run without Isaac Sim: python test/check_pure_stairs_mesh.py."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import trimesh


path = Path(__file__).resolve().parents[1] / (
    "source/humanoid_isaac_freq/humanoid_isaac_freq/terrains/trimesh/mesh_terrains.py"
)
spec = importlib.util.spec_from_file_location("stair_mesh", path)
terrain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(terrain)


class PureStairMeshTest(unittest.TestCase):
    def test_exterior_matches_boxes(self):
        rng = np.random.default_rng(42)
        for direction in ("up", "down"):
            for width in (0.3, 0.4):
                for border in (0.0, 1.0):
                    for difficulty in (0.0, 0.25, 1.0):
                        with self.subTest(direction=direction, width=width, border=border, difficulty=difficulty):
                            cfg = SimpleNamespace(
                                size=(9.0, 9.0),
                                step_height_range=(0.0, 0.23),
                                step_width=width,
                                platform_width=2.0,
                                border_width=border,
                                direction=direction,
                                base_thickness=0.1,
                                mesh_mode="boxes",
                                max_face_edge=1.0,
                            )
                            boxes, old_origin = terrain.pure_stairs_terrain(difficulty, cfg)
                            old = trimesh.util.concatenate(boxes)
                            cfg.mesh_mode = "surface"
                            meshes, new_origin = terrain.pure_stairs_terrain(difficulty, cfg)
                            new = meshes[0]
                            self.assertTrue(new.is_watertight)
                            self.assertTrue(new.is_winding_consistent)
                            np.testing.assert_allclose(new_origin, old_origin)
                            np.testing.assert_allclose(new.bounds, old.bounds)
                            np.testing.assert_allclose(new.volume, sum(box.volume for box in boxes))
                            self.assertLessEqual(new.edges_unique_length.max(), np.sqrt(2) + 1e-8)
                            # Probe all exterior directions, including stair risers and the bottom.
                            center = old.bounds.mean(axis=0)
                            radial = rng.normal(size=(100, 3))
                            radial /= np.linalg.norm(radial, axis=1, keepdims=True)
                            starts = center + radial * 20
                            targets = rng.uniform(old.bounds[0], old.bounds[1], size=(100, 3))
                            directions = targets - starts
                            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
                            distances = []
                            for mesh in (old, new):
                                loc, ids, _ = mesh.ray.intersects_location(starts, directions, multiple_hits=False)
                                distance = np.full(len(starts), np.inf)
                                distance[ids] = np.linalg.norm(loc - starts[ids], axis=1)
                                distances.append(distance)
                            np.testing.assert_allclose(*distances, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
