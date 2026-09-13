"""Diagnostic importer: unchanged visual/raycast mesh, spatially split colliders.

Import only after AppLauncher. Triangles are assigned by XY centroid, never
cut or duplicated; large border triangles therefore remain large.
"""

import numpy as np
from pxr import UsdGeom, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter
from isaaclab.terrains.utils import create_prim_from_mesh


class SplitCollisionTerrainImporter(TerrainImporter):
    chunk_size = 9.0

    def import_mesh(self, name, mesh):
        super().import_mesh(name, mesh)
        full_path = f"{self.cfg.prim_path}/{name}/mesh"
        full_prim = sim_utils.get_current_stage().GetPrimAtPath(full_path)
        UsdPhysics.CollisionAPI(full_prim).GetCollisionEnabledAttr().Set(False)

        cells = np.floor(mesh.triangles_center[:, :2] / self.chunk_size).astype(np.int64)
        _, assignments = np.unique(cells, axis=0, return_inverse=True)
        triangle_count = 0
        self.collision_chunk_count = int(assignments.max()) + 1
        for index in range(self.collision_chunk_count):
            face_ids = np.flatnonzero(assignments == index)
            chunk = mesh.submesh([face_ids], append=True, repair=False)
            triangle_count += len(chunk.faces)
            path = f"{self.cfg.prim_path}/collision_chunk_{index:04d}"
            create_prim_from_mesh(path, chunk, physics_material=self.cfg.physics_material)
            prim = sim_utils.get_current_stage().GetPrimAtPath(path)
            UsdGeom.Imageable(prim).MakeInvisible()
        if triangle_count != len(mesh.faces):
            raise RuntimeError("Collision partition changed the triangle count")
        print(f"[INFO] Split collision mesh: {self.collision_chunk_count} chunks, "
              f"{triangle_count} triangles; full raycast mesh preserved.", flush=True)
