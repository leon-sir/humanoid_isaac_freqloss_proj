"""Terrain import with explicit, serialized PhysX contact distances."""

import math

from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from pxr import Usd, UsdPhysics, PhysxSchema


class ContactOffsetTerrainImporter(TerrainImporter):
    """Set terrain offsets during import, before simulation initialization.

    Keep the imported geometry, materials, origin handling and raycast paths
    unchanged. Only terrain collision prims are modified, never robot shapes.
    """

    def __init__(self, cfg, **kwargs):
        if not math.isfinite(cfg.contact_offset) or cfg.contact_offset <= 0.0:
            raise ValueError("Terrain contact_offset must be finite and positive")
        if not math.isfinite(cfg.rest_offset) or cfg.rest_offset >= cfg.contact_offset:
            raise ValueError("Terrain rest_offset must be finite and below contact_offset")
        super().__init__(cfg, **kwargs)

    def _set_offsets(self, name):
        root = sim_utils.get_current_stage().GetPrimAtPath(f"{self.cfg.prim_path}/{name}")
        count = 0
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                api.CreateContactOffsetAttr().Set(self.cfg.contact_offset)
                api.CreateRestOffsetAttr().Set(self.cfg.rest_offset)
                count += 1
        if count == 0:
            raise RuntimeError(f"No collision prims found under {root.GetPath()}")

    def import_mesh(self, name, mesh):
        super().import_mesh(name, mesh)
        self._set_offsets(name)

    def import_usd(self, name, usd_path):
        super().import_usd(name, usd_path)
        self._set_offsets(name)

    def import_ground_plane(self, name, size=(2.0e6, 2.0e6)):
        super().import_ground_plane(name, size)
        self._set_offsets(name)


@configclass
class ContactOffsetTerrainImporterCfg(TerrainImporterCfg):
    """Explicit terrain offsets in meters; persisted in training env.yaml."""

    class_type: type = ContactOffsetTerrainImporter
    contact_offset: float = 0.02
    rest_offset: float = 0.0
