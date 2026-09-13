"""Test-only explicit static-terrain offsets, authored before PhysX starts."""

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter
from pxr import PhysxSchema


class OffsetTerrainImporter(TerrainImporter):
    contact_offset = None
    rest_offset = None

    def import_mesh(self, name, mesh):
        super().import_mesh(name, mesh)
        path = f"{self.cfg.prim_path}/{name}/mesh"
        prim = sim_utils.get_current_stage().GetPrimAtPath(path)
        api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        if self.contact_offset is not None:
            api.CreateContactOffsetAttr().Set(self.contact_offset)
        if self.rest_offset is not None:
            api.CreateRestOffsetAttr().Set(self.rest_offset)
        print(f"[INFO] Terrain offsets at {path}: contact={api.GetContactOffsetAttr().Get()}, "
              f"rest={api.GetRestOffsetAttr().Get()}", flush=True)
