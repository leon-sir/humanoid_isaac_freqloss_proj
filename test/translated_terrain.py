"""Diagnostic rigid Z translation; import after launching Isaac Sim."""

from humanoid_isaac_freq.terrains.terrain_generator_global_noise import TerrainGeneratorWithGlobalNoise


def translate_generated_terrain(generator, center_z=False, offset_z=0.0):
    """Translate the final mesh, world-space origins and sampled flat patches."""
    before = generator.terrain_mesh.bounds[:, 2].copy()
    shift = float(offset_z) - (float(before.mean()) if center_z else 0.0)
    generator.terrain_mesh.apply_translation((0.0, 0.0, shift))
    generator.terrain_origins[..., 2] += shift
    for patches in generator.flat_patches.values():
        patches[..., 2] += shift
    return {
        "shift_z_m": shift,
        "mesh_z_bounds_before_m": before.tolist(),
        "mesh_z_bounds_after_m": generator.terrain_mesh.bounds[:, 2].tolist(),
        "origin_z_bounds_after_m": [
            float(generator.terrain_origins[..., 2].min()),
            float(generator.terrain_origins[..., 2].max()),
        ],
    }


class TranslatedTerrainGenerator(TerrainGeneratorWithGlobalNoise):
    center_z = False
    offset_z = 0.0
    report = None

    def __init__(self, cfg, device="cpu"):
        super().__init__(cfg, device)
        type(self).report = translate_generated_terrain(self, self.center_z, self.offset_z)
        print(f"[INFO] Terrain Z translation: {type(self).report}", flush=True)
