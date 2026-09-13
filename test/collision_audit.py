"""Read-only USD collision audit plus resolved robot tensor-backend offsets."""


def audit_collisions(env):
    import numpy as np
    import isaaclab.sim as sim_utils
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema

    stage = sim_utils.get_current_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    transforms = UsdGeom.XformCache()

    def attribute(attr):
        value = attr.Get() if attr else None
        if isinstance(value, float) and not np.isfinite(value):
            value = str(value)  # Preserve automatic sentinels in valid JSON.
        if value is not None and not isinstance(value, (str, int, float, bool)):
            value = str(value)
        return {"value": value, "authored": bool(attr and attr.HasAuthoredValueOpinion())}

    records = []
    for root_path in (env.cfg.scene.terrain.prim_path, "/World/envs/env_0"):
        root = stage.GetPrimAtPath(root_path)
        if not root:
            continue
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            physics = PhysxSchema.PhysxCollisionAPI(prim)
            matrix = transforms.GetLocalToWorldTransform(prim)
            linear = np.array([[matrix[i][j] for j in range(3)] for i in range(3)])
            bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            record = {
                "path": str(prim.GetPath()), "type": prim.GetTypeName(),
                "collision_enabled": attribute(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()),
                "contact_offset_usd": attribute(physics.GetContactOffsetAttr()),
                "rest_offset_usd": attribute(physics.GetRestOffsetAttr()),
                "world_linear_singular_values": np.linalg.svd(linear, compute_uv=False).tolist(),
                "world_bounds": [list(bounds.GetMin()), list(bounds.GetMax())],
                "applied_schemas": list(prim.GetAppliedSchemas()),
            }
            if prim.IsA(UsdGeom.Mesh):
                record["approximation_usd"] = attribute(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr())
                counts = UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get()
                record["face_count"] = len(counts) if counts is not None else None
            records.append(record)

    runtime = {}
    view = env.scene["robot"].root_physx_view
    for name in ("contact_offsets", "rest_offsets"):
        try:
            data = getattr(view, f"get_{name}")()
            values = data.detach().cpu().numpy() if hasattr(data, "detach") else np.asarray(data)
            runtime[name] = {
                "shape": list(values.shape), "first_env_shape_slots": values[0].tolist(),
                "min": float(values.min()), "max": float(values.max()),
                "unique_values": np.unique(values).tolist(),
            }
        except Exception as exc:
            runtime[name] = {"unavailable": repr(exc)}
    return {
        "stage_meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "colliders": records, "robot_backend_offsets": runtime,
        "limitations": "USD schema values may be automatic sentinels, not resolved PhysX values. "
                        "Robot backend arrays use shape slots, not body indices, and may contain padding. "
                        "Static terrain backend offsets are not exposed by this audit. "
                        "USD robot transforms may lag GPU state; bounds are for scale inspection, not penetration diagnosis.",
    }
