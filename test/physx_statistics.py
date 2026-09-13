"""Best-effort final-step PhysX statistics, not rollout averages."""


def read_scene_statistics():
    try:
        import isaaclab.sim as sim_utils
        from omni.physx import get_physx_statistics_interface
        from omni.physx.bindings._physx import PhysicsSceneStats
        from pxr import UsdUtils, UsdPhysics, PhysicsSchemaTools

        stage = sim_utils.get_current_stage()
        stage_id = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
        result = {}
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Scene):
                continue
            stats = PhysicsSceneStats()
            path = str(prim.GetPath())
            available = get_physx_statistics_interface().get_physx_scene_statistics(
                stage_id, PhysicsSchemaTools.sdfPathToInt(path), stats
            )
            values = {}
            if available:
                for name in dir(type(stats)):
                    if isinstance(getattr(type(stats), name), property):
                        value = getattr(stats, name)
                        if isinstance(value, (int, float, bool)):
                            values[name] = value
            result[path] = {"available": available, "values": values}
        return result
    except Exception as exc:
        return {"available": False, "error": repr(exc)}
