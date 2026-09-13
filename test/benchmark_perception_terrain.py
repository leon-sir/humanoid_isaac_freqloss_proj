"""Compare terrain meshes and collision-stack capacities on identical rollouts.

Run each case in a separate process, with no concurrent training. A checkpoint
is optional; without one this measures zero-action stepping, not trained gait.
--profile synchronizes each measured section and adds profiling overhead.
"""

import argparse
import importlib.metadata as metadata
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

from isaaclab.app import AppLauncher

# Match train/play's import roots. Leaving scripts/ on sys.path would make its
# rsl_rl_dream directory part of the algorithm library's namespace package;
# Dream's class resolver could then accidentally import train.py and launch Kit.
scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
sys.path = [entry for entry in sys.path if Path(entry).resolve() != scripts_dir]
sys.path.insert(0, str(scripts_dir / "rsl_rl_dream"))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="DreamLab-Velocity-Rough-YMBOY12DOF-Perception-RNN-Base-v0")
parser.add_argument("--mesh", choices=("boxes", "surface", "perlin"), default="surface")
parser.add_argument("--stack_power", type=int, choices=(26, 27, 28, 29, 30), default=29)
parser.add_argument("--max_face_edge", type=float, default=1.0)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--warmup", type=int, default=48)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--profile", action="store_true")
parser.add_argument("--task_terrain_importer", action="store_true", help="Use the production task importer, including its explicit offsets.")
parser.add_argument("--collision_audit", action="store_true", help="Export collision USD attributes and robot backend offsets after measurement.")
parser.add_argument("--gpu_trace", action="store_true", help="Nsight capture range and NVTX labels; no section synchronizations.")
parser.add_argument("--split_collision", action="store_true",
                    help="Keep full raycast mesh, partition only PhysX colliders by XY.")
parser.add_argument("--collision_chunk_size", type=float, default=9.0)
parser.add_argument("--terrain_contact_offset", type=float, default=None, help="Test-only terrain contact offset in meters; default automatic.")
parser.add_argument("--terrain_rest_offset", type=float, default=None, help="Test-only terrain rest offset in meters; default automatic.")
parser.add_argument("--terrain_z_mode", choices=("original", "centered"), default="original")
parser.add_argument("--terrain_z_offset", type=float, default=0.0,
                    help="Additional rigid Z translation in meters; origins and flat patches move too.")
parser.add_argument("--align_rows", action=argparse.BooleanOptionalAction, default=True,
                    help="Accumulate pure-stair row elevations; --no-align_rows disables it.")
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.gpu_trace and args.profile:
    parser.error("Use --gpu_trace OR --profile, not both")
if args.steps <= 0 or args.warmup < 0:
    parser.error("steps must be positive and warmup must be non-negative")
if not 0.0 < args.collision_chunk_size < float("inf"):
    parser.error("collision_chunk_size must be finite and positive")
if not -float("inf") < args.terrain_z_offset < float("inf"):
    parser.error("terrain_z_offset must be finite")
if args.terrain_contact_offset is not None and not 0 < args.terrain_contact_offset < float("inf"):
    parser.error("terrain_contact_offset must be finite and positive")
if args.terrain_rest_offset is not None and not -float("inf") < args.terrain_rest_offset < float("inf"):
    parser.error("terrain_rest_offset must be finite")
if args.terrain_contact_offset is not None and args.terrain_rest_offset is not None:
    if args.terrain_contact_offset <= args.terrain_rest_offset:
        parser.error("terrain_contact_offset must exceed terrain_rest_offset")
if args.split_collision and (args.terrain_contact_offset is not None or args.terrain_rest_offset is not None):
    parser.error("Do not combine offset and split-collision experiments")
os.environ["RSL_RL_PLAY"] = "0"  # Preserve the full training terrain grid.
os.environ["TERRAIN_ABLATION"] = "none"  # This script controls terrain explicitly.
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
import humanoid_isaac_freq.tasks  # noqa: F401
from humanoid_isaac_freq.tasks.manager_based.locomotion.velocity.config.perception.terrain_cfg import (
    PERCEPTION_ROUGH_TERRAINS_CFG,
)
from humanoid_isaac_freq.terrains.height_field.perlin_terrain_cfg import PerlinPlaneTerrainCfg
from humanoid_isaac_freq.terrains.trimesh.mesh_terrains_cfg import MeshPureStairsTerrainCfg


def main():
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    if not args.task_terrain_importer:
        # Keep historical automatic-offset controls valid after the production fix.
        from isaaclab.terrains import TerrainImporter
        cfg.scene.terrain.class_type = TerrainImporter
    cfg.seed = 42
    cfg.scene.num_envs = args.num_envs
    if args.device is not None:
        cfg.sim.device = args.device
    cfg.sim.physx.gpu_collision_stack_size = 2**args.stack_power
    # Override the temporary Perlin ablation in the env cfg explicitly.
    cfg.scene.terrain.terrain_generator = PERCEPTION_ROUGH_TERRAINS_CFG.copy()
    generator = cfg.scene.terrain.terrain_generator
    from translated_terrain import TranslatedTerrainGenerator
    TranslatedTerrainGenerator.center_z = args.terrain_z_mode == "centered"
    TranslatedTerrainGenerator.offset_z = args.terrain_z_offset
    generator.class_type = TranslatedTerrainGenerator
    generator.align_pure_stair_rows = args.align_rows
    if args.mesh == "perlin":
        generator.sub_terrains = {"perlin_plane": PerlinPlaneTerrainCfg(proportion=1.0, noise_scale=0.02)}
    else:
        for terrain in generator.sub_terrains.values():
            if isinstance(terrain, MeshPureStairsTerrainCfg):
                terrain.mesh_mode = args.mesh
                terrain.max_face_edge = args.max_face_edge
    cfg.log_dir = None
    # Pin every terrain ray sensor to the original full mesh in BOTH cases.
    # Otherwise RayCaster's first-child lookup could select a collision chunk.
    for sensor in vars(cfg.scene).values():
        if hasattr(sensor, "mesh_prim_paths"):
            if sensor.mesh_prim_paths != [cfg.scene.terrain.prim_path]:
                raise ValueError(f"Unexpected raycast targets: {sensor.mesh_prim_paths}")
            sensor.mesh_prim_paths = [f"{cfg.scene.terrain.prim_path}/terrain"]
    if args.split_collision:
        from split_collision_terrain import SplitCollisionTerrainImporter
        SplitCollisionTerrainImporter.chunk_size = args.collision_chunk_size
        cfg.scene.terrain.class_type = SplitCollisionTerrainImporter
    if args.terrain_contact_offset is not None or args.terrain_rest_offset is not None:
        from offset_terrain import OffsetTerrainImporter
        OffsetTerrainImporter.contact_offset = args.terrain_contact_offset
        OffsetTerrainImporter.rest_offset = args.terrain_rest_offset
        cfg.scene.terrain.class_type = OffsetTerrainImporter
    agent = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent = handle_deprecated_rsl_rl_cfg(agent, metadata.version("rsl-rl-dream-lib"))
    env = gym.make(args.task, cfg=cfg)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        raw = env.unwrapped
        sync = lambda: torch.cuda.synchronize(raw.device) if str(raw.device).startswith("cuda") else None
        policy = None
        reset_policy = None
        if args.checkpoint:
            from rsl_rl_dream.runners import OnPolicyRunner

            torch.backends.cudnn.enabled = os.environ.get("RSL_RL_DREAM_ENABLE_CUDNN_RNN", "0") == "1"
            runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=raw.device)
            runner.load(args.checkpoint)
            policy = runner.get_inference_policy(device=raw.device)
            policy_module = getattr(runner.alg, "policy", getattr(runner.alg, "actor_critic", None))
            reset_policy = getattr(policy, "reset", getattr(policy_module, "reset", None))
            if reset_policy is None:
                raise RuntimeError("Cannot locate policy.reset for recurrent rollout evaluation")
        obs = wrapped.get_observations()
        zero = torch.zeros((raw.num_envs, raw.action_manager.total_action_dim), device=raw.device)
        measured = False
        totals = defaultdict(float)
        calls = defaultdict(int)
        exclusive = defaultdict(float)
        timing_stack = []
        reset_counts = []
        original_reset = raw._reset_idx

        def counted_reset(env_ids):
            if measured:
                reset_counts.append(len(env_ids))
            return original_reset(env_ids)

        raw._reset_idx = counted_reset

        def instrument(owner, name, label):
            original = getattr(owner, name)

            def timed(*a, **kw):
                if not measured:
                    return original(*a, **kw)
                if args.gpu_trace:
                    torch.cuda.nvtx.range_push(label)
                    try:
                        return original(*a, **kw)
                    finally:
                        torch.cuda.nvtx.range_pop()
                sync()
                start = time.perf_counter()
                frame = [0.0]
                timing_stack.append(frame)
                try:
                    return original(*a, **kw)
                finally:
                    sync()
                    elapsed = time.perf_counter() - start
                    timing_stack.pop()
                    totals[label] += elapsed
                    exclusive[label] += elapsed - frame[0]
                    calls[label] += 1
                    if timing_stack:
                        timing_stack[-1][0] += elapsed

            setattr(owner, name, timed)

        if args.profile or args.gpu_trace:
            for owner, name, label in (
                (raw.sim, "step", "physics"),
                (raw.scene, "update", "scene_update"),
                (raw.reward_manager, "compute", "rewards"),
                (raw.observation_manager, "compute", "observations"),
                (raw.termination_manager, "compute", "terminations"),
                (raw.scene, "write_data_to_sim", "write_data_to_sim"),
                (raw, "_reset_idx", "reset"),
            ):
                instrument(owner, name, label)
            # Sensors update lazily when .data is accessed, including inside
            # rewards/observations. Time their actual buffer update, not access.
            for name, sensor in raw.scene.sensors.items():
                instrument(sensor, "_update_buffers_impl", f"sensor/{name}")
            if policy is not None:
                holder = type("PolicyTimer", (), {})()
                holder.infer = policy
                instrument(holder, "infer", "policy_inference")
                policy = holder.infer
        durations = []
        for step in range(args.warmup + args.steps):
            measured = step >= args.warmup
            if args.gpu_trace and step == args.warmup:
                sync()
                torch.cuda.cudart().cudaProfilerStart()
            sync()
            start = time.perf_counter()
            with torch.inference_mode():
                action = zero if policy is None else policy(obs)
                obs, _, dones, _ = wrapped.step(action)
                if reset_policy is not None:
                    reset_policy(dones)
            sync()
            if measured:
                durations.append(time.perf_counter() - start)
        if args.gpu_trace:
            sync()
            torch.cuda.cudart().cudaProfilerStop()
        # Snapshot AFTER measurement: do not add statistics queries to the hot path.
        from physx_statistics import read_scene_statistics
        physics_statistics = read_scene_statistics()
        # This is a net-force snapshot, NOT contact-point or penetration data.
        # Reset bodies may already have cleared sensor buffers at this point.
        contact_snapshot = {}
        if "contact_forces" in raw.scene.sensors:
            sensor = raw.scene.sensors["contact_forces"]
            forces = torch.linalg.vector_norm(sensor.data.net_forces_w, dim=-1)
            for index, name in enumerate(sensor.body_names):
                values = forces[:, index]
                contact_snapshot[name] = {
                    "envs_above_1N": int((values > 1.0).sum().item()),
                    "mean_net_force_N": float(values.mean().item()),
                    "max_net_force_N": float(values.max().item()),
                }
        result = {
            "task_terrain_importer": args.task_terrain_importer,
            "terrain_contact_offset_requested_m": args.terrain_contact_offset,
            "terrain_rest_offset_requested_m": args.terrain_rest_offset,
            "reset_env_count_total": sum(reset_counts),
            "reset_env_counts_per_call": reset_counts,
            "net_contact_force_final_snapshot": contact_snapshot,
            "gpu_trace": args.gpu_trace,
            "physx_statistics_final_snapshot": physics_statistics,
            "mesh": args.mesh,
            "align_rows": args.align_rows,
            "terrain_z_mode": args.terrain_z_mode,
            "terrain_translation": TranslatedTerrainGenerator.report,
            "split_collision": args.split_collision,
            "collision_chunk_size": args.collision_chunk_size,
            "collision_chunk_count": getattr(raw.scene.terrain, "collision_chunk_count", 1),
            "seed": cfg.seed,
            "warmup": args.warmup,
            "steps": args.steps,
            "step_dt": raw.step_dt,
            "stack_bytes": 2**args.stack_power,
            "max_face_edge": args.max_face_edge,
            "num_envs": args.num_envs,
            "checkpoint": args.checkpoint,
            "profile": args.profile,
            "mean_step_s": sum(durations) / len(durations),
            "estimated_24_step_rollout_s": 24 * sum(durations) / len(durations),
            "section_mean_step_s": {k: v / args.steps for k, v in totals.items()},
            "section_exclusive_mean_step_s": {k: v / args.steps for k, v in exclusive.items()},
            "section_calls": dict(calls),
            "note": "Synchronized diagnostic: inclusive sections overlap. Exclusive times subtract instrumented children only.",
        }
        if args.collision_audit:
            from collision_audit import audit_collisions
            result["collision_audit"] = audit_collisions(raw)
        print(json.dumps(result, indent=2), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        env.close()


try:
    main()
except Exception:
    # Kit shutdown can terminate Python before an uncaught exception is printed.
    traceback.print_exc()
    raise
finally:
    app.close()
