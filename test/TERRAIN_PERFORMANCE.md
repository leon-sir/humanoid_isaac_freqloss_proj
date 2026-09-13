# Perception terrain performance

## Production fix

Both perception tasks now use ContactOffsetTerrainImporterCfg with explicit
contact_offset=0.02 m and rest_offset=0 m. These are config fields, not test-only
class attributes, and therefore appear in env.yaml and support Hydra overrides.
The importer sets collider attributes before simulation initialization, including
generator terrain and ground-plane imports. The default perception mixture now
explicitly uses boxes; continuity and a single collider are preserved.
The collision stack remains 2**29. Long-run training/contact stability still
requires validation; short captures alone do not justify lowering capacity.

IMPORTANT: benchmark_perception_terrain.py deliberately uses the original Lab
importer by default so existing `automatic` controls remain automatic. Pass
`--task_terrain_importer` to validate production defaults; do not combine that
with test-only importer overrides when testing production behavior.

## Current round: explicit terrain contact/rest offsets

Run `bash test/run_offset_gpu_trace.sh`. Four sequential cases use aligned boxes,
one collider, the same checkpoint, 4096 environments and a 2**29 stack:

| Case | Terrain contactOffset | Terrain restOffset |
| --- | --- | --- |
| automatic | unchanged/automatic | unchanged/automatic |
| contact_only | 0.02 m | unchanged/automatic |
| rest_only | unchanged/automatic | 0 m |
| both | 0.02 m | 0 m |

Offsets are authored on the imported terrain collision mesh BEFORE simulation
initialization, not after PhysX cooking. Robot offsets are unchanged. The full
collision audit in every JSON verifies authored settings. This does not expose
the automatic static-terrain backend values directly.

Return `test/output/offset_gpu_trace_*` plus any PhysX warnings/errors. Compare
narrowphase kernel time, collision-stack/contact demand, resets and forces.
The separate single-parameter cases avoid attributing a combined change to the
wrong parameter. 0.02 m is an experimental value, NOT a validated training
recommendation. Changing offsets can change trajectories/contact behavior.
Do not lower the collision stack or adopt this in production based on timing
alone. All code is test-only. Syntax checks passed; full simulation is pending.

## Current round: imported collision attributes and resolved robot offsets

Run `bash test/run_collision_audit.sh` in the Isaac Lab environment, with no
other GPU jobs. This does NOT change contact/rest offsets or collision settings.
It compares aligned/unaligned boxes and writes two JSON files under
`test/output/collision_audit_*`. No Nsight installation is needed for this round.

The audit runs after timing. It traverses terrain and env_0 collision prims,
including instance proxies, and records authored/default USD contact/rest
offsets, mesh approximation, world scaling singular values, bounds and units.
It also reads robot contact/rest offsets from the PhysX tensor backend across
all environments. Shape-slot ordering is not body ordering; padding may occur.

IMPORTANT: USD automatic/default values are not necessarily resolved PhysX
values. Static terrain backend offsets are not exposed by this audit. Missing
values or automatic sentinels must not be treated as literal contact distances.
USD robot world bounds may lag GPU motion and are not a penetration measurement.
The current robot spawn config does not explicitly set collision contact/rest
offsets, which makes checking imported/backend values useful but does not itself
indicate a defect.

## Next round: boxes versus exterior surface, kernel-level comparison

Run `bash test/run_mesh_gpu_trace.sh` on an idle GPU in the Isaac Lab conda
environment. Both cases retain aligned rows, a single collider, original Z,
4096 environments and a 2**29 stack. Only the pure-stair mesh representation
changes. The surface implementation both removes internal faces AND subdivides
external faces; this is not an isolated internal-face ablation.

Return `test/output/mesh_gpu_trace_*`. Compare convexTrimeshNarrowphase,
sortTriangleIndices, midphaseGeneratePairs and solver kernel times alongside
PhysX resource demand. Same checkpoint does not guarantee identical trajectories.
JSON now also includes the number of environments reset during capture and a
final per-body net contact force snapshot. Force magnitudes do not reveal
contact counts, opposing forces can cancel, and reset buffers may be cleared.
These metrics cannot by themselves demonstrate inter-environment collisions or
penetration. Statistics and force reductions run after GPU capture, not inside it.

Shell/Python syntax checks passed; the new full-scale capture awaits execution.

## Current round: GPU kernel timeline and PhysX snapshot

Rigid Z translation did not improve physics timing. Stop modifying geometry for
this round. Run `bash test/run_gpu_trace.sh` in the Isaac Lab environment with
no other GPU jobs. Nsight Systems (`nsys`) must be on PATH.

The script captures aligned/unaligned boxes, 4096 environments, the same policy,
48 warmup steps and 24 measured steps. Capture starts after warmup. NVTX labels
identify physics, sensors, reset and policy calls without synchronizing every
section (whole control-step synchronization remains). Do not use `--profile`
together with `--gpu_trace`.

Results are saved under a timestamped `test/output/gpu_trace_*` folder:
JSON timings plus best-effort final-step PhysX statistics, two `.nsys-rep`
timelines, and CSV kernel/API/NVTX summaries. Send the folder path and any
CUPTI/permission/PhysX errors. A statistics value of zero or unavailable must
not be interpreted as proof of zero GPU contacts; these are snapshots, not
rollout averages. Kernel durations may overlap and must not be equated directly
with wall time. The 24-step trace is for localization, not throughput validation.

Shell/Python syntax and installed Nsight option/report names were checked;
full GPU capture is pending user execution.

## Current experiment: absolute Z position (single collider)

Collision splitting failed: physics rose from 63.49 to 435.61 ms/control step.
Do NOT enable `--split_collision` for this experiment.

The diagnostic generator now supports rigid translation of the final terrain
mesh, terrain origins (and hence environment spawn origins), and world-space
flat patches. Triangle connectivity, relative stair heights, XY positions and
height span are unchanged. Normal training generators are not modified.

Run three cases sequentially on an idle GPU. The third is intentional: when
up/down columns nearly balance, centering may produce too small a translation
to be informative. A +20 m control tests a clearly different absolute position.

```bash
for zcase in original centered shifted; do
  z_args=(--terrain_z_mode original)
  if [ "$zcase" = centered ]; then
    z_args=(--terrain_z_mode centered)
  elif [ "$zcase" = shifted ]; then
    z_args=(--terrain_z_mode original --terrain_z_offset 20)
  fi
  RSL_RL_DREAM_ENABLE_CUDNN_RNN=1 python test/benchmark_perception_terrain.py \
    --headless --num_envs 4096 --mesh boxes --stack_power 29 --align_rows \
    "${z_args[@]}" --profile --warmup 48 --steps 240 \
    --checkpoint logs/rsl_rl/rough_12dof_perception_rnn/2026-09-08_13-44-33_Perf-PureBoxes/model_199.pt \
    --output "test/output/z_${zcase}.json"
done
```

Return all three JSON files and any simulation errors. `terrain_translation`
records the actual shift and mesh/origin Z bounds. First check that the shifts
are nonzero, then compare physics timing. This tests absolute placement, NOT
the effect of reducing terrain height span. It is not guaranteed to improve
performance. Custom terms depending on absolute world Z may also change behavior;
results are diagnostic, not proof of physically identical policy trajectories.

The translation helper passed checks for unchanged triangles and consistently
translated vertices, origins and patches; full simulation validation is pending.

## Previous test: partition collision shapes, preserve continuous terrain

The aligned/unaligned profile localized 49.86 ms of the 58.61 ms per-control-step
difference to `physics`; depth camera timing stayed approximately unchanged.
Test whether partitioning the single terrain collider helps PhysX.

Both cases below KEEP row alignment enabled and use identical generated
triangles, world positions, origins and the full raycast mesh. The split importer
disables collision on that full mesh and creates invisible collision-only meshes
grouped by triangle-centroid XY cells (9 m). It does not cut or duplicate faces.
This changes collision shape organization/cooking, not the intended surface.
It is diagnostic: changes in contact handling at shape boundaries remain possible.
Large border triangles are not subdivided by this test.

Run sequentially on an otherwise idle GPU:

```bash
for mode in monolithic split; do
  collision_args=()
  if [ "$mode" = split ]; then
    collision_args=(--split_collision)
  fi
  RSL_RL_DREAM_ENABLE_CUDNN_RNN=1 python test/benchmark_perception_terrain.py \
    --headless --num_envs 4096 --mesh boxes --stack_power 29 --align_rows \
    "${collision_args[@]}" --collision_chunk_size 9 --profile --warmup 48 --steps 240 \
    --checkpoint logs/rsl_rl/rough_12dof_perception_rnn/2026-09-08_13-44-33_Perf-PureBoxes/model_199.pt \
    --output "test/output/collision_${mode}.json"
done
```

Return `collision_monolithic.json` and `collision_split.json`, plus any PhysX
errors. Do not compare only total time: compare physics and sensor times and
check `collision_chunk_count`. A speedup here would support a collision-mesh
organization issue; it would not alone identify an internal PhysX algorithm.
The importer exists only under `test/`; normal task registration is unchanged.

## Next test: locate the accumulated-elevation overhead

The 200-iteration experiments showed mean collection times (iterations 50--199)
of 2.860 s for aligned boxes, 1.478 s for unaligned boxes, and 1.531 s for
pyramid stairs, all with the same 2**29 collision stack.

Run these two short diagnostic rollouts sequentially on an otherwise idle GPU.
They load the SAME policy; they do not train or write into the checkpoint run.
Outputs go to `test/output/`. Keep the full 4096 environments for the final test.

```bash
for alignment in align_rows no-align_rows; do
  RSL_RL_DREAM_ENABLE_CUDNN_RNN=1 python test/benchmark_perception_terrain.py \
    --headless --num_envs 4096 --mesh boxes --stack_power 29 \
    --"$alignment" --profile --warmup 48 --steps 240 \
    --checkpoint logs/rsl_rl/rough_12dof_perception_rnn/2026-09-08_13-44-33_Perf-PureBoxes/model_199.pt \
    --output "test/output/profile_${alignment}.json"
done
```

Send both JSON files and any PhysX overflow/errors. Profiling synchronizes GPU
work and changes throughput, so these numbers should NOT be compared directly
to TensorBoard training collection times. Compare the two profiled cases.

- `section_mean_step_s`: inclusive wall time per control step. Sensor updates
  can be inside rewards/observations, so these entries must not be summed.
- `section_exclusive_mean_step_s`: subtracts instrumented child calls; for
  example, observations exclude the separately timed sensor updates they invoke.
- `sensor/camera`: actual depth camera buffer updates, including its noise processing.
- Other `sensor/*`: actual height scanner/contact sensor buffer updates.
- `physics`: simulation step, including any work performed inside that call.
- `section_calls`: helps identify differing update/reset frequencies.

This retains sensor functionality; no zeroed images or disabled observations are
used. Alignment changes global geometry and can still change policy trajectories.
If a section dominates the difference, investigate that section with a more
targeted controlled query before changing production terrain behavior.

## Findings

The Base runs `2026-09-08_09-38-06` (stairs + Perlin) and
`2026-09-08_10-07-49` (Perlin only) have median collection times of 2.918 s and
1.605 s over iterations 50--120. Median learning times are 0.912 s and 0.953 s.
Both terrain geometry and collision-stack capacity changed, so these runs do
not isolate the effect of either variable. Startup flat-patch sampling also
changed; this is not repeated inside every rollout.

`gpu_collision_stack_size` reserves narrowphase scratch space. Increasing it
from 2**26 to 2**29 changes capacity from 64 MiB to 512 MiB; it does not request
eight times as much computation. More demanding mesh queries can cause both
larger scratch requirements and slower execution. Measure the capacity effect
using the SAME geometry and a capacity that does not overflow.

Pure stairs previously concatenated complete boxes. Each 9 m tile with 0.30 m
treads contained 23 boxes / 276 triangles, including hidden interfaces and
terrain-wide triangles. The new `mesh_mode="surface"` emits the exterior of
the same solid, retaining risers, side borders, outside walls and the bottom.
It subdivides faces with `max_face_edge=1.0` m (quad edges; diagonals <= sqrt(2)
m). This deliberately increases triangle count while removing internal faces
and bounding individual query primitives. It is an optimization candidate,
not a measured full-scale speedup yet. `mesh_mode="boxes"` selects the old mesh.

Validation covers 24 combinations of direction, tread width, border and
difficulty: bounds, origin, volume, watertightness, winding, maximum edge size
and exterior ray hits agree with the original. These comparisons assume global
vertex noise is disabled, as in the current configuration. Changing mesh
sampling changes the interpolation of global vertex noise if enabled.

A 16-env Isaac Sim smoke test with the full terrain grid, depth sensors and
2**26 stack completed successfully, both with zero actions and with the
`2026-09-08_09-38-06_RNN_CNN_Depth-Base/model_0.pt` policy loaded. This does NOT establish that 2**26 is
sufficient for 4096 environments, high difficulty or falling robots.

## Controlled measurement

Run in the Isaac Lab v23 conda environment with no other GPU training process.
Use separate processes and the same checkpoint, environment count, seed and
terrain configuration. The benchmark reconstructs the terrain mixture from
`perception/terrain_cfg.py`, overriding the temporary Perlin-only env setting.

```bash
RSL_RL_DREAM_ENABLE_CUDNN_RNN=1 python test/benchmark_perception_terrain.py \
  --headless --num_envs 4096 --mesh boxes --stack_power 29 \
  --checkpoint <checkpoint.pt> --output /tmp/stairs_boxes_29.json

RSL_RL_DREAM_ENABLE_CUDNN_RNN=1 python test/benchmark_perception_terrain.py \
  --headless --num_envs 4096 --mesh surface --stack_power 29 \
  --checkpoint <checkpoint.pt> --output /tmp/stairs_surface_29.json

RSL_RL_DREAM_ENABLE_CUDNN_RNN=1 python test/benchmark_perception_terrain.py \
  --headless --num_envs 4096 --mesh surface --stack_power 26 \
  --checkpoint <checkpoint.pt> --output /tmp/stairs_surface_26.json

RSL_RL_DREAM_ENABLE_CUDNN_RNN=1 python test/benchmark_perception_terrain.py \
  --headless --num_envs 4096 --mesh perlin --stack_power 29 \
  --checkpoint <checkpoint.pt> --output /tmp/perlin_29.json

RSL_RL_DREAM_ENABLE_CUDNN_RNN=1 python test/benchmark_perception_terrain.py \
  --headless --num_envs 4096 --mesh perlin --stack_power 26 \
  --checkpoint <checkpoint.pt> --output /tmp/perlin_26.json
```

Reject measurements with PhysX overflow/errors. Keep a larger stack if needed.
Repeat with `--profile` for synchronized physics, scene-update, reward,
observation and reset timings. Profiling adds synchronization overhead; use
unprofiled runs for total throughput. Section timers may nest, so do not sum
them to infer exact total time. Sensor work can occur lazily during rewards or
observations. Without `--checkpoint`, the benchmark uses zero actions and only
serves as an environment stepping diagnostic, not a trained-gait benchmark.

To train with the configured terrain mixture, leave `TERRAIN_ABLATION` unset:

```bash
RSL_RL_DREAM_ENABLE_CUDNN_RNN=1 python scripts/rsl_rl_dream/train.py \
  --task DreamLab-Velocity-Rough-YMBOY12DOF-Perception-RNN-Base-v0 \
  --headless --no-export_network_structure
```

This keeps the 2**29 stack. Do not lower the training stack until full-scale
measurements confirm adequate capacity.

```bash
python test/check_pure_stairs_mesh.py
```

## Follow-up: accumulated stair elevation versus pyramid stairs

The surface mesh did not improve collection time in the measured training runs.
Isaac Lab's pyramid stairs also use complete boxes, so hidden box faces alone
do not explain the difference. Test accumulated pure-stair row elevation next.
This is a hypothesis, not an established cause.

The perception env now accepts `TERRAIN_ABLATION` before configuration construction:

| Value | Terrain |
| --- | --- |
| `none` (default) | Unmodified configured mixture |
| `pure_boxes` | Original boxes, continuous height alignment between rows |
| `pure_unaligned` | Same boxes, without accumulated row elevation |
| `pyramid` | Replace each pure-stair entry with its up/down pyramid counterpart |
| `perlin` | Perlin-only control, noise scale 0.02 |

All ablation modes use a 2**29 collision stack. The first three experimental
mixtures preserve proportions, step widths, step-height ranges and the Perlin
component. Run the following sequentially, without other GPU jobs. Do not compare
against old runs with different command ranges or other settings.

```bash
TERRAIN_ABLATION=pure_boxes RSL_RL_DREAM_ENABLE_CUDNN_RNN=1 python scripts/rsl_rl_dream/train.py \
  --task DreamLab-Velocity-Rough-YMBOY12DOF-Perception-RNN-Base-v0 \
  --headless --no-export_network_structure --max_iterations 200 --run_name Perf-PureBoxes && \
TERRAIN_ABLATION=pure_unaligned RSL_RL_DREAM_ENABLE_CUDNN_RNN=1 python scripts/rsl_rl_dream/train.py \
  --task DreamLab-Velocity-Rough-YMBOY12DOF-Perception-RNN-Base-v0 \
  --headless --no-export_network_structure --max_iterations 200 --run_name Perf-PureUnaligned && \
TERRAIN_ABLATION=pyramid RSL_RL_DREAM_ENABLE_CUDNN_RNN=1 python scripts/rsl_rl_dream/train.py \
  --task DreamLab-Velocity-Rough-YMBOY12DOF-Perception-RNN-Base-v0 \
  --headless --no-export_network_structure --max_iterations 200 --run_name Perf-Pyramid
```

`pure_unaligned` deliberately introduces discontinuous tile seams and changes
global terrain elevations; it is NOT a replacement training terrain. Reset and
curriculum statistics may change, so a speedup would motivate a controlled
physics/raycast profile rather than prove a specific BVH or collision mechanism.
The selected mesh types and `align_pure_stair_rows` are saved in `env.yaml`.
The standalone benchmark above reconstructs its own terrain and does not use
these env ablation settings for measurement; use these training commands for
this experiment.

## References

- [PhysX GPU dynamics memory configuration](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/structPxGpuDynamicsMemoryConfig.html): collision stack is narrowphase scratch-space capacity.
- [PhysX geometry documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.1.3/docs/Geometry.html): discusses tessellation of large triangles; this motivates a controlled test, not a guaranteed speedup for this workload.
