"""Replay absolute retargeted CSV poses in Isaac Sim and analyze all 21 joints.

Run from repository root with the Isaac Lab conda environment. No policy needed.
120-Hz CSV -> linear interpolation at 50 Hz, matching policy collector timing.
Interpolation is NOT an anti-alias filter; content above 25 Hz can alias.
"""
import argparse
import csv
import json
import math
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

NUM_FRAMES = 750
WARMUP_STEPS = 250
SOURCE_FPS = 120.0
SAMPLE_FPS = 50.0
MAX_FREQUENCY_HZ = 25.0
ROOT = Path(__file__).resolve().parents[2]
# DEFAULT_CSV = ROOT / "source/humanoid_isaac_freq/humanoid_isaac_freq/datasets/motions/Neutral_walk_forward_002__A057/Neutral_walk_forward_002__A057.csv"
DEFAULT_CSV = ROOT / "source/humanoid_isaac_freq/humanoid_isaac_freq/datasets/motions/Neutral_walk_forward_002__A057/Neutral_walk_forward_002__A057_periodic_filtered.csv"

def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--num_frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--source_fps", type=float, default=SOURCE_FPS)
    parser.add_argument("--sample_fps", type=float, default=SAMPLE_FPS)
    parser.add_argument("--max_frequency_hz", type=float, default=MAX_FREQUENCY_HZ)
    parser.add_argument("--fast", action="store_true", help="Disable real-time pacing.")
    parser.add_argument("--fast_exit", action="store_true",
                        help="After output/resource cleanup, skip Isaac Sim's hanging Replicator shutdown workflow.")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=False,
                        help="Save an MP4 including warmup in the analysis output directory.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_frames < 3 or args.warmup_steps < 0 or min(args.source_fps, args.sample_fps, args.max_frequency_hz) <= 0:
        parser.error("Require num_frames >= 3, warmup_steps >= 0 and positive frequencies.")
    if args.video:
        args.enable_cameras = True
    launcher = AppLauncher(args)
    try:
        with ExitStack() as resources:
            run(args, launcher.app, resources)
    finally:
        launcher.app.close(wait_for_replicator=False, skip_cleanup=args.fast_exit)


def run(args, app, resources):
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation, Slerp
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from humanoid_isaac_freq.assets.ymbot_boy_21dof import YMBOT_BOY_21DOF_CFG
    from mocap_spectrum_utils import calculate_training_spectrum, write_spectrum_csv

    with args.csv.open(newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        data = np.asarray(list(reader), dtype=np.float64)
    names = [name[:-4] for name in header if name.endswith("_dof")]
    if len(names) != 21 or len(set(names)) != 21 or not np.isfinite(data).all():
        raise ValueError("Expected 21 unique joint columns and finite CSV data.")
    if not np.allclose(np.diff(data[:, 0]), 1):
        raise ValueError("CSV Frame indices must be consecutive; refusing to compress missing frames.")
    source_t = np.arange(len(data)) / args.source_fps
    dt = 1.0 / args.sample_fps
    times = np.arange(1, args.warmup_steps + args.num_frames + 1) * dt
    if times[-1] > source_t[-1]:
        raise ValueError("Motion is too short for warmup + recording; reduce the requested window.")
    q_source = np.deg2rad(data[:, [header.index(name + "_dof") for name in names]])
    q = np.stack([np.interp(times, source_t, col) for col in q_source.T], axis=1)
    xyz = np.stack([np.interp(times, source_t, data[:, header.index("root_translate" + axis)] * .01)
                    for axis in "XYZ"], axis=1)
    angles = data[:, [header.index("root_rotate" + axis) for axis in "XYZ"]]
    quat_xyzw = Slerp(source_t, Rotation.from_euler("xyz", angles, degrees=True))(times).as_quat()
    poses = np.concatenate([xyz, quat_xyzw[:, [3, 0, 1, 2]]], axis=1)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device or "cuda:0"))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    light = sim_utils.DomeLightCfg(intensity=2500.0)
    light.func("/World/light", light)
    cfg = YMBOT_BOY_21DOF_CFG.copy()
    cfg.prim_path = "/World/Robot"
    cfg.spawn.rigid_props.disable_gravity = True
    robot = Articulation(cfg)
    sim.reset()
    out = ROOT / "scripts/output/spectrum_mocap_analysis" / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f") + "_" + args.csv.stem)
    out.mkdir(parents=True)
    video_path = out / "runner_joint_angles_scale_rollout.mp4"
    if args.video:
        # Same camera RGB render-product approach as ManagerBasedRLEnv.render,
        # which the policy collector's Gym RecordVideo wrapper consumes.
        import omni.replicator.core as rep
        import imageio.v2 as imageio
        product = rep.create.render_product("/OmniverseKit_Persp", (1280, 720))
        resources.callback(product.destroy)
        rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        rgb.attach([product])
        resources.callback(rgb.detach, [product])
        writer = imageio.get_writer(str(video_path), fps=args.sample_fps, codec="libx264",
                                   macro_block_size=1)
        resources.callback(writer.close)
    if set(robot.joint_names) != set(names):
        raise ValueError(f"CSV/URDF mismatch: CSV={names}, robot={robot.joint_names}")
    ids = [robot.joint_names.index(name) for name in names]
    defaults = robot.data.default_joint_pos[0, ids].cpu().numpy()
    # Leg scales exactly match the policy collector. Added upper-body values
    # are provisional amplitudes in radians, not joint limits or fitted stats.
    leg_scales = dict(hip_pitch=.55, hip_roll=.10, hip_yaw=.05, knee=.85,
                      ankle_pitch=.55, ankle_roll=.25)
    scales = {name: (leg_scales[name.split("_", 1)[1][:-6]]
                    if name.split("_", 1)[1][:-6] in leg_scales else
                    (.25 if name == "waist_yaw_joint" else .5)) for name in names}
    joint_state = torch.zeros((1, 21), device=sim.device)
    zero_vel = torch.zeros_like(joint_state)
    samples = []
    for index, pose in enumerate(poses):
        if not app.is_running():
            raise RuntimeError("Playback interrupted; no complete analysis window was exported.")
        start = time.monotonic()
        joint_state[:, ids] = torch.as_tensor(q[index], device=sim.device, dtype=torch.float32)
        robot.write_root_pose_to_sim(torch.as_tensor(pose[None], device=sim.device, dtype=torch.float32))
        robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device))
        robot.write_joint_state_to_sim(joint_state, zero_vel)
        # Kinematic replay: render FK without physics stepping or limit clamping.
        sim.forward()
        robot.update(dt)
        sim.set_camera_view(eye=xyz[index] + np.array([3., 3., 2.]), target=xyz[index])
        sim.render()
        if args.video:
            frame = np.asarray(rgb.get_data())
            # Warm the renderer at the same pose, without advancing motion.
            for _ in range(20):
                if frame.size:
                    break
                sim.render()
                frame = np.asarray(rgb.get_data())
            if not frame.size:
                raise RuntimeError("Camera returned no RGB data after render warmup.")
            writer.append_data(np.ascontiguousarray(frame[:, :, :3]))
        if index >= args.warmup_steps:
            samples.append(robot.data.joint_pos[0, ids].detach().cpu().clone())
        if not args.fast and not args.headless:
            time.sleep(max(0., dt - (time.monotonic() - start)))

    if args.video:
        writer.close()
        print(f"[INFO] Video: {video_path} ({len(poses)} frames @ {args.sample_fps:g} FPS)")
    absolute = torch.stack(samples).double()
    z = (absolute - torch.tensor(defaults)) / torch.tensor(list(scales.values()))
    freq, spectrum, power, mean, window_energy = calculate_training_spectrum(z, dt)
    for filename, values, suffix in [("runner_joint_angles_scale.csv", z, "z_normalized"),
                                     ("runner_joint_angles_rad.csv", absolute, "rad")]:
        with (out / filename).open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["sample_index", "time_s", "source_time_s"] + [f"{n}_{suffix}" for n in names])
            for i, row in enumerate(values.tolist()):
                writer.writerow([i, i * dt, times[args.warmup_steps + i], *row])
    write_spectrum_csv(out / "runner_joint_power_scale.csv", freq, spectrum, power, names, args.max_frequency_hz)
    ac = power[1:].sum(0)
    with (out / "runner_joint_power_scale_summary.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["joint_name", "scale_rad", "window_mean_normalized", "total_ac_power_raw",
                         "hann_normalized_ac_energy", "peak_frequency_hz"])
        for j, name in enumerate(names):
            writer.writerow([name, scales[name], mean[j].item(), ac[j].item(),
                             (ac[j] / window_energy).item(), freq[power[1:, j].argmax() + 1].item()])
    for stem, x, y, xlabel, ylabel in [
        ("runner_joint_angles_scale", torch.arange(len(z)) * dt, z, "Time [s]", "(q - q_default) / scale"),
        ("runner_joint_power_scale", freq, power, "Frequency [Hz]", "Single-sided power (raw Hann FFT)")]:
        fig, axes = plt.subplots(math.ceil(len(names)/4), 4, figsize=(18, 24), constrained_layout=True)
        for j, ax in enumerate(axes.flat):
            if j >= len(names):
                ax.set_visible(False)
                continue
            ax.plot(x.numpy(), y[:, j].numpy(), linewidth=1)
            ax.set_title(names[j], fontsize=9)
            ax.set_xlabel(xlabel)
            ax.grid(alpha=.3)
            if "power" in stem:
                ax.set_xlim(0, min(args.max_frequency_hz, args.sample_fps / 2))
        fig.supylabel(ylabel)
        fig.savefig(out / (stem + ".png"), dpi=180)
        plt.close(fig)
    limits = robot.data.joint_pos_limits[0, ids].cpu().numpy()
    violations = ((absolute.numpy() < limits[:, 0]) | (absolute.numpy() > limits[:, 1])).sum(0)
    metadata = dict(video=str(video_path) if args.video else None,
        video_frames=len(poses) if args.video else 0, video_fps=args.sample_fps if args.video else None,
        csv=str(args.csv.resolve()), source_fps=args.source_fps, sample_frequency_hz=args.sample_fps,
        sample_dt_s=dt, num_frames=args.num_frames, warmup_steps=args.warmup_steps,
        window_duration_s=args.num_frames * dt, frequency_resolution_hz=args.sample_fps / args.num_frames,
        source_time_start_s=float(times[args.warmup_steps]), source_time_end_s=float(times[-1]),
        urdf=cfg.spawn.asset_path, joint_names=names, scales_rad=scales,
        default_joint_positions_rad=dict(zip(names, defaults.tolist())),
        joint_limit_violation_frames=dict(zip(names, violations.tolist())),
        interpolation="linear joint/translation; quaternion SLERP; no anti-alias filter",
        source_units="translation cm; joint/Euler angles degrees; root Euler xyz; converted to m/rad/wxyz",
        spectrum="mean removal, symmetric Hann, orthonormal rFFT, doubled interior one-sided power",
        window_energy=window_energy, upper_body_scales="provisional: waist .25 rad, arms .5 rad",
        replay="kinematic state assignment; no PD tracking, no clipping; simulated joint positions analyzed")
    (out / "rollout_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[INFO] Exported 21-DOF mocap analysis: {out}")
    if violations.any():
        print("[WARNING] Joint limit violations found; retained unchanged (see metadata).")


if __name__ == "__main__":
    main()
