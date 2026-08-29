# Copyright (c) 2026 Zheng Pan
# SPDX-License-Identifier: Apache-2.0

"""Collect a 12-DoF rollout and export spectra normalized exactly like training.

The analysis uses control-step samples and the same preprocessing as
``JointFrequencyAnalyzer``: ``z=(q-q_default)/scale``, unwindowed mean removal,
a non-periodic Hann window, orthonormal RFFT, and single-sided power.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata as metadata
import json
import math
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS_DIR / "rsl_rl"))
import cli_args  # noqa: E402, isort: skip


# -----------------------------------------------------------------------------
# Collection configuration
# -----------------------------------------------------------------------------
# Use a shared in-distribution operating point: this policy was trained with vx in [0, 1] m/s.
COMMAND_VEL = 0.6
VELOCITY_COMMAND = (COMMAND_VEL, 0.0, 0.0)
INITIAL_BASE_VELOCITY_BODY = (0.0, 0.0, 0.0)

TASK_NAME = "FreqLab-Velocity-Flat-YMBOY12DOF-FreqReward"
CHECK_POINT = "logs/rsl_rl/flat_12dof_freq_reward/2026-08-12_16-36-14_freq_rewards/model_2999.pt"
OUTPUT_ROOT = _SCRIPTS_DIR / "output/spectrum_analysis"

NUM_FRAMES = 750
WARMUP_STEPS = 250
MAX_FREQUENCY_HZ = 25.0
PLOT_DPI = 180
SPECTRUM_EPS = 1.0e-8
VIDEO = True

JOINT_SCALES = {
    "left_hip_pitch_joint": 0.55,
    "right_hip_pitch_joint": 0.55,
    "left_hip_roll_joint": 0.10,
    "right_hip_roll_joint": 0.10,
    "left_hip_yaw_joint": 0.05,
    "right_hip_yaw_joint": 0.05,
    "left_knee_joint": 0.85,
    "right_knee_joint": 0.85,
    "left_ankle_pitch_joint": 0.55,
    "right_ankle_pitch_joint": 0.55,
    "left_ankle_roll_joint": 0.25,
    "right_ankle_roll_joint": 0.25,
}
ACTUATED_JOINT_NAMES = list(JOINT_SCALES)


parser = argparse.ArgumentParser(description="Collect scale-normalized joint spectra from a policy rollout.")
parser.add_argument("--task", type=str, default=TASK_NAME, help="Gym task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config registry key.")
parser.add_argument("--num_envs", type=int, default=1, help="Must be 1 for a contiguous trajectory.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Normalized time-history CSV path. Defaults to <output root>/<checkpoint run>/runner_joint_angles_scale.csv.",
)
parser.add_argument(
    "--spectrum_output",
    type=Path,
    default=None,
    help="Single-sided power CSV path. Defaults to <output root>/<checkpoint run>/runner_joint_power_scale.csv.",
)
parser.add_argument("--num_frames", type=int, default=NUM_FRAMES, help="Recorded control-step samples.")
parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS, help="Unrecorded policy control steps.")
parser.add_argument(
    "--max_frequency_hz",
    type=float,
    default=MAX_FREQUENCY_HZ,
    help="Highest written FFT bin, capped at Nyquist.",
)
parser.add_argument("--keep_randomization", action="store_true", help="Keep domain-randomization events.")
parser.add_argument(
    "--video",
    action=argparse.BooleanOptionalAction,
    default=VIDEO,
    help="Record one camera-following MP4 rollout.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Approximately run in real time.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

if args_cli.num_envs != 1:
    parser.error("--num_envs must be 1 so one CSV is one contiguous trajectory.")
if args_cli.num_frames < 2:
    parser.error("--num_frames must be at least 2.")
if args_cli.warmup_steps < 0:
    parser.error("--warmup_steps cannot be negative.")
if not math.isfinite(args_cli.max_frequency_hz) or args_cli.max_frequency_hz <= 0.0:
    parser.error("--max_frequency_hz must be finite and positive.")

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402
import torch  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

os.environ["RSL_RL_PLAY"] = "1"

import humanoid_isaac_freq.tasks  # noqa: E402, F401
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


class FixedVelocityCommand:
    """Return one command and keep the command-manager buffer synchronized."""

    def __init__(self, command: tuple[float, float, float]):
        self.command = command

    def __call__(self, env) -> torch.Tensor:
        command = torch.tensor(self.command, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
        command_term = env.command_manager.get_term("base_velocity")
        if not hasattr(command_term, "vel_command_b"):
            raise RuntimeError("The base_velocity command term has no vel_command_b buffer.")
        command_term.vel_command_b.copy_(command)
        return command


def _disable_if_present(container, names: tuple[str, ...]) -> None:
    for name in names:
        if hasattr(container, name):
            setattr(container, name, None)


def _set_zero_range(event_cfg, parameter_name: str) -> None:
    if event_cfg is None:
        return
    ranges = event_cfg.params.get(parameter_name)
    if isinstance(ranges, dict):
        event_cfg.params[parameter_name] = {name: (0.0, 0.0) for name in ranges}
    elif ranges is not None:
        event_cfg.params[parameter_name] = (0.0, 0.0)


def _configure_environment(env_cfg, command_callable: FixedVelocityCommand) -> None:
    env_cfg.scene.num_envs = 1
    if args_cli.video:
        # IsaacLab's built-in asset-root follow mode preserves the configured
        # world-frame eye/lookat offsets while translating them with the robot.
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.asset_name = "robot"
    env_cfg.observations.policy.enable_corruption = False
    if getattr(env_cfg.observations, "critic", None) is not None:
        env_cfg.observations.critic.enable_corruption = False

    velocity_observation = env_cfg.observations.policy.velocity_commands
    velocity_observation.func = command_callable
    velocity_observation.params = {}

    command_cfg = env_cfg.commands.base_velocity
    vx, vy, wz = VELOCITY_COMMAND
    command_cfg.ranges.lin_vel_x = (vx, vx)
    command_cfg.ranges.lin_vel_y = (vy, vy)
    command_cfg.ranges.ang_vel_z = (wz, wz)
    command_cfg.heading_command = False
    command_cfg.rel_heading_envs = 0.0
    command_cfg.rel_standing_envs = 0.0
    command_cfg.resampling_time_range = (1.0e9, 1.0e9)

    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "commanded_no_progress"):
        env_cfg.terminations.commanded_no_progress = None

    initial_vx, initial_vy, initial_wz = map(float, INITIAL_BASE_VELOCITY_BODY)
    env_cfg.scene.robot.init_state.lin_vel = (initial_vx, initial_vy, 0.0)
    env_cfg.scene.robot.init_state.ang_vel = (0.0, 0.0, initial_wz)
    reset_base = getattr(env_cfg.events, "randomize_reset_base", None)
    if reset_base is not None:
        _set_zero_range(reset_base, "pose_range")
        reset_base.params["velocity_range"] = {
            "x": (initial_vx, initial_vx),
            "y": (initial_vy, initial_vy),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (initial_wz, initial_wz),
        }
    _set_zero_range(getattr(env_cfg.events, "randomize_reset_joints", None), "position_range")
    _set_zero_range(getattr(env_cfg.events, "randomize_reset_joints", None), "velocity_range")

    if not args_cli.keep_randomization:
        _disable_if_present(
            env_cfg.events,
            (
                "randomize_rigid_body_material",
                "randomize_rigid_feet_material",
                "randomize_rigid_body_mass",
                "randomize_each_rigid_body_mass",
                "randomize_rigid_body_inertia",
                "randomize_com_positions",
                "randomize_apply_external_force_torque",
                "randomize_actuator_gains",
                "randomize_joint_armature",
                "randomize_push_robot",
                "randomize_push_robot_helper",
                "randomize_push_robot_disturber",
            ),
        )


def _resolve_joints(env) -> tuple[object, list[int], list[str]]:
    robot = env.unwrapped.scene["robot"]
    joint_ids, joint_names = robot.find_joints(ACTUATED_JOINT_NAMES, preserve_order=True)
    if joint_names != ACTUATED_JOINT_NAMES:
        raise RuntimeError(f"Expected joints {ACTUATED_JOINT_NAMES}, resolved {joint_names}")
    if env.unwrapped.action_manager.total_action_dim != len(joint_names):
        raise RuntimeError(f"Expected {len(joint_names)} actions, got {env.unwrapped.action_manager.total_action_dim}.")
    return robot, joint_ids, joint_names


def calculate_training_spectrum(
    normalized_positions: torch.Tensor, sample_dt: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply the exact training analyzer spectrum definition along time dimension 0."""
    sample_count = normalized_positions.shape[0]
    mean = normalized_positions.mean(dim=0)
    centered = normalized_positions - mean.unsqueeze(0)
    window = torch.hann_window(sample_count, periodic=False, dtype=normalized_positions.dtype)
    spectrum = torch.fft.rfft(centered * window.unsqueeze(1), dim=0, norm="ortho")
    power = spectrum.abs().square()
    if sample_count % 2 == 0:
        power[1:-1] *= 2.0
    else:
        power[1:] *= 2.0
    frequencies = torch.fft.rfftfreq(sample_count, d=sample_dt, dtype=normalized_positions.dtype)
    window_energy = window.square().sum().clamp_min(SPECTRUM_EPS)
    return frequencies, spectrum, power, mean, float(window_energy)


def write_time_csv(path: Path, normalized_positions: torch.Tensor, joint_names: list[str], sample_dt: float) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["sample_index", "time_s", "command_vx_mps", "command_vy_mps", "command_wz_radps"]
    header += [f"{name}_z_normalized" for name in joint_names]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for index, positions in enumerate(normalized_positions):
            writer.writerow([index, index * sample_dt, *VELOCITY_COMMAND, *positions.tolist()])
    return path


def write_spectrum_csv(
    path: Path,
    frequencies: torch.Tensor,
    spectrum: torch.Tensor,
    power: torch.Tensor,
    joint_names: list[str],
    max_frequency_hz: float,
) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = frequencies <= min(max_frequency_hz, float(frequencies[-1])) + 1.0e-12
    header = ["frequency_hz"]
    for name in joint_names:
        header += [f"{name}_power_normalized", f"{name}_phase_rad"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for frequency, powers, phases in zip(frequencies[keep], power[keep], torch.angle(spectrum[keep]), strict=True):
            row = [frequency.item()]
            for joint_index in range(len(joint_names)):
                row += [powers[joint_index].item(), phases[joint_index].item()]
            writer.writerow(row)
    return path


def write_summary_csv(
    spectrum_path: Path,
    frequencies: torch.Tensor,
    power: torch.Tensor,
    mean: torch.Tensor,
    window_energy: float,
    joint_names: list[str],
) -> Path:
    path = spectrum_path.with_name(f"{spectrum_path.stem}_summary.csv")
    ac_power = power[1:].sum(dim=0)
    peak_indices = power[1:].argmax(dim=0) + 1
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "joint_name",
                "scale_rad",
                "window_mean_normalized",
                "total_ac_power_raw",
                "hann_normalized_ac_energy",
                "peak_frequency_hz",
            ]
        )
        for joint_index, name in enumerate(joint_names):
            writer.writerow(
                [
                    name,
                    JOINT_SCALES[name],
                    mean[joint_index].item(),
                    ac_power[joint_index].item(),
                    (ac_power[joint_index] / window_energy).item(),
                    frequencies[peak_indices[joint_index]].item(),
                ]
            )
    return path


def write_metadata(
    output_dir: Path,
    task_name: str,
    checkpoint_path: str,
    sample_dt: float,
) -> Path:
    """Record the exact policy and rollout settings associated with exported data."""
    path = output_dir / "rollout_metadata.json"
    payload = {
        "task": task_name,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_file": Path(checkpoint_path).name,
        "checkpoint_run": Path(checkpoint_path).parent.name,
        "velocity_command": list(VELOCITY_COMMAND),
        "sample_dt_s": sample_dt,
        "sample_frequency_hz": 1.0 / sample_dt,
        "warmup_steps": args_cli.warmup_steps,
        "num_frames": args_cli.num_frames,
        "keep_randomization": args_cli.keep_randomization,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
    return path


def plot_time(path: Path, positions: torch.Tensor, joint_names: list[str], sample_dt: float) -> Path:
    image_path = path.with_suffix(".png")
    time_s = torch.arange(positions.shape[0], dtype=positions.dtype) * sample_dt
    column_count = 4
    row_count = math.ceil(len(joint_names) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(18, 4 * row_count),
        sharex=True,
        constrained_layout=True,
    )
    flat_axes = axes.reshape(-1)
    for joint_index, name in enumerate(joint_names):
        flat_axes[joint_index].plot(time_s.numpy(), positions[:, joint_index].numpy(), linewidth=1.0)
        flat_axes[joint_index].axhline(0.0, color="tab:gray", linestyle="--", linewidth=0.8)
        flat_axes[joint_index].set_title(name, fontsize=9)
        flat_axes[joint_index].grid(True, alpha=0.3)
    for axis in flat_axes[len(joint_names) :]:
        axis.set_visible(False)
    figure.suptitle(f"Scale-normalized joint histories, command={VELOCITY_COMMAND}", fontsize=14)
    figure.supxlabel("Time [s]")
    figure.supylabel("z = (q - q_default) / scale")
    figure.savefig(image_path, dpi=PLOT_DPI)
    plt.close(figure)
    return image_path


def plot_spectrum(
    path: Path,
    frequencies: torch.Tensor,
    power: torch.Tensor,
    joint_names: list[str],
    max_frequency_hz: float,
) -> Path:
    image_path = path.with_suffix(".png")
    plot_max_frequency_hz = min(max_frequency_hz, float(frequencies[-1]))
    keep = frequencies <= plot_max_frequency_hz + 1.0e-12
    frequency_values = frequencies[keep].numpy()
    resolution_hz = float(frequencies[1] - frequencies[0])
    column_count = 4
    row_count = math.ceil(len(joint_names) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(18, 4 * row_count),
        sharex=True,
        constrained_layout=True,
    )
    flat_axes = axes.reshape(-1)
    for joint_index, name in enumerate(joint_names):
        axis = flat_axes[joint_index]
        axis.step(frequency_values, power[keep, joint_index].numpy(), where="mid", linewidth=1.0)
        axis.set_title(name, fontsize=9)
        axis.grid(True, alpha=0.3)
        axis.set_xlim(0.0, max(plot_max_frequency_hz, resolution_hz))
        axis.set_xticks(torch.linspace(0.0, plot_max_frequency_hz, steps=6).numpy())
    for axis in flat_axes[len(joint_names) :]:
        axis.set_visible(False)
    figure.suptitle(
        f"Training-normalized single-sided power, command={VELOCITY_COMMAND}, delta_f={resolution_hz:.4g} Hz",
        fontsize=14,
    )
    figure.supxlabel("Frequency [Hz]")
    figure.supylabel("Single-sided normalized power")
    figure.savefig(image_path, dpi=PLOT_DPI)
    plt.close(figure)
    return image_path


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
) -> None:
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    _configure_environment(env_cfg, FixedVelocityCommand(VELOCITY_COMMAND))
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    checkpoint_path = args_cli.checkpoint or CHECK_POINT
    resume_path = (
        retrieve_file_path(checkpoint_path)
        if checkpoint_path
        else get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    )
    checkpoint_run_name = Path(resume_path).parent.name
    default_output_dir = OUTPUT_ROOT / checkpoint_run_name
    time_output_path = args_cli.output or default_output_dir / "runner_joint_angles_scale.csv"
    spectrum_output_path = args_cli.spectrum_output or default_output_dir / "runner_joint_power_scale.csv"
    env_cfg.log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Checkpoint: {resume_path}")
    print(f"[INFO] Output directory: {default_output_dir.resolve()}")
    print(f"[INFO] Fixed velocity command: {VELOCITY_COMMAND}")

    env = None
    try:
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        if args_cli.video:
            video_folder = time_output_path.expanduser().resolve().parent
            video_folder.mkdir(parents=True, exist_ok=True)
            video_length = args_cli.warmup_steps + args_cli.num_frames
            video_prefix = f"{time_output_path.stem}_rollout"
            video_fps = max(1, round(1.0 / float(env.unwrapped.step_dt)))
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(video_folder),
                step_trigger=lambda step: step == 0,
                video_length=video_length,
                name_prefix=video_prefix,
                fps=video_fps,
                disable_logger=True,
            )
            print(
                f"[INFO] Recording {video_length} control steps at {video_fps} FPS to "
                f"{video_folder / f'{video_prefix}-step-0.mp4'}"
            )
            print(
                f"[INFO] IsaacLab viewer follow: asset_root=robot, eye={env_cfg.viewer.eye}, "
                f"lookat={env_cfg.viewer.lookat}"
            )
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        robot, joint_ids, joint_names = _resolve_joints(env)
        sample_dt = float(env.unwrapped.step_dt)
        default_positions = robot.data.default_joint_pos[0, joint_ids].detach().cpu().to(torch.float64)
        scales = torch.tensor([JOINT_SCALES[name] for name in joint_names], dtype=torch.float64)
        print(
            f"[INFO] Sampling: dt={sample_dt:.6f}s, fs={1.0 / sample_dt:.3f}Hz, "
            f"warm-up={args_cli.warmup_steps * sample_dt:.3f}s, "
            f"record={args_cli.num_frames * sample_dt:.3f}s"
        )
        print(f"[INFO] Joint order: {joint_names}")

        obs = env.get_observations()
        samples: list[torch.Tensor] = []
        for step_index in range(args_cli.warmup_steps + args_cli.num_frames):
            if not simulation_app.is_running():
                raise RuntimeError("Simulator stopped before collection completed.")
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
            if bool(dones[0].item()):
                phase = "recording" if step_index >= args_cli.warmup_steps else "warm-up"
                raise RuntimeError(f"Episode terminated during {phase} at policy step {step_index + 1}.")
            if step_index >= args_cli.warmup_steps:
                samples.append(robot.data.joint_pos[0, joint_ids].detach().cpu().to(torch.float64).clone())
                if len(samples) % 100 == 0 or len(samples) == args_cli.num_frames:
                    print(f"[INFO] Collected {len(samples)}/{args_cli.num_frames} samples.")
            if args_cli.real_time:
                sleep_time = sample_dt - (time.time() - start_time)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

        absolute_positions = torch.stack(samples)
        normalized_positions = (absolute_positions - default_positions.unsqueeze(0)) / scales.unsqueeze(0)
        frequencies, spectrum, power, mean, window_energy = calculate_training_spectrum(normalized_positions, sample_dt)
        time_path = write_time_csv(time_output_path, normalized_positions, joint_names, sample_dt)
        spectrum_path = write_spectrum_csv(
            spectrum_output_path, frequencies, spectrum, power, joint_names, args_cli.max_frequency_hz
        )
        summary_path = write_summary_csv(spectrum_path, frequencies, power, mean, window_energy, joint_names)
        time_plot = plot_time(time_path, normalized_positions, joint_names, sample_dt)
        spectrum_plot = plot_spectrum(spectrum_path, frequencies, power, joint_names, args_cli.max_frequency_hz)
        metadata_path = write_metadata(time_path.parent, args_cli.task, resume_path, sample_dt)
        print(f"[INFO] Normalized time CSV: {time_path}")
        print(f"[INFO] Power spectrum CSV: {spectrum_path}")
        print(f"[INFO] Per-joint summary CSV: {summary_path}")
        print(f"[INFO] Time plot: {time_plot}")
        print(f"[INFO] Spectrum plot: {spectrum_plot}")
        print(f"[INFO] Rollout metadata: {metadata_path}")
        print(
            f"[INFO] FFT: resolution={float(frequencies[1] - frequencies[0]):.6f}Hz, "
            f"Nyquist={float(frequencies[-1]):.3f}Hz, Hann energy={window_energy:.6f}."
        )
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
