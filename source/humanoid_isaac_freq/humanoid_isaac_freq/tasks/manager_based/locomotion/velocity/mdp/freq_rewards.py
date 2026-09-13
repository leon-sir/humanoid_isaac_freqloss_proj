# Copyright (c) 2026 Zheng Pan
# SPDX-License-Identifier: Apache-2.0

"""Shared joint-frequency analysis and frequency-domain reward terms.

All terms in this module reuse one :class:`JointFrequencyAnalyzer` registered on
the frequency-specific environment.  Joint histories, FFTs, and penalties stay
on the simulation device.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg


@dataclass
class JointFrequencyAnalyzerCfg:
    """Configuration shared by all joint-frequency reward terms."""

    analyzer_key: str
    asset_cfg: SceneEntityCfg
    joint_scales: dict[str, float]
    window_duration_s: float = 2.0
    fft_update_interval: int = 5
    window_type: str = "hann"
    eps: float = 1.0e-8


class JointFrequencyAnalyzer:
    """GPU-parallel ring buffer and cached single-sided joint spectrum."""

    def __init__(self, cfg: JointFrequencyAnalyzerCfg, env):
        self.cfg = cfg
        self.asset: Articulation = env.scene[cfg.asset_cfg.name]
        self.num_envs = env.num_envs
        self.device = env.device
        self.step_dt = float(env.step_dt)
        self.window_size = round(cfg.window_duration_s / self.step_dt)
        if self.window_size < 2:
            raise ValueError("window_duration_s must contain at least two environment steps")
        if cfg.fft_update_interval < 1:
            raise ValueError("fft_update_interval must be at least one")
        if cfg.window_type != "hann":
            raise ValueError(f"Unsupported frequency window type: {cfg.window_type!r}")
        if not cfg.joint_scales:
            raise ValueError("joint_scales must contain at least one actuated joint")

        requested_names = list(cfg.joint_scales)
        joint_ids, resolved_names = self.asset.find_joints(requested_names, preserve_order=True)
        if resolved_names != requested_names:
            raise ValueError(
                "Frequency analyzer joint names must resolve exactly and in configured order: "
                f"requested={requested_names}, resolved={resolved_names}"
            )
        scales = [float(cfg.joint_scales[name]) for name in requested_names]
        if any(scale <= 0.0 for scale in scales):
            raise ValueError(f"All joint frequency scales must be positive, got {cfg.joint_scales}")

        self.joint_names = tuple(resolved_names)
        self.joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=self.device)
        self.joint_name_to_index = {name: index for index, name in enumerate(self.joint_names)}
        self.scales = torch.tensor(scales, dtype=torch.float, device=self.device)
        self.num_joints = len(self.joint_names)

        self.history = torch.zeros((self.num_envs, self.window_size, self.num_joints), device=self.device)
        self.valid_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Per-environment write positions are required because environments reset asynchronously.
        self.write_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.cached_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cached_mean = torch.zeros((self.num_envs, self.num_joints), device=self.device)

        self.freq = torch.fft.rfftfreq(self.window_size, d=self.step_dt, device=self.device)
        self.num_freq_bins = self.freq.numel()
        self.cached_spectrum = torch.zeros(
            (self.num_envs, self.num_joints, self.num_freq_bins),
            dtype=torch.complex64,
            device=self.device,
        )
        self.cached_power = torch.zeros((self.num_envs, self.num_joints, self.num_freq_bins), device=self.device)
        self.window = torch.hann_window(self.window_size, periodic=False, dtype=torch.float, device=self.device)
        self.window_energy = torch.sum(self.window.square()).clamp_min(cfg.eps)
        self._env_indices = torch.arange(self.num_envs, device=self.device)
        self._time_indices = torch.arange(self.window_size, device=self.device)
        self.last_sample_step = -1
        self.last_fft_step = -cfg.fft_update_interval

    def fingerprint(self) -> tuple:
        """Return immutable fields used to reject conflicting analyzer configs."""
        return (
            self.cfg.asset_cfg.name,
            tuple(self.cfg.joint_scales.items()),
            self.cfg.window_duration_s,
            self.cfg.fft_update_interval,
            self.cfg.window_type,
            self.cfg.eps,
        )

    def assert_compatible(self, cfg: JointFrequencyAnalyzerCfg) -> None:
        candidate = (
            cfg.asset_cfg.name,
            tuple(cfg.joint_scales.items()),
            cfg.window_duration_s,
            cfg.fft_update_interval,
            cfg.window_type,
            cfg.eps,
        )
        if candidate != self.fingerprint():
            raise ValueError(f"Conflicting configurations use analyzer_key={cfg.analyzer_key!r}")

    def joint_indices(self, joint_names: Sequence[str]) -> torch.Tensor:
        """Resolve exact configured names to analyzer-local joint indices."""
        missing = [name for name in joint_names if name not in self.joint_name_to_index]
        if missing:
            raise ValueError(f"Joints are not configured in the frequency analyzer: {missing}")
        return torch.tensor(
            [self.joint_name_to_index[name] for name in joint_names],
            dtype=torch.long,
            device=self.device,
        )

    def reset(self, env_ids: torch.Tensor | Sequence[int] | slice | None = None) -> None:
        """Clear only the selected environments after an episodic reset."""
        if env_ids is None:
            env_ids = slice(None)
        self.history[env_ids] = 0.0
        self.valid_count[env_ids] = 0
        self.write_index[env_ids] = 0
        self.cached_ready[env_ids] = False
        self.cached_mean[env_ids] = 0.0
        self.cached_spectrum[env_ids] = 0.0
        self.cached_power[env_ids] = 0.0

    def update_once(self, env) -> None:
        """Sample once and perform at most one FFT for a common environment step."""
        step = int(env.common_step_counter)
        if step == self.last_sample_step:
            return

        joint_pos = self.asset.data.joint_pos[:, self.joint_ids]
        default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids]
        normalized_pos = (joint_pos - default_joint_pos) / self.scales.unsqueeze(0)
        self.history[self._env_indices, self.write_index] = normalized_pos
        self.write_index.remainder_(self.window_size).add_(1).remainder_(self.window_size)
        self.valid_count.add_(1).clamp_(max=self.window_size)
        self.last_sample_step = step

        if step - self.last_fft_step < self.cfg.fft_update_interval:
            return
        self.last_fft_step = step
        ready = self.valid_count >= self.window_size
        if not torch.any(ready):
            return

        # write_index points at the oldest sample once a ring is full.
        gather_index = (self.write_index[:, None] + self._time_indices[None, :]) % self.window_size
        ordered = torch.gather(
            self.history,
            1,
            gather_index[:, :, None].expand(-1, -1, self.num_joints),
        )
        mean = ordered.mean(dim=1)
        centered_windowed = (ordered - mean[:, None, :]) * self.window[None, :, None]
        spectrum = torch.fft.rfft(centered_windowed, dim=1, norm="ortho").transpose(1, 2)
        power = spectrum.abs().square()
        if self.window_size % 2 == 0:
            power[:, :, 1:-1] *= 2.0
        else:
            power[:, :, 1:] *= 2.0

        self.cached_mean[ready] = mean[ready]
        self.cached_spectrum[ready] = spectrum[ready]
        self.cached_power[ready] = power[ready]
        self.cached_ready.copy_(ready)


def _get_shared_analyzer(env, cfg: JointFrequencyAnalyzerCfg) -> JointFrequencyAnalyzer:
    if not hasattr(env, "joint_frequency_analyzers"):
        raise ValueError("Frequency analyzer registry is unavailable; use ManagerBasedRLYMFreqEnv for this task")
    analyzer = env.joint_frequency_analyzers.get(cfg.analyzer_key)
    if analyzer is None:
        analyzer = JointFrequencyAnalyzer(cfg, env)
        env.joint_frequency_analyzers[cfg.analyzer_key] = analyzer
    else:
        analyzer.assert_compatible(cfg)
    return analyzer


def _command_is_moving(env, command_name: str, k_omega: float, threshold: float) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    magnitude = torch.sqrt(command[:, 0].square() + command[:, 1].square() + (k_omega * command[:, 2]).square())
    return magnitude > threshold


def _frequency_band_indices(
    analyzer: JointFrequencyAnalyzer,
    frequency_band_hz: tuple[float, float],
    parameter_name: str,
) -> torch.Tensor:
    """Resolve an inclusive frequency band to discrete analyzer bins."""
    f_min, f_max = map(float, frequency_band_hz)
    if not 0.0 <= f_min < f_max:
        raise ValueError(f"Expected 0 <= f_min < f_max for {parameter_name}, got {frequency_band_hz}")
    indices = torch.nonzero((analyzer.freq >= f_min) & (analyzer.freq <= f_max), as_tuple=False).flatten()
    if indices.numel() == 0:
        frequency_resolution = 1.0 / (analyzer.window_size * analyzer.step_dt)
        raise ValueError(
            f"{parameter_name}={frequency_band_hz} contains no FFT bins; "
            f"frequency resolution is {frequency_resolution:.3f} Hz"
        )
    return indices


def _robot_fundamental(
    analyzer: JointFrequencyAnalyzer,
    joint_indices: torch.Tensor,
    search_band_hz: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return aggregate power, peak frequency, and non-DC energy."""
    search_ids = _frequency_band_indices(analyzer, search_band_hz, "fundamental_search_band_hz")
    robot_power = analyzer.cached_power[:, joint_indices].sum(dim=1)
    peak_local = robot_power[:, search_ids].argmax(dim=-1)
    peak_frequency = analyzer.freq[search_ids[peak_local]]
    total_ac_energy = robot_power[:, 1:].sum(dim=-1)
    return robot_power, peak_frequency, total_ac_energy


class JointDcPosturePenalty(ManagerTermBase):
    """Penalize window-mean error relative to a normalized DC target.

    ``target_dc`` is in (q - default_q) / joint_scale units, not radians.
    Zero preserves the original default-posture objective.
    """

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self.analyzer = _get_shared_analyzer(env, cfg.params["analyzer_cfg"])
        joint_patterns = cfg.params.get("joint_names")
        if joint_patterns is None:
            joint_names = self.analyzer.joint_names
        else:
            if isinstance(joint_patterns, str):
                joint_patterns = [joint_patterns]
            if not joint_patterns:
                raise ValueError("joint_names must contain at least one joint name or regular expression")
            joint_names = []
            for pattern in joint_patterns:
                matches = [name for name in self.analyzer.joint_names if re.fullmatch(pattern, name)]
                if not matches:
                    raise ValueError(f"joint_names pattern {pattern!r} matched no configured analyzer joints")
                joint_names.extend(name for name in matches if name not in joint_names)
        self.joint_indices = self.analyzer.joint_indices(joint_names)
        target_dc = cfg.params.get("target_dc", 0.0)
        if isinstance(target_dc, dict):
            missing = [name for name in joint_names if name not in target_dc]
            if missing:
                raise ValueError(f"target_dc is missing joints: {missing}")
            targets = [float(target_dc[name]) for name in joint_names]
        else:
            targets = [float(target_dc)] * len(joint_names)
        if not all(math.isfinite(value) for value in targets):
            raise ValueError("target_dc values must be finite")
        self.target_dc = torch.tensor(targets, device=env.device)
        moving_dc_limit = cfg.params["moving_dc_limit"]
        if isinstance(moving_dc_limit, dict):
            missing = [name for name in joint_names if name not in moving_dc_limit]
            if missing:
                raise ValueError(f"moving_dc_limit is missing joints: {missing}")
            limits = [moving_dc_limit[name] for name in joint_names]
        else:
            limits = [moving_dc_limit] * len(joint_names)
        if any(float(limit) < 0.0 for limit in limits):
            raise ValueError("moving_dc_limit values must be non-negative")
        self.moving_dc_limit = torch.tensor(limits, device=env.device)

    def __call__(
        self,
        env,
        analyzer_cfg: JointFrequencyAnalyzerCfg,
        command_name: str,
        k_omega: float,
        stand_command_threshold: float,
        moving_dc_limit: float | dict[str, float],
        joint_names: str | Sequence[str] | None = None,
        target_dc: float | dict[str, float] = 0.0,
    ) -> torch.Tensor:
        del analyzer_cfg, moving_dc_limit, joint_names, target_dc
        self.analyzer.update_once(env)
        mean = self.analyzer.cached_mean[:, self.joint_indices] - self.target_dc
        stand_penalty = mean.square().mean(dim=-1)
        move_excess = torch.relu(mean.abs() - self.moving_dc_limit)
        move_penalty = move_excess.square().mean(dim=-1)
        moving = _command_is_moving(env, command_name, k_omega, stand_command_threshold)
        result = torch.where(moving, move_penalty, stand_penalty)
        return torch.where(self.analyzer.cached_ready, result, torch.zeros_like(result))


class JointLeftRightDcMatchPenalty(ManagerTermBase):
    """Penalize bilateral mismatch between normalized joint-position window means."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self.analyzer = _get_shared_analyzer(env, cfg.params["analyzer_cfg"])

        pairs = cfg.params["dc_joint_pairs"]
        if not pairs:
            raise ValueError("dc_joint_pairs must contain at least one left/right joint pair")
        if any(len(pair) != 2 for pair in pairs):
            raise ValueError("Every dc_joint_pairs entry must contain exactly two joint names")

        mirror_signs = cfg.params["mirror_signs"]
        if len(pairs) != len(mirror_signs):
            raise ValueError("dc_joint_pairs and mirror_signs must have equal length")
        if any(float(sign) not in (-1.0, 1.0) for sign in mirror_signs):
            raise ValueError("Every mirror_sign must be either -1.0 or 1.0")

        max_abs_yaw_command = cfg.params.get("max_abs_yaw_command")
        if max_abs_yaw_command is not None and float(max_abs_yaw_command) < 0.0:
            raise ValueError("max_abs_yaw_command must be non-negative or None")

        self.left_indices = self.analyzer.joint_indices([pair[0] for pair in pairs])
        self.right_indices = self.analyzer.joint_indices([pair[1] for pair in pairs])
        self.mirror_signs = torch.tensor(mirror_signs, dtype=torch.float, device=env.device)
        self.max_abs_yaw_command = (
            None if max_abs_yaw_command is None else float(max_abs_yaw_command)
        )

    def __call__(
        self,
        env,
        analyzer_cfg: JointFrequencyAnalyzerCfg,
        command_name: str,
        k_omega: float,
        stand_command_threshold: float,
        dc_joint_pairs: Sequence[tuple[str, str]],
        mirror_signs: Sequence[float],
        max_abs_yaw_command: float | None = None,
    ) -> torch.Tensor:
        del analyzer_cfg, dc_joint_pairs, mirror_signs, max_abs_yaw_command
        self.analyzer.update_once(env)

        left_mean = self.analyzer.cached_mean[:, self.left_indices]
        right_mean = self.analyzer.cached_mean[:, self.right_indices]
        dc_error = left_mean - self.mirror_signs.unsqueeze(0) * right_mean
        result = dc_error.square().mean(dim=-1)

        moving = _command_is_moving(env, command_name, k_omega, stand_command_threshold)
        valid = self.analyzer.cached_ready & moving
        if self.max_abs_yaw_command is not None:
            command = env.command_manager.get_command(command_name)
            valid &= command[:, 2].abs() <= self.max_abs_yaw_command
        return torch.where(valid, result, torch.zeros_like(result))


class JointSpectralHighFrequencyPenalty(ManagerTermBase):
    """Penalize the high-frequency fraction for selected encouraged joints."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self.analyzer = _get_shared_analyzer(env, cfg.params["analyzer_cfg"])
        self.joint_indices = self.analyzer.joint_indices(cfg.params["joint_names"])
        alpha_s = float(cfg.params["alpha_s"])
        beta_s = float(cfg.params["beta_s"])
        if not 0.0 < alpha_s < beta_s:
            raise ValueError(f"Expected 0 < alpha_s < beta_s, got {alpha_s}, {beta_s}")
        omega = 2.0 * math.pi * self.analyzer.freq
        alpha_beta = (1.0 + (omega * beta_s).square()) / (1.0 + (omega * alpha_s).square())
        self.frequency_weight = (alpha_beta - 1.0) / (alpha_beta[-1] - 1.0 + self.analyzer.cfg.eps)
        self.frequency_weight[0] = 0.0

    def __call__(
        self,
        env,
        analyzer_cfg: JointFrequencyAnalyzerCfg,
        joint_names: Sequence[str],
        alpha_s: float,
        beta_s: float,
    ) -> torch.Tensor:
        del analyzer_cfg, joint_names, alpha_s, beta_s
        self.analyzer.update_once(env)
        power = self.analyzer.cached_power[:, self.joint_indices, 1:]
        weighted = (power * self.frequency_weight[None, None, 1:]).sum(dim=-1)
        total = power.sum(dim=-1)
        result = (weighted / (total + self.analyzer.cfg.eps)).mean(dim=-1)
        return torch.where(self.analyzer.cached_ready, result, torch.zeros_like(result))


class JointSpectralEnergyPenalty(ManagerTermBase):
    """Penalize total non-DC energy for joints discouraged from oscillating."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self.analyzer = _get_shared_analyzer(env, cfg.params["analyzer_cfg"])
        self.joint_indices = self.analyzer.joint_indices(cfg.params["joint_names"])

    def __call__(
        self,
        env,
        analyzer_cfg: JointFrequencyAnalyzerCfg,
        joint_names: Sequence[str],
    ) -> torch.Tensor:
        del analyzer_cfg, joint_names
        self.analyzer.update_once(env)
        power = self.analyzer.cached_power[:, self.joint_indices, 1:]
        # Hann-energy normalization makes the term an approximately window-length-independent
        # mean-square AC penalty in normalized joint coordinates.
        result = (power.sum(dim=-1) / self.analyzer.window_energy).mean(dim=-1)
        return torch.where(self.analyzer.cached_ready, result, torch.zeros_like(result))


class JointSpectralBandEnergyReward(ManagerTermBase):
    """Reward each selected joint for reaching a minimum normalized band energy.

    Band energy is divided by the Hann-window energy, so the configured target
    is independent of FFT window length.  Each joint contributes equally and
    saturates at one, preventing additional reward for excessive oscillation.
    """

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self.analyzer = _get_shared_analyzer(env, cfg.params["analyzer_cfg"])
        joint_names = cfg.params["joint_names"]
        self.joint_indices = self.analyzer.joint_indices(joint_names)
        energy_band_hz = cfg.params["energy_band_hz"]
        if float(energy_band_hz[0]) <= 0.0:
            raise ValueError("energy_band_hz must exclude the DC bin")
        self.band_indices = _frequency_band_indices(self.analyzer, energy_band_hz, "energy_band_hz")

        target_band_energy = cfg.params["target_band_energy"]
        if isinstance(target_band_energy, dict):
            missing = [name for name in joint_names if name not in target_band_energy]
            if missing:
                raise ValueError(f"target_band_energy is missing joints: {missing}")
            targets = [target_band_energy[name] for name in joint_names]
        else:
            targets = [target_band_energy] * len(joint_names)
        if any(float(target) <= 0.0 for target in targets):
            raise ValueError("target_band_energy values must be positive")
        self.target_band_energy = torch.tensor(targets, dtype=torch.float, device=env.device)

    def __call__(
        self,
        env,
        analyzer_cfg: JointFrequencyAnalyzerCfg,
        command_name: str,
        k_omega: float,
        stand_command_threshold: float,
        joint_names: Sequence[str],
        energy_band_hz: tuple[float, float],
        target_band_energy: float | dict[str, float],
    ) -> torch.Tensor:
        del analyzer_cfg, joint_names, energy_band_hz, target_band_energy
        self.analyzer.update_once(env)
        power = self.analyzer.cached_power[:, self.joint_indices]
        band_energy = power[:, :, self.band_indices].sum(dim=-1) / self.analyzer.window_energy
        # A bounded lower-target reward: zero at no oscillation and one once the
        # per-joint target is reached.  There is no incentive to exceed it.
        achievement = torch.clamp(band_energy / self.target_band_energy, min=0.0, max=1.0)
        result = achievement.mean(dim=-1)
        moving = _command_is_moving(env, command_name, k_omega, stand_command_threshold)
        valid = self.analyzer.cached_ready & moving
        return torch.where(valid, result, torch.zeros_like(result))


class JointFundamentalConcentrationPenalty(ManagerTermBase):
    """Penalize moving spectra that are not concentrated around one gait frequency."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self.analyzer = _get_shared_analyzer(env, cfg.params["analyzer_cfg"])
        self.joint_indices = self.analyzer.joint_indices(cfg.params["fundamental_joint_names"])

    def __call__(
        self,
        env,
        analyzer_cfg: JointFrequencyAnalyzerCfg,
        command_name: str,
        k_omega: float,
        stand_command_threshold: float,
        fundamental_search_band_hz: tuple[float, float],
        fundamental_half_width_hz: float,
        fundamental_joint_names: Sequence[str],
        spectrum_energy_floor: float,
    ) -> torch.Tensor:
        del analyzer_cfg, fundamental_joint_names
        self.analyzer.update_once(env)
        robot_power, f0, total_energy = _robot_fundamental(
            self.analyzer, self.joint_indices, fundamental_search_band_hz
        )
        fundamental_band = (self.analyzer.freq[None, :] - f0[:, None]).abs() <= fundamental_half_width_hz
        concentration = (robot_power * fundamental_band).sum(dim=-1) / (total_energy + self.analyzer.cfg.eps)
        moving = _command_is_moving(env, command_name, k_omega, stand_command_threshold)
        valid = self.analyzer.cached_ready & moving & (total_energy > spectrum_energy_floor)
        result = 1.0 - concentration
        return torch.where(valid, result, torch.zeros_like(result))


class JointLeftRightFrequencyEnergyMatchPenalty(ManagerTermBase):
    """Match left/right peak frequencies and normalized band energies."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self.analyzer = _get_shared_analyzer(env, cfg.params["analyzer_cfg"])
        pairs = cfg.params["frequency_joint_pairs"]
        self.left_indices = self.analyzer.joint_indices([pair[0] for pair in pairs])
        self.right_indices = self.analyzer.joint_indices([pair[1] for pair in pairs])
        energy_match_weight = float(cfg.params.get("energy_match_weight", 0.0))
        if energy_match_weight < 0.0:
            raise ValueError("energy_match_weight must be non-negative")
        self.energy_match_weight = energy_match_weight

    def __call__(
        self,
        env,
        analyzer_cfg: JointFrequencyAnalyzerCfg,
        command_name: str,
        k_omega: float,
        stand_command_threshold: float,
        fundamental_search_band_hz: tuple[float, float],
        frequency_joint_pairs: Sequence[tuple[str, str]],
        spectrum_energy_floor: float,
        energy_match_weight: float = 0.0,
    ) -> torch.Tensor:
        del analyzer_cfg, frequency_joint_pairs, energy_match_weight
        self.analyzer.update_once(env)
        f_min, f_max = fundamental_search_band_hz
        search_ids = _frequency_band_indices(self.analyzer, fundamental_search_band_hz, "fundamental_search_band_hz")
        left_power = self.analyzer.cached_power[:, self.left_indices][:, :, search_ids]
        right_power = self.analyzer.cached_power[:, self.right_indices][:, :, search_ids]
        left_peak = self.analyzer.freq[search_ids[left_power.argmax(dim=-1)]]
        right_peak = self.analyzer.freq[search_ids[right_power.argmax(dim=-1)]]
        frequency_penalty = ((left_peak - right_peak) / (f_max - f_min)).square()

        left_energy = left_power.sum(dim=-1)
        right_energy = right_power.sum(dim=-1)
        # This bounded symmetry index is scale-free: zero means equal energy,
        # while one is approached when only one side oscillates.
        energy_penalty = ((left_energy - right_energy) / (left_energy + right_energy + self.analyzer.cfg.eps)).square()
        both_sides_active = (left_energy > spectrum_energy_floor) & (right_energy > spectrum_energy_floor)
        pair_penalty = frequency_penalty * both_sides_active + self.energy_match_weight * energy_penalty
        # Energy asymmetry remains informative when only one side is active;
        # only an effectively silent pair is excluded.
        pair_valid = left_energy + right_energy > spectrum_energy_floor
        valid_count = pair_valid.sum(dim=-1)
        result = (pair_penalty * pair_valid).sum(dim=-1) / valid_count.clamp_min(1)
        moving = _command_is_moving(env, command_name, k_omega, stand_command_threshold)
        valid = self.analyzer.cached_ready & moving & (valid_count > 0)
        return torch.where(valid, result, torch.zeros_like(result))


# Backward-compatible name for older task configurations.
JointLeftRightFundamentalMatchPenalty = JointLeftRightFrequencyEnergyMatchPenalty


class JointLeftRightPhasePenalty(ManagerTermBase):
    """Penalize bilateral phase relations that deviate from a half-period delay.

    The term estimates one fundamental frequency per environment and compares the
    normalized left-right cross spectrum in a narrow band around that frequency.
    Frequency bins are weighted by the geometric mean of the two sides' power, so
    bins in which either side is nearly silent have little or no influence.
    """

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self.analyzer = _get_shared_analyzer(env, cfg.params["analyzer_cfg"])
        pairs = cfg.params["phase_joint_pairs"]
        mirror_signs = cfg.params["mirror_signs"]
        if len(pairs) != len(mirror_signs):
            raise ValueError("phase_joint_pairs and mirror_signs must have equal length")
        if any(float(sign) not in (-1.0, 1.0) for sign in mirror_signs):
            raise ValueError("Every mirror_sign must be either -1.0 or 1.0")
        self.left_indices = self.analyzer.joint_indices([pair[0] for pair in pairs])
        self.right_indices = self.analyzer.joint_indices([pair[1] for pair in pairs])
        self.mirror_signs = torch.tensor(mirror_signs, device=env.device)
        self.fundamental_indices = self.analyzer.joint_indices(cfg.params["fundamental_joint_names"])

    def __call__(
        self,
        env,
        analyzer_cfg: JointFrequencyAnalyzerCfg,
        command_name: str,
        k_omega: float,
        stand_command_threshold: float,
        fundamental_search_band_hz: tuple[float, float],
        fundamental_half_width_hz: float,
        fundamental_joint_names: Sequence[str],
        phase_joint_pairs: Sequence[tuple[str, str]],
        mirror_signs: Sequence[float],
        spectrum_energy_floor: float,
    ) -> torch.Tensor:
        """Compute the unweighted phase-mismatch penalty for every environment.

        Args:
            env: Manager-based environment providing commands and the shared
                joint-frequency analyzer state.
            analyzer_cfg: Configuration of the shared analyzer. The stateful term
                resolves and caches the analyzer during initialization; this
                argument remains in the call signature for the manager interface.
            command_name: Name of the velocity command used to decide whether the
                robot is moving.
            k_omega: Scale applied to commanded yaw velocity before combining it
                with commanded planar velocity.
            stand_command_threshold: Command-magnitude threshold below which the
                environment is treated as standing and receives zero penalty.
            fundamental_search_band_hz: Inclusive frequency range in which the
                aggregate joint spectrum is searched for its dominant frequency.
            fundamental_half_width_hz: Half-width of the frequency band evaluated
                around the detected fundamental frequency.
            fundamental_joint_names: Exact analyzer joint names whose summed power
                determines the robot-level fundamental frequency.
            phase_joint_pairs: Ordered ``(left_joint, right_joint)`` pairs whose
                cross-spectral phase is evaluated.
            mirror_signs: One ``+1`` or ``-1`` per joint pair. At the fundamental
                frequency, ``+1`` targets an anti-phase relation; ``-1`` adds a
                sign inversion and therefore targets an in-phase relation.
            spectrum_energy_floor: Minimum spectral energy required for the robot
                and for both members of a pair. Inactive signals are excluded.

        Returns:
            A tensor of shape ``(num_envs,)`` containing a non-negative penalty.
            The nominal range is zero (target phase) to two (opposite phase).
            Environments without a full FFT window, with standing commands, or
            without sufficient spectral energy return zero.
        """
        del analyzer_cfg, fundamental_joint_names, phase_joint_pairs, mirror_signs
        self.analyzer.update_once(env)
        _, f0, robot_energy = _robot_fundamental(self.analyzer, self.fundamental_indices, fundamental_search_band_hz)
        band = (self.analyzer.freq[None, :] - f0[:, None]).abs() <= fundamental_half_width_hz

        left_spectrum = self.analyzer.cached_spectrum[:, self.left_indices]
        right_spectrum = self.analyzer.cached_spectrum[:, self.right_indices]
        left_power = self.analyzer.cached_power[:, self.left_indices]
        right_power = self.analyzer.cached_power[:, self.right_indices]
        cross = left_spectrum * right_spectrum.conj()
        normalized_cross = cross / (cross.abs() + self.analyzer.cfg.eps)

        delay = 0.5 / f0.clamp_min(self.analyzer.cfg.eps)
        target = self.mirror_signs[None, :, None] * torch.exp(
            -1j * 2.0 * math.pi * self.analyzer.freq[None, None, :] * delay[:, None, None]
        )
        phase_loss = 1.0 - torch.real(normalized_cross * target.conj())
        amplitude_weight = torch.sqrt(left_power * right_power)
        active_weight = amplitude_weight * band[:, None, :]

        left_band_energy = (left_power * band[:, None, :]).sum(dim=-1)
        right_band_energy = (right_power * band[:, None, :]).sum(dim=-1)
        pair_valid = (left_band_energy > spectrum_energy_floor) & (right_band_energy > spectrum_energy_floor)
        active_weight = active_weight * pair_valid[:, :, None]
        denominator = active_weight.sum(dim=(1, 2))
        result = (phase_loss * active_weight).sum(dim=(1, 2)) / (denominator + self.analyzer.cfg.eps)
        moving = _command_is_moving(env, command_name, k_omega, stand_command_threshold)
        valid = (
            self.analyzer.cached_ready
            & moving
            & (robot_energy > spectrum_energy_floor)
            & (denominator > self.analyzer.cfg.eps)
        )
        return torch.where(valid, result, torch.zeros_like(result))
