# Copyright (c) 2026 Zheng Pan
# SPDX-License-Identifier: Apache-2.0

"""Sparse diagnostic recording for anomalous base-height rewards."""

from __future__ import annotations

import csv
import io
import json
import math
import os
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.managers import (
    DatasetExportMode,
    RecorderManagerBaseCfg,
    RecorderTerm,
    RecorderTermCfg,
)
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class BaseHeightAnomalyRecorder(RecorderTerm):
    """Record bounded pre-trigger clips for rare base-height anomalies.

    A compact ring buffer stays on the simulation device. Only selected
    anomalous environments are copied to CPU and written to disk. The term
    bypasses the standard per-environment episode dataset because that path is
    too expensive for large locomotion batches.
    """

    _FORMAT_VERSION = 1
    _HEIGHT_ERROR_BIT = 1
    _RAY_SPREAD_BIT = 2
    _INVALID_RAY_BIT = 4

    def __init__(self, cfg: BaseHeightAnomalyRecorderCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if cfg.history_steps <= 0:
            raise ValueError("history_steps must be positive.")
        if cfg.max_events <= 0 or cfg.max_events_per_step <= 0:
            raise ValueError("Event limits must be positive.")
        if cfg.events_per_shard <= 0 or cfg.max_disk_mb <= 0:
            raise ValueError("Shard and disk limits must be positive.")

        self.cfg = cfg
        self._num_envs = env.num_envs
        self._episode_serial = torch.zeros(
            self._num_envs, dtype=torch.long, device=env.device
        )
        self._recorded_episode_serial = torch.full(
            (self._num_envs,), -1, dtype=torch.long, device=env.device
        )
        self._initialized = False
        self._closed = False
        self._disk_full = False
        self._limit_reported = False
        self._write_index = 0
        self._steps_seen = 0
        self._accepted_event_count = 0
        self._saved_event_count = 0
        self._next_event_id = 0
        self._next_shard_id = 0
        self._pending_events: list[dict[str, Any]] = []
        self._pending_summaries: list[dict[str, Any]] = []

    def _initialize(self) -> None:
        """Allocate buffers after the RL managers have been constructed."""
        if self._initialized:
            return

        self._robot = self._env.scene[self.cfg.asset_name]
        self._sensor = self._env.scene[self.cfg.sensor_name]
        self._terrain = self._env.scene.terrain
        self._terrain_cfg = self._env.cfg.scene.terrain.terrain_generator
        if self._terrain_cfg is None:
            raise RuntimeError("BaseHeightAnomalyRecorder requires generator terrain.")

        self._num_rows = int(self._terrain_cfg.num_rows)
        self._num_cols = int(self._terrain_cfg.num_cols)
        self._terrain_size_x = float(self._terrain_cfg.size[0])
        self._terrain_size_y = float(self._terrain_cfg.size[1])
        self._global_length = self._num_rows * self._terrain_size_x
        self._global_width = self._num_cols * self._terrain_size_y
        self._column_names = self._resolve_column_names()

        self._ray_count = int(self._sensor.data.ray_hits_w.shape[1])
        if self._ray_count != 9:
            raise RuntimeError(
                "BaseHeightAnomalyRecorder expects a 3x3 height_scanner_base "
                f"pattern with 9 rays, received {self._ray_count}."
            )

        self._float_field_names = [
            "root_pos_w_x",
            "root_pos_w_y",
            "root_pos_w_z",
            "assigned_local_x",
            "assigned_local_y",
            "assigned_local_z",
            "root_quat_w_w",
            "root_quat_w_x",
            "root_quat_w_y",
            "root_quat_w_z",
            "root_lin_vel_w_x",
            "root_lin_vel_w_y",
            "root_lin_vel_w_z",
            "root_ang_vel_w_x",
            "root_ang_vel_w_y",
            "root_ang_vel_w_z",
            "projected_gravity_b_x",
            "projected_gravity_b_y",
            "projected_gravity_b_z",
            "command_x",
            "command_y",
            "command_yaw",
            *[f"ray_z_{index}" for index in range(self._ray_count)],
            "ray_mean_z",
            "ray_median_z",
            "ray_min_z",
            "ray_max_z",
            "ray_spread",
            "relative_height",
            "height_error",
            "base_height_reward_rate",
            "base_height_episode_sum",
        ]
        self._int_field_names = [
            "assigned_level",
            "assigned_column",
            "geometric_row",
            "geometric_column",
            "episode_step",
            "valid_ray_count",
            "trigger_code",
        ]
        self._float_index = {
            name: index for index, name in enumerate(self._float_field_names)
        }
        self._int_index = {
            name: index for index, name in enumerate(self._int_field_names)
        }

        history_shape = (self.cfg.history_steps, self._num_envs)
        self._float_history = torch.empty(
            (*history_shape, len(self._float_field_names)),
            dtype=torch.float32,
            device=self._env.device,
        )
        self._int_history = torch.empty(
            (*history_shape, len(self._int_field_names)),
            dtype=torch.int32,
            device=self._env.device,
        )
        self._trigger_counts = torch.zeros(
            (self._num_rows, self._num_cols, 8),
            dtype=torch.long,
            device=self._env.device,
        )

        reward_names = list(self._env.reward_manager.active_terms)
        self._base_height_reward_index = (
            reward_names.index(self.cfg.reward_term_name)
            if self.cfg.reward_term_name in reward_names
            else None
        )
        self._termination_names = list(self._env.termination_manager.active_terms)

        mode = "play" if os.environ.get("RSL_RL_PLAY", "0") == "1" else "train"
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
        session_name = datetime.now().strftime(
            f"%Y-%m-%d_%H-%M-%S_%f_{mode}_rank{rank}"
        )
        log_dir = getattr(self._env.cfg, "log_dir", None) or "/tmp/isaaclab/logs"
        output_root = self.cfg.output_dir or os.path.join(
            log_dir, "base_height_anomalies"
        )
        self._output_dir = os.path.join(output_root, session_name)
        os.makedirs(self._output_dir, exist_ok=False)
        self._metadata_path = os.path.join(self._output_dir, "metadata.json")
        self._csv_path = os.path.join(self._output_dir, "events.csv")
        self._counts_path = os.path.join(self._output_dir, "trigger_counts.pt")
        self._max_disk_bytes = int(self.cfg.max_disk_mb * 1024 * 1024)

        self._write_metadata()
        self._write_csv_header()
        self._initialized = True

        ring_bytes = (
            self._float_history.numel() * self._float_history.element_size()
            + self._int_history.numel() * self._int_history.element_size()
        )
        print(
            "[INFO] Base-height anomaly recorder enabled: "
            f"output={self._output_dir}, ring_buffer={ring_bytes / 1024**2:.1f} MiB, "
            f"max_events={self.cfg.max_events}, max_disk={self.cfg.max_disk_mb} MiB"
        )

    def _resolve_column_names(self) -> list[str]:
        """Reproduce the generator's curriculum column assignment."""
        sub_terrains = list(self._terrain_cfg.sub_terrains.items())
        total = sum(float(sub_cfg.proportion) for _, sub_cfg in sub_terrains)
        if total <= 0.0:
            return ["unknown"] * self._num_cols

        cumulative: list[float] = []
        running = 0.0
        for _, sub_cfg in sub_terrains:
            running += float(sub_cfg.proportion) / total
            cumulative.append(running)

        names = []
        for column in range(self._num_cols):
            sample = column / self._num_cols + 0.001
            index = next(
                (
                    index
                    for index, upper_bound in enumerate(cumulative)
                    if sample < upper_bound
                ),
                len(sub_terrains) - 1,
            )
            names.append(sub_terrains[index][0])
        return names

    def _write_metadata(self) -> None:
        metadata = {
            "format_version": self._FORMAT_VERSION,
            "step_dt": float(self._env.step_dt),
            "target_height": self.cfg.target_height,
            "height_error_threshold": self.cfg.height_error_threshold,
            "ray_spread_threshold": self.cfg.ray_spread_threshold,
            "history_steps": self.cfg.history_steps,
            "history_duration_s": self.cfg.history_steps * float(self._env.step_dt),
            "max_events": self.cfg.max_events,
            "max_events_per_step": self.cfg.max_events_per_step,
            "events_per_shard": self.cfg.events_per_shard,
            "max_disk_mb": self.cfg.max_disk_mb,
            "float_history_fields": self._float_field_names,
            "int_history_fields": self._int_field_names,
            "termination_names": self._termination_names,
            "trigger_bits": {
                "1": "absolute_height_error",
                "2": "ray_height_spread",
                "4": "invalid_ray",
            },
            "terrain": {
                "num_rows": self._num_rows,
                "num_cols": self._num_cols,
                "size": [self._terrain_size_x, self._terrain_size_y],
                "global_border_width": float(self._terrain_cfg.border_width),
                "column_to_subterrain": self._column_names,
                "terrain_origins": self._terrain.terrain_origins.detach()
                .cpu()
                .tolist(),
            },
            "height_scanner_base": {
                "ray_count": self._ray_count,
                "ray_starts_local": self._sensor.ray_starts[0]
                .detach()
                .cpu()
                .tolist(),
                "ray_alignment": str(self._sensor.cfg.ray_alignment),
            },
        }
        self._atomic_write_bytes(
            self._metadata_path,
            json.dumps(metadata, indent=2, ensure_ascii=False).encode("utf-8"),
        )

    @staticmethod
    def _summary_field_names() -> list[str]:
        return [
            "event_id",
            "env_id",
            "global_policy_step",
            "episode_id",
            "episode_step",
            "history_length",
            "trigger_code",
            "trigger_reasons",
            "terrain_name",
            "assigned_level",
            "assigned_column",
            "geometric_row",
            "geometric_column",
            "local_x",
            "local_y",
            "local_z",
            "root_z",
            "root_vz",
            "relative_height",
            "height_error",
            "ray_mean_z",
            "ray_median_z",
            "ray_min_z",
            "ray_max_z",
            "ray_spread",
            "valid_ray_count",
            "distance_to_row_seam",
            "distance_to_column_seam",
            "distance_to_pure_x_border_start",
            "distance_to_pure_y_border_start",
            "signed_distance_to_grid_x_edge",
            "signed_distance_to_grid_y_edge",
            "signed_distance_to_global_x_edge",
            "signed_distance_to_global_y_edge",
            "base_height_reward_rate",
            "base_height_episode_sum",
            "reset",
            "terminated",
            "time_out",
            "termination_reasons",
        ]

    def _write_csv_header(self) -> None:
        stream = io.StringIO()
        csv.DictWriter(
            stream, fieldnames=self._summary_field_names()
        ).writeheader()
        self._atomic_write_bytes(self._csv_path, stream.getvalue().encode("utf-8"))

    def _snapshot(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build compact per-environment state and trigger masks."""
        root_pos = self._robot.data.root_pos_w
        local_pos = root_pos - self._terrain.env_origins
        ray_hits = self._sensor.data.ray_hits_w
        ray_z = ray_hits[..., 2]
        valid_rays = torch.isfinite(ray_hits).all(dim=-1) & (torch.abs(ray_z) < 1.0e6)
        valid_count = valid_rays.sum(dim=1)

        nan_ray_z = torch.where(
            valid_rays, ray_z, torch.full_like(ray_z, torch.nan)
        )
        ray_mean = torch.nanmean(nan_ray_z, dim=1)
        ray_median = torch.nanmedian(nan_ray_z, dim=1).values
        ray_min = torch.where(
            valid_rays, ray_z, torch.full_like(ray_z, torch.inf)
        ).min(dim=1).values
        ray_max = torch.where(
            valid_rays, ray_z, torch.full_like(ray_z, -torch.inf)
        ).max(dim=1).values
        no_valid_ray = valid_count == 0
        ray_min = torch.where(
            no_valid_ray, torch.full_like(ray_min, torch.nan), ray_min
        )
        ray_max = torch.where(
            no_valid_ray, torch.full_like(ray_max, torch.nan), ray_max
        )
        ray_spread = ray_max - ray_min
        relative_height = root_pos[:, 2] - ray_mean
        height_error = relative_height - self.cfg.target_height

        command = self._env.command_manager.get_command(self.cfg.command_name)
        if command.shape[1] < 3:
            command = torch.cat(
                (
                    command,
                    torch.zeros(
                        (self._num_envs, 3 - command.shape[1]),
                        dtype=command.dtype,
                        device=command.device,
                    ),
                ),
                dim=1,
            )
        command = command[:, :3]

        nan_column = torch.full(
            (self._num_envs,),
            float("nan"),
            dtype=root_pos.dtype,
            device=root_pos.device,
        )
        reward_rate = nan_column
        episode_sum = nan_column
        if self._base_height_reward_index is not None:
            reward_rate = self._env.reward_manager._step_reward[
                :, self._base_height_reward_index
            ]
            episode_sum = getattr(
                self._env.reward_manager, "_episode_sums", {}
            ).get(self.cfg.reward_term_name, nan_column)

        float_snapshot = torch.cat(
            (
                root_pos,
                local_pos,
                self._robot.data.root_quat_w,
                self._robot.data.root_lin_vel_w,
                self._robot.data.root_ang_vel_w,
                self._robot.data.projected_gravity_b,
                command,
                ray_z,
                ray_mean.unsqueeze(1),
                ray_median.unsqueeze(1),
                ray_min.unsqueeze(1),
                ray_max.unsqueeze(1),
                ray_spread.unsqueeze(1),
                relative_height.unsqueeze(1),
                height_error.unsqueeze(1),
                reward_rate.unsqueeze(1),
                episode_sum.unsqueeze(1),
            ),
            dim=1,
        ).to(dtype=torch.float32)

        # TerrainGenerator centers the complete grid around the world origin.
        # Keep out-of-grid indices (-1 or num_rows/cols) instead of clamping so
        # border excursions remain distinguishable from the terminal tile.
        geometric_row = torch.floor(
            (root_pos[:, 0] + 0.5 * self._global_length)
            / self._terrain_size_x
        ).to(torch.long)
        geometric_column = torch.floor(
            (root_pos[:, 1] + 0.5 * self._global_width)
            / self._terrain_size_y
        ).to(torch.long)

        height_bad = torch.isfinite(height_error) & (
            torch.abs(height_error) > self.cfg.height_error_threshold
        )
        spread_bad = torch.isfinite(ray_spread) & (
            ray_spread > self.cfg.ray_spread_threshold
        )
        invalid_bad = valid_count < self._ray_count
        trigger_code = (
            height_bad.to(torch.long) * self._HEIGHT_ERROR_BIT
            + spread_bad.to(torch.long) * self._RAY_SPREAD_BIT
            + invalid_bad.to(torch.long) * self._INVALID_RAY_BIT
        )
        int_snapshot = torch.stack(
            (
                self._terrain.terrain_levels,
                self._terrain.terrain_types,
                geometric_row,
                geometric_column,
                self._env.episode_length_buf,
                valid_count,
                trigger_code,
            ),
            dim=1,
        ).to(dtype=torch.int32)
        return float_snapshot, int_snapshot, trigger_code

    def record_post_step(self):
        """Update history and select anomalies before environment reset."""
        self._initialize()
        float_snapshot, int_snapshot, trigger_code = self._snapshot()
        self._float_history[self._write_index].copy_(float_snapshot)
        self._int_history[self._write_index].copy_(int_snapshot)

        novel = (trigger_code > 0) & (
            self._recorded_episode_serial != self._episode_serial
        )
        novel_ids = novel.nonzero(as_tuple=False).flatten()
        if novel_ids.numel() > 0:
            levels = self._terrain.terrain_levels[novel_ids].clamp(
                0, self._num_rows - 1
            )
            columns = self._terrain.terrain_types[novel_ids].clamp(
                0, self._num_cols - 1
            )
            codes = trigger_code[novel_ids].clamp(0, 7)
            flat_indices = (levels * self._num_cols + columns) * 8 + codes
            self._trigger_counts += torch.bincount(
                flat_indices, minlength=self._num_rows * self._num_cols * 8
            ).reshape(self._num_rows, self._num_cols, 8)

            # Persistent anomalies count once per environment episode.
            self._recorded_episode_serial[novel_ids] = self._episode_serial[novel_ids]
            remaining = max(
                0, self.cfg.max_events - self._accepted_event_count
            )
            selection_count = min(
                int(novel_ids.numel()),
                self.cfg.max_events_per_step,
                remaining,
            )
            if selection_count > 0 and not self._disk_full:
                if novel_ids.numel() > selection_count:
                    positions = torch.linspace(
                        0,
                        novel_ids.numel() - 1,
                        steps=selection_count,
                        device=novel_ids.device,
                    ).round().to(torch.long)
                    selected_ids = novel_ids[positions]
                else:
                    selected_ids = novel_ids
                for env_id in selected_ids.tolist():
                    self._append_event(env_id)

        if len(self._pending_events) >= self.cfg.events_per_shard:
            self._flush_pending_events()
        if (
            self._accepted_event_count >= self.cfg.max_events
            and not self._limit_reported
        ):
            print(
                "[INFO] Base-height anomaly recorder reached max_events; "
                "only aggregate trigger counts will continue."
            )
            self._limit_reported = True

        self._write_index = (self._write_index + 1) % self.cfg.history_steps
        self._steps_seen += 1
        return None, None

    def _append_event(self, env_id: int) -> None:
        """Copy one selected environment's same-episode history to CPU."""
        episode_step = int(self._env.episode_length_buf[env_id].item())
        history_length = min(
            self.cfg.history_steps,
            max(1, episode_step),
            self._steps_seen + 1,
        )
        indices = torch.remainder(
            torch.arange(
                self._write_index - history_length + 1,
                self._write_index + 1,
                device=self._env.device,
            ),
            self.cfg.history_steps,
        )
        float_history = self._float_history[indices, env_id].detach().cpu()
        int_history = self._int_history[indices, env_id].detach().cpu()
        current_float = float_history[-1]
        current_int = int_history[-1]

        termination_values = torch.stack(
            [
                self._env.termination_manager.get_term(name)[env_id]
                for name in self._termination_names
            ]
        ).to(dtype=torch.bool).detach().cpu()
        termination_reasons = "|".join(
            name
            for name, value in zip(self._termination_names, termination_values)
            if bool(value)
        )

        summary = self._build_summary(
            env_id,
            history_length,
            current_float,
            current_int,
            termination_reasons,
        )
        event = {
            "event_id": self._next_event_id,
            "summary": summary,
            "float_history": float_history,
            "int_history": int_history,
            "terrain_origin": self._terrain.env_origins[env_id]
            .detach()
            .cpu(),
            "termination_values": termination_values,
        }
        self._pending_events.append(event)
        self._pending_summaries.append(summary)
        self._next_event_id += 1
        self._accepted_event_count += 1

    def _build_summary(
        self,
        env_id: int,
        history_length: int,
        floats: torch.Tensor,
        integers: torch.Tensor,
        termination_reasons: str,
    ) -> dict[str, Any]:
        """Create a flat event row for fast filtering and plotting."""
        f = lambda name: float(floats[self._float_index[name]].item())
        i = lambda name: int(integers[self._int_index[name]].item())
        local_x = f("assigned_local_x")
        local_y = f("assigned_local_y")
        root_x = f("root_pos_w_x")
        root_y = f("root_pos_w_y")
        column = i("assigned_column")
        terrain_name = (
            self._column_names[column]
            if 0 <= column < len(self._column_names)
            else "outside"
        )

        x_from_grid_min = root_x + 0.5 * self._global_length
        y_from_grid_min = root_y + 0.5 * self._global_width
        row_mod = x_from_grid_min % self._terrain_size_x
        column_mod = y_from_grid_min % self._terrain_size_y
        row_seam_distance = min(
            row_mod, self._terrain_size_x - row_mod
        )
        column_seam_distance = min(
            column_mod, self._terrain_size_y - column_mod
        )
        pure_cfg = self._terrain_cfg.sub_terrains.get(terrain_name)
        pure_border_width = (
            float(pure_cfg.border_width)
            if terrain_name.startswith("pure_stair") and pure_cfg is not None
            else float("nan")
        )
        if math.isfinite(pure_border_width):
            x_border_start = (
                self._terrain_size_x * 0.5 - pure_border_width
            )
            y_border_start = (
                self._terrain_size_y * 0.5 - pure_border_width
            )
            distance_to_pure_x_border = abs(abs(local_x) - x_border_start)
            distance_to_pure_y_border = abs(abs(local_y) - y_border_start)
        else:
            distance_to_pure_x_border = float("nan")
            distance_to_pure_y_border = float("nan")

        code = i("trigger_code")
        trigger_reasons = []
        if code & self._HEIGHT_ERROR_BIT:
            trigger_reasons.append("absolute_height_error")
        if code & self._RAY_SPREAD_BIT:
            trigger_reasons.append("ray_height_spread")
        if code & self._INVALID_RAY_BIT:
            trigger_reasons.append("invalid_ray")

        return {
            "event_id": self._next_event_id,
            "env_id": env_id,
            "global_policy_step": int(self._env.common_step_counter),
            "episode_id": int(self._episode_serial[env_id].item()),
            "episode_step": i("episode_step"),
            "history_length": history_length,
            "trigger_code": code,
            "trigger_reasons": "|".join(trigger_reasons),
            "terrain_name": terrain_name,
            "assigned_level": i("assigned_level"),
            "assigned_column": column,
            "geometric_row": i("geometric_row"),
            "geometric_column": i("geometric_column"),
            "local_x": local_x,
            "local_y": local_y,
            "local_z": f("assigned_local_z"),
            "root_z": f("root_pos_w_z"),
            "root_vz": f("root_lin_vel_w_z"),
            "relative_height": f("relative_height"),
            "height_error": f("height_error"),
            "ray_mean_z": f("ray_mean_z"),
            "ray_median_z": f("ray_median_z"),
            "ray_min_z": f("ray_min_z"),
            "ray_max_z": f("ray_max_z"),
            "ray_spread": f("ray_spread"),
            "valid_ray_count": i("valid_ray_count"),
            "distance_to_row_seam": row_seam_distance,
            "distance_to_column_seam": column_seam_distance,
            "distance_to_pure_x_border_start": distance_to_pure_x_border,
            "distance_to_pure_y_border_start": distance_to_pure_y_border,
            "signed_distance_to_grid_x_edge": min(
                x_from_grid_min, self._global_length - x_from_grid_min
            ),
            "signed_distance_to_grid_y_edge": min(
                y_from_grid_min, self._global_width - y_from_grid_min
            ),
            "signed_distance_to_global_x_edge": min(
                x_from_grid_min + float(self._terrain_cfg.border_width),
                self._global_length
                + float(self._terrain_cfg.border_width)
                - x_from_grid_min,
            ),
            "signed_distance_to_global_y_edge": min(
                y_from_grid_min + float(self._terrain_cfg.border_width),
                self._global_width
                + float(self._terrain_cfg.border_width)
                - y_from_grid_min,
            ),
            "base_height_reward_rate": f("base_height_reward_rate"),
            "base_height_episode_sum": f("base_height_episode_sum"),
            "reset": bool(self._env.reset_buf[env_id].item()),
            "terminated": bool(self._env.reset_terminated[env_id].item()),
            "time_out": bool(self._env.reset_time_outs[env_id].item()),
            "termination_reasons": termination_reasons,
        }

    def _flush_pending_events(self) -> None:
        """Write one atomic shard without crossing the disk hard limit."""
        if not self._pending_events or self._disk_full:
            return

        events = self._pending_events
        summaries = self._pending_summaries
        shard_bytes = self._serialize(
            {
                "format_version": self._FORMAT_VERSION,
                "events": events,
            }
        )
        csv_stream = io.StringIO()
        writer = csv.DictWriter(
            csv_stream, fieldnames=self._summary_field_names()
        )
        writer.writerows(summaries)
        csv_bytes = csv_stream.getvalue().encode("utf-8")

        # Reserve 64 KiB for the small aggregate-count file.
        projected_size = (
            self._directory_size()
            + len(shard_bytes)
            + len(csv_bytes)
            + 64 * 1024
        )
        if projected_size > self._max_disk_bytes:
            self._disk_full = True
            self._pending_events = []
            self._pending_summaries = []
            print(
                "[WARN] Base-height anomaly recorder reached its "
                f"{self.cfg.max_disk_mb} MiB hard limit; detailed output stopped."
            )
            self._write_trigger_counts()
            return

        shard_path = os.path.join(
            self._output_dir, f"shard_{self._next_shard_id:05d}.pt"
        )
        self._atomic_write_bytes(shard_path, shard_bytes)
        with open(self._csv_path, "ab") as csv_file:
            csv_file.write(csv_bytes)
            csv_file.flush()
        self._saved_event_count += len(events)
        self._next_shard_id += 1
        self._pending_events = []
        self._pending_summaries = []
        self._write_trigger_counts()

    def _write_trigger_counts(self) -> None:
        """Atomically refresh the small aggregate counter file."""
        if not self._initialized:
            return
        counts_bytes = self._serialize(
            {
                "format_version": self._FORMAT_VERSION,
                "counts": self._trigger_counts.detach().cpu(),
                "column_to_subterrain": self._column_names,
            }
        )
        existing_size = (
            os.path.getsize(self._counts_path)
            if os.path.exists(self._counts_path)
            else 0
        )
        if (
            self._directory_size() - existing_size + len(counts_bytes)
            <= self._max_disk_bytes
        ):
            self._atomic_write_bytes(self._counts_path, counts_bytes)

    @staticmethod
    def _serialize(value: Any) -> bytes:
        stream = io.BytesIO()
        torch.save(value, stream)
        return stream.getvalue()

    def _directory_size(self) -> int:
        return sum(
            os.path.getsize(entry.path)
            for entry in os.scandir(self._output_dir)
            if entry.is_file()
        )

    @staticmethod
    def _atomic_write_bytes(path: str, data: bytes) -> None:
        temporary_path = f"{path}.tmp"
        with open(temporary_path, "wb") as output_file:
            output_file.write(data)
            output_file.flush()
        os.replace(temporary_path, path)

    def record_post_reset(
        self, env_ids: Sequence[int] | None
    ) -> tuple[None, None]:
        """Advance episode identities so each episode triggers at most once."""
        if env_ids is None:
            env_ids = slice(None)
        self._episode_serial[env_ids] += 1
        return None, None

    def close(self, file_path: str) -> None:
        """Flush pending clips and aggregate counters when the environment closes."""
        del file_path
        if self._closed:
            return
        self._closed = True
        if not self._initialized:
            return
        self._flush_pending_events()
        self._write_trigger_counts()
        print(
            "[INFO] Base-height anomaly recorder closed: "
            f"saved_events={self._saved_event_count}, output={self._output_dir}, "
            f"disk={self._directory_size() / 1024**2:.2f} MiB"
        )


@configclass
class BaseHeightAnomalyRecorderCfg(RecorderTermCfg):
    """Configuration for sparse base-height anomaly recording."""

    class_type: type[RecorderTerm] = BaseHeightAnomalyRecorder
    asset_name: str = "robot"
    sensor_name: str = "height_scanner_base"
    command_name: str = "base_velocity"
    reward_term_name: str = "base_height_l2"
    target_height: float = 0.65
    height_error_threshold: float = 0.30
    ray_spread_threshold: float = 0.50
    history_steps: int = 64
    max_events: int = 500
    max_events_per_step: int = 32
    events_per_shard: int = 32
    max_disk_mb: int = 64
    output_dir: str | None = None


@configclass
class BaseHeightAnomalyRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder manager with bounded sparse base-height diagnostics."""

    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset: bool = False
    export_in_close: bool = False
    base_height_anomaly: BaseHeightAnomalyRecorderCfg = (
        BaseHeightAnomalyRecorderCfg()
    )
