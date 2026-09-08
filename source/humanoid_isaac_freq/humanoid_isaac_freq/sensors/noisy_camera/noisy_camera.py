"""Ordered image transforms and asynchronous history support for cameras."""

from __future__ import annotations

import inspect
from collections.abc import Sequence

import torch
from isaaclab.utils import string_to_callable

from ...utils.buffers import AsyncCircularBuffer
from ...utils.noise import ImageNoiseCfg


class NoisyCameraMixin:
    """Post-process camera outputs and retain configured history tensors."""

    def build_noise_pipeline(self) -> None:
        self.noise_pipeline: list[ImageNoiseCfg] = []
        for noise_name, noise_cfg in self.cfg.noise_pipeline.items():
            if not isinstance(noise_cfg, ImageNoiseCfg):
                raise ValueError(f"Invalid image transform '{noise_name}': {noise_cfg}")
            noise_cfg.device = self.device
            if isinstance(noise_cfg.func, str):
                noise_cfg.func = string_to_callable(noise_cfg.func)
            if inspect.isclass(noise_cfg.func):
                noise_cfg.func = noise_cfg.func(
                    noise_cfg, num_envs=self.num_instances, device=self.device
                )
            self.noise_pipeline.append(noise_cfg)

        for data_type in self.cfg.data_types:
            self._data.output[f"{data_type}_noised"] = self.apply_noise_pipeline(
                self._data.output[data_type], self._ALL_INDICES
            )

    def apply_noise_pipeline(
        self,
        data: torch.Tensor,
        env_ids: torch.Tensor | Sequence[int],
    ) -> torch.Tensor:
        transformed = data.clone()
        for noise_cfg in self.noise_pipeline:
            transformed = noise_cfg.func(transformed, noise_cfg, env_ids)
        return transformed

    def apply_noise_pipeline_to_all_data_types(
        self, env_ids: torch.Tensor | Sequence[int]
    ) -> None:
        for data_type in self.cfg.data_types:
            self._data.output[f"{data_type}_noised"][env_ids] = (
                self.apply_noise_pipeline(
                    self._data.output[data_type][env_ids], env_ids
                )
            )

    def reset_noise_pipeline(self, env_ids: Sequence[int] | None = None) -> None:
        for noise_cfg in self.noise_pipeline:
            if hasattr(noise_cfg.func, "reset"):
                noise_cfg.func.reset(env_ids)

    def build_history_buffers(self) -> None:
        self.output_history_buffers: dict[str, AsyncCircularBuffer] = {}
        for data_type, history_length in self.cfg.data_histories.items():
            if data_type not in self._data.output:
                raise KeyError(
                    f"Cannot create camera history for '{data_type}'. "
                    f"Available outputs: {tuple(self._data.output.keys())}"
                )
            self.output_history_buffers[data_type] = AsyncCircularBuffer(
                history_length, self.num_instances, self.device
            )
            data_shape = self._data.output[data_type].shape
            self._data.output[f"{data_type}_history"] = torch.zeros(
                (data_shape[0], history_length, *data_shape[1:]),
                device=self.device,
                dtype=self._data.output[data_type].dtype,
            )

    def update_history_buffers(self, env_ids: torch.Tensor | Sequence[int]) -> None:
        for data_type, history_buffer in self.output_history_buffers.items():
            history_buffer.append(self._data.output[data_type][env_ids], env_ids)
            self._data.output[f"{data_type}_history"][env_ids] = (
                history_buffer.get_by_batch_ids(env_ids)
            )

    def reset_history_buffers(self, env_ids: Sequence[int] | None = None) -> None:
        for history_buffer in self.output_history_buffers.values():
            history_buffer.reset(env_ids)
