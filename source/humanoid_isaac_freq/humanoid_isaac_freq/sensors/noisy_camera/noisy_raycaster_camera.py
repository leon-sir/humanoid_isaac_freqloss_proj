"""Ray-caster camera with ordered preprocessing and history outputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.sensors.ray_caster import RayCasterCamera

from .noisy_camera import NoisyCameraMixin

if TYPE_CHECKING:
    from .noisy_raycaster_camera_cfg import NoisyRayCasterCameraCfg


class NoisyRayCasterCamera(NoisyCameraMixin, RayCasterCamera):
    """Apply the configured image pipeline after each terrain ray cast."""

    cfg: NoisyRayCasterCameraCfg

    def _initialize_impl(self) -> None:
        super()._initialize_impl()
        self.build_noise_pipeline()
        self.build_history_buffers()

    def _initialize_rays_impl(self) -> None:
        RayCasterCamera._initialize_rays_impl(self)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self.reset_noise_pipeline(env_ids)
        self.reset_history_buffers(env_ids)

    def _update_buffers_impl(self, env_ids: Sequence[int]) -> None:
        super()._update_buffers_impl(env_ids)
        self.apply_noise_pipeline_to_all_data_types(env_ids)
        self.update_history_buffers(env_ids)
