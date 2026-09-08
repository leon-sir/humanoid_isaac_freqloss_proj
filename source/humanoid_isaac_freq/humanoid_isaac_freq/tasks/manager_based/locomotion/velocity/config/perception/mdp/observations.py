"""Depth image observation terms compatible with InstinctLab parkour."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def visualizable_image(
    env: "ManagerBasedEnv",
    data_type: str,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
    debug_vis: bool = False,
) -> torch.Tensor:
    """Return the latest processed camera frame in NCHW layout."""
    sensor = env.scene.sensors[sensor_cfg.name]
    images = sensor.data.output[data_type].clone()
    if images.ndim != 4:
        raise ValueError(
            f"Current camera output '{data_type}' must have shape (N,H,W,C), got {tuple(images.shape)}."
        )
    images = images.permute(0, 3, 1, 2)
    if debug_vis:
        delayed_visualizable_image._debug_visualize(images)
    return images


class delayed_visualizable_image(ManagerTermBase):
    """Select delayed, evenly spaced frames from a camera history.

    The returned tensor has shape ``(N, num_output_frames, H, W)`` and is
    ordered from oldest to newest. Its indexing and per-reset delay sampling
    intentionally match InstinctLab's ``delayed_visualizable_image`` term.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self.sensor_cfg = cfg.params.get("sensor_cfg", SceneEntityCfg("camera"))
        self.data_type = cfg.params["data_type"]
        if "history" not in self.data_type:
            raise ValueError("data_type must refer to a camera history output")

        self.sensor = env.scene.sensors[self.sensor_cfg.name]
        self.delayed_frame_ranges = cfg.params.get("delayed_frame_ranges", (0, 0))
        self.delayed_frame_distribution: Literal["uniform", "log_uniform"] = (
            cfg.params.get("delayed_frame_distribution", "uniform")
        )
        self._num_delayed_frames = torch.zeros(env.num_envs, device=env.device)
        self.history_skip_frames = max(cfg.params.get("history_skip_frames", 1), 1)
        self.num_output_frames = max(cfg.params.get("num_output_frames", 1), 1)

        data_shape = self.sensor._data.output[self.data_type].shape
        if len(data_shape) < 5:
            raise ValueError(
                f"Camera history '{self.data_type}' must have shape (N,T,H,W,C), got {data_shape}."
            )
        self.sensor_history_length = data_shape[1]
        self.frame_offset = torch.flip(
            torch.arange(
                0,
                self.num_output_frames * self.history_skip_frames,
                self.history_skip_frames,
                device=env.device,
            ),
            dims=(0,),
        )
        self._check_delay_bounds()

    def _check_delay_bounds(self) -> None:
        max_delay = self.delayed_frame_ranges[1]
        frames_needed = (self.num_output_frames - 1) * self.history_skip_frames + 1
        if frames_needed + max_delay > self.sensor_history_length:
            raise ValueError(
                "Camera history is too short for the requested depth frames: "
                f"need {frames_needed + max_delay}, have {self.sensor_history_length}."
            )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._num_delayed_frames[env_ids] = _randomize_prop_by_op(
            self._num_delayed_frames[env_ids].unsqueeze(-1),
            self.delayed_frame_ranges,
            None,
            slice(None),
            operation="abs",
            distribution=self.delayed_frame_distribution,
        ).squeeze(-1)

    def __call__(
        self,
        env: "ManagerBasedEnv",
        data_type: str,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
        history_skip_frames: int = 1,
        num_output_frames: int = 1,
        delayed_frame_ranges: tuple[int, int] = (0, 0),
        delayed_frame_distribution: Literal["uniform", "log_uniform"] = "uniform",
        flatten: bool = False,
        debug_vis: bool = False,
        scale_up_vis: int = 5,
    ) -> torch.Tensor:
        del env, data_type, sensor_cfg, history_skip_frames, num_output_frames
        del delayed_frame_ranges, delayed_frame_distribution, scale_up_vis

        images = self.sensor.data.output[self.data_type].clone().squeeze(-1)
        frame_indices = (
            self.sensor_history_length
            - self.frame_offset.unsqueeze(0)
            - self._num_delayed_frames.unsqueeze(1)
            - 1
        ).long()
        if not torch.all(frame_indices >= 0) or not torch.all(
            frame_indices < self.sensor_history_length
        ):
            raise RuntimeError(
                f"Depth history indices are out of bounds: {frame_indices}"
            )

        batch_indices = torch.arange(images.shape[0], device=images.device).unsqueeze(1)
        batch_indices = batch_indices.expand_as(frame_indices)
        delayed_frames = images[batch_indices, frame_indices]

        if debug_vis:
            self._debug_visualize(delayed_frames)
        if flatten:
            return delayed_frames.flatten(start_dim=1)
        return delayed_frames

    @staticmethod
    def _debug_visualize(images: torch.Tensor) -> None:
        """Display a depth mosaic when explicitly requested."""
        import cv2

        mosaic = images.permute(1, 2, 0, 3).flatten(0, 1).flatten(1, 2)
        max_value = torch.clamp(mosaic.max(), min=torch.finfo(mosaic.dtype).eps)
        frame = (mosaic * 255.0 / max_value).cpu().numpy().astype("uint8")
        cv2.imshow("depth_image", frame)
        cv2.waitKey(1)
