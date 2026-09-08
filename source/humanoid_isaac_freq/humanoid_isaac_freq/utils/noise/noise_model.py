"""Active image transforms from the InstinctLab parkour depth pipeline."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur

if TYPE_CHECKING:
    from .noise_cfg import CropAndResizeCfg, DepthNormalizationCfg, GaussianBlurNoiseCfg


def crop_and_resize(
    data: torch.Tensor,
    cfg: "CropAndResizeCfg",
    env_ids: torch.Tensor | Sequence[int],
) -> torch.Tensor:
    """Crop an ``(N,H,W,C)`` image and optionally bilinearly resize it."""
    del env_ids
    top, bottom, left, right = cfg.crop_region
    cropped = data[:, top : data.shape[1] - bottom, left : data.shape[2] - right, :]
    if cfg.resize_shape is None:
        return cropped
    resized = F.interpolate(
        cropped.permute(0, 3, 1, 2),
        size=cfg.resize_shape,
        mode="bilinear",
        align_corners=False,
    )
    return resized.permute(0, 2, 3, 1)


def gaussian_blur_noise(
    data: torch.Tensor,
    cfg: "GaussianBlurNoiseCfg",
    env_ids: torch.Tensor | Sequence[int],
) -> torch.Tensor:
    """Apply torchvision's Gaussian blur, matching InstinctLab."""
    del env_ids
    blurred = GaussianBlur(kernel_size=cfg.kernel_size, sigma=cfg.sigma)(
        data.permute(0, 3, 1, 2)
    )
    return blurred.permute(0, 2, 3, 1)


def depth_normalization(
    data: torch.Tensor,
    cfg: "DepthNormalizationCfg",
    env_ids: torch.Tensor | Sequence[int],
) -> torch.Tensor:
    """Clip depth and map it linearly to the requested output range."""
    del env_ids
    min_depth, max_depth = cfg.depth_range
    data = data.clip(min_depth, max_depth)
    if cfg.normalize:
        data = (data - min_depth) / (max_depth - min_depth)
        out_min, out_max = cfg.output_range
        data = data * (out_max - out_min) + out_min
    return data
