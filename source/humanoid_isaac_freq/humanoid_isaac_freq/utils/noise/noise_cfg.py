"""Configuration objects for image preprocessing pipelines."""

from collections.abc import Callable

import torch
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg

from .noise_model import crop_and_resize, depth_normalization, gaussian_blur_noise


@configclass
class ImageNoiseCfg(NoiseCfg):
    """Base configuration for an image transform."""

    func: Callable = lambda data, cfg, env_ids: data
    device: str | torch.device = "cpu"


@configclass
class CropAndResizeCfg(ImageNoiseCfg):
    """Crop ``(top, bottom, left, right)`` pixels and optionally resize."""

    crop_region: tuple[int, int, int, int] = (0, 0, 0, 0)
    resize_shape: tuple[int, int] | None = None
    func = crop_and_resize


@configclass
class GaussianBlurNoiseCfg(ImageNoiseCfg):
    """Apply a fixed Gaussian blur."""

    kernel_size: int = 3
    sigma: float = 1.0
    func = gaussian_blur_noise


@configclass
class DepthNormalizationCfg(ImageNoiseCfg):
    """Clip metric depth and optionally normalize it."""

    depth_range: tuple[float, float] = (0.0, 10.0)
    normalize: bool = True
    output_range: tuple[float, float] = (0.0, 1.0)
    func = depth_normalization
