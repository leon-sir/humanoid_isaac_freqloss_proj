"""Configuration mixin for noisy camera outputs and histories."""

from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg


@configclass
class NoisyCameraCfgMixin:
    """Add an ordered image pipeline and history declarations to a camera cfg."""

    noise_pipeline: dict[str, NoiseCfg] = {}
    data_histories: dict[str, int] = {}
