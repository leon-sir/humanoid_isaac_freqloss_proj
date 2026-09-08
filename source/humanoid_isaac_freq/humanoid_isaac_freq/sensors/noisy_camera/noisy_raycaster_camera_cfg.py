"""Configuration for :class:`NoisyRayCasterCamera`."""

from isaaclab.sensors.ray_caster import RayCasterCameraCfg
from isaaclab.utils import configclass

from .noisy_camera_cfg import NoisyCameraCfgMixin
from .noisy_raycaster_camera import NoisyRayCasterCamera


@configclass
class NoisyRayCasterCameraCfg(NoisyCameraCfgMixin, RayCasterCameraCfg):
    """Terrain ray-cast camera configuration with transforms and histories."""

    class_type: type = NoisyRayCasterCamera
