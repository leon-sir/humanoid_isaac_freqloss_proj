"""Perception-local environment owning shared frequency analyzer state."""

from collections.abc import Sequence

import torch

from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg

from .manager_based_rl_env import ManagerBasedRLYMEnv


class ManagerBasedRLYMFreqEnv(ManagerBasedRLYMEnv):
    """Reset all lazily-created shared frequency analyzers with each environment."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        self.joint_frequency_analyzers: dict[str, object] = {}
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def _reset_idx(self, env_ids: torch.Tensor | Sequence[int]):
        super()._reset_idx(env_ids)
        for analyzer in self.joint_frequency_analyzers.values():
            analyzer.reset(env_ids)
