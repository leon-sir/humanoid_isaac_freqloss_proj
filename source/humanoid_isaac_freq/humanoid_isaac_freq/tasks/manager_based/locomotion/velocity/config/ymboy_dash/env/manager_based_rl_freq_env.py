# Copyright (c) 2026 Zheng Pan
# SPDX-License-Identifier: Apache-2.0

"""Frequency-reward environment with one shared analyzer registry."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg

from .manager_based_rl_env import ManagerBasedRLYMEnv


class ManagerBasedRLYMFreqEnv(ManagerBasedRLYMEnv):
    """YMBOY environment that owns and resets shared joint-frequency analyzers."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        self.joint_frequency_analyzers: dict[str, object] = {}
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def _reset_idx(self, env_ids: torch.Tensor | Sequence[int]):
        super()._reset_idx(env_ids)
        for analyzer in self.joint_frequency_analyzers.values():
            analyzer.reset(env_ids)
