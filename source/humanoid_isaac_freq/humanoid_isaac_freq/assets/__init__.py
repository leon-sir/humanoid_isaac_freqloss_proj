# Copyright (c) 2026 Zheng Pan
# SPDX-License-Identifier: Apache-2.0

"""Assets bundled with the humanoid Isaac frequency project."""

from pathlib import Path

HUMANOID_ISAAC_FREQ_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

from .ymbot_boy_12dof import YMBOT_BOY_12DOF_CFG  # noqa: E402, F401
