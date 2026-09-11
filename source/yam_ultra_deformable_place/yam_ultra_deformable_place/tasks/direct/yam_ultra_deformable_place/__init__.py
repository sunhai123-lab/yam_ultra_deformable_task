# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Yam-Ultra-Deformable-Place-Direct-v0",
    entry_point=f"{__name__}.yam_ultra_deformable_place_env:YamUltraDeformablePlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.yam_ultra_deformable_place_env_cfg:YamUltraDeformablePlaceEnvCfg",
    },
)