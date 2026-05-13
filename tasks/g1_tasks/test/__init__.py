# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import gymnasium as gym

from . import test_env_cfg


gym.register(
    id="test",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": test_env_cfg.TestG129DEX1BaseFixEnvCfg,
    },
    disable_env_checker=True,
)
