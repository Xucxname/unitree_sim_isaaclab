# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Gym registrations for selected GenieSim tasks on G129 Dex1 fixed base."""

from __future__ import annotations

import gymnasium as gym

from .genie_task_cfg import GENIE_TASK_SPECS, get_task_spec, make_genie_env_cfg
from .scene_events import load_genie_selected_task_assets


def _make_entry_point(task_key: str):
    def _entry_point():
        return make_genie_env_cfg(task_key)

    return _entry_point


for task_key, task_spec in GENIE_TASK_SPECS.items():
    gym.register(
        id=task_spec.gym_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": _make_entry_point(task_key),
        },
        disable_env_checker=True,
    )


__all__ = [
    "GENIE_TASK_SPECS",
    "get_task_spec",
    "load_genie_selected_task_assets",
    "make_genie_env_cfg",
]
