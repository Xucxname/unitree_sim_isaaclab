# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Scene configuration for testing migrated GenieSim USD assets."""

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass


@configclass
class TestAssetTableCylinderSceneCfg(InteractiveSceneCfg):
    """Scene for testing the GenieSim ICRA popcorn task assets.

    Genie USD layers are loaded from sim_main.py before simulation reset,
    because the popcorn PointInstancer expects the background layer to be
    mounted exactly at /World/background.
    """
