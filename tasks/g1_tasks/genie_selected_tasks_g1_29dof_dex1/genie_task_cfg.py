# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Shared G1 Dex1 fixed-base configs for selected GenieSim tasks."""

from __future__ import annotations

from dataclasses import dataclass

import torch

import isaaclab.envs.mdp as base_mdp
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from tasks.common_config import CameraPresets, G1RobotPresets
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from tasks.common_observations.camera_state import get_camera_image
from tasks.common_observations.dex3_state import get_robot_dex3_joint_states
from tasks.common_observations.g1_29dof_state import get_robot_boy_joint_states
from tasks.common_observations.gripper_state import get_robot_gipper_joint_states
from tasks.common_scene.base_scene_test_asset import TestAssetTableCylinderSceneCfg


ROBOT_BASE_HEIGHT = 0.55
DUMP_TRASH_DEX1_ROBOT_BASE_HEIGHT = 0.8


@dataclass(frozen=True)
class GenieTaskSpec:
    key: str
    gym_id: str
    background_usd: str
    subtask_usd: str
    robot_base_position: tuple[float, float, float]
    robot_base_quaternion: tuple[float, float, float, float]

    @property
    def g1_base_position(self) -> tuple[float, float, float]:
        return (self.robot_base_position[0], self.robot_base_position[1], ROBOT_BASE_HEIGHT)


GENIE_TASK_SPECS: dict[str, GenieTaskSpec] = {
    "scoop_popcorn": GenieTaskSpec(
        key="scoop_popcorn",
        gym_id="Isaac-Scoop-Popcorn-G129-Dex1-Joint",
        background_usd="genie/assets/background/popcorn/popcorn_1/background.usda",
        subtask_usd="genie/benchmark/config/llm_task/scoop_popcorn/0/scene.usda",
        robot_base_position=(0.05481, -0.06354, 0.0024519748985767365),
        robot_base_quaternion=(1.0, 0.0, 0.0, 0.0),
    ),
    "clean_the_desktop": GenieTaskSpec(
        key="clean_the_desktop",
        gym_id="Isaac-Clean-The-Desktop-G129-Dex1-Joint",
        background_usd="genie/assets/background/study_room/study_4/background.usda",
        subtask_usd="genie/benchmark/config/llm_task/clean_the_desktop/0/scene.usda",
        robot_base_position=(-0.17, 0.08, 0.0),
        robot_base_quaternion=(1.0, 0.0, 0.0, 0.0),
    ),
    "empty_desktop_bin": GenieTaskSpec(
        key="empty_desktop_bin",
        gym_id="Isaac-Empty-Desktop-Bin-G129-Dex1-Joint",
        background_usd="genie/assets/background/study_room/study_3/background.usda",
        subtask_usd="genie/benchmark/config/llm_task/empty_desktop_bin/0/scene.usda",
        robot_base_position=(-0.25331934405082956, -0.26856191723415573, 0.0),
        robot_base_quaternion=(0.94382, 0.0, 0.0, 0.33045),
    ),
    "dump_trash_kitchen": GenieTaskSpec(
        key="dump_trash_kitchen",
        gym_id="Isaac-Dump-Trash-Kitchen-G129-Dex1-Joint",
        background_usd="genie/assets/background/kitchen/kitchen_1/background.usda",
        subtask_usd="genie/benchmark/config/llm_task/dump_trash_kitchen/0/scene.usda",
        robot_base_position=(0.34510623800766926, 0.13254566779200472, 0.0),
        robot_base_quaternion=(1.0, 0.0, 0.0, 0.0),
    ),
    "dump_trash_kitchen_dex3_debug": GenieTaskSpec(
        key="dump_trash_kitchen",
        gym_id="Isaac-Dump-Trash-Kitchen-G129-Dex3-Joint",
        background_usd="genie/assets/background/kitchen/kitchen_1/background.usda",
        subtask_usd="genie/benchmark/config/llm_task/dump_trash_kitchen/0/scene.usda",
        robot_base_position=(0.34510623800766926, 0.13254566779200472, 0.85),
        robot_base_quaternion=(1.0, 0.0, 0.0, 0.0),
    ),
    "heat_food": GenieTaskSpec(
        key="heat_food",
        gym_id="Isaac-Heat-Food-G129-Dex1-Joint",
        background_usd="genie/assets/background/kitchen/kitchen_1/background.usda",
        subtask_usd="genie/benchmark/config/llm_task/heat_food/0/scene.usda",
        robot_base_position=(0.15, 0.44296056032180786, 0.0),
        robot_base_quaternion=(1.0, 0.0, 0.0, 0.0),
    ),
    "hold_pot": GenieTaskSpec(
        key="hold_pot",
        gym_id="Isaac-Hold-Pot-G129-Dex1-Joint",
        background_usd="genie/assets/background/kitchen/kitchen_1/background.usda",
        subtask_usd="genie/benchmark/config/llm_task/hold_pot/0/scene.usda",
        robot_base_position=(0.41677864189035635, -0.3424178548514593, 0.0),
        robot_base_quaternion=(0.707, 0.0, 0.0, -0.707),
    ),
    "hang_tableware": GenieTaskSpec(
        key="hang_tableware",
        gym_id="Isaac-Hang-Tableware-G129-Dex1-Joint",
        background_usd="genie/assets/background/kitchen/kitchen_1/background.usda",
        subtask_usd="genie/benchmark/config/llm_task/hang_tableware/0/scene.usda",
        robot_base_position=(0.3764646733889869, 0.44296057285398793, 0.0),
        robot_base_quaternion=(1.0, 0.0, 0.0, 0.0),
    ),
    "open_door": GenieTaskSpec(
        key="open_door",
        gym_id="Isaac-Open-Door-G129-Dex1-Joint",
        background_usd="genie/assets/background/home/home_3/background.usda",
        subtask_usd="genie/benchmark/config/llm_task/open_door/0/scene.usda",
        robot_base_position=(-0.1, -0.0, 0.0),
        robot_base_quaternion=(1.0, 0.0, 0.0, 0.0),
    ),
    "take_book": GenieTaskSpec(
        key="take_book",
        gym_id="Isaac-Take-Book-G129-Dex1-Joint",
        background_usd="genie/assets/background/study_room/study_1/background.usda",
        subtask_usd="genie/benchmark/config/llm_task/take_book/0/scene.usda",
        robot_base_position=(-0.074, -0.06107266992330551, 0.0),
        robot_base_quaternion=(1.0, 0.0, 0.0, 0.0),
    ),
    "place_book": GenieTaskSpec(
        key="place_book",
        gym_id="Isaac-Place-Book-G129-Dex1-Joint",
        background_usd="genie/assets/background/study_room/study_1/background.usda",
        subtask_usd="genie/benchmark/config/llm_task/place_book/0/scene.usda",
        robot_base_position=(-0.074, -0.06107266992330551, 0.0),
        robot_base_quaternion=(1.0, 0.0, 0.0, 0.0),
    ),
    "store_objects_in_drawer": GenieTaskSpec(
        key="store_objects_in_drawer",
        gym_id="Isaac-Store-Objects-In-Drawer-G129-Dex1-Joint",
        background_usd="genie/assets/background/home/home_2/background.usda",
        subtask_usd="genie/benchmark/config/llm_task/store_objects_in_drawer/0/scene.usda",
        robot_base_position=(1.53725, -0.050541069358587265, 0.17135),
        robot_base_quaternion=(1.0, 0.0, 0.0, 0.0),
    ),
    "put_pen_into_penholder": GenieTaskSpec(
        key="put_pen_into_penholder",
        gym_id="Isaac-Put-Pen-Into-Penholder-G129-Dex1-Joint",
        background_usd="genie/assets/background/room/room_3/background.usda",
        subtask_usd="genie/benchmark/config/llm_task/put_pen_into_penholder/0/scene.usda",
        robot_base_position=(-0.71, 0.0, 0.0),
        robot_base_quaternion=(1.0, 0.0, 0.0, 0.0),
    ),
    "pick_block_color": GenieTaskSpec(
        key="pick_block_color",
        gym_id="Isaac-Pick-Block-Color-G129-Dex1-Joint",
        background_usd="genie/assets/background/home/home_b_aligned/background.usda",
        subtask_usd="genie/benchmark/config/llm_task/pick_block_color/0/scene.usda",
        robot_base_position=(-0.71, 0.0, 0.0),
        robot_base_quaternion=(1.0, 0.0, 0.0, 0.0),
    ),
    "pick_common_sense": GenieTaskSpec(
        key="pick_common_sense",
        gym_id="Isaac-Pick-Common-Sense-G129-Dex1-Joint",
        background_usd="genie/assets/background/home/home_b_aligned/background.usda",
        subtask_usd="genie/benchmark/config/llm_task/pick_common_sense/0/scene.usda",
        robot_base_position=(-0.66, 0.0, 0.0),
        robot_base_quaternion=(1.0, 0.0, 0.0, 0.0),
    ),
    "sorting_packages": GenieTaskSpec(
        key="sorting_packages",
        gym_id="Isaac-Sorting-Packages-G129-Dex1-Joint",
        background_usd="genie/assets/background/room/room_3/background.usda",
        subtask_usd="genie/benchmark/config/llm_task/sorting_packages/0/scene.usda",
        robot_base_position=(-0.71, 0.0, 0.0),
        robot_base_quaternion=(1.0, 0.0, 0.0, 0.0),
    ),
}

GENIE_TASK_IDS = {spec.gym_id: spec for spec in GENIE_TASK_SPECS.values()}


def get_task_spec(task_name: str) -> GenieTaskSpec | None:
    return GENIE_TASK_IDS.get(task_name)


@configclass
class GenieObjectSceneCfg(TestAssetTableCylinderSceneCfg):
    """G1 Dex1 fixed-base scene for GenieSim task foreground/background USDs."""

    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex1_base_fix(
        init_pos=(0.0, 0.0, ROBOT_BASE_HEIGHT),
        init_rot=(1.0, 0.0, 0.0, 0.0),
    )

    front_camera = CameraPresets.g1_front_camera()
    left_wrist_camera = CameraPresets.left_gripper_wrist_camera()
    right_wrist_camera = CameraPresets.right_gripper_wrist_camera()


@configclass
class GenieDex3DebugObjectSceneCfg(TestAssetTableCylinderSceneCfg):
    """G1 Dex3 fixed-base scene for debug GenieSim task foreground/background USDs."""

    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex3_base_fix(
        init_pos=(0.0, 0.0, 0.85),
        init_rot=(1.0, 0.0, 0.0, 0.0),
    )

    front_camera = CameraPresets.g1_front_camera()
    left_wrist_camera = CameraPresets.left_dex3_wrist_camera()
    right_wrist_camera = CameraPresets.right_dex3_wrist_camera()


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        robot_joint_state = ObsTerm(func=get_robot_boy_joint_states)
        robot_gipper_state = ObsTerm(func=get_robot_gipper_joint_states)
        camera_image = ObsTerm(func=get_camera_image)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class Dex3ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        robot_joint_state = ObsTerm(func=get_robot_boy_joint_states)
        robot_gipper_state = ObsTerm(func=get_robot_dex3_joint_states)
        camera_image = ObsTerm(func=get_camera_image)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    pass


@configclass
class RewardsCfg:
    pass


@configclass
class EventCfg:
    pass


@configclass
class GenieSelectedG129DEX1BaseFixEnvCfg(ManagerBasedRLEnvCfg):
    """Base env cfg reused by selected GenieSim tasks."""

    scene: GenieObjectSceneCfg = GenieObjectSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 20.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        self.event_manager = SimpleEventManager()
        self.event_manager.register(
            "reset_object_self",
            SimpleEvent(
                func=lambda env: base_mdp.reset_scene_to_default(
                    env,
                    torch.arange(env.num_envs, device=env.device),
                )
            ),
        )
        self.event_manager.register(
            "reset_all_self",
            SimpleEvent(
                func=lambda env: base_mdp.reset_scene_to_default(
                    env,
                    torch.arange(env.num_envs, device=env.device),
                )
            ),
        )


@configclass
class GenieSelectedG129DEX3DebugBaseFixEnvCfg(GenieSelectedG129DEX1BaseFixEnvCfg):
    """Debug Dex3 env cfg reused by selected GenieSim tasks."""

    scene: GenieDex3DebugObjectSceneCfg = GenieDex3DebugObjectSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )
    observations: Dex3ObservationsCfg = Dex3ObservationsCfg()


def make_genie_env_cfg(task_key: str) -> GenieSelectedG129DEX1BaseFixEnvCfg:
    spec = GENIE_TASK_SPECS[task_key]
    if "Dex3" in spec.gym_id:
        cfg = GenieSelectedG129DEX3DebugBaseFixEnvCfg()
        cfg.scene.robot = G1RobotPresets.g1_29dof_dex3_base_fix(
            init_pos=(
                spec.robot_base_position[0],
                spec.robot_base_position[1],
                0.85,
            ),
            init_rot=spec.robot_base_quaternion,
        )
        return cfg

    cfg = GenieSelectedG129DEX1BaseFixEnvCfg()
    init_pos = spec.g1_base_position
    if spec.gym_id == "Isaac-Dump-Trash-Kitchen-G129-Dex1-Joint":
        init_pos = (
            spec.robot_base_position[0],
            spec.robot_base_position[1],
            DUMP_TRASH_DEX1_ROBOT_BASE_HEIGHT,
        )
    cfg.scene.robot = G1RobotPresets.g1_29dof_dex1_base_fix(
        init_pos=init_pos,
        init_rot=spec.robot_base_quaternion,
    )
    return cfg
