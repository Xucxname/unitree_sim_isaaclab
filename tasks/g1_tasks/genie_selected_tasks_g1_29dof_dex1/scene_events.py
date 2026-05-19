# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Runtime USD loading helpers for selected GenieSim tasks."""

from __future__ import annotations

import os
from pathlib import Path

from .genie_task_cfg import get_task_spec


def _get_stage_utils():
    try:
        from isaacsim.core.utils.prims import get_prim_at_path
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    except ModuleNotFoundError:
        from omni.isaac.core.utils.prims import get_prim_at_path
        from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage

    return add_reference_to_stage, get_prim_at_path, get_current_stage


def _set_attr(prim, attr_name: str, value, value_type):
    attr = prim.GetAttribute(attr_name)
    if not attr.IsValid():
        attr = prim.CreateAttribute(attr_name, value_type)
    attr.Set(value)


def _ensure_popcorn_physics(stage, get_prim_at_path):
    from pxr import PhysxSchema, Sdf, UsdPhysics

    popcorn_proto = get_prim_at_path("/World/background/benchmark_popcorn_001")
    if popcorn_proto.IsValid():
        stage.Load(popcorn_proto.GetPath())
        UsdPhysics.RigidBodyAPI.Apply(popcorn_proto)
        UsdPhysics.CollisionAPI.Apply(popcorn_proto)
        UsdPhysics.MeshCollisionAPI.Apply(popcorn_proto)
        UsdPhysics.MassAPI.Apply(popcorn_proto)
        PhysxSchema.PhysxRigidBodyAPI.Apply(popcorn_proto)
        _set_attr(popcorn_proto, "physics:rigidBodyEnabled", True, Sdf.ValueTypeNames.Bool)
        _set_attr(popcorn_proto, "physics:kinematicEnabled", False, Sdf.ValueTypeNames.Bool)
        _set_attr(popcorn_proto, "physics:collisionEnabled", True, Sdf.ValueTypeNames.Bool)
        _set_attr(popcorn_proto, "physics:approximation", "boundingSphere", Sdf.ValueTypeNames.Token)
        _set_attr(popcorn_proto, "physics:mass", 0.003, Sdf.ValueTypeNames.Float)
        _set_attr(popcorn_proto, "physxRigidBody:disableGravity", False, Sdf.ValueTypeNames.Bool)


def _expand_popcorn_instancer(stage, get_prim_at_path, project_root: Path):
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    instancer_prim = get_prim_at_path("/World/background/popcornInstancer")
    proto_prim = get_prim_at_path("/World/background/benchmark_popcorn_001")
    if not instancer_prim.IsValid() or not proto_prim.IsValid():
        return

    instances_root_path = Sdf.Path("/World/background/popcornRigidInstances")
    instances_root = stage.GetPrimAtPath(instances_root_path)
    if instances_root.IsValid():
        return

    instancer = UsdGeom.PointInstancer(instancer_prim)
    positions = instancer.GetPositionsAttr().Get()
    proto_indices = instancer.GetProtoIndicesAttr().Get()
    if positions is None:
        positions = []
    if proto_indices is None:
        proto_indices = []
    if len(positions) == 0:
        return

    proto_xform = UsdGeom.Xformable(proto_prim)
    proto_matrix = proto_xform.ComputeLocalToWorldTransform(0.0)
    proto_translation = proto_matrix.ExtractTranslation()
    popcorn_usd = project_root / "genie/assets/background/common/popcorn/benchmark_popcorn_001/Aligned.usd"

    UsdGeom.Scope.Define(stage, instances_root_path)
    for index, point in enumerate(positions):
        if len(proto_indices) > 0 and int(proto_indices[index]) != 0:
            continue

        instance_path = instances_root_path.AppendChild(f"popcorn_{index:04d}")
        instance = UsdGeom.Xform.Define(stage, instance_path)
        prim = instance.GetPrim()
        prim.GetReferences().AddReference(str(popcorn_usd))

        position = proto_translation + Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
        instance.ClearXformOpOrder()
        instance.AddTranslateOp().Set(position)
        instance.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        instance.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))

        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim)
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        _set_attr(prim, "physics:rigidBodyEnabled", True, Sdf.ValueTypeNames.Bool)
        _set_attr(prim, "physics:kinematicEnabled", False, Sdf.ValueTypeNames.Bool)
        _set_attr(prim, "physics:collisionEnabled", True, Sdf.ValueTypeNames.Bool)
        _set_attr(prim, "physics:approximation", "boundingSphere", Sdf.ValueTypeNames.Token)
        _set_attr(prim, "physics:mass", 0.003, Sdf.ValueTypeNames.Float)
        _set_attr(prim, "physxRigidBody:disableGravity", False, Sdf.ValueTypeNames.Bool)

    instancer_prim.SetActive(False)
    proto_prim.SetActive(False)


def load_genie_selected_task_assets(task_name: str, instance_id: int = 0):
    spec = get_task_spec(task_name)
    if spec is None:
        return False

    project_root = Path(os.environ["PROJECT_ROOT"])
    background_usd = project_root / spec.background_usd
    subtask_usd = project_root / "genie/benchmark/config/llm_task" / spec.key / str(instance_id) / "scene.usda"

    from pxr import Sdf

    add_reference_to_stage, get_prim_at_path, get_current_stage = _get_stage_utils()
    stage = get_current_stage()

    if not get_prim_at_path("/World/background").IsValid():
        add_reference_to_stage(str(background_usd), "/World")

    if not get_prim_at_path("/Workspace").IsValid():
        add_reference_to_stage(str(subtask_usd), "/Workspace")

    stage.Load(Sdf.Path("/World"))
    stage.Load(Sdf.Path("/Workspace"))

    if spec.key == "scoop_popcorn":
        _ensure_popcorn_physics(stage, get_prim_at_path)
        _expand_popcorn_instancer(stage, get_prim_at_path, project_root)

    return True
