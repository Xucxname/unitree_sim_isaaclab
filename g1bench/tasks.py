# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Task discovery and episode metadata loading for Unitree benchmark runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tasks.g1_tasks.genie_selected_tasks_g1_29dof_dex1.genie_task_cfg import (
    GENIE_TASK_IDS,
    GENIE_TASK_SPECS,
    GenieTaskSpec,
)


@dataclass(frozen=True)
class BenchmarkEpisode:
    task_key: str
    gym_id: str
    instance_id: int
    instruction: str
    eval_type: str
    max_steps: int
    instructions_path: Path
    problems_path: Path | None


def list_task_ids() -> list[str]:
    return [spec.gym_id for spec in GENIE_TASK_SPECS.values()]


def resolve_task_specs(task_names: str | list[str]) -> list[GenieTaskSpec]:
    if isinstance(task_names, str):
        names = [name.strip() for name in task_names.split(",") if name.strip()]
    else:
        names = task_names

    if not names or names == ["all"]:
        return list(GENIE_TASK_SPECS.values())

    specs: list[GenieTaskSpec] = []
    for name in names:
        if name in GENIE_TASK_SPECS:
            specs.append(GENIE_TASK_SPECS[name])
        elif name in GENIE_TASK_IDS:
            specs.append(GENIE_TASK_IDS[name])
        else:
            raise ValueError(f"Unknown benchmark task '{name}'.")
    return specs


def parse_instance_ids(raw: str | None, task_root: Path) -> list[int]:
    if raw is None or raw.strip() == "":
        return [0]
    if raw.strip() == "all":
        return sorted(int(path.name) for path in task_root.iterdir() if path.is_dir() and path.name.isdigit())
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def load_benchmark_episode(project_root: Path, spec: GenieTaskSpec, instance_id: int, instruction_index: int = 0) -> BenchmarkEpisode:
    task_root = project_root / "genie/benchmark/config/llm_task" / spec.key / str(instance_id)
    instructions_path = task_root / "instructions.json"
    problems_path = task_root / "problems.json"

    instructions_data = _load_json(instructions_path)
    problems_data = _load_json(problems_path) if problems_path.is_file() else {}
    instructions = instructions_data.get("instructions", [])
    if not instructions:
        instruction = ""
    else:
        instruction_item = instructions[instruction_index % len(instructions)]
        instruction = str(instruction_item.get("instruction", ""))

    return BenchmarkEpisode(
        task_key=spec.key,
        gym_id=spec.gym_id,
        instance_id=instance_id,
        instruction=instruction,
        eval_type=str(instructions_data.get("eval_type", "")),
        max_steps=_extract_max_steps(problems_data, default=900),
        instructions_path=instructions_path,
        problems_path=problems_path if problems_path.is_file() else None,
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_max_steps(data: Any, default: int) -> int:
    step_outs: list[int] = []

    def visit(value: Any):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "StepOut":
                    try:
                        step_outs.append(int(item))
                    except (TypeError, ValueError):
                        pass
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(data)
    return max(step_outs) if step_outs else default
