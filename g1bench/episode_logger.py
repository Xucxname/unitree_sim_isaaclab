# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""JSONL/summary output for Unitree benchmark runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class EpisodeResult:
    task_key: str
    gym_id: str
    instance_id: int
    episode_index: int
    instruction: str
    policy: str
    status: str
    steps: int
    total_reward: float
    final_reward: float
    done: bool
    start_time: str
    end_time: str
    error: str | None = None


class BenchmarkLogger:
    def __init__(self, output_dir: str | Path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(output_dir) / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_path = self.run_dir / "episodes.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self._results: list[EpisodeResult] = []

    def add(self, result: EpisodeResult):
        self._results.append(result)
        with self.episodes_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    def write_summary(self, extra: dict[str, Any] | None = None):
        completed = [item for item in self._results if item.status == "completed"]
        failed = [item for item in self._results if item.status == "error"]
        summary = {
            "run_dir": str(self.run_dir),
            "num_episodes": len(self._results),
            "completed": len(completed),
            "failed": len(failed),
            "average_total_reward": _mean(item.total_reward for item in completed),
            "average_steps": _mean(item.steps for item in completed),
            "episodes_path": str(self.episodes_path),
        }
        if extra:
            summary.update(extra)
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary


def _mean(values):
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))
