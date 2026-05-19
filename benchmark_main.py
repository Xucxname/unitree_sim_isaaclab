# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

#!/usr/bin/env python3
"""Run Unitree-adapted benchmark evaluation for migrated GenieSim tasks."""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

project_root = Path(__file__).resolve().parent
os.environ["PROJECT_ROOT"] = str(project_root)

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description="Unitree benchmark evaluation")
    parser.add_argument("--tasks", default="all", help="Comma-separated task keys or Gym ids. Use 'all' for all migrated tasks.")
    parser.add_argument("--instances", default="0", help="Comma-separated instance ids or 'all'.")
    parser.add_argument("--episodes", type=int, default=1, help="Repeat count per task instruction.")
    parser.add_argument("--max_steps", type=int, default=None, help="Override problem StepOut/max steps.")
    parser.add_argument("--policy", choices=["zero", "replay"], default="zero", help="Benchmark policy adapter.")
    parser.add_argument("--replay_path", default=None, help="JSON replay file for --policy replay.")
    parser.add_argument("--output_dir", default="output/unitree_benchmark", help="Directory for benchmark results.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true", help="Render every benchmark step.")
    parser.add_argument("--list_tasks", action="store_true", help="List migrated benchmark Gym ids and exit.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
if getattr(args_cli, "headless", False):
    os.environ["LIVESTREAM"] = "0"
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import tasks  # noqa: F401
from g1bench.episode_logger import BenchmarkLogger, EpisodeResult
from g1bench.evaluator import UnitreeBenchmarkEvaluator
from g1bench.policies import make_policy
from g1bench.tasks import list_task_ids, load_benchmark_episode, parse_instance_ids, resolve_task_specs
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from tasks.g1_tasks.genie_selected_tasks_g1_29dof_dex1.scene_events import load_genie_selected_task_assets


def main():
    if args_cli.list_tasks:
        print("\n".join(list_task_ids()))
        return

    specs = resolve_task_specs(args_cli.tasks)
    logger = BenchmarkLogger(project_root / args_cli.output_dir)
    policy = make_policy(args_cli.policy, args_cli.replay_path)
    episode_index = 0

    for spec in specs:
        task_root = project_root / "genie/benchmark/config/llm_task" / spec.key
        instance_ids = parse_instance_ids(args_cli.instances, task_root)
        for instance_id in instance_ids:
            print(f"[benchmark] task={spec.key} gym_id={spec.gym_id} instance={instance_id}")
            episode = load_benchmark_episode(project_root, spec, instance_id, instruction_index=0)
            env = None
            try:
                _reset_stage()
                load_genie_selected_task_assets(spec.gym_id, instance_id=instance_id)
                env = _create_env(spec.gym_id)
                evaluator = UnitreeBenchmarkEvaluator(env=env, policy=policy, render=args_cli.render)
                for repeat in range(args_cli.episodes):
                    episode = load_benchmark_episode(project_root, spec, instance_id, instruction_index=repeat)
                    result = evaluator.run_episode(
                        episode=episode,
                        episode_index=episode_index,
                        max_steps=args_cli.max_steps,
                    )
                    logger.add(result)
                    print(
                        f"[benchmark] episode={episode_index} status={result.status} "
                        f"steps={result.steps} total_reward={result.total_reward:.4f}"
                    )
                    episode_index += 1
            except Exception as exc:
                traceback.print_exc()
                logger.add(_make_error_result(spec, episode, episode_index, policy.name, exc))
                episode_index += 1
            finally:
                if env is not None:
                    env.close()

    summary = logger.write_summary(
        {
            "tasks": [spec.gym_id for spec in specs],
            "policy": policy.name,
            "episodes_per_instruction": args_cli.episodes,
        }
    )
    print(f"[benchmark] results: {summary['run_dir']}")


def _create_env(task_name: str):
    env_cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=1)
    env_cfg.env_name = task_name
    env_cfg.seed = args_cli.seed
    env = gym.make(task_name, cfg=env_cfg).unwrapped
    env.seed(args_cli.seed)
    env.sim.reset()
    return env


def _make_error_result(spec, episode, episode_index: int, policy_name: str, exc: Exception) -> EpisodeResult:
    from datetime import datetime

    now = datetime.now().isoformat(timespec="seconds")
    return EpisodeResult(
        task_key=spec.key,
        gym_id=spec.gym_id,
        instance_id=episode.instance_id,
        episode_index=episode_index,
        instruction=episode.instruction,
        policy=policy_name,
        status="error",
        steps=0,
        total_reward=0.0,
        final_reward=0.0,
        done=False,
        start_time=now,
        end_time=now,
        error=str(exc),
    )


def _reset_stage():
    try:
        from isaacsim.core.utils.stage import create_new_stage

        create_new_stage()
    except Exception:
        try:
            import omni.usd

            omni.usd.get_context().new_stage()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
