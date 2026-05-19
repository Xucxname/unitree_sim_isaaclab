# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Evaluation loop for Unitree-adapted GenieSim benchmark tasks."""

from __future__ import annotations

from datetime import datetime

import torch

from .episode_logger import EpisodeResult


class UnitreeBenchmarkEvaluator:
    def __init__(self, env, policy, render: bool = False):
        self.env = env
        self.policy = policy
        self.render = render

    def run_episode(self, episode, episode_index: int, max_steps: int | None = None) -> EpisodeResult:
        step_limit = int(max_steps or episode.max_steps)
        start_time = _now()
        total_reward = 0.0
        final_reward = 0.0
        steps = 0
        done = False
        error = None

        try:
            observation, _ = _reset_env(self.env)
            self.policy.reset(self.env, episode.instruction)

            with torch.inference_mode():
                for step in range(step_limit):
                    action = self.policy.act(self.env, observation, step)
                    observation, reward, done, _info = _step_env(self.env, action)
                    final_reward = _scalar_reward(reward)
                    total_reward += final_reward
                    steps = step + 1
                    if self.render:
                        self.env.sim.render()
                    if done:
                        break
            status = "completed"
        except Exception as exc:
            status = "error"
            error = str(exc)

        return EpisodeResult(
            task_key=episode.task_key,
            gym_id=episode.gym_id,
            instance_id=episode.instance_id,
            episode_index=episode_index,
            instruction=episode.instruction,
            policy=self.policy.name,
            status=status,
            steps=steps,
            total_reward=float(total_reward),
            final_reward=float(final_reward),
            done=bool(done),
            start_time=start_time,
            end_time=_now(),
            error=error,
        )


def _reset_env(env):
    result = env.reset()
    if isinstance(result, tuple):
        return result
    return result, {}


def _step_env(env, action):
    result = env.step(action)
    if not isinstance(result, tuple):
        return result, 0.0, False, {}
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        done = _to_bool(terminated) or _to_bool(truncated)
        return observation, reward, done, info
    if len(result) == 4:
        observation, reward, done, info = result
        return observation, reward, _to_bool(done), info
    raise RuntimeError(f"Unsupported env.step result length: {len(result)}")


def _scalar_reward(reward) -> float:
    if reward is None:
        return 0.0
    if isinstance(reward, torch.Tensor):
        if reward.numel() == 0:
            return 0.0
        return float(reward.detach().flatten()[0].cpu())
    try:
        return float(reward)
    except (TypeError, ValueError):
        return 0.0


def _to_bool(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.detach().flatten()[0].cpu()) if value.numel() else False
    return bool(value)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
