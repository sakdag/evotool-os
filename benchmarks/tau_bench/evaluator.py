"""
τ-Bench evaluator: computes Pass@1 on the test split.

Pass@1 = fraction of tasks where reward = 1.0 (database state matches ground truth
and all required outputs are present in the agent's responses).

This module provides both:
  1. A single-episode evaluator used by the EVOTOOL loop (reward from tau_bench step).
  2. A bulk test-set evaluator for final reporting.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evotool.agent.executor import EpisodeRecord, TaskInstance
from evotool.agent.modules import PolicySpec
from evotool.llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class TauBenchResult:
    env_name: str
    task_split: str
    total_tasks: int
    successful_tasks: int
    pass_at_1: float
    per_task_rewards: list[dict] = field(default_factory=list)

    def display(self) -> str:
        return (
            f"τ-Bench [{self.env_name}/{self.task_split}]  "
            f"Pass@1 = {self.pass_at_1:.1f}  "
            f"({self.successful_tasks}/{self.total_tasks})"
        )

    def to_dict(self) -> dict:
        return {
            "env_name": self.env_name,
            "task_split": self.task_split,
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "pass_at_1": self.pass_at_1,
            "per_task_rewards": self.per_task_rewards,
        }


def evaluate_tau_bench(
    policy: PolicySpec,
    env_name: str,
    task_indices: list[int],
    task_split: str,
    llm: LLMClient,
    cfg: dict,
    max_steps: int = 30,
) -> TauBenchResult:
    """
    Run the modular agent on all tasks in task_indices and compute Pass@1.

    This is the final evaluation function called from run_eval.py.
    """
    from benchmarks.tau_bench.runner import load_tau_bench_tasks, execute_tau_bench_episode

    tasks, wrapper = load_tau_bench_tasks(
        env_name=env_name,
        task_split=task_split,
        task_indices=task_indices,
        user_model=cfg.get("tau_bench", {}).get("user_model", "Qwen/Qwen3-8B"),
        user_strategy=cfg.get("tau_bench", {}).get("user_strategy", "llm"),
        user_provider=cfg.get("tau_bench", {}).get("user_provider", "openai"),
    )

    per_task_rewards: list[dict] = []
    n_success = 0

    for i, task in enumerate(tasks):
        logger.info("  Task %d/%d: %s", i + 1, len(tasks), task.task_id)
        try:
            ep = execute_tau_bench_episode(policy, task, wrapper, llm, max_steps)
            reward = ep.reward
        except Exception as exc:
            logger.warning("  Episode failed: %s", exc)
            reward = 0.0
        if reward >= 1.0 - 1e-6:
            n_success += 1
        per_task_rewards.append({"task_id": task.task_id, "reward": reward})

    total = len(tasks)
    pass_at_1 = 100.0 * n_success / total if total > 0 else 0.0

    result = TauBenchResult(
        env_name=env_name,
        task_split=task_split,
        total_tasks=total,
        successful_tasks=n_success,
        pass_at_1=pass_at_1,
        per_task_rewards=per_task_rewards,
    )
    logger.info(result.display())
    return result


def save_result(result: TauBenchResult, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info("Saved τ-Bench result to %s", path)
