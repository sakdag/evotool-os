"""
BFCL evaluator: computes Accuracy on Single-turn and Multi-turn subsets.

Two evaluation modes:
  1. Lightweight (used during evolution): AST matching via our own evaluator.
  2. Official (for final reporting): write BFCL-format result files and
     call `bfcl evaluate` CLI, which gives the official leaderboard numbers.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evotool.agent.executor import EpisodeRecord, TaskInstance
from evotool.agent.modules import PolicySpec
from evotool.llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class BFCLResult:
    subset: str   # "single" or "multi"
    categories: list[str]
    total_instances: int
    correct_instances: int
    accuracy: float
    per_category: dict[str, float] = field(default_factory=dict)
    per_instance: list[dict] = field(default_factory=list)

    def display(self) -> str:
        return (
            f"BFCL [{self.subset}]  "
            f"Accuracy = {self.accuracy:.1f}  "
            f"({self.correct_instances}/{self.total_instances})"
        )

    def to_dict(self) -> dict:
        return {
            "subset": self.subset,
            "categories": self.categories,
            "total_instances": self.total_instances,
            "correct_instances": self.correct_instances,
            "accuracy": self.accuracy,
            "per_category": self.per_category,
            "per_instance": self.per_instance,
        }


def evaluate_bfcl(
    policy: PolicySpec,
    categories: list[str],
    test_instances: list[dict],
    ground_truth: dict[str, Any],
    llm: LLMClient,
    subset: str = "single",
    max_steps: int = 15,
) -> BFCLResult:
    """
    Run the modular agent on all test instances and compute Accuracy.
    Uses our lightweight AST-matching evaluator.
    """
    from benchmarks.bfcl.runner import BFCLEnv, execute_bfcl_episode, load_bfcl_tasks

    is_multi_turn = subset == "multi"
    tasks = load_bfcl_tasks(test_instances, ground_truth, is_multi_turn=is_multi_turn)
    env = BFCLEnv(ground_truth=ground_truth, is_multi_turn=is_multi_turn)

    per_instance: list[dict] = []
    per_category: dict[str, list[float]] = {cat: [] for cat in categories}
    n_correct = 0

    for i, task in enumerate(tasks):
        logger.info("  BFCL %s %d/%d: %s", subset, i + 1, len(tasks), task.task_id)
        try:
            ep = execute_bfcl_episode(policy, task, env, llm, max_steps)
            reward = ep.reward
        except Exception as exc:
            logger.warning("  Episode failed: %s", exc)
            reward = 0.0

        if reward >= 1.0 - 1e-6:
            n_correct += 1
        cat = task.metadata.get("bfcl_instance", {}).get("category", "unknown")
        if cat in per_category:
            per_category[cat].append(reward)

        per_instance.append({"task_id": task.task_id, "reward": reward})

    total = len(tasks)
    accuracy = 100.0 * n_correct / total if total > 0 else 0.0
    per_cat_acc = {
        cat: (100.0 * sum(vals) / len(vals) if vals else 0.0)
        for cat, vals in per_category.items()
    }

    result = BFCLResult(
        subset=subset,
        categories=categories,
        total_instances=total,
        correct_instances=n_correct,
        accuracy=accuracy,
        per_category=per_cat_acc,
        per_instance=per_instance,
    )
    logger.info(result.display())
    return result


def write_bfcl_result_file(
    episodes: list[EpisodeRecord],
    model_name: str,
    category: str,
    result_dir: str | Path,
) -> Path:
    """
    Write BFCL-format result file for use with official `bfcl evaluate` CLI.
    Format: one JSON object per line with id and result fields.
    """
    result_dir = Path(result_dir)
    (result_dir / model_name).mkdir(parents=True, exist_ok=True)

    filename = f"BFCL_v3_{category}_result.json"
    out_path = result_dir / model_name / filename

    with open(out_path, "w") as f:
        for ep in episodes:
            # Extract the last Caller action as the function call prediction
            caller_actions = [s.action for s in ep.trajectory if s.module == "caller"]
            prediction = caller_actions[-1] if caller_actions else ep.prediction

            # Try to format as BFCL result format
            try:
                import json as _json
                parsed = _json.loads(prediction)
                fn_name = parsed.get("tool_name") or parsed.get("function") or ""
                args = parsed.get("arguments") or {}
                result_str = f"{fn_name}({', '.join(f'{k}={v!r}' for k,v in args.items())})"
            except Exception:
                result_str = prediction[:500]

            f.write(json.dumps({"id": ep.task_id, "result": [result_str]}) + "\n")

    logger.info("Wrote BFCL result file: %s", out_path)
    return out_path


def run_official_bfcl_eval(
    model_name: str,
    category: str,
    project_root: str | Path,
) -> dict:
    """
    Call `bfcl evaluate` CLI to get official accuracy scores.
    Returns parsed score dict.
    """
    cmd = [
        "bfcl", "evaluate",
        "--model", model_name,
        "--test-category", category,
    ]
    env_vars = {"BFCL_PROJECT_ROOT": str(project_root)}
    import os
    env = {**os.environ, **env_vars}

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
        if result.returncode != 0:
            logger.warning("bfcl evaluate stderr: %s", result.stderr[:500])
        return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
    except Exception as exc:
        logger.warning("bfcl evaluate failed: %s", exc)
        return {"error": str(exc)}


def save_result(result: BFCLResult, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info("Saved BFCL result to %s", path)
