#!/usr/bin/env python3
"""
Run baseline methods for comparison against EVOTOOL.

Supported baselines:
  - react:   ReAct (Yao et al., 2022) — interleaved reasoning + tool use
  - static:  INITIAL_POLICY with no evolution (same as run_eval --use-initial)

Expected numbers (Qwen3-8B, from Table 1):
  τ-Bench:   ReAct Retail=33.1, Airline=14.4
  BFCL:      ReAct Single=70.6, Multi=33.3

Usage:
  python scripts/run_baseline.py --method react --benchmark tau_bench_retail
  python scripts/run_baseline.py --method react --benchmark bfcl_single
  python scripts/run_baseline.py --method static --benchmark bfcl_multi
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logger = logging.getLogger("run_baseline")


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a baseline on a benchmark.")
    parser.add_argument(
        "--method",
        required=True,
        choices=["react", "static"],
        help="Baseline method.",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=["tau_bench_retail", "tau_bench_airline", "bfcl_single", "bfcl_multi"],
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Re-run even if output already exists.")
    parser.add_argument("--fast", action="store_true", help="Fast mode: limit test set to 6 tasks.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.fast and not args.limit:
        args.limit = 6
    llm_cfg = cfg.get("llm", {})

    from evotool.llm.client import LLMClient, LLMConfig
    llm = LLMClient(LLMConfig.from_dict(llm_cfg))

    results_dir = Path(cfg.get("paths", {}).get("results_dir", "experiments/results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else \
        results_dir / f"baseline_{args.method}_{args.benchmark}.json"

    if output_path.exists() and not args.force:
        logger.info("Output already exists, skipping: %s  (use --force to re-run)", output_path)
        llm.close()
        return

    if args.method == "static":
        # Reuse run_eval with INITIAL_POLICY
        _run_static(args.benchmark, cfg, llm, args.max_steps, args.limit, output_path)
    elif args.method == "react":
        _run_react(args.benchmark, cfg, llm, args.max_steps, args.limit, output_path)

    llm.close()


def _run_static(benchmark, cfg, llm, max_steps, limit, output_path):
    from evotool.agent.modules import INITIAL_POLICY
    logger.info("Running STATIC baseline (INITIAL_POLICY) on %s", benchmark)

    # Delegate to run_eval logic
    import subprocess, sys
    cmd = [
        sys.executable, "scripts/run_eval.py",
        "--benchmark", benchmark,
        "--use-initial",
        "--output", str(output_path),
        "--max-steps", str(max_steps),
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    subprocess.run(cmd, check=True)


def _run_react(benchmark, cfg, llm, max_steps, limit, output_path):
    from baselines.react import run_react_episode
    from benchmarks.tau_bench.data_split import make_splits
    logger.info("Running ReAct baseline on %s", benchmark)

    if benchmark.startswith("tau_bench"):
        _react_tau_bench(benchmark, cfg, llm, max_steps, limit, output_path)
    else:
        _react_bfcl(benchmark, cfg, llm, max_steps, limit, output_path)


def _react_tau_bench(benchmark, cfg, llm, max_steps, limit, output_path):
    from benchmarks.tau_bench.runner import load_tau_bench_tasks, TauBenchEnvWrapper
    from benchmarks.tau_bench.data_split import make_splits
    from benchmarks.tau_bench.evaluator import TauBenchResult, save_result
    from baselines.react import run_react_episode

    env_name = "retail" if "retail" in benchmark else "airline"
    tau_cfg = cfg.get("tau_bench", {})
    split_cfg = cfg.get("data_splits", {}).get(f"tau_bench_{env_name}", {})
    test_split = split_cfg.get("test_split", "test")
    test_offset = split_cfg.get("test_offset", 0)
    seed = split_cfg.get("seed", 42)

    all_tasks, wrapper = load_tau_bench_tasks(
        env_name=env_name,
        task_split=test_split,
        task_indices=None,
        user_model=tau_cfg.get("user_model", "Qwen/Qwen3-8B-AWQ"),
        user_strategy=tau_cfg.get("user_strategy", "llm"),
        user_provider=tau_cfg.get("user_provider", "openai"),
    )

    if test_offset > 0:
        import random as _random
        rng = _random.Random(seed)
        pool = list(range(len(all_tasks)))
        rng.shuffle(pool)
        test_tasks = [all_tasks[i] for i in pool[test_offset:] if i < len(all_tasks)]
    else:
        test_tasks = all_tasks

    if limit:
        test_tasks = test_tasks[:limit]

    per_task_rewards = []
    n_success = 0

    for i, task in enumerate(test_tasks):
        logger.info("  [%d/%d] %s", i + 1, len(test_tasks), task.task_id)
        try:
            wrapper.reset_for_task(task)
            ep = run_react_episode(task, wrapper, llm, max_steps, env_wiki=wrapper.wiki)
            reward = ep.reward
        except Exception as exc:
            logger.warning("  Failed: %s", exc)
            reward = 0.0
        if reward >= 1.0 - 1e-6:
            n_success += 1
        per_task_rewards.append({"task_id": task.task_id, "reward": reward})

    total = len(test_tasks)
    pass_at_1 = 100.0 * n_success / total if total > 0 else 0.0
    result = TauBenchResult(
        env_name=env_name, task_split=test_split, total_tasks=total,
        successful_tasks=n_success, pass_at_1=pass_at_1,
        per_task_rewards=per_task_rewards,
    )
    logger.info("\n" + "=" * 50)
    logger.info("[ReAct] %s", result.display())
    logger.info("=" * 50)
    save_result(result, output_path)


def _react_bfcl(benchmark, cfg, llm, max_steps, limit, output_path):
    from benchmarks.bfcl.data_split import (
        SINGLE_TURN_CATEGORIES, MULTI_TURN_CATEGORIES,
        load_all_bfcl, make_bfcl_splits,
    )
    from benchmarks.bfcl.runner import BFCLEnv, load_bfcl_tasks
    from benchmarks.bfcl.evaluator import BFCLResult, save_result
    from baselines.react import run_react_episode

    subset = "single" if "single" in benchmark else "multi"
    categories = SINGLE_TURN_CATEGORIES if subset == "single" else MULTI_TURN_CATEGORIES
    split_cfg = cfg.get("data_splits", {}).get(f"bfcl_{subset}", {})
    seed = split_cfg.get("seed", 42)
    train_size = split_cfg.get("train_size", 200)
    sel_size = split_cfg.get("sel_size", 100)
    is_multi_turn = (subset == "multi")

    all_instances, all_gt = load_all_bfcl(categories)
    splits = make_bfcl_splits(categories, all_instances, train_size, sel_size, seed=seed)
    test_instances = [all_instances[i] for _, i in splits.test_indices]
    if limit:
        test_instances = test_instances[:limit]

    test_tasks = load_bfcl_tasks(test_instances, all_gt, is_multi_turn)
    env = BFCLEnv(ground_truth=all_gt, is_multi_turn=is_multi_turn)

    per_instance = []
    n_correct = 0

    for i, task in enumerate(test_tasks):
        logger.info("  [%d/%d] %s", i + 1, len(test_tasks), task.task_id)
        env.reset_episode()
        try:
            ep = run_react_episode(task, env, llm, max_steps)
            reward = ep.reward
        except Exception as exc:
            logger.warning("  Failed: %s", exc)
            reward = 0.0
        if reward >= 1.0 - 1e-6:
            n_correct += 1
        per_instance.append({"task_id": task.task_id, "reward": reward})

    total = len(test_tasks)
    accuracy = 100.0 * n_correct / total if total > 0 else 0.0
    result = BFCLResult(
        subset=subset, categories=categories, total_instances=total,
        correct_instances=n_correct, accuracy=accuracy,
        per_instance=per_instance,
    )
    logger.info("\n" + "=" * 50)
    logger.info("[ReAct] %s", result.display())
    logger.info("=" * 50)
    save_result(result, output_path)


if __name__ == "__main__":
    main()
