#!/usr/bin/env python3
"""
Evaluate a saved policy (or INITIAL_POLICY) on a benchmark test split.

Usage:
  # Evaluate evolved policy
  python scripts/run_eval.py \
    --policy experiments/results/evotool_tau_bench_retail_full.json \
    --benchmark tau_bench_retail

  # Evaluate initial policy (static baseline)
  python scripts/run_eval.py --benchmark tau_bench_retail --use-initial

  # Evaluate on BFCL
  python scripts/run_eval.py \
    --policy experiments/results/evotool_bfcl_single_full.json \
    --benchmark bfcl_single
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
logger = logging.getLogger("run_eval")


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved EVOTOOL policy.")
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=["tau_bench_retail", "tau_bench_airline", "bfcl_single", "bfcl_multi"],
    )
    parser.add_argument("--policy", default=None, help="Path to saved policy JSON.")
    parser.add_argument("--use-initial", action="store_true", help="Use INITIAL_POLICY (static baseline).")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default=None, help="Output JSON file for results.")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test tasks (for quick tests).")
    args = parser.parse_args()

    if not args.policy and not args.use_initial:
        parser.error("Provide --policy or --use-initial.")

    cfg = load_config(args.config)
    llm_cfg = cfg.get("llm", {})

    from evotool.llm.client import LLMClient, LLMConfig
    from evotool.agent.modules import INITIAL_POLICY
    from evotool.evolution.loop import EvoToolResult

    # Load policy
    if args.use_initial:
        policy = INITIAL_POLICY
        policy_label = "initial"
    else:
        result = EvoToolResult.load(args.policy)
        policy = result.best_policy
        policy_label = Path(args.policy).stem

    llm = LLMClient(LLMConfig.from_dict(llm_cfg))

    # Auto output path
    if args.output:
        output_path = Path(args.output)
    else:
        results_dir = Path(cfg.get("paths", {}).get("results_dir", "experiments/results"))
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"eval_{args.benchmark}_{policy_label}.json"

    if args.benchmark.startswith("tau_bench"):
        env_name = "retail" if "retail" in args.benchmark else "airline"
        _eval_tau_bench(env_name, policy, cfg, llm, args.max_steps, args.limit, output_path)
    else:
        subset = "single" if "single" in args.benchmark else "multi"
        _eval_bfcl(subset, policy, cfg, llm, args.max_steps, args.limit, output_path)

    llm.close()


def _eval_tau_bench(env_name, policy, cfg, llm, max_steps, limit, output_path):
    from benchmarks.tau_bench.data_split import make_splits
    from benchmarks.tau_bench.runner import load_tau_bench_tasks, execute_tau_bench_episode
    from benchmarks.tau_bench.evaluator import TauBenchResult, save_result

    tau_cfg = cfg.get("tau_bench", {})
    split_cfg = cfg.get("data_splits", {}).get(f"tau_bench_{env_name}", {})

    test_split = split_cfg.get("test_split", "test")
    test_offset = split_cfg.get("test_offset", 0)

    logger.info("Loading %s test split...", env_name)
    all_tasks, wrapper = load_tau_bench_tasks(
        env_name=env_name,
        task_split=test_split,
        task_indices=None,
        user_model=tau_cfg.get("user_model", "Qwen/Qwen3-8B-AWQ"),
        user_strategy=tau_cfg.get("user_strategy", "llm"),
        user_provider=tau_cfg.get("user_provider", "openai"),
    )

    # For airline: use test indices starting from test_offset
    if test_offset > 0:
        import random as _random
        seed = split_cfg.get("seed", 42)
        rng = _random.Random(seed)
        pool = list(range(len(all_tasks)))
        rng.shuffle(pool)
        train_size = split_cfg.get("train_size", 200)
        sel_size = split_cfg.get("sel_size", 100)
        test_indices = pool[test_offset:]
        test_tasks = [all_tasks[i] for i in test_indices if i < len(all_tasks)]
    else:
        test_tasks = all_tasks

    if limit:
        test_tasks = test_tasks[:limit]

    logger.info("Evaluating on %d test tasks...", len(test_tasks))
    per_task_rewards = []
    n_success = 0

    for i, task in enumerate(test_tasks):
        logger.info("  [%d/%d] %s", i + 1, len(test_tasks), task.task_id)
        try:
            ep = execute_tau_bench_episode(policy, task, wrapper, llm, max_steps)
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
        env_name=env_name,
        task_split=test_split,
        total_tasks=total,
        successful_tasks=n_success,
        pass_at_1=pass_at_1,
        per_task_rewards=per_task_rewards,
    )
    logger.info("\n" + "=" * 50)
    logger.info(result.display())
    logger.info("=" * 50)
    save_result(result, output_path)


def _eval_bfcl(subset, policy, cfg, llm, max_steps, limit, output_path):
    from benchmarks.bfcl.data_split import (
        SINGLE_TURN_CATEGORIES, MULTI_TURN_CATEGORIES,
        load_all_bfcl, make_bfcl_splits,
    )
    from benchmarks.bfcl.runner import BFCLEnv, execute_bfcl_episode, load_bfcl_tasks
    from benchmarks.bfcl.evaluator import BFCLResult, save_result

    categories = SINGLE_TURN_CATEGORIES if subset == "single" else MULTI_TURN_CATEGORIES
    split_cfg = cfg.get("data_splits", {}).get(f"bfcl_{subset}", {})
    seed = split_cfg.get("seed", 42)
    train_size = split_cfg.get("train_size", 200)
    sel_size = split_cfg.get("sel_size", 100)
    is_multi_turn = (subset == "multi")

    logger.info("Loading BFCL %s...", subset)
    all_instances, all_gt = load_all_bfcl(categories)
    splits = make_bfcl_splits(categories, all_instances, train_size, sel_size, seed=seed)
    test_instances = [all_instances[i] for _, i in splits.test_indices]

    if limit:
        test_instances = test_instances[:limit]

    test_tasks = load_bfcl_tasks(test_instances, all_gt, is_multi_turn)
    env = BFCLEnv(ground_truth=all_gt, is_multi_turn=is_multi_turn)

    logger.info("Evaluating on %d test instances...", len(test_tasks))
    per_instance = []
    per_category: dict = {cat: [] for cat in categories}
    n_correct = 0

    for i, task in enumerate(test_tasks):
        logger.info("  [%d/%d] %s", i + 1, len(test_tasks), task.task_id)
        try:
            ep = execute_bfcl_episode(policy, task, env, llm, max_steps)
            reward = ep.reward
        except Exception as exc:
            logger.warning("  Failed: %s", exc)
            reward = 0.0
        if reward >= 1.0 - 1e-6:
            n_correct += 1
        cat = task.metadata.get("bfcl_instance", {}).get("category", "unknown")
        if cat in per_category:
            per_category[cat].append(reward)
        per_instance.append({"task_id": task.task_id, "reward": reward})

    total = len(test_tasks)
    accuracy = 100.0 * n_correct / total if total > 0 else 0.0
    per_cat_acc = {
        cat: (100.0 * sum(v) / len(v) if v else 0.0)
        for cat, v in per_category.items()
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
    logger.info("\n" + "=" * 50)
    logger.info(result.display())
    for cat, acc in per_cat_acc.items():
        logger.info("  %s: %.1f", cat, acc)
    logger.info("=" * 50)
    save_result(result, output_path)


if __name__ == "__main__":
    main()
