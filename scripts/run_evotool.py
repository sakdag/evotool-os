#!/usr/bin/env python3
"""
Main EVOTOOL evolution script.

Runs the full EVOTOOL loop (Algorithm 1) on a specified benchmark,
saves the best evolved policy and full generation logs.

Usage examples:
  python scripts/run_evotool.py --benchmark tau_bench_retail
  python scripts/run_evotool.py --benchmark tau_bench_airline --generations 8
  python scripts/run_evotool.py --benchmark bfcl_single --ablation random
  python scripts/run_evotool.py --benchmark bfcl_multi --ablation monolithic
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logger = logging.getLogger("run_evotool")


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EVOTOOL on a benchmark.")
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=["tau_bench_retail", "tau_bench_airline", "bfcl_single", "bfcl_multi"],
        help="Which benchmark to run.",
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config.yaml."
    )
    parser.add_argument(
        "--generations", type=int, default=None, help="Max generations G (overrides config)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Mini-batch size B (overrides config)."
    )
    parser.add_argument(
        "--ablation",
        default="full",
        choices=["full", "static", "random", "plan_only", "sel_only", "call_only", "syn_only", "monolithic"],
        help="Ablation mode (default: full).",
    )
    parser.add_argument(
        "--output", default=None, help="Output JSON file path (auto-generated if not set)."
    )
    parser.add_argument(
        "--llm-model", default=None, help="Override LLM model name."
    )
    parser.add_argument(
        "--max-steps", type=int, default=30, help="Max steps per episode."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed."
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Fast mode: tiny splits + 2 generations for quick smoke-testing.",
    )
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    cfg = load_config(args.config)
    llm_cfg = cfg.get("llm", {})
    evotool_cfg = cfg.get("evotool", {})
    splits_cfg = cfg.get("data_splits", {})

    if args.llm_model:
        llm_cfg["agent_model"] = args.llm_model

    G = args.generations or evotool_cfg.get("max_generations", 8)
    B = args.batch_size or evotool_cfg.get("mini_batch_size", 3)
    max_steps = args.max_steps or evotool_cfg.get("max_steps_per_episode", 30)
    max_workers = evotool_cfg.get("max_workers", 1)

    # Fast mode: override splits and generations with tiny values
    if args.fast:
        logger.info("FAST MODE enabled — using tiny data splits and 2 generations.")
        G = 2
        B = 2
        for key in splits_cfg:
            splits_cfg[key]["train_size"] = 4
            splits_cfg[key]["sel_size"] = 3

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        results_dir = Path(cfg.get("paths", {}).get("results_dir", "experiments/results"))
        results_dir.mkdir(parents=True, exist_ok=True)
        label = f"{args.benchmark}_{args.ablation}"
        output_path = results_dir / f"evotool_{label}.json"

    log_dir = Path(cfg.get("paths", {}).get("logs_dir", "experiments/logs"))

    from evotool.llm.client import LLMClient, LLMConfig
    from evotool.agent.modules import INITIAL_POLICY
    from evotool.evolution.loop import run_evotool

    llm = LLMClient(LLMConfig.from_dict(llm_cfg))

    # ----------------------------------------------------------------
    # Load tasks and environment
    # ----------------------------------------------------------------
    if args.benchmark.startswith("tau_bench"):
        env_name = "retail" if "retail" in args.benchmark else "airline"
        _run_tau_bench(
            env_name=env_name,
            cfg=cfg,
            splits_cfg=splits_cfg,
            llm=llm,
            G=G, B=B, max_steps=max_steps,
            ablation=args.ablation,
            output_path=output_path,
            log_dir=log_dir,
            benchmark_name=args.benchmark,
            max_workers=max_workers,
        )
    else:
        subset = "single" if "single" in args.benchmark else "multi"
        _run_bfcl(
            subset=subset,
            cfg=cfg,
            splits_cfg=splits_cfg,
            llm=llm,
            G=G, B=B, max_steps=max_steps,
            ablation=args.ablation,
            output_path=output_path,
            log_dir=log_dir,
            benchmark_name=args.benchmark,
            max_workers=max_workers,
        )

    llm.close()


def _run_tau_bench(env_name, cfg, splits_cfg, llm, G, B, max_steps, ablation, output_path, log_dir, benchmark_name, max_workers=1):
    from benchmarks.tau_bench.data_split import make_splits
    from benchmarks.tau_bench.runner import load_tau_bench_tasks, execute_tau_bench_episode, TauBenchEnvWrapper
    from evotool.evolution.loop import run_evotool
    from evotool.agent.executor import TaskInstance

    tau_cfg = cfg.get("tau_bench", {})
    split_cfg = splits_cfg.get(f"tau_bench_{env_name}", {})
    seed = split_cfg.get("seed", 42)
    train_size = split_cfg.get("train_size", 100)
    sel_size = split_cfg.get("sel_size", 50)
    test_offset = split_cfg.get("test_offset", 0)
    train_split = split_cfg.get("train_split", "train")

    # Load training tasks
    logger.info("Loading %s tasks (split=%s)...", env_name, train_split)
    train_tasks, wrapper = load_tau_bench_tasks(
        env_name=env_name,
        task_split=train_split,
        task_indices=None,
        user_model=tau_cfg.get("user_model", "Qwen/Qwen3-8B-AWQ"),
        user_strategy=tau_cfg.get("user_strategy", "llm"),
        user_provider=tau_cfg.get("user_provider", "openai"),
    )

    splits = make_splits(
        total=len(train_tasks),
        train_size=train_size,
        sel_size=sel_size,
        seed=seed,
        test_offset=test_offset,
    )

    train_pool = [train_tasks[i] for i in splits.train_indices if i < len(train_tasks)]
    sel_set = [train_tasks[i] for i in splits.sel_indices if i < len(train_tasks)]

    logger.info("S_train: %d tasks, S_sel: %d tasks", len(train_pool), len(sel_set))

    # Create a TauBench-compatible BenchmarkEnv that uses execute_tau_bench_episode
    class TauBenchLoopEnv:
        """Minimal BenchmarkEnv for the EVOTOOL loop (delegates to wrapper)."""

        def __init__(self, w: TauBenchEnvWrapper, llm_client):
            self.wrapper = w
            self.llm = llm_client

        def execute_tool(self, task, tool_call):
            return self.wrapper.execute_tool(task, tool_call)

        def evaluate(self, task, prediction, trajectory):
            return self.wrapper.evaluate(task, prediction, trajectory)

        def find_tool_spec(self, tools, name):
            return self.wrapper.find_tool_spec(tools, name)

    # Patch execute_policy to use tau-bench conversation loop
    import evotool.agent.executor as _exec
    _orig_execute = _exec.execute_policy

    def _tau_execute_policy(policy, task, env, llm, max_steps=30):
        return execute_tau_bench_episode(policy, task, wrapper, llm, max_steps)

    _exec.execute_policy = _tau_execute_policy  # monkey-patch for the loop

    loop_env = TauBenchLoopEnv(wrapper, llm)

    ckpt_path = output_path.with_suffix(".ckpt.json")
    result = run_evotool(
        train_pool=train_pool,
        sel_set=sel_set,
        env=loop_env,
        llm=llm,
        max_generations=G,
        mini_batch_size=B,
        max_steps_per_episode=max_steps,
        ablation=ablation,
        log_dir=log_dir,
        benchmark_name=benchmark_name,
        max_workers=max_workers,
        checkpoint_path=ckpt_path,
    )

    _exec.execute_policy = _orig_execute  # restore

    result.save(output_path)
    logger.info("Done. Best policy saved to %s", output_path)


def _run_bfcl(subset, cfg, splits_cfg, llm, G, B, max_steps, ablation, output_path, log_dir, benchmark_name, max_workers=1):
    from benchmarks.bfcl.data_split import (
        SINGLE_TURN_CATEGORIES, MULTI_TURN_CATEGORIES,
        load_all_bfcl, make_bfcl_splits,
    )
    from benchmarks.bfcl.runner import BFCLEnv, execute_bfcl_episode, load_bfcl_tasks
    from evotool.evolution.loop import run_evotool
    import evotool.agent.executor as _exec

    categories = SINGLE_TURN_CATEGORIES if subset == "single" else MULTI_TURN_CATEGORIES
    split_cfg = splits_cfg.get(f"bfcl_{subset}", {})
    seed = split_cfg.get("seed", 42)
    train_size = split_cfg.get("train_size", 200)
    sel_size = split_cfg.get("sel_size", 100)
    is_multi_turn = (subset == "multi")

    logger.info("Loading BFCL %s categories: %s", subset, categories)
    all_instances, all_gt = load_all_bfcl(categories)

    if not all_instances:
        logger.error("No BFCL instances loaded. Check bfcl-eval installation.")
        return

    splits = make_bfcl_splits(categories, all_instances, train_size, sel_size, seed=seed)
    train_instances = [all_instances[i] for _, i in splits.train_indices]
    sel_instances = [all_instances[i] for _, i in splits.sel_indices]

    train_pool = load_bfcl_tasks(train_instances, all_gt, is_multi_turn)
    sel_set = load_bfcl_tasks(sel_instances, all_gt, is_multi_turn)

    logger.info("S_train: %d, S_sel: %d", len(train_pool), len(sel_set))

    env = BFCLEnv(ground_truth=all_gt, is_multi_turn=is_multi_turn)

    # Patch execute_policy for BFCL
    _orig_execute = _exec.execute_policy

    def _bfcl_execute_policy(policy, task, loop_env, llm, max_steps=15):
        return execute_bfcl_episode(policy, task, env, llm, max_steps)

    _exec.execute_policy = _bfcl_execute_policy

    ckpt_path = output_path.with_suffix(".ckpt.json")
    result = run_evotool(
        train_pool=train_pool,
        sel_set=sel_set,
        env=env,
        llm=llm,
        max_generations=G,
        mini_batch_size=B,
        max_steps_per_episode=max_steps,
        ablation=ablation,
        log_dir=log_dir,
        benchmark_name=benchmark_name,
        max_workers=max_workers,
        checkpoint_path=ckpt_path,
    )

    _exec.execute_policy = _orig_execute

    result.save(output_path)
    logger.info("Done. Best policy saved to %s", output_path)


if __name__ == "__main__":
    main()
