"""
Quick smoke test for all critical code paths.

Runs in ~5-10 minutes (vLLM must be up).  Checks:
  1. Airline split → non-zero S_sel (ZeroDivisionError fix)
  2. Diversity selection with n_sel=0 → no crash (guard fix)
  3. BlameResult.scores attribute (attribute-name bug fix)
  4. Parallel evaluation (max_workers=2) completes without errors
  5. 1-generation EVOTOOL mini-loop on retail (3 train, 3 sel tasks)
  6. 1-generation EVOTOOL mini-loop on airline (5 train, 3 sel tasks)

Usage:
    conda activate evotool
    python scripts/smoke_test.py
"""
import logging
import os
import sys
import traceback
from pathlib import Path

# Ensure project root is on sys.path regardless of how the script is invoked
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("smoke_test")

PASS = "PASS"
FAIL = "FAIL"
results: list[tuple[str, str, str]] = []  # (test_name, status, detail)


def check(name: str, fn):
    try:
        detail = fn() or ""
        results.append((name, PASS, str(detail)))
        log.info("  [PASS] %s  %s", name, detail)
    except Exception as exc:
        results.append((name, FAIL, str(exc)))
        log.error("  [FAIL] %s: %s", name, exc)
        traceback.print_exc()


# ─────────────────────────────────────────────
# 1. Airline split produces non-zero S_sel
# ─────────────────────────────────────────────
def test_airline_split():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    ac = cfg["data_splits"]["tau_bench_airline"]

    from benchmarks.tau_bench.data_split import make_splits
    from benchmarks.tau_bench.runner import load_tau_bench_tasks

    all_tasks, _ = load_tau_bench_tasks("airline", "test")
    total = len(all_tasks)

    splits = make_splits(
        total=total,
        train_size=ac["train_size"],
        sel_size=ac["sel_size"],
        seed=ac.get("seed", 42),
        test_offset=ac.get("test_offset", 0),
    )
    assert len(splits.sel_indices) > 0, (
        f"S_sel is empty! total={total}, train_size={ac['train_size']}, "
        f"sel_size={ac['sel_size']}, test_offset={ac.get('test_offset',0)}"
    )
    return (
        f"total={total}, train={len(splits.train_indices)}, "
        f"sel={len(splits.sel_indices)}, test={len(splits.test_indices)}"
    )


check("airline_split_nonzero_sel", test_airline_split)


# ─────────────────────────────────────────────
# 2. Diversity selection with n_sel=0 → no crash
# ─────────────────────────────────────────────
def test_empty_sel_no_crash():
    from evotool.agent.modules import INITIAL_POLICY
    from evotool.selection.population import Population

    pop = Population([INITIAL_POLICY])
    # Pass empty sel_set — should not raise
    pop.initialize_win_frequencies(sel_set=[], env=None, llm=None, max_steps=1)
    pop.diversity_selection(sel_set=[], env=None, llm=None, max_steps=1)
    assert pop.size() == 1
    return "n_sel=0 handled gracefully"


check("empty_sel_no_crash", test_empty_sel_no_crash)


# ─────────────────────────────────────────────
# 3. BlameResult.scores attribute exists
# ─────────────────────────────────────────────
def test_blame_result_attribute():
    from evotool.blamer.blamer import BlameResult
    br = BlameResult(
        scores={"planner": 0.1, "selector": 0.2, "caller": 0.6, "synthesizer": 0.1},
        evidence="test",
        diagnosis="test",
        blamed_module="caller",
    )
    # This was the bug: code tried blame_result.blame_scores
    _ = br.scores          # must not raise AttributeError
    _ = br.blamed_module   # must not raise
    assert br.blamed_module == "caller"
    return f"scores={br.scores}, blamed={br.blamed_module}"


check("blame_result_scores_attr", test_blame_result_attribute)


# ─────────────────────────────────────────────
# 4-6. Mini EVOTOOL loop (requires vLLM)
# ─────────────────────────────────────────────
with open("config.yaml") as _f:
    _cfg = yaml.safe_load(_f)

from evotool.llm.client import LLMClient, LLMConfig

_llm_cfg = LLMConfig.from_dict(_cfg["llm"])
_llm = LLMClient(_llm_cfg)

# Quick connectivity check
def test_vllm_reachable():
    import openai
    client = openai.OpenAI(
        base_url=_llm_cfg.base_url, api_key=_llm_cfg.api_key
    )
    models = client.models.list()
    names = [m.id for m in models.data]
    assert names, "No models returned by vLLM"
    return f"models={names}"


check("vllm_reachable", test_vllm_reachable)

if any(r[1] == FAIL for r in results if r[0] == "vllm_reachable"):
    log.error("vLLM not reachable — skipping LLM-dependent tests.")
    sys.exit(1)


# ─────────────────────────────────────────────
# 4. Parallel evaluation: 2 tasks in parallel
# ─────────────────────────────────────────────
def test_parallel_eval():
    from benchmarks.tau_bench.runner import load_tau_bench_tasks, execute_tau_bench_episode
    from evotool.agent.modules import INITIAL_POLICY
    from evotool.agent.executor import execute_policy
    from evotool.selection.population import Population

    tasks, wrapper = load_tau_bench_tasks("retail", "test")
    sel_tasks = tasks[:3]

    pop = Population([INITIAL_POLICY])

    # This exercises parallel path with max_workers=2
    pop.initialize_win_frequencies(
        sel_set=sel_tasks,
        env=wrapper,
        llm=_llm,
        max_steps=5,
        max_workers=2,
    )
    return f"sel_score={pop.candidates[0].sel_score:.3f}, win_freq={pop.candidates[0].win_frequency:.3f}"


check("parallel_eval_3_tasks", test_parallel_eval)


# ─────────────────────────────────────────────
# 5. Mini retail EVOTOOL (G=1, B=1, 3 sel)
# ─────────────────────────────────────────────
def test_mini_evotool_retail():
    from benchmarks.tau_bench.runner import load_tau_bench_tasks
    from evotool.agent.modules import INITIAL_POLICY
    from evotool.evolution.loop import run_evotool

    all_tasks, wrapper = load_tau_bench_tasks("retail", "train")
    train_pool = all_tasks[:5]
    sel_set = all_tasks[5:8]

    result = run_evotool(
        train_pool=train_pool,
        sel_set=sel_set,
        env=wrapper,
        llm=_llm,
        max_generations=1,
        mini_batch_size=1,
        max_steps_per_episode=5,
        ablation="full",
        benchmark_name="smoke_retail",
        max_workers=2,
    )
    return (
        f"gen_logs={len(result.generation_logs)}, "
        f"pop={len(result.final_population)}, "
        f"sel_score={result.final_population[0]['sel_score']:.3f}"
    )


check("mini_evotool_retail_G1", test_mini_evotool_retail)


# ─────────────────────────────────────────────
# 6. Mini airline EVOTOOL (G=1, B=1, 3 sel)
# ─────────────────────────────────────────────
def test_mini_evotool_airline():
    from benchmarks.tau_bench.runner import load_tau_bench_tasks
    from benchmarks.tau_bench.data_split import make_splits
    from evotool.evolution.loop import run_evotool
    import yaml

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    ac = cfg["data_splits"]["tau_bench_airline"]

    all_tasks, wrapper = load_tau_bench_tasks("airline", "test")
    splits = make_splits(
        total=len(all_tasks),
        train_size=ac["train_size"],
        sel_size=ac["sel_size"],
        seed=ac.get("seed", 42),
        test_offset=ac.get("test_offset", 0),
    )
    train_pool = [all_tasks[i] for i in splits.train_indices[:5]]
    sel_set = [all_tasks[i] for i in splits.sel_indices[:3]]

    assert len(sel_set) > 0, "sel_set still empty after config fix!"

    result = run_evotool(
        train_pool=train_pool,
        sel_set=sel_set,
        env=wrapper,
        llm=_llm,
        max_generations=1,
        mini_batch_size=1,
        max_steps_per_episode=5,
        ablation="full",
        benchmark_name="smoke_airline",
        max_workers=2,
    )
    return (
        f"gen_logs={len(result.generation_logs)}, "
        f"pop={len(result.final_population)}, "
        f"sel_score={result.final_population[0]['sel_score']:.3f}"
    )


check("mini_evotool_airline_G1", test_mini_evotool_airline)


# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
print()
print("=" * 60)
print("SMOKE TEST RESULTS")
print("=" * 60)
passed = sum(1 for _, s, _ in results if s == PASS)
failed = sum(1 for _, s, _ in results if s == FAIL)
for name, status, detail in results:
    icon = "✓" if status == PASS else "✗"
    print(f"  {icon} [{status}] {name}")
    if detail:
        print(f"       {detail}")
print()
print(f"  {passed}/{passed+failed} passed")
print("=" * 60)

if failed:
    sys.exit(1)
