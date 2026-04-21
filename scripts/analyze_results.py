#!/usr/bin/env python3
"""
Analysis script: generates learning curves, error-progression plots,
and compiles a final results table vs. the paper's reported numbers.

Usage:
  python scripts/analyze_results.py
  python scripts/analyze_results.py --results-dir experiments/results
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("analyze")

# Paper's reported numbers for Qwen3-8B (Table 1)
PAPER_NUMBERS = {
    "tau_bench_retail":   {"evotool": 35.9, "react": 33.1, "static": 21.0},
    "tau_bench_airline":  {"evotool": 15.7, "react": 14.4, "static": 21.0},
    "bfcl_single":        {"evotool": 76.9, "react": 70.6, "static": 52.0},
    "bfcl_multi":         {"evotool": 36.4, "react": 33.3, "static": 52.0},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--logs-dir", default="experiments/logs")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib plots.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    logs_dir = Path(args.logs_dir)

    # ----------------------------------------------------------------
    # 1. Collect all result files
    # ----------------------------------------------------------------
    results: dict[str, dict] = {}
    for f in results_dir.glob("*.json"):
        try:
            with open(f) as fh:
                data = json.load(fh)
            results[f.stem] = data
        except Exception as exc:
            logger.warning("Skipping %s: %s", f.name, exc)

    # ----------------------------------------------------------------
    # 2. Build comparison table
    # ----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EVOTOOL RESULTS vs. PAPER (Qwen3-8B)")
    print("=" * 80)
    print(f"{'Benchmark':<25} {'Ours':>8} {'Paper':>8} {'Delta':>8} {'React':>8}")
    print("-" * 60)

    our_scores: dict[str, float] = {}

    benchmarks = [
        ("tau_bench_retail",  "τ-Bench Retail",   "pass_at_1"),
        ("tau_bench_airline", "τ-Bench Airline",  "pass_at_1"),
        ("bfcl_single",       "BFCL Single",      "accuracy"),
        ("bfcl_multi",        "BFCL Multi",       "accuracy"),
    ]

    for bench_key, bench_label, metric in benchmarks:
        paper = PAPER_NUMBERS.get(bench_key, {})
        paper_evotool = paper.get("evotool", None)
        paper_react = paper.get("react", None)

        # Find our evotool result
        our_score = None
        for stem, data in results.items():
            if bench_key in stem and "baseline" not in stem and "ablation" not in stem:
                our_score = data.get(metric, data.get("pass_at_1", data.get("accuracy")))
                if our_score is not None:
                    break

        our_scores[bench_key] = our_score

        our_str = f"{our_score:.1f}" if our_score is not None else "N/A"
        paper_str = f"{paper_evotool:.1f}" if paper_evotool else "N/A"
        delta_str = f"{our_score - paper_evotool:+.1f}" if (our_score and paper_evotool) else "N/A"
        react_str = f"{paper_react:.1f}" if paper_react else "N/A"

        print(f"{bench_label:<25} {our_str:>8} {paper_str:>8} {delta_str:>8} {react_str:>8}")

    # Overall averages
    tau_scores = [our_scores.get("tau_bench_retail"), our_scores.get("tau_bench_airline")]
    bfcl_scores = [our_scores.get("bfcl_single"), our_scores.get("bfcl_multi")]
    tau_avg = sum(s for s in tau_scores if s) / max(len([s for s in tau_scores if s]), 1)
    bfcl_avg = sum(s for s in bfcl_scores if s) / max(len([s for s in bfcl_scores if s]), 1)
    print("-" * 60)
    print(f"{'τ-Bench Avg':<25} {tau_avg:>8.1f} {'25.8':>8} {'':>8} {'23.8':>8}")
    print(f"{'BFCL Avg':<25} {bfcl_avg:>8.1f} {'56.7':>8} {'':>8} {'52.0':>8}")
    print("=" * 80)

    # ----------------------------------------------------------------
    # 3. Learning curves from generation logs
    # ----------------------------------------------------------------
    gen_logs = _load_gen_logs(logs_dir)
    if gen_logs and not args.no_plots:
        _plot_learning_curves(gen_logs)
        _plot_error_progression(gen_logs)

    # ----------------------------------------------------------------
    # 4. Ablation table (Table 2)
    # ----------------------------------------------------------------
    _print_ablation_table(results)

    # ----------------------------------------------------------------
    # 5. Save summary
    # ----------------------------------------------------------------
    summary = {
        "our_scores": {k: v for k, v in our_scores.items() if v is not None},
        "paper_numbers": PAPER_NUMBERS,
        "tau_bench_avg": tau_avg,
        "bfcl_avg": bfcl_avg,
    }
    out = results_dir / "summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", out)


def _load_gen_logs(logs_dir: Path) -> dict[str, list[dict]]:
    """Load per-generation JSON logs."""
    logs: dict[str, list[dict]] = {}
    for f in sorted(logs_dir.glob("*.json")):
        parts = f.stem.rsplit("_gen", 1)
        if len(parts) != 2:
            continue
        benchmark = parts[0]
        if benchmark not in logs:
            logs[benchmark] = []
        try:
            with open(f) as fh:
                logs[benchmark].append(json.load(fh))
        except Exception:
            pass
    return logs


def _plot_learning_curves(gen_logs: dict[str, list[dict]]) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend for WSL

        fig, axes = plt.subplots(1, len(gen_logs), figsize=(5 * len(gen_logs), 4), squeeze=False)
        for ax, (benchmark, logs) in zip(axes[0], gen_logs.items()):
            gens = [l["generation"] for l in logs]
            parent_rewards = [l.get("batch_parent_reward", 0) for l in logs]
            child_rewards = [l.get("batch_child_reward", 0) for l in logs]
            ax.plot(gens, parent_rewards, "b-o", label="Parent (batch)")
            ax.plot(gens, child_rewards, "r-s", label="Child (batch)")
            ax.set_title(benchmark.replace("_", " ").title())
            ax.set_xlabel("Generation")
            ax.set_ylabel("Avg Reward (mini-batch)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle("EVOTOOL Learning Curves (Qwen3-8B)", fontsize=12)
        plt.tight_layout()
        out = Path("experiments/results/learning_curves.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        logger.info("Learning curves saved to %s", out)
        plt.close()
    except ImportError:
        logger.warning("matplotlib not available; skipping plots.")


def _plot_error_progression(gen_logs: dict[str, list[dict]]) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        modules = ["planner", "selector", "caller", "synthesizer"]
        colors = {"planner": "blue", "selector": "orange", "caller": "red", "synthesizer": "green"}

        n = len(gen_logs)
        if n == 0:
            return

        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
        for ax, (benchmark, logs) in zip(axes[0], gen_logs.items()):
            gens = [l["generation"] for l in logs]
            for mod in modules:
                scores = [l.get("blame_scores", {}).get(mod, 0) for l in logs]
                ax.plot(gens, scores, "-o", label=mod.capitalize(), color=colors[mod])
            ax.set_title(f"{benchmark.replace('_', ' ').title()}\nBlame Scores")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Blame Score")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle("Module-Level Error Progression (Qwen3-8B)", fontsize=12)
        plt.tight_layout()
        out = Path("experiments/results/error_progression.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        logger.info("Error progression saved to %s", out)
        plt.close()
    except ImportError:
        logger.warning("matplotlib not available; skipping plots.")


def _print_ablation_table(results: dict[str, dict]) -> None:
    """Print ablation results matching Table 2 format."""
    ablation_variants = ["static", "random", "plan_only", "sel_only", "call_only", "syn_only", "monolithic", "full"]
    paper_ablation = {
        "static":     {"tau_bench": 21.0, "bfcl": 52.0},
        "random":     {"tau_bench": 15.9, "bfcl": 43.6},
        "plan_only":  {"tau_bench": 24.7, "bfcl": 50.1},
        "sel_only":   {"tau_bench": 20.4, "bfcl": 48.2},
        "call_only":  {"tau_bench": 20.2, "bfcl": 55.6},
        "syn_only":   {"tau_bench": 21.2, "bfcl": 50.7},
        "monolithic": {"tau_bench": 20.6, "bfcl": 48.8},
        "full":       {"tau_bench": 25.8, "bfcl": 56.7},
    }

    ablation_results: dict[str, dict] = {}
    for stem, data in results.items():
        for variant in ablation_variants:
            if f"ablation_{variant}" in stem or (variant == "full" and "evotool" in stem and "ablation" not in stem):
                if variant not in ablation_results:
                    ablation_results[variant] = {}
                if "retail" in stem or "airline" in stem:
                    ablation_results[variant]["tau_bench"] = data.get("pass_at_1", 0)
                elif "bfcl" in stem:
                    ablation_results[variant]["bfcl"] = data.get("accuracy", 0)

    if not ablation_results:
        return

    print("\n" + "=" * 60)
    print("ABLATION RESULTS (Table 2 style, Qwen3-8B)")
    print("=" * 60)
    print(f"{'Variant':<15} {'τ-Bench (Ours)':>15} {'τ-Bench (Paper)':>16} {'BFCL (Ours)':>12} {'BFCL (Paper)':>13}")
    print("-" * 75)
    for variant in ablation_variants:
        ours = ablation_results.get(variant, {})
        paper = paper_ablation.get(variant, {})
        tb_ours = f"{ours.get('tau_bench', 0):.1f}" if "tau_bench" in ours else "N/A"
        tb_paper = f"{paper.get('tau_bench', 0):.1f}" if "tau_bench" in paper else "N/A"
        bfcl_ours = f"{ours.get('bfcl', 0):.1f}" if "bfcl" in ours else "N/A"
        bfcl_paper = f"{paper.get('bfcl', 0):.1f}" if "bfcl" in paper else "N/A"
        print(f"{variant:<15} {tb_ours:>15} {tb_paper:>16} {bfcl_ours:>12} {bfcl_paper:>13}")
    print("=" * 60)


if __name__ == "__main__":
    main()
