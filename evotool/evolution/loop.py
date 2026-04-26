"""
EVOTOOL self-evolving optimization loop (Algorithm 1 from the paper).

Four phases per generation:
  1. Trajectory Collection   — run parent policy on a mini-batch from S_train
  2. Blame Attribution       — Blamer LLM assigns module-level blame scores
  3. Targeted Mutation       — Mutator LLM edits only the blamed module
  4. Diversity-Aware Selection — retain candidates that win ≥1 S_sel instance

Returns the best-performing PolicySpec on S_sel after G generations.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Literal

from evotool.agent.executor import BenchmarkEnv, EpisodeRecord, TaskInstance, execute_policy
from evotool.agent.modules import INITIAL_POLICY, PolicySpec
from evotool.blamer.blamer import Blamer, BlameResult
from evotool.blamer.diagnostics import DiagnosticEvents, extract_diagnostics
from evotool.llm.client import LLMClient
from evotool.mutator.mutator import Mutator, apply_mutation
from evotool.selection.population import Candidate, Population

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generation log (for analysis / learning curves)
# ---------------------------------------------------------------------------


@dataclass
class GenerationLog:
    generation: int
    parent_uid: str
    blamed_module: str
    blame_scores: dict[str, float]
    batch_parent_reward: float
    batch_child_reward: float
    child_accepted: bool
    population_size: int
    sel_scores: dict[str, float] = field(default_factory=dict)  # uid → sel score
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "parent_uid": self.parent_uid,
            "blamed_module": self.blamed_module,
            "blame_scores": self.blame_scores,
            "batch_parent_reward": self.batch_parent_reward,
            "batch_child_reward": self.batch_child_reward,
            "child_accepted": self.child_accepted,
            "population_size": self.population_size,
            "sel_scores": self.sel_scores,
            "timestamp": self.timestamp,
        }


@dataclass
class EvoToolResult:
    best_policy: PolicySpec
    generation_logs: list[GenerationLog]
    final_population: list[dict]   # uid → {sel_score, win_frequency}
    benchmark: str = ""

    def to_dict(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "best_policy": self.best_policy.to_dict(),
            "generation_logs": [g.to_dict() for g in self.generation_logs],
            "final_population": self.final_population,
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved EvoTool result to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "EvoToolResult":
        with open(path) as f:
            d = json.load(f)
        return cls(
            best_policy=PolicySpec.from_dict(d["best_policy"]),
            generation_logs=[],
            final_population=d.get("final_population", []),
            benchmark=d.get("benchmark", ""),
        )


# ---------------------------------------------------------------------------
# Ablation modes (Table 2)
# ---------------------------------------------------------------------------

AblationMode = Literal[
    "full",        # EVOTOOL (default)
    "static",      # No evolution
    "random",      # Random module mutation
    "plan_only",   # Always mutate planner
    "sel_only",    # Always mutate selector
    "call_only",   # Always mutate caller
    "syn_only",    # Always mutate synthesizer
    "monolithic",  # Mutate all modules at once as a single string
]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_evotool(
    train_pool: list[TaskInstance],
    sel_set: list[TaskInstance],
    env: BenchmarkEnv,
    llm: LLMClient,
    max_generations: int = 8,
    mini_batch_size: int = 3,
    max_steps_per_episode: int = 30,
    initial_policy: PolicySpec | None = None,
    ablation: AblationMode = "full",
    log_dir: str | Path | None = None,
    benchmark_name: str = "",
    max_workers: int = 1,
    checkpoint_path: str | Path | None = None,
) -> EvoToolResult:
    """
    EVOTOOL Algorithm 1.

    Args:
        train_pool:          S_train — task instances used during training
        sel_set:             S_sel   — held-out set for population selection
        env:                 Tool environment (τ-Bench or BFCL)
        llm:                 Shared LLM client
        max_generations:     G (paper default: 8)
        mini_batch_size:     B (paper default: 3)
        max_steps_per_episode: Per-episode rollout limit
        initial_policy:      Θ_0 (defaults to INITIAL_POLICY from Appendix A.7)
        ablation:            One of AblationMode variants for Table 2 experiments
        log_dir:             Directory to write per-generation JSON logs
        benchmark_name:      Label for logging / output files

    Returns:
        EvoToolResult with best policy and full generation logs.
    """
    if initial_policy is None:
        initial_policy = INITIAL_POLICY

    blamer = Blamer(llm)
    mutator = Mutator(llm)
    generation_logs: list[GenerationLog] = []
    start_generation = 1

    # ----------------------------------------------------------------
    # Static baseline: no evolution
    # ----------------------------------------------------------------
    if ablation == "static":
        logger.info("Ablation=static: returning INITIAL_POLICY without evolution.")
        return EvoToolResult(
            best_policy=initial_policy,
            generation_logs=[],
            final_population=[],
            benchmark=benchmark_name,
        )

    env_factory = getattr(env, "clone", None) or (lambda: env)

    # ----------------------------------------------------------------
    # Resume from checkpoint if one exists; otherwise start fresh
    # ----------------------------------------------------------------
    ckpt_path = Path(checkpoint_path) if checkpoint_path else None
    if ckpt_path and ckpt_path.exists():
        population, generation_logs, start_generation = _load_checkpoint(ckpt_path)
        logger.info(
            "Resumed from checkpoint %s (completed %d/%d gens, pop=%d).",
            ckpt_path, start_generation - 1, max_generations, population.size(),
        )
    else:
        population = Population([initial_policy])
        logger.info("=== EVOTOOL [%s, ablation=%s] ===", benchmark_name or "?", ablation)
        logger.info("Initialising population on S_sel (%d instances)...", len(sel_set))
        population.initialize_win_frequencies(sel_set, env, llm, max_steps_per_episode, max_workers)

    # ----------------------------------------------------------------
    # Main loop: G generations
    # ----------------------------------------------------------------
    for g in range(start_generation, max_generations + 1):
        logger.info("\n--- Generation %d/%d ---", g, max_generations)

        # --------------------------------------------------------
        # PHASE 1: Trajectory Collection
        # --------------------------------------------------------
        parent_cand = population.sample_parent()
        parent_policy = parent_cand.policy
        batch = random.sample(train_pool, min(mini_batch_size, len(train_pool)))

        logger.info("  Parent: %s  Batch size: %d", parent_cand.uid, len(batch))
        batch_records: list[EpisodeRecord] = _run_batch(
            parent_policy, batch, env_factory, llm, max_steps_per_episode, max_workers
        )

        if not batch_records:
            logger.warning("  No valid episodes; skipping generation %d.", g)
            continue

        batch_parent_reward = sum(e.reward for e in batch_records) / len(batch_records)
        logger.info("  Batch parent reward: %.3f", batch_parent_reward)

        # --------------------------------------------------------
        # PHASE 2: Blame Attribution
        # --------------------------------------------------------
        diagnostics: list[DiagnosticEvents] = [extract_diagnostics(e) for e in batch_records]

        blamed_module = _determine_blamed_module(
            blamer, batch_records, diagnostics, ablation
        )
        blame_scores: dict[str, float] = {}

        if ablation == "full":
            try:
                blame_result = blamer.run(batch_records, diagnostics)
                blamed_module = blame_result.blamed_module
                blame_scores = blame_result.scores  # field is `scores`, not `blame_scores`
                logger.info("  Blame → %s  (scores: %s)", blamed_module, blame_scores)
            except Exception as exc:
                logger.warning("  Blamer failed: %s", exc)
                blame_result = _fallback_blame(blamed_module)
        else:
            blame_result = _fallback_blame(blamed_module)

        # --------------------------------------------------------
        # PHASE 3: Feedback-Guided Targeted Mutation
        # --------------------------------------------------------
        if ablation == "monolithic":
            child_policy, child_records = _monolithic_mutate(
                mutator, batch_records, diagnostics, parent_policy, blame_result,
                env, llm, batch, max_steps_per_episode
            )
        else:
            try:
                mutation = mutator.run(
                    batch_records, diagnostics, blamed_module, parent_policy, blame_result
                )
                child_policy = apply_mutation(parent_policy, mutation)
                logger.info(
                    "  Mutation: %s → %d chars (was %d chars)",
                    mutation.target_module,
                    len(mutation.revised_spec),
                    len(parent_policy.get(mutation.target_module)),
                )
            except Exception as exc:
                logger.warning("  Mutation failed: %s. Keeping parent.", exc)
                child_policy = parent_policy.copy()

            # Evaluate child on mini-batch
            child_records: list[EpisodeRecord] = _run_batch(
                child_policy, batch, env_factory, llm, max_steps_per_episode, max_workers
            )

        batch_child_reward = sum(e.reward for e in child_records) / len(child_records) if child_records else 0.0
        logger.info("  Batch child reward: %.3f", batch_child_reward)

        # Accept child only if it improves on the batch
        accepted = population.add_if_better(
            child_policy, parent_policy, batch_records, child_records
        )

        # --------------------------------------------------------
        # PHASE 4: Diversity-Aware Population Selection on S_sel
        # --------------------------------------------------------
        logger.info("  Running diversity selection on S_sel...")
        population.diversity_selection(sel_set, env, llm, max_steps_per_episode, max_workers)

        sel_scores = {c.uid: c.sel_score for c in population.candidates}

        # Log this generation
        gen_log = GenerationLog(
            generation=g,
            parent_uid=parent_cand.uid,
            blamed_module=blamed_module,
            blame_scores=blame_scores or {m: 0.0 for m in PolicySpec.MODULES},
            batch_parent_reward=batch_parent_reward,
            batch_child_reward=batch_child_reward,
            child_accepted=accepted,
            population_size=population.size(),
            sel_scores=sel_scores,
        )
        generation_logs.append(gen_log)

        if log_dir:
            _write_gen_log(log_dir, benchmark_name, g, gen_log)

        # Save checkpoint after every generation so the run can be resumed
        if ckpt_path:
            _save_checkpoint(ckpt_path, population, generation_logs, g + 1)
            logger.debug("  Checkpoint saved to %s (next gen=%d).", ckpt_path, g + 1)

        logger.info(
            "  Generation %d done | pop=%d | best S_sel=%.3f",
            g,
            population.size(),
            max(c.sel_score for c in population.candidates),
        )

    # ----------------------------------------------------------------
    # Return best candidate
    # ----------------------------------------------------------------
    best = population.best()
    logger.info("\n=== EVOTOOL done. Best candidate: %s (sel_score=%.3f) ===",
                best.uid, best.sel_score)

    return EvoToolResult(
        best_policy=best.policy,
        generation_logs=generation_logs,
        final_population=[
            {"uid": c.uid, "sel_score": c.sel_score, "win_frequency": c.win_frequency}
            for c in population.candidates
        ],
        benchmark=benchmark_name,
    )


# ---------------------------------------------------------------------------
# Ablation helpers
# ---------------------------------------------------------------------------


def _determine_blamed_module(
    blamer: Blamer,
    batch_records: list[EpisodeRecord],
    diagnostics: list[DiagnosticEvents],
    ablation: AblationMode,
) -> str:
    """Return the blame target based on the ablation mode (bypasses Blamer for non-full variants)."""
    if ablation == "random":
        return random.choice(list(PolicySpec.MODULES))
    if ablation == "plan_only":
        return "planner"
    if ablation == "sel_only":
        return "selector"
    if ablation == "call_only":
        return "caller"
    if ablation == "syn_only":
        return "synthesizer"
    # "full" and "monolithic": Blamer decides
    return "caller"  # placeholder; replaced in the main loop for "full"


def _fallback_blame(module: str) -> BlameResult:
    from evotool.blamer.blamer import BlameResult
    scores = {m: 0.1 for m in PolicySpec.MODULES}
    scores[module] = 0.7
    return BlameResult(scores=scores, evidence="", diagnosis="", blamed_module=module)


def _monolithic_mutate(
    mutator: Mutator,
    batch_records: list[EpisodeRecord],
    diagnostics: list[DiagnosticEvents],
    parent: PolicySpec,
    blame: BlameResult,
    env: BenchmarkEnv,
    llm: LLMClient,
    batch: list[TaskInstance],
    max_steps: int,
) -> tuple[PolicySpec, list[EpisodeRecord]]:
    """
    Monolithic ablation: concatenate all four module specs into one string,
    mutate it, and redistribute back to the four modules.
    """
    combined = "\n\n".join([
        f"[PLANNER]\n{parent.planner}",
        f"[SELECTOR]\n{parent.selector}",
        f"[CALLER]\n{parent.caller}",
        f"[SYNTHESIZER]\n{parent.synthesizer}",
    ])
    mono_parent = PolicySpec(
        planner=combined, selector="", caller="", synthesizer=""
    )
    import re
    # Run mutator targeting "planner" (which holds the combined spec)
    mutation = mutator.run(batch_records, diagnostics, "planner", mono_parent, blame)
    revised = mutation.revised_spec
    # Try to split back into modules
    planner = _extract_section(revised, "PLANNER") or parent.planner
    selector = _extract_section(revised, "SELECTOR") or parent.selector
    caller = _extract_section(revised, "CALLER") or parent.caller
    synthesizer = _extract_section(revised, "SYNTHESIZER") or parent.synthesizer

    child_policy = PolicySpec(
        planner=planner, selector=selector, caller=caller, synthesizer=synthesizer
    )
    child_records: list[EpisodeRecord] = []
    for task in batch:
        try:
            ep = execute_policy(child_policy, task, env, llm, max_steps)
            child_records.append(ep)
        except Exception:
            pass
    return child_policy, child_records


def _extract_section(text: str, tag: str) -> str:
    import re
    pattern = rf"\[{tag}\]\s*\n(.*?)(?:\[(?:PLANNER|SELECTOR|CALLER|SYNTHESIZER)\]|$)"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def _save_checkpoint(
    path: Path,
    population: "Population",
    generation_logs: list[GenerationLog],
    next_generation: int,
) -> None:
    """Persist current population + logs so the run can be resumed after a crash."""
    from evotool.selection.population import Candidate
    ckpt = {
        "next_generation": next_generation,
        "generation_logs": [g.to_dict() for g in generation_logs],
        "population": [
            {
                "uid": c.uid,
                "win_frequency": c.win_frequency,
                "sel_score": c.sel_score,
                "policy": c.policy.to_dict(),
            }
            for c in population.candidates
        ],
    }
    tmp = path.with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(ckpt, f, indent=2)
    tmp.replace(path)  # atomic replace


def _load_checkpoint(
    path: Path,
) -> tuple["Population", list[GenerationLog], int]:
    """Restore population and logs from a checkpoint file."""
    from evotool.selection.population import Candidate, Population
    with open(path) as f:
        ckpt = json.load(f)

    candidates = []
    for cd in ckpt["population"]:
        cand = Candidate(policy=PolicySpec.from_dict(cd["policy"]))
        cand.uid = cd["uid"]
        cand.win_frequency = cd["win_frequency"]
        cand.sel_score = cd["sel_score"]
        candidates.append(cand)

    pop = Population([])
    pop.candidates = candidates

    logs: list[GenerationLog] = []
    for ld in ckpt.get("generation_logs", []):
        logs.append(GenerationLog(
            generation=ld["generation"],
            parent_uid=ld["parent_uid"],
            blamed_module=ld["blamed_module"],
            blame_scores=ld["blame_scores"],
            batch_parent_reward=ld["batch_parent_reward"],
            batch_child_reward=ld["batch_child_reward"],
            child_accepted=ld["child_accepted"],
            population_size=ld["population_size"],
            sel_scores=ld.get("sel_scores", {}),
            timestamp=ld.get("timestamp", 0.0),
        ))

    return pop, logs, ckpt["next_generation"]


def _run_batch(
    policy: PolicySpec,
    batch: list[TaskInstance],
    env_factory: Callable[[], BenchmarkEnv],
    llm: LLMClient,
    max_steps: int,
    max_workers: int = 1,
) -> list[EpisodeRecord]:
    """Run a mini-batch of episodes, optionally in parallel."""
    records: list[EpisodeRecord] = []

    if max_workers > 1 and len(batch) > 1:
        futures: dict = {}
        with ThreadPoolExecutor(max_workers=min(max_workers, len(batch))) as pool:
            for task in batch:
                fut = pool.submit(execute_policy, policy, task, env_factory(), llm, max_steps)
                futures[fut] = task.task_id
            for fut in as_completed(futures):
                try:
                    records.append(fut.result())
                except Exception as exc:
                    logger.warning("    Episode failed for task %s: %s", futures[fut], exc)
    else:
        for task in batch:
            try:
                records.append(execute_policy(policy, task, env_factory(), llm, max_steps))
            except Exception as exc:
                logger.warning("    Episode failed for task %s: %s", task.task_id, exc)

    return records


def _write_gen_log(log_dir: str | Path, benchmark: str, gen: int, log: GenerationLog) -> None:
    p = Path(log_dir) / f"{benchmark}_gen{gen:02d}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(log.to_dict(), f, indent=2)
