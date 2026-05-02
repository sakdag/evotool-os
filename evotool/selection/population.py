"""
Diversity-Aware Population Selection (§4.3 of the paper).

Maintains a population P = {Θ(i)}^N_i=1 of candidate PolicySpecs.
After each generation:
  - Retain only candidates that achieve the highest score on ≥1 S_sel instance.
  - Update sampling probabilities proportional to winner frequency.
  - Sample parents proportional to win frequency for the next generation.

This prevents the population from collapsing into a single mode and
preserves complementary competencies across the task distribution.
"""

from __future__ import annotations

import logging
import random
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

import evotool.agent.executor as _executor_mod
from evotool.agent.executor import BenchmarkEnv, EpisodeRecord, TaskInstance
from evotool.agent.modules import PolicySpec
from evotool.llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """A single policy in the population pool."""
    policy: PolicySpec
    uid: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    win_frequency: float = 0.0          # w(Θ): fraction of S_sel instances won
    sel_score: float = 0.0              # average reward on S_sel


class Population:
    """
    P = {Θ(i)}^N_i=1 with diversity-aware selection.

    Sampling distribution for parent selection:
        Pr(Θ_p = Θ) ∝ w(Θ)   (Algorithm 1, line 4)

    If all win_frequencies are zero (initial state), sample uniformly.
    """

    def __init__(self, initial_policies: list[PolicySpec]) -> None:
        self.candidates: list[Candidate] = [Candidate(policy=p) for p in initial_policies]

    # ------------------------------------------------------------------
    # Phase 4 initialisation: compute initial win frequencies on S_sel
    # ------------------------------------------------------------------

    def initialize_win_frequencies(
        self,
        sel_set: list[TaskInstance],
        env: BenchmarkEnv,
        llm: LLMClient,
        max_steps: int = 30,
        max_workers: int = 1,
    ) -> None:
        """
        Compute win frequencies on S_sel before the first generation.
        Required by Algorithm 1 (line 2: compute win frequency on S_sel).
        """
        logger.info("Initialising win frequencies on %d S_sel instances...", len(sel_set))
        if not sel_set:
            logger.warning("S_sel is empty — skipping win-frequency initialisation.")
            for c in self.candidates:
                c.win_frequency = 1.0
                c.sel_score = 0.0
            return
        self._run_diversity_selection(sel_set, env, llm, max_steps, max_workers)

    # ------------------------------------------------------------------
    # Phase 3 helper: add child if it improves on the mini-batch
    # ------------------------------------------------------------------

    def add_if_better(
        self,
        child: PolicySpec,
        parent: PolicySpec,
        batch_records: list[EpisodeRecord],
        child_records: list[EpisodeRecord],
    ) -> bool:
        """
        Algorithm 1 lines 21-23:
        Add Θ_child to P only if R̄(Θ_child; B) > R̄(Θ_parent; B).
        """
        parent_score = _average_reward(batch_records)
        child_score = _average_reward(child_records)
        logger.debug("  Parent batch score: %.3f  Child batch score: %.3f", parent_score, child_score)
        if child_score > parent_score:
            self.candidates.append(Candidate(policy=child))
            logger.info("  Child accepted (%.3f > %.3f). Population size: %d",
                        child_score, parent_score, len(self.candidates))
            return True
        logger.info("  Child rejected (%.3f <= %.3f).", child_score, parent_score)
        return False

    # ------------------------------------------------------------------
    # Phase 4: diversity-aware population selection on S_sel
    # ------------------------------------------------------------------

    def diversity_selection(
        self,
        sel_set: list[TaskInstance],
        env: BenchmarkEnv,
        llm: LLMClient,
        max_steps: int = 30,
        max_workers: int = 1,
    ) -> None:
        """
        Algorithm 1 lines 24-32:
        1. For each x ∈ S_sel, find the winning candidate W(x) = argmax_Θ r_x(Θ).
        2. Retain only candidates that win ≥1 instance.
        3. Update win frequencies = fraction of S_sel instances won.
        """
        if not sel_set:
            logger.warning("S_sel is empty — skipping diversity selection.")
            return
        self._run_diversity_selection(sel_set, env, llm, max_steps, max_workers)

    # ------------------------------------------------------------------
    # Parent sampling
    # ------------------------------------------------------------------

    def sample_parent(self) -> Candidate:
        """
        Algorithm 1 line 4: sample parent ∝ w(Θ).
        Fall back to uniform if all frequencies are zero.
        """
        weights = [max(c.win_frequency, 1e-6) for c in self.candidates]
        total = sum(weights)
        probs = [w / total for w in weights]
        return random.choices(self.candidates, weights=probs, k=1)[0]

    # ------------------------------------------------------------------
    # Best candidate (returned after G generations)
    # ------------------------------------------------------------------

    def best(self) -> Candidate:
        """
        Algorithm 1 line 34: return argmax_Θ ∈ P  (1/|S_sel|) Σ R(x, ŷ_Θ(x)).
        """
        return max(self.candidates, key=lambda c: c.sel_score)

    def size(self) -> int:
        return len(self.candidates)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_diversity_selection(
        self,
        sel_set: list[TaskInstance],
        env: BenchmarkEnv,
        llm: LLMClient,
        max_steps: int,
        max_workers: int = 1,
    ) -> None:
        if not self.candidates:
            return

        n_cand = len(self.candidates)
        n_sel = len(sel_set)

        if n_sel == 0:
            logger.warning("S_sel is empty — keeping all candidates with score=0.")
            for c in self.candidates:
                c.win_frequency = 1.0 / max(n_cand, 1)
                c.sel_score = 0.0
            return

        # rewards[i][j] = reward of candidate i on sel instance j
        rewards: list[list[float]] = [[0.0] * n_sel for _ in range(n_cand)]

        logger.info(
            "  Running diversity selection: %d candidates × %d S_sel instances (workers=%d)...",
            n_cand, n_sel, max_workers,
        )

        # env.clone() gives a fresh env per thread (no shared mutable state).
        env_factory: Callable[[], BenchmarkEnv] = getattr(env, "clone", None) or (lambda: env)

        if max_workers > 1:
            # Submit all (candidate, task) pairs in parallel
            futures: dict = {}
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                for i, cand in enumerate(self.candidates):
                    for j, task in enumerate(sel_set):
                        fut = pool.submit(
                            _eval_single, cand.policy, task, env_factory, llm, max_steps
                        )
                        futures[fut] = (i, j)

                for fut in as_completed(futures):
                    i, j = futures[fut]
                    try:
                        rewards[i][j] = fut.result()
                    except Exception as exc:
                        logger.warning(
                            "    Eval failed for cand %s, task %s: %s",
                            self.candidates[i].uid, sel_set[j].task_id, exc,
                        )
        else:
            for j, task in enumerate(sel_set):
                for i, cand in enumerate(self.candidates):
                    try:
                        fresh_env = env_factory()
                        ep = _executor_mod.execute_policy(cand.policy, task, fresh_env, llm, max_steps)
                        rewards[i][j] = ep.reward
                    except Exception as exc:
                        logger.warning(
                            "    Eval failed for cand %s, task %s: %s",
                            cand.uid, task.task_id, exc,
                        )

        # Determine winner per S_sel instance
        win_counts = [0] * n_cand
        for j in range(n_sel):
            col = [rewards[i][j] for i in range(n_cand)]
            best_score = max(col)
            for i in range(n_cand):
                if rewards[i][j] >= best_score - 1e-9:
                    win_counts[i] += 1

        # Retain only candidates with ≥1 win; update frequencies
        retained: list[Candidate] = []
        for i, cand in enumerate(self.candidates):
            if win_counts[i] > 0:
                cand.win_frequency = win_counts[i] / n_sel
                cand.sel_score = sum(rewards[i]) / n_sel
                retained.append(cand)
            else:
                logger.info("    Candidate %s discarded (0 wins on S_sel).", cand.uid)

        if not retained:
            # Safety: keep the best candidate even if it never strictly won
            best_idx = max(range(n_cand), key=lambda i: sum(rewards[i]))
            retained = [self.candidates[best_idx]]
            retained[0].win_frequency = 1.0
            retained[0].sel_score = sum(rewards[best_idx]) / n_sel  # n_sel > 0 here
            logger.warning("All candidates discarded; keeping best by average S_sel reward.")

        self.candidates = retained
        logger.info(
            "  Population after selection: %d candidates.", len(self.candidates)
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eval_single(
    policy: PolicySpec,
    task: TaskInstance,
    env_factory: Callable[[], BenchmarkEnv],
    llm: LLMClient,
    max_steps: int,
) -> float:
    """Run one episode and return its reward. Used by the ThreadPoolExecutor."""
    env = env_factory()
    ep = _executor_mod.execute_policy(policy, task, env, llm, max_steps)
    return ep.reward


def _average_reward(records: list[EpisodeRecord]) -> float:
    if not records:
        return 0.0
    return sum(r.reward for r in records) / len(records)
