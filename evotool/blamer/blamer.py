"""
Trajectory-Grounded Blame Attribution (§4.1 of the paper).

The Blamer LLM receives a batch of episode records and their structured
diagnostic events, and outputs module-level blame scores in [0, 1].
The module with the highest score is selected as the mutation target π*.

Meta-prompt is reproduced verbatim from Appendix A.4.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from evotool.agent.executor import EpisodeRecord
from evotool.blamer.diagnostics import DiagnosticEvents
from evotool.llm.client import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exact Blamer meta-prompt from Appendix A.4
# ---------------------------------------------------------------------------

BLAMER_META_PROMPT = """\
# ROLE
You are a diagnostic judge for a modular tool-using agent.

# GOAL
Given (i) a task, (ii) a full execution trajectory, (iii) structured events for
each module in Planner, Selector, Caller, and Synthesizer extracted from the
trajectory, and (iv) an outcome signal with either 0 (fail) or 1 (success), your
task is to assign module-level blame to one of the four modules that is most
responsible for the errors or suboptimality in the trajectory.

# ATTRIBUTION CRITERIA
- Planner: missing or incorrect decomposition; incorrect ordering;
  dropped constraints or lost state.
- Selector: wrong tool choice; missing tool choice when necessary.
- Caller: schema or format violations; wrong parameters; malformed calls.
- Synthesizer: ungrounded final response; contradiction with tool outputs;
  missing integration of key observations.

# BLAME ASSIGNMENT RULES
- Give each module a score in 0 to 1.
- Blame the most causal module that most directly caused failure or quality loss.
- Use the extracted events for each module first, then confirm with trajectory evidence.
- Prefer the earliest causal mistake. If multi causal, still pick one primary.

# OUTPUT FORMAT Output plain text using following format:
1. Scores
planner <number>
selector <number>
caller <number>
synthesizer <number>
2. Evidence
Provide evidence for each module. Each line must include information from the
extracted events and a short reason grounded in the trajectory.
3. One sentence diagnosis
Write one sentence explaining why the primary module is blamed.\
"""


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class BlameResult:
    scores: dict[str, float] = field(
        default_factory=lambda: {"planner": 0.25, "selector": 0.25, "caller": 0.25, "synthesizer": 0.25}
    )
    evidence: str = ""
    diagnosis: str = ""
    blamed_module: str = "caller"  # argmax(scores)

    def to_dict(self) -> dict:
        return {
            "scores": self.scores,
            "evidence": self.evidence,
            "diagnosis": self.diagnosis,
            "blamed_module": self.blamed_module,
        }


# ---------------------------------------------------------------------------
# Blamer
# ---------------------------------------------------------------------------


class Blamer:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(
        self,
        episodes: list[EpisodeRecord],
        diagnostics: list[DiagnosticEvents],
    ) -> BlameResult:
        """
        Given a mini-batch of B episode records and their diagnostics,
        call the Blamer LLM and return aggregated blame scores.

        The Blamer receives all episodes in one prompt for holistic attribution.
        """
        prompt = self._build_prompt(episodes, diagnostics)
        messages = [
            {"role": "system", "content": BLAMER_META_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            raw = self.llm.call_blamer(messages)
            result = _parse_blame_output(raw)
        except Exception as exc:
            logger.warning("Blamer LLM call failed: %s. Using heuristic blame.", exc)
            result = _heuristic_blame(episodes)
        return result

    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        episodes: list[EpisodeRecord],
        diagnostics: list[DiagnosticEvents],
    ) -> str:
        parts: list[str] = []
        for i, (ep, diag) in enumerate(zip(episodes, diagnostics)):
            trajectory_text = _format_trajectory(ep)
            parts.append(
                f"=== EPISODE {i + 1} (reward={ep.reward}) ===\n"
                f"Task: {ep.task}\n\n"
                f"Trajectory:\n{trajectory_text}\n\n"
                f"Structured Diagnostic Events:\n{diag.to_text()}\n"
            )
        return (
            "\n\n".join(parts)
            + "\n\nBased on ALL episodes above, assign blame scores and identify the "
            "primary responsible module across the batch."
        )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_blame_output(text: str) -> BlameResult:
    """
    Parse Blamer output with format:
    1. Scores
    planner <number>
    selector <number>
    caller <number>
    synthesizer <number>
    2. Evidence
    ...
    3. One sentence diagnosis
    ...
    """
    logger.debug("Blamer raw output (first 600 chars): %s", text[:600])

    scores: dict[str, float] = {}
    evidence = ""
    diagnosis = ""

    # --- Step 1: isolate the Scores section to avoid false matches in Evidence ---
    section_match = re.search(
        r"(?:1\.\s*)?Scores?\s*\n(.*?)(?:\n\s*2\.|\n\s*Evidence|$)",
        text, re.DOTALL | re.IGNORECASE,
    )
    scores_text = section_match.group(1) if section_match else text

    # --- Step 2: line-anchored pattern inside scores section ---
    # Handles: "planner 0.7", "- planner: 0.7", "* Planner - 0.7", "planner: 0.7"
    score_pattern = re.compile(
        r"^\s*[-*•]?\s*\b(planner|selector|caller|synthesizer)\b\s*[:\-]?\s*(\d+(?:\.\d+)?)",
        re.IGNORECASE | re.MULTILINE,
    )
    for m in score_pattern.finditer(scores_text):
        module = m.group(1).lower()
        if module in scores:
            continue  # keep first match per module
        try:
            scores[module] = min(max(float(m.group(2)), 0.0), 1.0)
        except ValueError:
            scores[module] = 0.0

    # Ensure all four modules have a score
    for mod in ("planner", "selector", "caller", "synthesizer"):
        if mod not in scores:
            scores[mod] = 0.0

    # Warn when all zeros — likely the section boundary wasn't found
    if all(v == 0.0 for v in scores.values()):
        logger.warning(
            "Blamer returned all-zero scores — parse may have failed. "
            "Raw output (first 500 chars):\n%s", text[:500]
        )

    # --- Step 3: evidence and diagnosis ---
    ev_match = re.search(r"2\.\s*Evidence\s*\n(.*?)(?:3\.|$)", text, re.DOTALL | re.IGNORECASE)
    if ev_match:
        evidence = ev_match.group(1).strip()

    diag_match = re.search(r"3\.\s*One sentence diagnosis\s*\n(.+)", text, re.DOTALL | re.IGNORECASE)
    if diag_match:
        diagnosis = diag_match.group(1).strip().split("\n")[0]

    blamed = max(scores, key=lambda k: scores[k]) if scores else "caller"
    return BlameResult(scores=scores, evidence=evidence, diagnosis=diagnosis, blamed_module=blamed)


def _heuristic_blame(episodes: list[EpisodeRecord]) -> BlameResult:
    """
    Fallback: assign blame based on simple heuristics when the LLM fails.
    """
    caller_failures = sum(
        1 for ep in episodes for s in ep.trajectory if s.module == "caller" and not s.success
    )
    selector_failures = sum(
        1 for ep in episodes for s in ep.trajectory if s.module == "selector" and not s.success
    )
    n = len(episodes)
    failed = sum(1 for ep in episodes if ep.reward < 0.5)

    if caller_failures > 0:
        blamed = "caller"
        scores = {"planner": 0.1, "selector": 0.2, "caller": 0.6, "synthesizer": 0.1}
    elif selector_failures > 0:
        blamed = "selector"
        scores = {"planner": 0.1, "selector": 0.6, "caller": 0.2, "synthesizer": 0.1}
    elif failed > n // 2:
        blamed = "planner"
        scores = {"planner": 0.6, "selector": 0.2, "caller": 0.1, "synthesizer": 0.1}
    else:
        blamed = "synthesizer"
        scores = {"planner": 0.1, "selector": 0.1, "caller": 0.2, "synthesizer": 0.6}

    return BlameResult(
        scores=scores,
        evidence="Heuristic fallback blame attribution.",
        diagnosis=f"Heuristic: {blamed} module appears most problematic.",
        blamed_module=blamed,
    )


def _format_trajectory(ep: EpisodeRecord) -> str:
    lines: list[str] = []
    for i, step in enumerate(ep.trajectory):
        obs = f" → {step.observation[:200]}" if step.observation else ""
        lines.append(f"  [{step.module.upper()}] {step.action[:300]}{obs}")
    return "\n".join(lines)
