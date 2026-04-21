"""
Feedback-Guided Targeted Mutation (§4.2 of the paper).

Given the blamed module π* and its current specification, the Mutator LLM
produces a minimal, trajectory-grounded edit to that module's prompt.
Only the blamed module is changed; all others are frozen.

Meta-prompt is reproduced verbatim from Appendix A.5.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from evotool.agent.executor import EpisodeRecord
from evotool.agent.modules import PolicySpec
from evotool.blamer.blamer import BlameResult
from evotool.blamer.diagnostics import DiagnosticEvents
from evotool.llm.client import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exact Mutator meta-prompt from Appendix A.5
# ---------------------------------------------------------------------------

MUTATOR_META_PROMPT = """\
# ROLE
You are a targeted prompt editor for exactly one module of a modular tool-using agent.

# GOAL
Given (i) a target module chosen from Planner, Selector, Caller, and
Synthesizer, (ii) the current specification of that module, (iii) a
failure episode packet containing the task input x, the module-local trajectory slice
(the target module's outputs plus nearby context), the final outcome and verifier
feedback, and (iv) the blamer's rationale and blame scores, produce a single minimal
and general edit to the selected module that addresses the diagnosed failure mode
while preserving the module's interface contract and output format.

# EDITING RULES
- Edit only target module specification; do not modify other modules.
- Do not add new tools or environments.
- Ground the edit in the trajectory
- Make the smallest change that fixes the error or suboptimality.

# HEURISTIC EDIT PATTERNS
- Schema/format error → add argument checklist, schema verification.
- Wrong tool selection → add decision rubric mapping subgoals to tools.
- Planning error → add explicit subgoals, state fields, ordering constraints,
  prerequisite checks.
- Ungrounded synthesis → require attribution to tool outputs, prohibit
  unsupported facts.

# OUTPUT FORMAT
Output plain text with the following sections in this order:
1. Target module
<planner or selector or caller or synthesizer>
2. Diagnosed error mode
<1-2 sentences describing the failure mode grounded in the trajectory>
3. Minimal edit summary
<1-2 short sentences describing the minimal change and why>
4. Revised target module spec
<updated specification text for the target module only>\
"""


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class MutationResult:
    target_module: str
    error_mode: str
    edit_summary: str
    revised_spec: str

    def to_dict(self) -> dict:
        return {
            "target_module": self.target_module,
            "error_mode": self.error_mode,
            "edit_summary": self.edit_summary,
            "revised_spec": self.revised_spec,
        }


# ---------------------------------------------------------------------------
# Mutator
# ---------------------------------------------------------------------------


class Mutator:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(
        self,
        episodes: list[EpisodeRecord],
        diagnostics: list[DiagnosticEvents],
        blamed_module: str,
        parent: PolicySpec,
        blame: BlameResult,
    ) -> MutationResult:
        """
        Given the blamed module and parent policy, produce a child spec
        by editing only the blamed module's prompt.
        """
        current_spec = parent.get(blamed_module)
        prompt = self._build_prompt(episodes, diagnostics, blamed_module, current_spec, blame)
        messages = [
            {"role": "system", "content": MUTATOR_META_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            raw = self.llm.call_mutator(messages)
            result = _parse_mutation_output(raw, blamed_module, current_spec)
        except Exception as exc:
            logger.warning("Mutator LLM call failed: %s. Keeping current spec.", exc)
            result = MutationResult(
                target_module=blamed_module,
                error_mode="Unknown (LLM call failed).",
                edit_summary="No edit applied.",
                revised_spec=current_spec,
            )
        return result

    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        episodes: list[EpisodeRecord],
        diagnostics: list[DiagnosticEvents],
        blamed_module: str,
        current_spec: str,
        blame: BlameResult,
    ) -> str:
        # Collect module-local trajectory slices from all episodes
        slices: list[str] = []
        for i, (ep, diag) in enumerate(zip(episodes, diagnostics)):
            local_steps = [
                s for s in ep.trajectory if s.module == blamed_module
            ]
            nearby = _nearby_context(ep, blamed_module)
            slice_text = "\n".join(
                f"  [{s.module.upper()}] {s.action[:400]}"
                + (f" → {s.observation[:200]}" if s.observation else "")
                for s in local_steps
            )
            slices.append(
                f"Episode {i + 1} (reward={ep.reward}):\n"
                f"  Task: {ep.task[:200]}\n"
                f"  {blamed_module.capitalize()} outputs:\n{slice_text or '  (none)'}\n"
                f"  Nearby context:\n{nearby}"
            )

        blamer_summary = (
            f"Blame scores: {json.dumps(blame.scores)}\n"
            f"Evidence: {blame.evidence[:400]}\n"
            f"Diagnosis: {blame.diagnosis}"
        )

        return (
            f"Target module: {blamed_module}\n\n"
            f"Current {blamed_module} specification:\n{current_spec}\n\n"
            f"Failure episode packets ({len(episodes)} episodes):\n"
            + "\n\n".join(slices)
            + f"\n\nBlamer rationale:\n{blamer_summary}\n\n"
            "Now produce a minimal, general edit to the target module specification."
        )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_mutation_output(text: str, fallback_module: str, fallback_spec: str) -> MutationResult:
    """
    Parse Mutator output with four sections:
    1. Target module
    2. Diagnosed error mode
    3. Minimal edit summary
    4. Revised target module spec
    """
    target_module = fallback_module
    error_mode = ""
    edit_summary = ""
    revised_spec = fallback_spec

    # Section 1: target module
    m = re.search(
        r"1\.\s*Target module\s*\n\s*(planner|selector|caller|synthesizer)",
        text, re.IGNORECASE,
    )
    if m:
        target_module = m.group(1).lower()

    # Section 2: diagnosed error mode
    m = re.search(
        r"2\.\s*Diagnosed error mode\s*\n(.*?)(?:3\.|$)", text, re.DOTALL | re.IGNORECASE
    )
    if m:
        error_mode = m.group(1).strip()[:500]

    # Section 3: minimal edit summary
    m = re.search(
        r"3\.\s*Minimal edit summary\s*\n(.*?)(?:4\.|$)", text, re.DOTALL | re.IGNORECASE
    )
    if m:
        edit_summary = m.group(1).strip()[:500]

    # Section 4: revised spec
    m = re.search(
        r"4\.\s*Revised target module spec\s*\n(.*?)$", text, re.DOTALL | re.IGNORECASE
    )
    if m:
        revised_spec = m.group(1).strip()

    # Fallback: if no section 4 found but text is long, take the last paragraph
    if revised_spec == fallback_spec and len(text) > len(fallback_spec) + 50:
        sections = re.split(r"\n{2,}", text.strip())
        if sections:
            last = sections[-1].strip()
            if len(last) > 20:
                revised_spec = last

    return MutationResult(
        target_module=target_module,
        error_mode=error_mode,
        edit_summary=edit_summary,
        revised_spec=revised_spec,
    )


def apply_mutation(parent: PolicySpec, result: MutationResult) -> PolicySpec:
    """
    Create a child PolicySpec by replacing only the blamed module's spec.
    All other modules are frozen (copied from parent).
    """
    return parent.set(result.target_module, result.revised_spec)


def _nearby_context(ep: EpisodeRecord, target_module: str) -> str:
    """Return steps immediately before/after each target-module step."""
    lines: list[str] = []
    for i, step in enumerate(ep.trajectory):
        if step.module == target_module:
            before = ep.trajectory[i - 1] if i > 0 else None
            after = ep.trajectory[i + 1] if i < len(ep.trajectory) - 1 else None
            if before:
                lines.append(f"    (before) [{before.module.upper()}] {before.action[:200]}")
            lines.append(f"    (target) [{step.module.upper()}] {step.action[:300]}")
            if after:
                lines.append(f"    (after)  [{after.module.upper()}] {after.action[:200]}")
    return "\n".join(lines) or "  (no context)"
