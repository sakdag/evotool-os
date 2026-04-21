"""
Diagnostic event extraction from episode trajectories.

The Blamer LLM requires structured events per module extracted from
the full trajectory before it can assign blame scores.

Four event categories (from §4.1 of the paper):
  - Planner events:    subgoal validity, ordering issues, lost state
  - Selector events:   wrong / missed tool choices
  - Caller events:     schema violations, malformed calls, execution errors
  - Synthesizer events: ungrounded claims, missing key observations
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from evotool.agent.executor import EpisodeRecord
from evotool.agent.modular_agent import AgentStep


@dataclass
class PlannerEvent:
    step_index: int
    plan_text: str
    issues: list[str] = field(default_factory=list)  # detected problems
    note: str = ""


@dataclass
class SelectorEvent:
    step_index: int
    selected_tool: str
    available_tools: list[str] = field(default_factory=list)
    tool_exists: bool = True
    subsequent_caller_failed: bool = False
    note: str = ""


@dataclass
class CallerEvent:
    step_index: int
    tool_name: str
    arguments: dict = field(default_factory=dict)
    execution_success: bool = True
    error_message: str = ""
    note: str = ""


@dataclass
class SynthesizerEvent:
    step_index: int
    answer: str
    tool_results_referenced: int = 0
    total_tool_results: int = 0
    grounded: bool = True
    note: str = ""


@dataclass
class DiagnosticEvents:
    """Structured events extracted from a single trajectory."""
    episode_id: str
    reward: float
    planner_events: list[PlannerEvent] = field(default_factory=list)
    selector_events: list[SelectorEvent] = field(default_factory=list)
    caller_events: list[CallerEvent] = field(default_factory=list)
    synthesizer_events: list[SynthesizerEvent] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "reward": self.reward,
            "planner_events": [_asdict(e) for e in self.planner_events],
            "selector_events": [_asdict(e) for e in self.selector_events],
            "caller_events": [_asdict(e) for e in self.caller_events],
            "synthesizer_events": [_asdict(e) for e in self.synthesizer_events],
        }

    def to_text(self) -> str:
        """Human-readable summary for the Blamer LLM prompt."""
        lines: list[str] = [f"Outcome: {'SUCCESS' if self.reward >= 1.0 else 'FAILURE'} (reward={self.reward})"]

        lines.append("\n--- PLANNER EVENTS ---")
        if self.planner_events:
            for e in self.planner_events:
                lines.append(f"  Step {e.step_index}: Plan produced.")
                if e.issues:
                    lines.append(f"    Issues detected: {'; '.join(e.issues)}")
        else:
            lines.append("  No planner events recorded.")

        lines.append("\n--- SELECTOR EVENTS ---")
        if self.selector_events:
            for e in self.selector_events:
                lines.append(f"  Step {e.step_index}: Selected '{e.selected_tool}'.")
                if not e.tool_exists:
                    lines.append(f"    ERROR: Tool '{e.selected_tool}' not found in available tools.")
                if e.subsequent_caller_failed:
                    lines.append(f"    NOTE: Caller subsequently failed for this selection.")
        else:
            lines.append("  No tool selections made.")

        lines.append("\n--- CALLER EVENTS ---")
        if self.caller_events:
            for e in self.caller_events:
                status = "SUCCESS" if e.execution_success else "FAILURE"
                lines.append(
                    f"  Step {e.step_index}: Called '{e.tool_name}' with args "
                    f"{json.dumps(e.arguments)} → {status}."
                )
                if e.error_message:
                    lines.append(f"    Error: {e.error_message}")
        else:
            lines.append("  No tool calls made.")

        lines.append("\n--- SYNTHESIZER EVENTS ---")
        if self.synthesizer_events:
            for e in self.synthesizer_events:
                lines.append(
                    f"  Step {e.step_index}: Synthesized answer. "
                    f"Referenced {e.tool_results_referenced}/{e.total_tool_results} tool results. "
                    f"Grounded: {e.grounded}."
                )
        else:
            lines.append("  No synthesizer events recorded.")

        return "\n".join(lines)


def extract_diagnostics(episode: EpisodeRecord) -> DiagnosticEvents:
    """
    Walk the trajectory and extract structured diagnostic events.
    These are passed to the Blamer LLM alongside the full episode.
    """
    diag = DiagnosticEvents(episode_id=episode.task_id, reward=episode.reward)

    tool_call_count = 0
    total_tool_calls = sum(1 for s in episode.trajectory if s.module == "caller")
    successful_tool_calls = sum(1 for s in episode.trajectory if s.module == "caller" and s.success)

    selector_indices: dict[int, SelectorEvent] = {}

    for idx, step in enumerate(episode.trajectory):
        if step.module == "planner":
            issues = _detect_planner_issues(step)
            diag.planner_events.append(
                PlannerEvent(step_index=idx, plan_text=step.action, issues=issues)
            )

        elif step.module == "selector":
            tool_name = _extract_selected_tool(step.action)
            available_tools = _extract_available_tools(step.state)
            tool_exists = (not tool_name) or (tool_name in available_tools) or (not available_tools)
            ev = SelectorEvent(
                step_index=idx,
                selected_tool=tool_name,
                available_tools=available_tools,
                tool_exists=tool_exists,
            )
            diag.selector_events.append(ev)
            selector_indices[idx] = ev

        elif step.module == "caller":
            tool_name, arguments = _extract_caller_info(step.action)
            error_msg = "" if step.success else _extract_error(step.observation)
            ev = CallerEvent(
                step_index=idx,
                tool_name=tool_name,
                arguments=arguments,
                execution_success=step.success,
                error_message=error_msg,
            )
            diag.caller_events.append(ev)
            # Back-propagate: mark the preceding selector event if this call failed
            if not step.success and selector_indices:
                last_sel_idx = max(selector_indices.keys())
                if last_sel_idx < idx:
                    selector_indices[last_sel_idx].subsequent_caller_failed = True
            tool_call_count += 1

        elif step.module == "synthesizer":
            tool_results_referenced = _count_tool_references(step.action, total_tool_calls)
            grounded = _check_grounding(step.action, step.state)
            diag.synthesizer_events.append(
                SynthesizerEvent(
                    step_index=idx,
                    answer=step.action,
                    tool_results_referenced=tool_results_referenced,
                    total_tool_results=total_tool_calls,
                    grounded=grounded,
                )
            )

    return diag


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_planner_issues(step: AgentStep) -> list[str]:
    issues: list[str] = []
    plan = step.action.lower()
    if len(plan) < 30:
        issues.append("Plan is very short; may lack subgoal detail.")
    if "error" in plan or "cannot" in plan or "unable" in plan:
        issues.append("Plan text contains error/failure language.")
    return issues


def _extract_selected_tool(action: str) -> str:
    import re, json as _json
    try:
        d = _json.loads(action)
        return d.get("tool_name", d.get("tool", ""))
    except Exception:
        pass
    m = re.search(r'"tool_name"\s*:\s*"([^"]+)"', action)
    if m:
        return m.group(1)
    return ""


def _extract_available_tools(state: str) -> list[str]:
    import re
    m = re.search(r"AVAILABLE TOOLS:\s*(.+)", state)
    if m:
        return [t.strip() for t in m.group(1).split(",")]
    return []


def _extract_caller_info(action: str) -> tuple[str, dict]:
    import json as _json
    try:
        d = _json.loads(action)
        name = d.get("tool_name", d.get("tool", ""))
        args = d.get("arguments", d.get("tool_arguments", {}))
        return name, args if isinstance(args, dict) else {}
    except Exception:
        return "", {}


def _extract_error(observation: str) -> str:
    if "ERROR:" in observation:
        return observation.split("ERROR:", 1)[1].strip()[:200]
    return observation[:200] if observation else ""


def _count_tool_references(answer: str, total_tool_calls: int) -> int:
    if total_tool_calls == 0:
        return 0
    # Heuristic: count numeric mentions or "result" / "tool" references
    import re
    refs = len(re.findall(r"\d+(?:\.\d+)?", answer))
    return min(refs, total_tool_calls)


def _check_grounding(answer: str, state: str) -> bool:
    """
    Heuristic: if the answer is very short and the state shows tool results
    were present, the answer may not be grounded.
    """
    if len(answer) < 10:
        return False
    if "CALL" in state and len(answer) < 20:
        return False
    return True


def _asdict(obj: Any) -> dict:
    import dataclasses
    return dataclasses.asdict(obj)
