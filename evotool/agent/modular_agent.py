"""
Modular agent: Planner → Selector → Caller → Synthesizer pipeline.

The agent decomposes a task using the Planner, then iteratively
Selects tools and Calls them until termination, and finally
Synthesizes a grounded response.

Both τ-Bench (conversation loop) and BFCL (single/multi-turn function-calling)
share this pipeline; only the tool environment differs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from evotool.agent.modules import PolicySpec
from evotool.llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class AgentStep:
    """One step in the trajectory: state → action → observation."""
    state: str          # textual summary of history at this point
    action: str         # agent output (JSON tool call or final text)
    observation: str    # tool result or "" for final answer steps
    module: str         # "planner" | "selector" | "caller" | "synthesizer"
    success: bool = True  # whether the tool call succeeded


class ModularAgent:
    """
    Planner / Selector / Caller / Synthesizer modular agent.

    Each module receives the current state and its module-specific prompt,
    and the LLM produces the module's output.
    """

    def __init__(self, policy: PolicySpec, llm: LLMClient, max_steps: int = 30) -> None:
        self.policy = policy
        self.llm = llm
        self.max_steps = max_steps

    # ------------------------------------------------------------------
    # Module invocations
    # ------------------------------------------------------------------

    def plan(self, task: str, tools: list[dict]) -> tuple[str, AgentStep]:
        """
        Planner: given the task and available tools, produce a subgoal plan.
        Returns (plan_text, step).
        """
        tool_names = ", ".join(t.get("name", t.get("function", {}).get("name", "?")) for t in tools)
        state = f"TASK: {task}\nAVAILABLE TOOLS: {tool_names}"
        messages = [
            {"role": "system", "content": self.policy.planner},
            {
                "role": "user",
                "content": (
                    f"Task: {task}\n\n"
                    f"Available tools:\n{json.dumps(tools, indent=2)}\n\n"
                    "Decompose this task into a sequential list of clear, executable subgoals."
                ),
            },
        ]
        plan = self.llm.call_agent(messages)
        step = AgentStep(state=state, action=plan, observation="", module="planner")
        return plan, step

    def select(
        self,
        task: str,
        plan: str,
        history: str,
        tools: list[dict],
    ) -> tuple[str, AgentStep]:
        """
        Selector: given current state and plan, choose the next tool to call.
        Returns (selection_text, step).
        """
        state = f"TASK: {task}\nPLAN: {plan}\nHISTORY:\n{history}"
        tool_names_json = json.dumps(
            [{"name": t.get("name", t.get("function", {}).get("name", "?")),
              "description": t.get("description", t.get("function", {}).get("description", ""))}
             for t in tools],
            indent=2,
        )
        messages = [
            {"role": "system", "content": self.policy.selector},
            {
                "role": "user",
                "content": (
                    # Label fields to match the selector system prompt language
                    # ("Given the current subgoal and the list of available tools")
                    # and put the output instruction first so it is not buried.
                    'Respond with JSON only — {"tool_name": "<name>", "rationale": "<one line>"}\n'
                    'Or {"tool_name": "DONE", "rationale": "<why>"} if the task is complete.\n\n'
                    f"Current subgoal (next uncompleted step from the plan):\n{plan}\n\n"
                    f"Available tools:\n{tool_names_json}\n\n"
                    f"Execution history:\n{history or 'None'}\n\n"
                    f"Task context: {task}"
                ),
            },
        ]
        selection = self.llm.call_agent(messages)
        step = AgentStep(state=state, action=selection, observation="", module="selector")
        return selection, step

    def call_tool(
        self,
        task: str,
        plan: str,
        history: str,
        tool_name: str,
        tool_spec: dict,
    ) -> tuple[ToolCall, AgentStep]:
        """
        Caller: construct valid arguments for the selected tool.
        Returns (ToolCall, step).
        """
        state = (
            f"TASK: {task}\nPLAN: {plan}\n"
            f"SELECTED TOOL: {tool_name}\nHISTORY:\n{history}"
        )
        messages = [
            {"role": "system", "content": self.policy.caller},
            {
                "role": "user",
                "content": (
                    # Put output format first; use field names matching the caller
                    # system prompt ("selected tool and its documentation").
                    f'Respond with JSON only — {{"tool_name": "{tool_name}", "arguments": {{<key: value>}}}}\n\n'
                    f"Selected tool: {tool_name}\n"
                    f"Tool documentation:\n{json.dumps(tool_spec, indent=2)}\n\n"
                    f"Execution history (use values from previous tool results as arguments):\n{history or 'None'}\n\n"
                    f"Task context: {task}"
                ),
            },
        ]
        raw = self.llm.call_agent(messages)
        tool_call = _parse_tool_call(raw, tool_name)
        step = AgentStep(state=state, action=raw, observation="", module="caller")
        return tool_call, step

    def synthesize(
        self,
        task: str,
        history: str,
        tool_results: list[dict],
    ) -> tuple[str, AgentStep]:
        """
        Synthesizer: produce the final grounded response.
        Returns (answer, step).
        """
        state = f"TASK: {task}\nHISTORY:\n{history}"
        results_str = json.dumps(tool_results, indent=2) if tool_results else "No tool calls made."
        messages = [
            {"role": "system", "content": self.policy.synthesizer},
            {
                "role": "user",
                "content": (
                    f"Original task: {task}\n\n"
                    f"Tool execution results:\n{results_str}\n\n"
                    f"Full execution history:\n{history}\n\n"
                    "Synthesize this information to provide a grounded, accurate final answer."
                ),
            },
        ]
        answer = self.llm.call_agent(messages)
        step = AgentStep(state=state, action=answer, observation="", module="synthesizer")
        return answer, step


# ---------------------------------------------------------------------------
# Helper: parse tool call from Caller output
# ---------------------------------------------------------------------------

_RESPOND_NAMES = frozenset({"respond", "answer", "done"})


def _parse_tool_call(raw: str, fallback_name: str) -> ToolCall:
    """
    Extract tool name and arguments from the Caller's output.

    Tool-name resolution:
    - If the Caller says "respond"/"done", honour it (signals task complete).
    - If the Caller names a *different* tool than the Selector chose, ignore it
      and use the Selector's choice (fallback_name) — prevents the Caller from
      silently re-selecting a tool the Selector didn't pick.
    - Arguments are always taken from the Caller's output.
    """
    import re

    def _resolve(caller_name: str, args: dict) -> ToolCall:
        if caller_name.lower() in _RESPOND_NAMES:
            return ToolCall(name=caller_name, arguments=args)
        # Use Selector's tool; ignore arbitrary Caller overrides
        clean = {k: v for k, v in args.items() if k != "tool_name"}
        return ToolCall(name=fallback_name, arguments=clean)

    text = raw.strip()
    # Try direct JSON parse
    try:
        parsed = json.loads(text)
        caller_name = parsed.get("tool_name", fallback_name)
        args = parsed.get("arguments", parsed.get("tool_arguments", parsed))
        return _resolve(caller_name, args if isinstance(args, dict) else {})
    except json.JSONDecodeError:
        pass

    # Try extracting first JSON object from text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            caller_name = parsed.get("tool_name", fallback_name)
            args = parsed.get("arguments", parsed.get("tool_arguments", parsed))
            return _resolve(caller_name, args if isinstance(args, dict) else {})
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse tool call from Caller output, using empty args")
    return ToolCall(name=fallback_name, arguments={})


def _parse_selector_output(raw: str) -> tuple[str, bool]:
    """
    Parse Selector output. Returns (tool_name, is_done).
    'is_done' is True when the selector says the task is complete.
    """
    import re

    text = raw.strip().upper()
    if "DONE" in text and len(text) < 50:
        return "", True

    # Try JSON
    try:
        parsed = json.loads(raw.strip())
        name = parsed.get("tool_name", "")
        if name.upper() == "DONE" or name == "":
            return "", True
        return name, False
    except json.JSONDecodeError:
        pass

    # Fallback: extract first JSON block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            name = parsed.get("tool_name", "")
            if name.upper() == "DONE" or name == "":
                return "", True
            return name, False
        except json.JSONDecodeError:
            pass

    # Last resort: look for a quoted tool name in any format
    match = re.search(r'"tool_name"\s*:\s*"([^"]+)"', raw)
    if match:
        name = match.group(1)
        if name.upper() == "DONE":
            return "", True
        return name, False

    # Try to find any tool name mentioned after common prefixes
    # e.g. "I will call get_order_details" or "Tool: find_user_id_by_name_zip"
    match = re.search(
        r'(?:call|use|tool|select|invoke|run|execute)[:\s]+["\']?(\w+)["\']?',
        raw, re.IGNORECASE,
    )
    if match:
        name = match.group(1)
        if name.upper() not in ("DONE", "NONE", "NULL", "THE", "A", "AN"):
            return name, False

    # If output explicitly says the task is complete in plain English
    done_phrases = ("task is complete", "no further", "nothing more", "all done",
                    "completed", "finished", "no tool", "no action")
    if any(p in raw.lower() for p in done_phrases):
        return "", True

    # Cannot parse and no done signal — default to done to avoid infinite loops
    logger.warning("Selector output unparseable, defaulting to done. Raw: %s", raw[:200])
    return "", True
