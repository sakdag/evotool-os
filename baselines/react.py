"""
ReAct baseline (Yao et al., 2022).

Implements the standard ReAct prompting framework that interleaves reasoning
traces with tool actions. This is used as a sanity-check baseline to verify
our environment setup produces numbers consistent with Table 1 (Qwen3-8B row):
  τ-Bench: Retail=33.1, Airline=14.4
  BFCL:    Single=70.6, Multi=33.3

ReAct does NOT use the modular Planner/Selector/Caller/Synthesizer pipeline;
instead, it uses a single unified prompt with the ReAct thought-action-observe loop.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from evotool.agent.executor import BenchmarkEnv, EpisodeRecord, TaskInstance
from evotool.agent.modular_agent import AgentStep, ToolCall
from evotool.llm.client import LLMClient

logger = logging.getLogger(__name__)

REACT_SYSTEM_PROMPT = """\
You are a ReAct agent that interleaves reasoning and tool use.

At each step, follow this exact format:
Thought: <your reasoning about what to do next>
Action: <tool_name>(<json_arguments>)

When you have enough information to answer, use:
Thought: I now have the information to answer.
Action: respond({"content": "<your final answer>"})

Rules:
- Always write Thought before Action.
- Action must be a valid function call with JSON arguments.
- Use tool outputs to ground your reasoning.
- Do not hallucinate tool outputs.
"""


@dataclass
class ReActStep:
    thought: str
    action_name: str
    action_args: dict
    observation: str = ""
    success: bool = True


def run_react_episode(
    task: TaskInstance,
    env: BenchmarkEnv,
    llm: LLMClient,
    max_steps: int = 30,
    env_wiki: str = "",
) -> EpisodeRecord:
    """
    Run a single ReAct episode.

    The agent receives the task + tools and iteratively produces
    Thought / Action / Observation triples until it calls respond().
    """
    trajectory: list[AgentStep] = []
    react_history: list[str] = []
    tool_results: list[dict] = []

    system = REACT_SYSTEM_PROMPT
    if env_wiki:
        system = f"Policy document:\n{env_wiki[:1000]}\n\n" + system

    # Build initial prompt
    tool_descs = _format_tools_for_react(task.tools)

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"Task: {task.task_description}\n\n"
                f"Available tools:\n{tool_descs}\n\n"
                "Begin. Write Thought then Action."
            ),
        },
    ]

    steps_taken = 0
    final_answer = ""

    while steps_taken < max_steps:
        try:
            raw = llm.call_agent(messages)
        except Exception as exc:
            logger.warning("ReAct LLM call failed: %s", exc)
            break

        thought, action_name, action_args = _parse_react_output(raw)
        react_history.append(f"Thought: {thought}\nAction: {action_name}({json.dumps(action_args)})")

        # Convert to AgentStep (module=selector to get blame attribution)
        step = AgentStep(
            state="\n".join(react_history[-10:]),
            action=raw,
            observation="",
            module="selector",  # ReAct uses a unified loop; map to selector for diagnostics
        )

        if action_name in ("respond", "final_answer", "answer"):
            final_answer = action_args.get("content", str(action_args))
            step.observation = "[FINAL ANSWER]"
            step.module = "synthesizer"
            trajectory.append(step)
            break

        # Execute tool
        tool_call = ToolCall(name=action_name, arguments=action_args)
        try:
            obs, success = env.execute_tool(task, tool_call)
        except Exception as exc:
            obs = f"Error: {exc}"
            success = False

        step.observation = obs
        step.success = success
        step.module = "caller"
        trajectory.append(step)

        tool_results.append({"tool": action_name, "result": obs})
        react_history.append(f"Observation: {obs[:300]}")

        # Add to message history
        messages.append({"role": "assistant", "content": raw})
        messages.append(
            {"role": "user", "content": f"Observation: {obs[:500]}\n\nContinue."}
        )
        steps_taken += 1

    if not final_answer and tool_results:
        final_answer = str(tool_results[-1])

    reward = env.evaluate(task, final_answer, trajectory)

    return EpisodeRecord(
        task=task.task_description[:500],
        task_id=task.task_id,
        trajectory=trajectory,
        prediction=final_answer,
        reward=reward,
        metadata={**task.metadata, "baseline": "react"},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_tools_for_react(tools: list[dict]) -> str:
    lines = []
    for t in tools:
        fn = t.get("function", t)
        name = fn.get("name", t.get("name", ""))
        desc = fn.get("description", t.get("description", ""))
        params = fn.get("parameters", {})
        props = params.get("properties", {})
        args_str = ", ".join(
            f"{k}: {v.get('type', 'any')}" for k, v in props.items()
        )
        lines.append(f"  {name}({args_str}) — {desc}")
    return "\n".join(lines) or "  (no tools)"


def _parse_react_output(raw: str) -> tuple[str, str, dict]:
    """Parse Thought / Action from ReAct output."""
    thought = ""
    action_name = "respond"
    action_args: dict = {}

    t_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", raw, re.DOTALL)
    if t_match:
        thought = t_match.group(1).strip()

    a_match = re.search(r"Action:\s*(\w+)\s*\((.+)\)\s*$", raw, re.DOTALL)
    if a_match:
        action_name = a_match.group(1).strip()
        args_str = a_match.group(2).strip()
        try:
            action_args = json.loads(args_str)
        except json.JSONDecodeError:
            # Try wrapping in braces
            try:
                action_args = json.loads(f"{{{args_str}}}")
            except json.JSONDecodeError:
                action_args = {"raw": args_str}
    else:
        # Fallback: look for any JSON in the output
        j_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if j_match:
            try:
                action_args = json.loads(j_match.group())
            except json.JSONDecodeError:
                pass

    return thought, action_name, action_args
