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

# System prompt for BFCL multi-turn ReAct — mirrors execute_bfcl_episode's turn iteration
REACT_SYSTEM_PROMPT_BFCL_MULTI = """\
You are a function-calling agent with access to a stateful environment.

# Available tools
{tools_json}

# Instruction
For each user request, call the appropriate function(s) with the correct arguments.

At each step, your generation should have exactly the following format:
Thought:
<A single line of reasoning.>
Action:
{{"name": <function name>, "arguments": <arguments as JSON object>}}

Use values from previous function results where needed.
You should not use made-up or placeholder arguments.
"""

# System prompt matching official ChatReActAgent exactly (wiki + tools + instruction)
REACT_SYSTEM_PROMPT_TAU = """\
{wiki}

# Available tools
{tools_json}

# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform the decision making. Do not include extra lines.>
Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}

The Action will be parsed, so it must be valid JSON.

You should not use made-up or placeholder arguments.
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

    For τ-Bench environments (TauBenchEnvWrapper), delegates to
    _run_react_tau_bench_episode which mirrors ChatReActAgent.solve() exactly.
    For all other environments (BFCL, etc.), uses the standard BFCL ReAct path.
    """
    from benchmarks.tau_bench.runner import TauBenchEnvWrapper
    if isinstance(env, TauBenchEnvWrapper):
        logger.debug("Using τ-bench ReAct path (ChatReActAgent-compatible) for task %s", task.task_id)
        return _run_react_tau_bench_episode(task, env, llm, max_steps, env_wiki)

    if task.metadata.get("is_multi_turn", False):
        logger.debug("Using BFCL multi-turn ReAct path for task %s", task.task_id)
        return _run_react_bfcl_multi_episode(task, env, llm, max_steps)

    # --- BFCL single-turn / other benchmark path (unchanged) ---
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


def _run_react_tau_bench_episode(
    task: TaskInstance,
    env: "TauBenchEnvWrapper",
    llm: LLMClient,
    max_steps: int,
    wiki: str,
) -> EpisodeRecord:
    """
    τ-Bench-specific ReAct loop matching ChatReActAgent.solve() exactly.

    - System prompt = full WIKI + tools JSON + ReAct instruction
    - Action format: {"name": tool_name, "arguments": {...}}
    - All actions including respond go through env.execute_tool
    - Loop continues until env._env_done (done=True from tau-bench step)
    """
    from benchmarks.tau_bench.runner import TauBenchEnvWrapper

    tools_json = json.dumps(task.tools, indent=2)
    system = REACT_SYSTEM_PROMPT_TAU.format(wiki=wiki, tools_json=tools_json)
    initial_obs = env._initial_observation or ""

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": initial_obs},
    ]
    trajectory: list[AgentStep] = []
    tool_results: list[dict] = []

    for _ in range(max_steps):
        try:
            raw = llm.call_agent(messages)
        except Exception as exc:
            logger.warning("ReAct τ-bench LLM call failed: %s", exc)
            break

        action_name, action_args = _parse_react_json_action(raw)
        tool_call = ToolCall(name=action_name, arguments=action_args)

        try:
            obs, success = env.execute_tool(task, tool_call)
        except Exception as exc:
            obs = f"Tool execution error: {exc}"
            success = False

        step = AgentStep(
            state=messages[-1]["content"],
            action=raw,
            observation=obs,
            module="caller",
            success=success,
        )
        trajectory.append(step)
        tool_results.append({"tool": action_name, "result": obs})

        messages.append({"role": "assistant", "content": raw})
        if action_name == "respond":
            obs_for_user = obs
        else:
            obs_for_user = f"API output: {obs}"
        messages.append({"role": "user", "content": obs_for_user})

        if isinstance(env, TauBenchEnvWrapper) and env._env_done:
            break

    reward = env.evaluate(task, "", trajectory)
    return EpisodeRecord(
        task=task.task_description[:500],
        task_id=task.task_id,
        trajectory=trajectory,
        prediction="",
        reward=reward,
        metadata={**task.metadata, "baseline": "react"},
    )


def _run_react_bfcl_multi_episode(
    task: TaskInstance,
    env: BenchmarkEnv,
    llm: LLMClient,
    max_steps: int,
    max_calls_per_turn: int = 6,
) -> EpisodeRecord:
    """
    BFCL multi-turn ReAct loop.

    Mirrors execute_bfcl_episode's turn iteration: for each conversation turn
    (task.metadata["question"]), show the user message to the agent and let it
    make up to max_calls_per_turn function calls before advancing to the next turn.
    All calls accumulate in env._episode_calls for final evaluation.
    """
    question = task.metadata.get("question", [[]])
    tools_json = json.dumps(task.tools, indent=2)
    system = REACT_SYSTEM_PROMPT_BFCL_MULTI.format(tools_json=tools_json)

    messages: list[dict] = [{"role": "system", "content": system}]
    trajectory: list[AgentStep] = []

    for turn_idx, turn_messages in enumerate(question):
        if not turn_messages:
            continue

        if isinstance(turn_messages, list) and turn_messages:
            last_msg = turn_messages[-1]
            user_msg = (
                last_msg.get("content", "") if isinstance(last_msg, dict) else str(last_msg)
            )
        else:
            user_msg = str(turn_messages)

        messages.append({"role": "user", "content": user_msg})
        logger.debug("BFCL multi ReAct turn=%d: %s", turn_idx + 1, user_msg[:120])

        turn_calls: dict[str, int] = {}

        for _ in range(max_calls_per_turn):
            try:
                raw = llm.call_agent(messages)
            except Exception as exc:
                logger.warning("BFCL multi ReAct LLM call failed: %s", exc)
                break

            action_name, action_args = _parse_react_json_action(raw)

            if not action_name or action_name in ("respond", "done", "finish"):
                messages.append({"role": "assistant", "content": raw})
                break

            tool_call = ToolCall(name=action_name, arguments=action_args)
            try:
                obs, success = env.execute_tool(task, tool_call)
            except Exception as exc:
                obs = f"Error: {exc}"
                success = False

            step = AgentStep(
                state=user_msg,
                action=raw,
                observation=obs,
                module="caller",
                success=success,
            )
            trajectory.append(step)

            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Result: {obs}"})

            # Repeated-call guard
            call_key = f"{action_name}:{json.dumps(action_args, sort_keys=True)}"
            turn_calls[call_key] = turn_calls.get(call_key, 0) + 1
            if turn_calls[call_key] >= 2:
                break

    reward = env.evaluate(task, "", trajectory)
    return EpisodeRecord(
        task=task.task_description[:500],
        task_id=task.task_id,
        trajectory=trajectory,
        prediction="",
        reward=reward,
        metadata={**task.metadata, "baseline": "react"},
    )


def _parse_react_json_action(raw: str) -> tuple[str, dict]:
    """Parse Action: {"name": ..., "arguments": ...} format used by ChatReActAgent."""
    m = re.search(r"Action:\s*(\{.*\})", raw, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(1))
            return d.get("name", "respond"), d.get("arguments", {})
        except json.JSONDecodeError:
            pass
    return "respond", {"content": raw}


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

    # Match "Action: func_name(...)" on a single line (most reliable)
    a_match = re.search(r"Action:\s*(\w+)\s*\(([^)]*)\)", raw)
    if not a_match:
        # Fallback: allow multi-line args (JSON objects spanning lines)
        a_match = re.search(r"Action:\s*(\w+)\s*\((\{.*?\})\)", raw, re.DOTALL)
    if a_match:
        action_name = a_match.group(1).strip()
        args_str = a_match.group(2).strip()
        try:
            action_args = json.loads(args_str)
        except json.JSONDecodeError:
            # Try wrapping bare key=value pairs in braces
            try:
                action_args = json.loads(f"{{{args_str}}}")
            except json.JSONDecodeError:
                # Try replacing Python-style single quotes with double quotes
                try:
                    action_args = json.loads(args_str.replace("'", '"'))
                except json.JSONDecodeError:
                    action_args = {"raw": args_str}
    else:
        # Fallback: look for any JSON object in the output
        j_match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if j_match:
            try:
                action_args = json.loads(j_match.group())
            except json.JSONDecodeError:
                pass

    return thought, action_name, action_args
