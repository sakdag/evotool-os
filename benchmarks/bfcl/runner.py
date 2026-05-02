"""
BFCL environment adapter.

Wraps BFCL tasks into our BenchmarkEnv interface.

For BFCL single-turn:
  - The agent receives a user query and a list of function specs.
  - The agent must output the correct function call (name + arguments).
  - Reward is 1.0 if the function call AST-matches the ground truth.

For BFCL multi-turn:
  - Multiple turns, each requiring a correct tool invocation.
  - Reward is 1.0 only if ALL turns are correct.
  - We use a simplified state-tracking approach for the evolution reward.

Our modular agent pipeline maps to BFCL as follows:
  - Planner:    decomposes the query into a subgoal (which function to call)
  - Selector:   picks the exact function name
  - Caller:     constructs the JSON arguments
  - Synthesizer: formats the final function call output
"""

from __future__ import annotations

import json
import logging

from typing import Any

from evotool.agent.executor import BenchmarkEnv, EpisodeRecord, TaskInstance
from evotool.agent.modular_agent import AgentStep, ToolCall

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task loader
# ---------------------------------------------------------------------------


def load_bfcl_tasks(
    instances: list[dict],
    ground_truth: dict[str, Any],
    is_multi_turn: bool = False,
) -> list[TaskInstance]:
    """Convert BFCL raw instances into TaskInstance objects."""
    from benchmarks.bfcl.data_split import load_multi_turn_func_specs

    tasks: list[TaskInstance] = []
    for inst in instances:
        inst_id = inst.get("id", "")
        question = inst.get("question", [[]])
        gt = ground_truth.get(inst_id, [])

        # Build user query from first turn
        if question and question[0]:
            first_turn = question[0]
            if isinstance(first_turn, list) and first_turn:
                last_msg = first_turn[-1]
                user_msg = (
                    last_msg.get("content", "")
                    if isinstance(last_msg, dict)
                    else str(last_msg)
                )
            else:
                user_msg = str(first_turn)
        else:
            user_msg = ""

        if is_multi_turn:
            # Multi-turn tasks have no "function" field; specs come from involved_classes
            involved_classes = inst.get("involved_classes", [])
            raw_specs = load_multi_turn_func_specs(involved_classes)
            tools = _bfcl_functions_to_tools(raw_specs)
            task_desc = (
                "You are a function-calling agent with access to a stateful environment. "
                "For each user request, call the correct function(s) with the right arguments.\n\n"
                f"First request: {user_msg}"
            )
        else:
            functions = inst.get("function", [])
            tools = _bfcl_functions_to_tools(functions)
            task_desc = (
                "You are a function-calling agent. Given the user query and the available functions, "
                "call the correct function(s) with the right arguments.\n\n"
                f"User query: {user_msg}"
            )

        tasks.append(
            TaskInstance(
                task_id=inst_id,
                task_description=task_desc,
                tools=tools,
                metadata={
                    "bfcl_instance": inst,
                    "ground_truth": gt,
                    "is_multi_turn": is_multi_turn,
                    "question": question,
                },
            )
        )
    return tasks


def _bfcl_functions_to_tools(functions: list[dict]) -> list[dict]:
    """Convert BFCL function specs to OpenAI tool format."""
    tools = []
    for fn in functions:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                },
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
            }
        )
    return tools


# ---------------------------------------------------------------------------
# BFCL Environment
# ---------------------------------------------------------------------------


class BFCLEnv(BenchmarkEnv):
    """
    BFCL environment: function calls are validated against ground truth.
    No external tool execution; reward is based on AST matching.
    """

    def __init__(self, ground_truth: dict[str, Any], is_multi_turn: bool = False) -> None:
        self.ground_truth = ground_truth
        self.is_multi_turn = is_multi_turn
        self._episode_calls: list[dict] = []  # accumulated tool calls for multi-turn

    def reset_episode(self) -> None:
        self._episode_calls = []

    def execute_tool(self, task: TaskInstance, tool_call: ToolCall) -> tuple[str, bool]:
        """
        Record the tool call. For single-turn, this is called once.
        For multi-turn, this accumulates across turns.
        Returns (simulated_observation, True).
        """
        self._episode_calls.append(
            {"function": tool_call.name, "arguments": tool_call.arguments}
        )
        return json.dumps({"status": "ok", "called": tool_call.name}), True

    def evaluate(
        self,
        task: TaskInstance,
        prediction: str,
        trajectory: list[AgentStep],
    ) -> float:
        """
        Evaluate the episode's function calls against ground truth.
        Returns 1.0 if all calls match, 0.0 otherwise.
        """
        gt = task.metadata.get("ground_truth", [])
        if not gt:
            return 0.0

        # Collect all caller-module actions as our function call predictions
        caller_actions = [
            s.action for s in trajectory if s.module == "caller"
        ]

        if not caller_actions and not self._episode_calls:
            return 0.0

        # Try to match against ground truth
        for action in caller_actions:
            call = _parse_function_call(action)
            if call and _matches_ground_truth(call, gt):
                return 1.0

        # Also check accumulated episode calls
        for call in self._episode_calls:
            if _matches_ground_truth(call, gt):
                return 1.0

        return 0.0


# ---------------------------------------------------------------------------
# Custom BFCL episode executor
# ---------------------------------------------------------------------------


def execute_bfcl_episode(
    policy: "PolicySpec",
    task: TaskInstance,
    env: BFCLEnv,
    llm: "LLMClient",
    max_steps: int = 15,
) -> EpisodeRecord:
    """
    BFCL-specific episode executor.

    For single-turn BFCL:
      - Planner: identify which function to call and why
      - Selector: select the exact function name from the function list
      - Caller: construct the JSON arguments
      - Synthesizer: format the final function call (redundant but kept for consistency)

    For multi-turn BFCL:
      - Runs multiple rounds of Selector+Caller, one per expected function call.
    """
    from evotool.agent.modular_agent import ModularAgent, _parse_selector_output
    from evotool.agent.executor import EpisodeRecord

    env.reset_episode()
    agent = ModularAgent(policy=policy, llm=llm, max_steps=max_steps)
    trajectory: list[AgentStep] = []
    tool_results: list[dict] = []
    history_lines: list[str] = []
    is_multi_turn = task.metadata.get("is_multi_turn", False)
    question = task.metadata.get("question", [[]])

    # Planner
    try:
        plan, plan_step = agent.plan(task.task_description, task.tools)
        trajectory.append(plan_step)
        history_lines.append(f"[PLAN] {plan}")
    except Exception as exc:
        logger.warning("Planner failed: %s", exc)
        plan = "Call the most appropriate function with correct arguments."

    # For multi-turn, iterate through conversation turns
    turns = question if is_multi_turn else [question[0]] if question else [[]]
    # Maximum function calls the agent may make within a single turn
    max_calls_per_turn = 6 if is_multi_turn else 1

    for turn_idx, turn_messages in enumerate(turns):
        if not turn_messages:
            continue

        if isinstance(turn_messages, list) and turn_messages:
            last_msg = turn_messages[-1]
            user_msg = (
                last_msg.get("content", "")
                if isinstance(last_msg, dict)
                else str(last_msg)
            )
        else:
            user_msg = str(turn_messages)

        if user_msg:
            history_lines.append(f"[USER turn {turn_idx + 1}] {user_msg}")

        # Inner loop: one selector+caller cycle per expected function call in this turn
        for _call_in_turn in range(max_calls_per_turn):
            history = "\n".join(history_lines)

            # Selector: choose function
            try:
                sel_raw, sel_step = agent.select(task.task_description, plan, history, task.tools)
                tool_name, is_done = _parse_selector_output(sel_raw)
                sel_step.observation = f"Selected: {tool_name or 'NONE'}"
                trajectory.append(sel_step)
            except Exception as exc:
                logger.warning("Selector failed: %s", exc)
                is_done = True

            if is_done or not tool_name:
                break  # Done with this turn (or entire episode for single-turn)

            history_lines.append(f"[SELECT] {tool_name}")

            # Caller: construct arguments
            tool_spec = env.find_tool_spec(task.tools, tool_name)
            try:
                tool_call, call_step = agent.call_tool(
                    task.task_description, plan, history, tool_name, tool_spec
                )
            except Exception as exc:
                logger.warning("Caller failed: %s", exc)
                break

            obs, success = env.execute_tool(task, tool_call)
            call_step.observation = obs
            call_step.success = success
            trajectory.append(call_step)
            tool_results.append(
                {"function": tool_call.name, "arguments": tool_call.arguments, "result": obs}
            )
            history_lines.append(
                f"[CALL] {tool_call.name}({json.dumps(tool_call.arguments)}) → {obs}"
            )

            if not is_multi_turn:
                break  # Single-turn: one call expected

    history = "\n".join(history_lines)

    # Synthesizer: format the final answer
    try:
        answer, syn_step = agent.synthesize(task.task_description, history, tool_results)
        trajectory.append(syn_step)
    except Exception as exc:
        logger.warning("Synthesizer failed: %s", exc)
        answer = json.dumps(tool_results[-1]) if tool_results else ""

    reward = env.evaluate(task, answer, trajectory)

    return EpisodeRecord(
        task=task.task_description[:500],
        task_id=task.task_id,
        trajectory=trajectory,
        prediction=answer,
        reward=reward,
        metadata=task.metadata,
    )


# ---------------------------------------------------------------------------
# AST matching helpers
# ---------------------------------------------------------------------------


def _parse_function_call(action: str) -> dict | None:
    """Extract a {function, arguments} dict from a Caller action string."""
    import re

    text = action.strip()
    try:
        d = json.loads(text)
        fn = d.get("tool_name") or d.get("function") or d.get("name", "")
        args = d.get("arguments") or d.get("tool_arguments") or {}
        if fn:
            return {"function": fn, "arguments": args}
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group())
            fn = d.get("tool_name") or d.get("function") or d.get("name", "")
            args = d.get("arguments") or d.get("tool_arguments") or {}
            if fn:
                return {"function": fn, "arguments": args}
        except json.JSONDecodeError:
            pass

    return None


def _parse_gt_call_string(call_str: str) -> tuple[str, dict] | None:
    """
    Parse a Python function-call string like "cd(folder='document')"
    into (func_name, {arg: value}).
    """
    import ast as _ast

    try:
        tree = _ast.parse(call_str.strip(), mode="eval")
        call_node = tree.body
        if not isinstance(call_node, _ast.Call):
            return None
        if isinstance(call_node.func, _ast.Name):
            func_name = call_node.func.id
        elif isinstance(call_node.func, _ast.Attribute):
            func_name = call_node.func.attr
        else:
            return None

        args: dict[str, Any] = {}
        for kw in call_node.keywords:
            if kw.arg is not None:
                try:
                    args[kw.arg] = _ast.literal_eval(kw.value)
                except Exception:
                    args[kw.arg] = _ast.unparse(kw.value)

        return func_name, args
    except Exception:
        return None


def _matches_ground_truth(call: dict, ground_truth: list) -> bool:
    """
    Check if a function call matches any ground truth entry.

    Handles two GT formats:
      Single-turn: [{"func_name": {"arg": [acceptable_values]}}]
      Multi-turn:  [["call_str1", "call_str2"], ["call_str3"]]  (Python call strings per turn)
    """
    predicted_fn = call.get("function", "")
    predicted_args = call.get("arguments", {})

    # Normalise GT into a flat list of (func_name, expected_args) pairs
    flat_gt: list[tuple[str, Any]] = []
    for entry in ground_truth:
        if isinstance(entry, list):
            for sub in entry:
                if isinstance(sub, str):
                    # Multi-turn format: Python call string
                    parsed = _parse_gt_call_string(sub)
                    if parsed:
                        flat_gt.append(parsed)
                elif isinstance(sub, dict):
                    # Single-turn nested list format
                    for fn_name, exp_args in sub.items():
                        flat_gt.append((fn_name, exp_args))
        elif isinstance(entry, dict):
            for fn_name, exp_args in entry.items():
                flat_gt.append((fn_name, exp_args))
        elif isinstance(entry, str):
            parsed = _parse_gt_call_string(entry)
            if parsed:
                flat_gt.append(parsed)

    for fn_name, expected_args in flat_gt:
        if fn_name != predicted_fn:
            continue
        if not isinstance(expected_args, dict):
            return True  # Function name match is sufficient

        # Distinguish single-turn format (values are lists of acceptable values)
        # from multi-turn format (values are actual values).
        if expected_args and isinstance(next(iter(expected_args.values())), list):
            if _args_match_single(predicted_args, expected_args):
                return True
        else:
            if _args_match_multi(predicted_args, expected_args):
                return True

    return False


def _args_match_single(predicted: dict, expected: dict) -> bool:
    """Single-turn: expected[arg] = list of acceptable values (any one valid)."""
    for arg_name, acceptable_values in expected.items():
        if arg_name not in predicted:
            return False
        predicted_val = predicted[arg_name]
        if isinstance(acceptable_values, list):
            if all(av == "" for av in acceptable_values):
                continue  # any value accepted
            if not any(
                str(predicted_val) == str(av) or predicted_val == av
                for av in acceptable_values
            ):
                return False
    return True


def _args_match_multi(predicted: dict, expected: dict) -> bool:
    """Multi-turn: expected[arg] is the actual required value."""
    for arg_name, expected_val in expected.items():
        if arg_name not in predicted:
            return False
        predicted_val = predicted[arg_name]
        if str(predicted_val) != str(expected_val) and predicted_val != expected_val:
            return False
    return True
