"""
Episode executor: run a PolicySpec on a single task instance and collect
the full EpisodeRecord.

The executor is benchmark-agnostic: concrete environments implement
BenchmarkEnv and are passed in at call time.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from evotool.agent.modular_agent import (
    AgentStep,
    ModularAgent,
    ToolCall,
    _parse_selector_output,
)
from evotool.agent.modules import PolicySpec
from evotool.llm.client import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class TaskInstance:
    """A single task to be solved by the agent."""
    task_id: str
    task_description: str
    tools: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class EpisodeRecord:
    """
    e = (x, τ, ŷ, R(x, ŷ))  — the unit of training signal for EVOTOOL.
    """
    task: str
    task_id: str
    trajectory: list[AgentStep]
    prediction: str
    reward: float           # 0.0 or 1.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task": self.task,
            "prediction": self.prediction,
            "reward": self.reward,
            "trajectory": [
                {
                    "state": s.state,
                    "action": s.action,
                    "observation": s.observation,
                    "module": s.module,
                    "success": s.success,
                }
                for s in self.trajectory
            ],
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Abstract benchmark environment interface
# ---------------------------------------------------------------------------


class BenchmarkEnv(ABC):
    """
    Thin wrapper around a benchmark environment.
    Implementations live in benchmarks/tau_bench/ and benchmarks/bfcl/.
    """

    @abstractmethod
    def execute_tool(self, task_instance: TaskInstance, tool_call: ToolCall) -> tuple[str, bool]:
        """
        Execute a tool call in the environment.
        Returns (observation_str, success_bool).
        """

    @abstractmethod
    def evaluate(self, task_instance: TaskInstance, prediction: str, trajectory: list[AgentStep]) -> float:
        """Return reward in [0, 1]."""

    def find_tool_spec(self, tools: list[dict], tool_name: str) -> dict:
        """Look up a tool spec by name from the task's tool list."""
        for t in tools:
            name = t.get("name") or t.get("function", {}).get("name", "")
            if name == tool_name:
                return t
        return {}


# ---------------------------------------------------------------------------
# Core executor function
# ---------------------------------------------------------------------------


def execute_policy(
    policy: PolicySpec,
    task: TaskInstance,
    env: BenchmarkEnv,
    llm: LLMClient,
    max_steps: int = 30,
) -> EpisodeRecord:
    """
    Run πΘ,W on task instance x in environment E.
    Collects (s_t, a_t, o_t) tuples into a full trajectory τ.

    Returns an EpisodeRecord e = (x, τ, ŷ, R(x, ŷ)).
    """
    agent = ModularAgent(policy=policy, llm=llm, max_steps=max_steps)
    trajectory: list[AgentStep] = []
    tool_results: list[dict] = []
    history_lines: list[str] = []

    # ----------------------------------------------------------------
    # Phase 1: Planner
    # ----------------------------------------------------------------
    try:
        plan, plan_step = agent.plan(task.task_description, task.tools)
        trajectory.append(plan_step)
        history_lines.append(f"[PLAN]\n{plan}")
    except Exception as exc:
        logger.warning("Planner failed for task %s: %s", task.task_id, exc)
        plan = "Execute the task step by step using the available tools."
        trajectory.append(
            AgentStep(state=task.task_description, action=plan, observation="", module="planner", success=False)
        )

    # ----------------------------------------------------------------
    # Phase 2: Selector → Caller loop
    # ----------------------------------------------------------------
    steps_taken = 0
    while steps_taken < max_steps:
        history = "\n".join(history_lines)

        # Selector
        try:
            sel_raw, sel_step = agent.select(task.task_description, plan, history, task.tools)
            tool_name, is_done = _parse_selector_output(sel_raw)
            sel_step.observation = f"Selected: {tool_name or 'DONE'}"
            trajectory.append(sel_step)
        except Exception as exc:
            logger.warning("Selector failed for task %s: %s", task.task_id, exc)
            break

        if is_done or not tool_name:
            break

        history_lines.append(f"[SELECT] {tool_name}")

        # Caller
        tool_spec = env.find_tool_spec(task.tools, tool_name)
        try:
            tool_call, call_step = agent.call_tool(
                task.task_description, plan, history, tool_name, tool_spec
            )
        except Exception as exc:
            logger.warning("Caller failed for task %s: %s", task.task_id, exc)
            call_step = AgentStep(
                state=history, action=str(exc), observation="", module="caller", success=False
            )
            trajectory.append(call_step)
            break

        # Execute the tool call in the environment
        try:
            obs, success = env.execute_tool(task, tool_call)
        except Exception as exc:
            obs = f"ERROR: {exc}"
            success = False

        call_step.observation = obs
        call_step.success = success
        trajectory.append(call_step)

        tool_results.append(
            {"tool": tool_call.name, "arguments": tool_call.arguments, "result": obs, "success": success}
        )
        history_lines.append(f"[CALL] {tool_call.name}({tool_call.arguments}) → {obs}")
        steps_taken += 1

    # ----------------------------------------------------------------
    # Phase 3: Synthesizer
    # ----------------------------------------------------------------
    history = "\n".join(history_lines)
    try:
        answer, syn_step = agent.synthesize(task.task_description, history, tool_results)
        trajectory.append(syn_step)
    except Exception as exc:
        logger.warning("Synthesizer failed for task %s: %s", task.task_id, exc)
        answer = history_lines[-1] if history_lines else ""
        trajectory.append(
            AgentStep(state=history, action=answer, observation="", module="synthesizer", success=False)
        )

    # ----------------------------------------------------------------
    # Evaluate
    # ----------------------------------------------------------------
    reward = env.evaluate(task, answer, trajectory)

    return EpisodeRecord(
        task=task.task_description,
        task_id=task.task_id,
        trajectory=trajectory,
        prediction=answer,
        reward=reward,
        metadata=task.metadata,
    )
