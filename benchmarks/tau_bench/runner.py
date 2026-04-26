"""
τ-Bench environment adapter.

Wraps the tau_bench Python API into our BenchmarkEnv interface.

Key design decision
-------------------
tau_bench's ``get_env()`` constructor immediately fires an LLM call through
``LLMUserSimulationEnv.__init__() → self.reset() → generate_next_message()``.
That initialisation call is **redundant**: the user simulator is properly
reinitialised for each task by ``env.reset(task_index=N)``.

We therefore apply a one-time monkey-patch that skips the eager ``self.reset()``
inside ``LLMUserSimulationEnv.__init__``.  This makes task loading free of LLM
calls, while every actual episode still receives a fully initialised user.

Task / tool metadata is loaded directly from tau_bench's importable modules
(no env object required), so ``load_tau_bench_tasks()`` works even before the
vLLM server is started.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from evotool.agent.executor import BenchmarkEnv, EpisodeRecord, TaskInstance
from evotool.agent.modular_agent import AgentStep, ToolCall

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# One-time patch: skip the eager LLM call in LLMUserSimulationEnv.__init__
# ---------------------------------------------------------------------------

def _patch_tau_bench_user_init() -> None:
    """
    Apply two patches to ``LLMUserSimulationEnv``:

    1. Skip the eager ``self.reset()`` in ``__init__`` (which fires an LLM
       request before the vLLM server is guaranteed to be up).  The simulator
       is properly re-initialised for each task by ``env.reset(task_index=N)``.

    2. Wrap ``generate_next_message`` to:
       - Keep only system message + last 8 conversation turns before calling
         the LLM, preventing context-window overflow on long episodes.
       - Pass ``max_tokens=512`` — user simulator responses are always short
         single-turn messages, so we don't need more output capacity.
    """
    try:
        import tau_bench.envs.user as tau_user
        from litellm import completion as _litellm_completion

        if getattr(tau_user, "_init_patched", False):
            return  # already patched

        # --- patch 1: suppress eager __init__ reset ---
        def _patched_init(self: Any, model: str, provider: str) -> None:
            self.model = model
            self.provider = provider
            self.messages: list = []
            self.total_cost = 0.0
            # Skip self.reset() — properly called by env.reset(task_index=...)

        # --- patch 2: trim history + add max_tokens ---
        _orig_generate = tau_user.LLMUserSimulationEnv.generate_next_message

        def _safe_generate(self: Any, messages: list) -> str:
            system = [m for m in messages if m.get("role") == "system"]
            non_sys = [m for m in messages if m.get("role") != "system"]
            trimmed = system + non_sys[-8:]  # keep last 8 turns
            # Call completion directly with max_tokens so we never overflow
            res = _litellm_completion(
                model=self.model,
                custom_llm_provider=self.provider,
                messages=trimmed,
                max_tokens=512,
            )
            message = res.choices[0].message
            self.messages.append(message.model_dump())
            try:
                self.total_cost = res._hidden_params.get("response_cost", 0.0)
            except Exception:
                pass
            return message.content

        tau_user.LLMUserSimulationEnv.__init__ = _patched_init  # type: ignore[method-assign]
        tau_user.LLMUserSimulationEnv.generate_next_message = _safe_generate  # type: ignore[method-assign]
        tau_user._init_patched = True
        logger.debug("tau_bench LLMUserSimulationEnv patched (eager init + context-window guard).")
    except (ImportError, AttributeError) as exc:
        logger.warning("Could not patch tau_bench user init: %s", exc)


# ---------------------------------------------------------------------------
# Direct metadata loaders (no env creation, no LLM call)
# ---------------------------------------------------------------------------

def _load_retail_metadata(task_split: str) -> tuple[list, list[dict], str]:
    """Return (tasks, tools_info, wiki) for the retail env."""
    from tau_bench.envs.retail.tools import ALL_TOOLS
    from tau_bench.envs.retail.wiki import WIKI

    if task_split == "train":
        from tau_bench.envs.retail.tasks_train import TASKS_TRAIN as tasks
    elif task_split == "dev":
        from tau_bench.envs.retail.tasks_dev import TASKS_DEV as tasks
    else:
        from tau_bench.envs.retail.tasks_test import TASKS_TEST as tasks

    tools_info = [t.get_info() for t in ALL_TOOLS]
    return list(tasks), tools_info, WIKI


def _load_airline_metadata(task_split: str) -> tuple[list, list[dict], str]:
    """Return (tasks, tools_info, wiki) for the airline env."""
    from tau_bench.envs.airline.tools import ALL_TOOLS
    from tau_bench.envs.airline.wiki import WIKI
    from tau_bench.envs.airline.tasks_test import TASKS  # airline only has one split

    tools_info = [t.get_info() for t in ALL_TOOLS]
    return list(TASKS), tools_info, WIKI


# ---------------------------------------------------------------------------
# Task loader  (no LLM calls)
# ---------------------------------------------------------------------------


def load_tau_bench_tasks(
    env_name: str,
    task_split: str,
    task_indices: list[int] | None = None,
    user_model: str = "Qwen/Qwen3-8B-AWQ",
    user_strategy: str = "llm",
    user_provider: str = "openai",
) -> tuple[list[TaskInstance], "TauBenchEnvWrapper"]:
    """
    Load τ-Bench tasks and return ``(task_list, env_wrapper)``.

    This function does NOT create the tau-bench env (and thus does NOT call the
    LLM).  The env is created lazily inside ``execute_tau_bench_episode()``.
    """
    _configure_openai_env()

    if env_name == "retail":
        all_tasks, tools_info, wiki = _load_retail_metadata(task_split)
    elif env_name == "airline":
        all_tasks, tools_info, wiki = _load_airline_metadata(task_split)
    else:
        raise ValueError(f"Unknown tau-bench env: {env_name}")

    if task_indices is None:
        task_indices = list(range(len(all_tasks)))

    task_instances: list[TaskInstance] = []
    for idx in task_indices:
        if idx >= len(all_tasks):
            logger.warning("Task index %d out of range (%d tasks); skipping.", idx, len(all_tasks))
            continue
        t = all_tasks[idx]
        task_instances.append(
            TaskInstance(
                task_id=f"{env_name}_{task_split}_{idx}",
                task_description=_build_task_description(t, wiki),
                tools=tools_info,
                metadata={
                    "tau_bench_index": idx,
                    "env_name": env_name,
                    "task_split": task_split,
                    "user_id": getattr(t, "user_id", ""),
                    "ground_truth_actions": [
                        {"name": a.name, "kwargs": a.kwargs}
                        for a in getattr(t, "actions", [])
                    ],
                    "required_outputs": getattr(t, "outputs", []),
                },
            )
        )

    wrapper = TauBenchEnvWrapper(
        env_name=env_name,
        task_split=task_split,
        user_model=user_model,
        user_strategy=user_strategy,
        user_provider=user_provider,
        tools_info=tools_info,
        wiki=wiki,
    )

    logger.info("Loaded %d τ-Bench tasks (%s/%s).", len(task_instances), env_name, task_split)
    return task_instances, wrapper


def _build_task_description(task: Any, wiki: str) -> str:
    """Build a natural-language task description from a τ-Bench Task object."""
    instruction = getattr(task, "instruction", "")
    parts = [
        f"Policy document (excerpt):\n{wiki[:400]}",
        "You are a customer service agent. Assist the user by calling the appropriate "
        "backend tools and responding helpfully. When complete, call respond() with a "
        "final confirmation.",
        f"(User simulator instruction: {instruction})",
    ]
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# τ-Bench Environment Wrapper
# ---------------------------------------------------------------------------


class TauBenchEnvWrapper(BenchmarkEnv):
    """
    Lazy tau-bench env wrapper.

    The actual tau_bench env is created fresh for each episode inside
    ``execute_tau_bench_episode()``, so vLLM must be running by then.
    """

    def __init__(
        self,
        env_name: str,
        task_split: str,
        user_model: str,
        user_strategy: str,
        user_provider: str,
        tools_info: list[dict],
        wiki: str,
    ) -> None:
        self.env_name = env_name
        self.task_split = task_split
        self.user_model = user_model
        self.user_strategy = user_strategy
        self.user_provider = user_provider
        self.tools_info = tools_info
        self.wiki = wiki

        # Per-episode mutable state (set by execute_tau_bench_episode)
        self._tau_env: Any = None
        self._env_done: bool = False
        self._last_reward: float = 0.0
        self._initial_observation: str = ""

    def clone(self) -> "TauBenchEnvWrapper":
        """Return a fresh wrapper with the same config but no shared tau env.
        Used by parallel evaluation (each thread needs its own env instance)."""
        return TauBenchEnvWrapper(
            env_name=self.env_name,
            task_split=self.task_split,
            user_model=self.user_model,
            user_strategy=self.user_strategy,
            user_provider=self.user_provider,
            tools_info=self.tools_info,
            wiki=self.wiki,
        )

    def _create_env(self) -> Any:
        """Create a fresh tau-bench env (requires vLLM to be running)."""
        _patch_tau_bench_user_init()
        from tau_bench.envs import get_env as tau_get_env
        return tau_get_env(
            env_name=self.env_name,
            user_strategy=self.user_strategy,
            user_model=self.user_model,
            user_provider=self.user_provider,
            task_split=self.task_split,
        )

    def reset_for_task(self, task: TaskInstance) -> str:
        """Create (or reuse) the env and reset it for a specific task index."""
        idx = task.metadata.get("tau_bench_index", 0)
        self._env_done = False
        self._last_reward = 0.0

        if self._tau_env is None:
            self._tau_env = self._create_env()

        try:
            reset_resp = self._tau_env.reset(task_index=idx)
            self._initial_observation = reset_resp.observation
        except Exception as exc:
            logger.warning("tau-bench reset failed for idx %d: %s", idx, exc)
            self._initial_observation = "Hello, I need help with my request."
        return self._initial_observation

    def execute_tool(self, task: TaskInstance, tool_call: ToolCall) -> tuple[str, bool]:
        from tau_bench.types import Action as TauAction
        if self._env_done or self._tau_env is None:
            return "Episode already done.", False
        try:
            action = TauAction(name=tool_call.name, kwargs=tool_call.arguments)
            resp = self._tau_env.step(action)
            obs = resp.observation or ""
            if resp.done:
                self._env_done = True
                self._last_reward = resp.reward
            return obs, True
        except Exception as exc:
            logger.warning("tau-bench step failed (tool=%s): %s", tool_call.name, exc)
            return f"Tool execution error: {exc}", False

    def evaluate(self, task: TaskInstance, prediction: str, trajectory: list[AgentStep]) -> float:
        return self._last_reward

    def find_tool_spec(self, tools: list[dict], tool_name: str) -> dict:
        for t in tools:
            fn = t.get("function", t)
            if fn.get("name", t.get("name", "")) == tool_name:
                return t
        return {}


# ---------------------------------------------------------------------------
# τ-Bench episode executor
# ---------------------------------------------------------------------------


def execute_tau_bench_episode(
    policy: "PolicySpec",
    task: TaskInstance,
    wrapper: TauBenchEnvWrapper,
    llm: "LLMClient",
    max_steps: int = 30,
) -> EpisodeRecord:
    """
    Run a full τ-Bench conversation episode using the modular agent.

    Conversation structure:
      1. env.reset() → initial user message
      2. Planner produces a plan
      3. Loop: Selector picks tool or "respond" → Caller executes → Synthesizer replies
      4. Repeat until env.done
    """
    from evotool.agent.modular_agent import ModularAgent, _parse_selector_output
    from evotool.agent.executor import EpisodeRecord

    initial_obs = wrapper.reset_for_task(task)
    agent = ModularAgent(policy=policy, llm=llm, max_steps=max_steps)
    trajectory: list[AgentStep] = []
    tool_results: list[dict] = []
    conversation_history: list[str] = [f"[USER] {initial_obs}"]
    steps_taken = 0

    # Planner
    try:
        plan, plan_step = agent.plan(
            f"{task.task_description}\n\nUser's first message: {initial_obs}",
            task.tools,
        )
        trajectory.append(plan_step)
        conversation_history.append(f"[PLAN] {plan}")
    except Exception as exc:
        logger.warning("Planner failed: %s", exc)
        plan = "Handle the customer request step by step using available tools."

    while steps_taken < max_steps and not wrapper._env_done:
        history_text = "\n".join(conversation_history[-10:])

        # Selector
        try:
            sel_raw, sel_step = agent.select(
                task.task_description, plan, history_text, task.tools
            )
            tool_name, is_done = _parse_selector_output(sel_raw)
            sel_step.observation = f"Selected: {tool_name or 'respond'}"
            trajectory.append(sel_step)
        except Exception as exc:
            logger.warning("Selector failed: %s", exc)
            break

        # Synthesize and respond to user
        if is_done or not tool_name or tool_name.lower() in ("respond", "answer"):
            try:
                answer, syn_step = agent.synthesize(
                    task.task_description, history_text, tool_results
                )
                trajectory.append(syn_step)
            except Exception as exc:
                logger.warning("Synthesizer failed: %s", exc)
                answer = "I have completed your request."

            from tau_bench.types import Action as TauAction
            try:
                resp = wrapper._tau_env.step(
                    TauAction(name="respond", kwargs={"content": answer})
                )
                obs = resp.observation or ""
                if resp.done:
                    wrapper._env_done = True
                    wrapper._last_reward = resp.reward
                    conversation_history.append(f"[AGENT] {answer}")
                    break
                conversation_history.append(f"[AGENT] {answer}")
                conversation_history.append(f"[USER] {obs}")
                tool_results = []
            except Exception as exc:
                logger.warning("Respond step failed: %s", exc)
                break
        else:
            # Caller
            tool_spec = wrapper.find_tool_spec(task.tools, tool_name)
            try:
                tool_call, call_step = agent.call_tool(
                    task.task_description, plan, history_text, tool_name, tool_spec
                )
            except Exception as exc:
                logger.warning("Caller failed: %s", exc)
                break

            obs, success = wrapper.execute_tool(task, tool_call)
            call_step.observation = obs
            call_step.success = success
            trajectory.append(call_step)
            tool_results.append(
                {"tool": tool_call.name, "arguments": tool_call.arguments,
                 "result": obs, "success": success}
            )
            conversation_history.append(
                f"[TOOL:{tool_call.name}] {json.dumps(tool_call.arguments)} → {obs[:200]}"
            )

        steps_taken += 1

    return EpisodeRecord(
        task=task.task_description[:500],
        task_id=task.task_id,
        trajectory=trajectory,
        prediction=conversation_history[-1] if conversation_history else "",
        reward=wrapper._last_reward,
        metadata=task.metadata,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _configure_openai_env() -> None:
    if not os.getenv("OPENAI_BASE_URL") and not os.getenv("OPENAI_API_BASE"):
        os.environ["OPENAI_BASE_URL"] = "http://localhost:8000/v1"
        logger.info("Set OPENAI_BASE_URL=http://localhost:8000/v1 for tau-bench user simulator.")
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "dummy"
