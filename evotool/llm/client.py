"""
LLM client wrapping the OpenAI-compatible API served by vLLM.

Supports:
  - Qwen3-8B thinking-mode toggle (/think vs /no_think prefix)
  - Optional fallback to a remote model (e.g. gpt-4.1) via standard OPENAI_API_KEY
  - Both chat-completion and tool-calling modes
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    agent_model: str = "Qwen/Qwen3-8B-AWQ"
    blamer_model: str = "Qwen/Qwen3-8B-AWQ"
    mutator_model: str = "Qwen/Qwen3-8B-AWQ"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "dummy"
    agent_thinking: bool = False
    blamer_thinking: bool = True
    mutator_thinking: bool = True
    max_tokens: int = 2048
    temperature: float = 0.7
    fallback_model: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "LLMConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class LLMClient:
    """
    Unified LLM client. Wraps the vLLM OpenAI-compatible server.
    A second client pointing to api.openai.com is initialised lazily
    if a fallback_model is configured and OPENAI_API_KEY is set.
    """

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._local = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
        self._remote: OpenAI | None = None
        if cfg.fallback_model and os.getenv("OPENAI_API_KEY"):
            self._remote = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def call_agent(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
    ) -> str:
        """Call the agent backbone (thinking disabled by default)."""
        return self._chat(
            model=self.cfg.agent_model,
            messages=self._inject_thinking(messages, self.cfg.agent_thinking),
            tools=tools,
            tool_choice=tool_choice if tools else None,
            thinking=self.cfg.agent_thinking,
        )

    def call_blamer(self, messages: list[dict]) -> str:
        """Call the Blamer LLM (thinking enabled by default)."""
        return self._chat(
            model=self.cfg.blamer_model,
            messages=self._inject_thinking(messages, self.cfg.blamer_thinking),
            thinking=self.cfg.blamer_thinking,
        )

    def call_mutator(self, messages: list[dict]) -> str:
        """Call the Mutator LLM (thinking enabled by default)."""
        return self._chat(
            model=self.cfg.mutator_model,
            messages=self._inject_thinking(messages, self.cfg.mutator_thinking),
            thinking=self.cfg.mutator_thinking,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _inject_thinking(self, messages: list[dict], thinking: bool) -> list[dict]:
        """
        Qwen3 thinking mode is toggled via a /think or /no_think suffix
        on the last user message.  We inject a lightweight system hint
        instead so any future message can still be appended cleanly.
        """
        if not messages:
            return messages
        tag = "/think" if thinking else "/no_think"
        msgs = list(messages)
        last = dict(msgs[-1])
        if last.get("role") == "user":
            content = last.get("content", "")
            if not content.endswith(tag):
                last["content"] = f"{content} {tag}"
            msgs[-1] = last
        return msgs

    def _chat(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        thinking: bool = False,
    ) -> str:
        """
        Make a chat completion request.  Returns the plain text content
        of the first choice, or a JSON-serialised tool call if tools are
        returned by the model.
        """
        messages = _trim_messages(messages, max_chars=28_000)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"

        client = self._local
        # Use fallback remote model when explicitly requested
        if self.cfg.fallback_model and model == self.cfg.fallback_model and self._remote:
            client = self._remote
            kwargs["model"] = self.cfg.fallback_model

        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            raise

        choice = resp.choices[0]

        # Tool call response
        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            tc = choice.message.tool_calls[0]
            return json.dumps(
                {
                    "tool_name": tc.function.name,
                    "tool_arguments": json.loads(tc.function.arguments),
                }
            )

        text = choice.message.content or ""
        # Strip Qwen3 thinking tags <think>...</think> from content
        text = _strip_thinking_tags(text)
        return text.strip()

    def close(self) -> None:
        self._local.close()
        if self._remote:
            self._remote.close()


def _strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks that Qwen3 may emit in thinking mode."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _trim_messages(messages: list[dict], max_chars: int = 28_000) -> list[dict]:
    """
    Trim message list so total character count stays under max_chars.

    Strategy: keep the system message intact; drop the oldest non-system
    messages from the middle until we're under budget.  This preserves the
    most recent context which is most relevant for generation.
    ~4 chars ≈ 1 token, so 28 000 chars ≈ 7 000 tokens leaving ample room
    for a 2 048-token reply within a 16 384-token context window.
    """
    total = sum(len(str(m.get("content", ""))) for m in messages)
    if total <= max_chars:
        return messages

    system = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    # Drop oldest non-system messages one at a time until under budget
    while non_system and total > max_chars:
        removed = non_system.pop(0)
        total -= len(str(removed.get("content", "")))

    trimmed = system + non_system
    logger.debug("Trimmed messages from %d chars to %d chars.", total + sum(len(str(m.get('content',''))) for m in system), total)
    return trimmed
