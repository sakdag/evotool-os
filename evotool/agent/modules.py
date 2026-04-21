"""
Module specification dataclass and initial prompts.

Each PolicySpec holds four evolvable prompt strings – one per module.
The INITIAL_POLICY contains exactly the prompts listed in Appendix A.7
of the EVOTOOL paper.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass


@dataclass
class PolicySpec:
    """
    Θ = {θ_plan, θ_sel, θ_call, θ_syn}

    Each field is a plain-text prompt string for the corresponding module.
    Only the blamed module is updated per generation; the rest are frozen.
    """

    planner: str
    selector: str
    caller: str
    synthesizer: str

    # ---------------------------------------------------------------
    # Module names (used throughout blame / mutation code)
    # ---------------------------------------------------------------
    MODULES: tuple[str, ...] = ("planner", "selector", "caller", "synthesizer")

    def get(self, module: str) -> str:
        return getattr(self, module)

    def set(self, module: str, spec: str) -> "PolicySpec":
        """Return a new PolicySpec with only *module* replaced."""
        d = asdict(self)
        d.pop("MODULES", None)
        d[module] = spec
        return PolicySpec(**d)

    def copy(self) -> "PolicySpec":
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        return {
            "planner": self.planner,
            "selector": self.selector,
            "caller": self.caller,
            "synthesizer": self.synthesizer,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PolicySpec":
        return cls(
            planner=d["planner"],
            selector=d["selector"],
            caller=d["caller"],
            synthesizer=d["synthesizer"],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "PolicySpec":
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# Initial prompts – verbatim from Appendix A.7 of the paper
# ---------------------------------------------------------------------------

INITIAL_POLICY = PolicySpec(
    planner=(
        "You are a planning agent. Your task is to decompose the user's complex "
        "instruction into a sequential list of clear, executable subgoals"
    ),
    selector=(
        "You are a tool selection agent. Given the current subgoal and the list of "
        "available tools, select the most appropriate tool."
    ),
    caller=(
        "You are a tool calling agent. Given the selected tool and its documentation, "
        "generate the specific arguments required to execute it."
    ),
    synthesizer=(
        "You are a synthesis agent. Review the user's original query and the history "
        "of tool executions, then synthesize this information to provide the answer."
    ),
)
