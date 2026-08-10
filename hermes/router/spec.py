"""`TaskSpec` -- the structured description a route decision is made from.

"Coding" and "research" are not routing labels. The best model for *writing* a kernel is
often not the best for *reviewing its numerical correctness*, and a task needing an
`ncu` profile on `sm_120` is unroutable to an expert without a GPU regardless of how good
that expert is at CUDA. So a task is described along several independent axes, and the
`action` axis matters as much as `domain`.

Everything here is deliberately declarative and boring. Phase A exists to fix the schemas
and the logging before any learned router is introduced, because a learned router whose
inputs were never pinned down is not debuggable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --- taxonomy ----------------------------------------------------------------------

DOMAINS = ("swe", "cuda", "firmware", "browser", "research", "math", "documents", "vision")

ACTIONS = ("explain", "plan", "implement", "debug", "optimize", "review", "verify")

# Horizon drives budget, harness configuration and how much a mistake costs to unwind.
HORIZONS = ("direct", "short", "medium", "long", "multi_session")
HORIZON_TOOL_CALLS: dict[str, tuple[int, int]] = {
    "direct": (0, 0),
    "short": (1, 5),
    "medium": (6, 20),
    "long": (20, 100),
    "multi_session": (100, 100_000),
}

ENVIRONMENTS = ("repository", "terminal", "browser", "gpu", "emulator", "hardware_in_loop")

# Ordered weakest -> strongest. A route that can be checked by running something beats
# one that can only be checked by asking a model, so this ordering is load-bearing when
# choosing between otherwise-equal candidates.
VERIFICATION_KINDS = (
    "judge_only",
    "citation_check",
    "exact_answer",
    "compile",
    "unit_tests",
    "numerical_comparison",
    "benchmark",
)
DETERMINISTIC_VERIFICATION = frozenset(k for k in VERIFICATION_KINDS if k != "judge_only")

RISKS = ("low", "medium", "high")


class TaskSpecError(ValueError):
    """A task specification is malformed or uses an unknown taxonomy value."""


def _check(values: tuple[str, ...], allowed: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    unknown = [v for v in values if v not in allowed]
    if unknown:
        raise TaskSpecError(f"unknown {field_name}: {unknown}; expected values from {list(allowed)}")
    return values


@dataclass(frozen=True)
class TaskSpec:
    """A routable unit of work.

    `verification` names how the result will be checked, not how it will be produced.
    That distinction is what lets the router prefer a candidate whose output can be
    proven over one whose output can only be believed.
    """

    task_id: str
    prompt: str
    domain: tuple[str, ...]
    action: tuple[str, ...]
    horizon: str = "medium"
    language: tuple[str, ...] = ()
    modality: tuple[str, ...] = ("text",)
    environment: dict[str, Any] = field(default_factory=dict)
    required_tools: tuple[str, ...] = ()
    required_environments: tuple[str, ...] = ()
    verification: str = "judge_only"
    risk: str = "low"
    fresh_information_required: bool = False
    context_tokens_required: int = 0
    expected_tool_calls: int = 0
    target_student: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id or not self.prompt.strip():
            raise TaskSpecError("task spec needs a task_id and a prompt")
        _check(self.domain, DOMAINS, "domain")
        _check(self.action, ACTIONS, "action")
        _check(self.required_environments, ENVIRONMENTS, "environment")
        if not self.domain:
            raise TaskSpecError("task spec needs at least one domain")
        if not self.action:
            raise TaskSpecError("task spec needs at least one action")
        if self.horizon not in HORIZONS:
            raise TaskSpecError(f"unknown horizon {self.horizon!r}; expected one of {list(HORIZONS)}")
        if self.verification not in VERIFICATION_KINDS:
            raise TaskSpecError(f"unknown verification {self.verification!r}")
        if self.risk not in RISKS:
            raise TaskSpecError(f"unknown risk {self.risk!r}")

    @property
    def deterministically_verifiable(self) -> bool:
        """Whether the outcome can be checked by running something rather than judged."""
        return self.verification in DETERMINISTIC_VERIFICATION

    @property
    def requires_gpu(self) -> bool:
        return "gpu" in self.required_environments

    @property
    def bucket(self) -> str:
        """Coarse key for capability statistics: `domain.action.horizon`.

        Deliberately coarse. Per-task statistics never accumulate enough samples to mean
        anything, and the whole point of bucketing is to have a denominator.
        """
        return f"{self.domain[0]}.{self.action[0]}.{self.horizon}"

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "domain": list(self.domain),
            "action": list(self.action),
            "horizon": self.horizon,
            "language": list(self.language),
            "modality": list(self.modality),
            "environment": self.environment,
            "required_tools": list(self.required_tools),
            "required_environments": list(self.required_environments),
            "verification": self.verification,
            "risk": self.risk,
            "fresh_information_required": self.fresh_information_required,
            "context_tokens_required": self.context_tokens_required,
            "expected_tool_calls": self.expected_tool_calls,
            "target_student": self.target_student,
            "bucket": self.bucket,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> TaskSpec:
        return cls(
            task_id=str(record.get("task_id") or ""),
            prompt=str(record.get("prompt") or ""),
            domain=tuple(record.get("domain") or ()),
            action=tuple(record.get("action") or ()),
            horizon=str(record.get("horizon", "medium")),
            language=tuple(record.get("language") or ()),
            modality=tuple(record.get("modality") or ("text",)),
            environment=record.get("environment") or {},
            required_tools=tuple(record.get("required_tools") or ()),
            required_environments=tuple(record.get("required_environments") or ()),
            verification=str(record.get("verification", "judge_only")),
            risk=str(record.get("risk", "low")),
            fresh_information_required=bool(record.get("fresh_information_required", False)),
            context_tokens_required=int(record.get("context_tokens_required", 0)),
            expected_tool_calls=int(record.get("expected_tool_calls", 0)),
            target_student=record.get("target_student"),
            metadata=record.get("metadata") or {},
        )
