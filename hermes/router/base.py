"""Routing decision type and the Router protocol.

Routing is a far easier problem than solving -- which is why a 3B-7B model can do it --
but "easier" is not "free", and a router that silently guesses is worse than no router
at all, because a specialist acting outside its domain acts *confidently*. Every
decision therefore carries a confidence and a reason, and the router is expected to fall
back to `general` rather than pick a specialist it is unsure about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from hermes.router.domains import GENERAL, is_valid_target, model_for


class RoutingError(ValueError):
    """A router produced a decision that cannot be acted on."""


# Why a router abstained. Structured because the cascade has to tell *uncertainty* from
# a *decision*: "I have no idea what this is" is worth escalating to a model, while
# "this task genuinely spans three specialists" is already the right answer and paying
# for a second opinion on it just burns a call. String-matching the human-readable
# reason to make that distinction would break the moment someone reworded it.
NO_EVIDENCE = "no_evidence"
TOO_CLOSE = "too_close"
CROSS_DOMAIN = "cross_domain"
EMPTY_TASK = "empty_task"
LOW_CONFIDENCE = "low_confidence"
UNKNOWN_TARGET = "unknown_target"
MALFORMED_REPLY = "malformed_reply"
ROUTER_ERROR = "router_error"

# Abstentions a second opinion could plausibly resolve.
ESCALATABLE = frozenset({NO_EVIDENCE, TOO_CLOSE})

# Tier labels, for attributing a decision to whoever actually made it.
TIER_KEYWORD = "keyword"
TIER_MODEL = "model"


@dataclass(frozen=True)
class RoutingDecision:
    """Where a task should go, how sure the router is, and why.

    `abstained` records that the router fell back to `general` because nothing cleared
    its threshold -- distinct from confidently deciding a task really is generalist
    work. The two look identical in the `target` field and score very differently: an
    abstention is the router working as designed, a confident `general` is a claim.
    """

    target: str
    confidence: float
    reason: str
    abstained: bool = False
    scores: dict[str, float] = field(default_factory=dict)
    # Machine-readable counterpart to `reason`; empty on a positive decision.
    reason_code: str = ""
    # Which tier actually decided. A cascade needs this to report what it spent: a
    # decision made by the free tier and one made by the escalation model are worth
    # very different amounts even when they agree.
    tier: str = ""
    escalated: bool = False

    def __post_init__(self) -> None:
        if not is_valid_target(self.target):
            raise RoutingError(f"unknown routing target {self.target!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise RoutingError(f"confidence must be in [0, 1], got {self.confidence}")

    @property
    def model(self) -> str:
        return model_for(self.target)

    @property
    def escalatable(self) -> bool:
        """Whether a second opinion could plausibly resolve this abstention."""
        return self.abstained and self.reason_code in ESCALATABLE

    def to_record(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "model": self.model,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "reason_code": self.reason_code,
            "abstained": self.abstained,
            "tier": self.tier,
            "escalated": self.escalated,
            "scores": {k: round(v, 4) for k, v in sorted(self.scores.items(), key=lambda kv: -kv[1])},
        }


def abstain(
    reason: str,
    scores: dict[str, float] | None = None,
    confidence: float = 0.0,
    *,
    reason_code: str = "",
    tier: str = "",
) -> RoutingDecision:
    """Fall back to the generalist. The safe answer when nothing is clearly indicated."""
    return RoutingDecision(
        target=GENERAL,
        confidence=confidence,
        reason=reason,
        abstained=True,
        scores=scores or {},
        reason_code=reason_code,
        tier=tier,
    )


class Router(Protocol):
    """Anything that can pick a specialist for a task."""

    def route(self, task: str) -> RoutingDecision: ...
