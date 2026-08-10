"""Two-tier routing: free rules first, a tiny LLM only when they are unsure.

```
task -> KeywordRouter ──confident──> specialist          (free)
             │
             └──unsure──> tiny router (3B-7B) ──> specialist or generalist   (paid)
```

The economics are the point. Most real tasks say `cutlass` or `freertos` or `pytest`
somewhere, and matching a word costs nothing; the hard minority is where a model earns
its keep. Routing everything through an LLM pays full price for the easy majority, and
routing nothing through one sends every ambiguous task to the generalist.

**Only uncertainty escalates.** `KeywordRouter` abstains for two different reasons, and
conflating them wastes calls:

- `no_evidence` / `too_close` -- the rules genuinely cannot tell. A model might. **Escalate.**
- `cross_domain` -- the task provably spans three specialists, so the generalist *is* the
  right answer. **Do not escalate**; a second opinion has nothing to add and costs a call.
- `empty_task` -- there is nothing to route. **Do not escalate.**

That distinction is carried by `RoutingDecision.reason_code`, not by matching the
human-readable reason string, so rewording a message cannot silently change what gets
billed.

The tier that actually decided is recorded on every decision (`tier`, `escalated`), which
is what lets `evaluate()` report the escalation rate -- the number that says whether the
cascade is saving anything.
"""

from __future__ import annotations

from hermes.router.base import TIER_KEYWORD, Router, RoutingDecision
from hermes.router.keyword import KeywordRouter


class CascadeRouter:
    """Cheap deterministic routing, escalating to a small model only when unsure.

    `escalate_to` is any `Router` -- typically a `ModelRouter` wrapping a served 3B-7B
    checkpoint. Passing `None` degrades the cascade to the free tier alone, which is the
    correct behavior when no tiny router is deployed yet rather than an error.
    """

    def __init__(
        self,
        escalate_to: Router | None = None,
        *,
        fast: Router | None = None,
    ) -> None:
        self.fast = fast if fast is not None else KeywordRouter()
        self.escalate_to = escalate_to

    def route(self, task: str) -> RoutingDecision:
        decision = self.fast.route(task)

        if not decision.escalatable or self.escalate_to is None:
            return decision

        escalated = self.escalate_to.route(task)

        if escalated.abstained:
            # The model could not decide either. Keep *its* verdict rather than the
            # keyword tier's: both land on the generalist, but the reason that survives
            # should be the last one actually consulted, or the logs will claim the
            # cheap tier made a call it did not make.
            return RoutingDecision(
                target=escalated.target,
                confidence=escalated.confidence,
                reason=f"{decision.reason}; escalated, and {escalated.reason}",
                abstained=True,
                scores=decision.scores,
                reason_code=escalated.reason_code,
                tier=escalated.tier,
                escalated=True,
            )

        return RoutingDecision(
            target=escalated.target,
            confidence=escalated.confidence,
            reason=f"{decision.reason}; escalated -> {escalated.reason}",
            abstained=False,
            # Keyword scores are kept for debuggability: they are why this escalated.
            scores=decision.scores,
            reason_code=escalated.reason_code,
            tier=escalated.tier,
            escalated=True,
        )

    @property
    def has_escalation(self) -> bool:
        return self.escalate_to is not None

    def __repr__(self) -> str:
        target = type(self.escalate_to).__name__ if self.escalate_to else "none"
        fast = type(self.fast).__name__ if self.fast else TIER_KEYWORD
        return f"CascadeRouter(fast={fast}, escalate_to={target})"
