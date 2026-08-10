"""Deterministic keyword router -- the baseline a learned router must beat.

This exists to be a floor, not a product. Without it, "the 3B router is 71% accurate"
is an unreadable number: it could be excellent or it could be worse than counting words.
A model that cannot beat this does not justify its inference cost, and shipping the
comparison alongside the model is what keeps that honest.

Scoring: strong terms (ones that essentially only occur in their domain -- `cutlass`,
`freertos`, `use-after-free`) count fully; weak terms (`optimize`, `driver`, `test`)
count a fraction. Weighting is what stops a task that says "optimize" and "test" four
times from outvoting one that says "cutlass" once.
"""

from __future__ import annotations

import re

from hermes.router.base import (
    CROSS_DOMAIN,
    EMPTY_TASK,
    NO_EVIDENCE,
    TIER_KEYWORD,
    TOO_CLOSE,
    RoutingDecision,
    abstain,
)
from hermes.router.domains import DOMAINS

STRONG_WEIGHT = 1.0
WEAK_WEIGHT = 0.25

# A specialist is only chosen when it clears this score AND leads the runner-up by
# MIN_MARGIN. Two domains scoring 1.0 and 0.95 is not a decision, it is a coin flip
# wearing a confidence value -- and the whole point of the fallback is to not flip it.
MIN_SCORE = 1.0
MIN_MARGIN = 0.5

# Evidence weight at which a domain counts as "present in this task" for the
# cross-domain check below.
EVIDENCE_FLOOR = 0.5
# A task showing evidence from this many domains is cross-domain work, and goes to the
# generalist even when one domain leads clearly. "Our CUDA kernel has a buffer overflow
# a fuzzer found; patch it and open a PR" is not a CUDA task with noise -- it is one
# task spanning three specialists, and handing it to any one of them means the other two
# thirds get done by a model outside its domain.
MAX_DOMAINS_BEFORE_ABSTAIN = 3


def _count(haystack: str, term: str) -> int:
    """Count occurrences of `term`, matching on word boundaries where meaningful.

    Word boundaries matter: without them `spi` matches "inspired" and `poc` matches
    "process", which is how a keyword router quietly becomes a random one. Multi-word
    terms are matched as phrases.
    """
    pattern = r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])"
    return len(re.findall(pattern, haystack))


def score_domains(task: str) -> dict[str, float]:
    """Weighted keyword score per domain. Not normalized -- raw evidence weight."""
    text = task.lower()
    scores: dict[str, float] = {}
    for key, domain in DOMAINS.items():
        strong = sum(_count(text, term) for term in domain.strong_terms)
        weak = sum(_count(text, term) for term in domain.weak_terms)
        scores[key] = strong * STRONG_WEIGHT + weak * WEAK_WEIGHT
    return scores


class KeywordRouter:
    """Rule-based routing baseline. Deterministic, explainable, and free."""

    def __init__(
        self,
        *,
        min_score: float = MIN_SCORE,
        min_margin: float = MIN_MARGIN,
        max_domains: int = MAX_DOMAINS_BEFORE_ABSTAIN,
    ) -> None:
        self.min_score = min_score
        self.min_margin = min_margin
        self.max_domains = max_domains

    def route(self, task: str) -> RoutingDecision:
        if not task or not task.strip():
            return abstain("empty task", reason_code=EMPTY_TASK, tier=TIER_KEYWORD)

        scores = score_domains(task)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        (top_key, top_score), (_, runner_up) = ranked[0], ranked[1]

        if top_score < self.min_score:
            return abstain(
                f"no domain cleared the evidence threshold (best {top_key}={top_score:.2f})",
                scores,
                reason_code=NO_EVIDENCE,
                tier=TIER_KEYWORD,
            )

        present = [key for key, score in scores.items() if score >= EVIDENCE_FLOOR]
        if len(present) >= self.max_domains:
            # A decision, not a doubt: this really is generalist work. Marked
            # non-escalatable so a cascade does not spend a model call re-deciding it.
            return abstain(
                f"task spans {len(present)} domains ({', '.join(sorted(present))}); generalist work",
                scores,
                reason_code=CROSS_DOMAIN,
                tier=TIER_KEYWORD,
            )

        margin = top_score - runner_up
        if margin < self.min_margin:
            return abstain(
                f"{top_key} and the runner-up are within {margin:.2f}; too close to call",
                scores,
                reason_code=TOO_CLOSE,
                tier=TIER_KEYWORD,
            )

        # Confidence from the *margin*, not the absolute score: a task naming `cuda`
        # twenty times is not more clearly CUDA than one naming it twice, but a task
        # where CUDA leads by a mile genuinely is.
        confidence = min(1.0, margin / (top_score + 1e-9))
        return RoutingDecision(
            target=top_key,
            confidence=confidence,
            reason=f"{top_key} led with {top_score:.2f} (margin {margin:.2f})",
            scores=scores,
            tier=TIER_KEYWORD,
        )
