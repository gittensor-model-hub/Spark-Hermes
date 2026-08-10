"""Routing evaluation.

Accuracy alone is the wrong headline number for a router, for two reasons.

**The errors cost different amounts.** Sending a firmware task to `cyber` puts a
specialist to work confidently outside its training. Sending it to `general` costs some
quality and nothing else. Pooling both into "29% wrong" hides which kind you have, so
they are counted separately: `misroute_rate` is the expensive one, `abstention_rate` is
the cheap one, and a router that trades misroutes for abstentions has improved even
though its accuracy did not move.

**A majority class flatters a lazy router.** If half the set is `swe`, a router that
answers `swe` unconditionally scores 50% and is useless. Per-domain recall is reported
so that router reads as 1.00 on `swe` and 0.00 on everything else.

Caveat on the shipped suite: `routing_v0`'s tasks and `domains.py`'s keyword lists were
authored together, so `KeywordRouter` scores 100% on it by construction. That makes the
suite a regression harness, not validation, and leaves a learned router no headroom to
prove itself here -- see hermes/router/README.md before quoting any number from it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hermes.router.base import Router, RoutingDecision
from hermes.router.domains import ALL_TARGETS, GENERAL

TASKS_ROOT = Path(__file__).parent / "tasks"


class RoutingDatasetError(ValueError):
    """A labeled routing example is malformed."""


@dataclass(frozen=True)
class RoutingExample:
    task: str
    gold: str
    example_id: str | None = None

    @classmethod
    def from_record(cls, record: dict[str, Any], *, origin: str = "<memory>") -> RoutingExample:
        task = str(record.get("task") or "").strip()
        gold = str(record.get("gold") or "").strip().lower()
        if not task:
            raise RoutingDatasetError(f"{origin}: example has no task")
        if gold not in ALL_TARGETS:
            raise RoutingDatasetError(f"{origin}: unknown gold label {gold!r}; expected one of {list(ALL_TARGETS)}")
        return cls(task=task, gold=gold, example_id=record.get("example_id"))


@dataclass(frozen=True)
class Outcome:
    """One routed example, classified by *what kind* of right or wrong it was."""

    example_id: str | None
    gold: str
    decision: RoutingDecision

    @property
    def correct(self) -> bool:
        return self.decision.target == self.gold

    @property
    def misrouted(self) -> bool:
        """Dispatched to the wrong specialist -- the expensive error."""
        return not self.correct and self.decision.target != GENERAL

    @property
    def abstained_with_specialist_available(self) -> bool:
        """Fell back when a specialist existed -- the cheap miss."""
        return self.decision.abstained and self.gold != GENERAL

    @property
    def confident_general_miss(self) -> bool:
        """Claimed `general` outright when a specialist existed.

        Worse than an abstention: an abstention is the router reporting that it does not
        know, this is the router reporting that it does.
        """
        return not self.correct and self.decision.target == GENERAL and not self.decision.abstained

    def to_record(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "gold": self.gold,
            "correct": self.correct,
            "misrouted": self.misrouted,
            **self.decision.to_record(),
        }


@dataclass(frozen=True)
class RouterMetrics:
    accuracy: float
    misroute_rate: float
    abstention_rate: float
    confident_general_miss_rate: float
    specialist_recall: dict[str, float]
    per_domain_support: dict[str, int]
    examples: int
    # Share of tasks that reached the paid tier. For a cascade this is the cost: at 0.0
    # the tiny router is never consulted and is not earning anything; at 1.0 the cheap
    # tier is doing no work and the cascade is just an LLM router with extra steps.
    escalation_rate: float = 0.0
    outcomes: tuple[Outcome, ...] = field(default=())

    def to_record(self) -> dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "misroute_rate": round(self.misroute_rate, 4),
            "abstention_rate": round(self.abstention_rate, 4),
            "confident_general_miss_rate": round(self.confident_general_miss_rate, 4),
            "escalation_rate": round(self.escalation_rate, 4),
            "specialist_recall": {k: round(v, 4) for k, v in sorted(self.specialist_recall.items())},
            "per_domain_support": dict(sorted(self.per_domain_support.items())),
            "examples": self.examples,
        }


def _rate(n: int, d: int) -> float:
    return n / d if d else 0.0


def evaluate(router: Router, examples: Iterable[RoutingExample]) -> RouterMetrics:
    """Score a router over labeled examples."""
    outcomes = [Outcome(example_id=e.example_id, gold=e.gold, decision=router.route(e.task)) for e in examples]
    total = len(outcomes)
    if not total:
        return RouterMetrics(0.0, 0.0, 0.0, 0.0, {}, {}, 0, 0.0, ())

    support: dict[str, int] = {}
    hits: dict[str, int] = {}
    for outcome in outcomes:
        support[outcome.gold] = support.get(outcome.gold, 0) + 1
        if outcome.correct:
            hits[outcome.gold] = hits.get(outcome.gold, 0) + 1

    return RouterMetrics(
        accuracy=_rate(sum(1 for o in outcomes if o.correct), total),
        misroute_rate=_rate(sum(1 for o in outcomes if o.misrouted), total),
        abstention_rate=_rate(sum(1 for o in outcomes if o.decision.abstained), total),
        confident_general_miss_rate=_rate(sum(1 for o in outcomes if o.confident_general_miss), total),
        specialist_recall={domain: _rate(hits.get(domain, 0), count) for domain, count in support.items()},
        per_domain_support=support,
        examples=total,
        escalation_rate=_rate(sum(1 for o in outcomes if o.decision.escalated), total),
        outcomes=tuple(outcomes),
    )


def compare(baseline: RouterMetrics, candidate: RouterMetrics) -> dict[str, Any]:
    """Whether a candidate router earns its inference cost against the baseline.

    `verdict` is deliberately conservative: matching the baseline's accuracy is not a
    win, because the baseline is free. A candidate also fails if it bought its accuracy
    with extra misroutes -- trading a cheap error for an expensive one is a regression
    however the headline number moves.
    """
    accuracy_delta = candidate.accuracy - baseline.accuracy
    misroute_delta = candidate.misroute_rate - baseline.misroute_rate
    beats = accuracy_delta > 0 and misroute_delta <= 0
    if beats:
        verdict = "candidate beats baseline"
    elif accuracy_delta > 0:
        verdict = "candidate is more accurate but misroutes more; not an improvement"
    elif accuracy_delta == 0:
        verdict = "candidate matches the free baseline; it does not justify its cost"
    else:
        verdict = "candidate is worse than the free baseline"
    return {
        "accuracy_delta": round(accuracy_delta, 4),
        "misroute_delta": round(misroute_delta, 4),
        "escalation_rate": round(candidate.escalation_rate, 4),
        "beats_baseline": beats,
        "verdict": verdict,
        "baseline": baseline.to_record(),
        "candidate": candidate.to_record(),
    }


def load_examples(path: Path) -> Iterator[RoutingExample]:
    """Yield labeled examples from a JSONL file, reporting bad rows with their line."""
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield RoutingExample.from_record(json.loads(line), origin=f"{path}:{lineno}")
            except (json.JSONDecodeError, RoutingDatasetError) as exc:
                raise RoutingDatasetError(f"{path}:{lineno}: {exc}") from exc


def load_suite(version: str = "v0", root: Path | None = None) -> list[RoutingExample]:
    path = (root or TASKS_ROOT) / f"routing_{version}.jsonl"
    if not path.is_file():
        raise RoutingDatasetError(f"no such routing suite: {path}")
    return list(load_examples(path))
