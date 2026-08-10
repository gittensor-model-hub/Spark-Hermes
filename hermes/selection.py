"""Winner selection: what the evidence decides, and what a policy decides.

`select_winner` applies tie-breaks in a fixed order -- evidence, then fewer tool calls,
then cost, then recovery, then model id. That order is a *policy*. It is not derived from
anything; someone chose it, and it silently decides which teacher wins every tournament
where the leading candidates are not strictly comparable.

Two problems follow, and they are different.

**The order is hard-coded, so it can be changed after the results are known.** Reordering
cost above tool calls because one teacher keeps losing is a rule change made with the
answers in hand, and nothing in the record would show it happened. So a policy is declared
as data with a digest, and that digest belongs in the harness digest: changing the order
then makes prior results *incomparable* rather than quietly re-rankable, which is the
honest consequence and the one `hermes/harness.py` already knows how to express.

**A tie-break reports which rule fired, not whether a rule was needed at all.** A winner
that beat every rival on every measured dimension and a winner picked from four mutually
incomparable candidates both come back saying `fewer_tool_calls`. Those are very different
claims about a teacher, and the second one is mostly a claim about the policy. So the
Pareto frontier is computed first: attempts that another attempt beat outright are removed
as dominated, and the report says how many survived. A frontier of one means the evidence
decided. A frontier of four means the policy did.

**A dimension only participates when both sides can be measured on it.** An unpriced run
has no cost, and treating that as zero would let the unmeasured dominate the measured --
the same failure `PriceBook.price_of` refuses and the efficiency-margin builder refuses.
Here it takes a third form: a pair that cannot be compared on cost is compared on
everything else, and neither dominates the other on the strength of a missing number.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

# Dimensions a policy may order. Each names a direction, because "better" is not obvious
# from the field name: fewer tool calls is better, more recovery is better.
EVIDENCE = "evidence"
TOOL_CALLS = "tool_calls"
COST = "cost"
WALL_TIME = "wall_time"
RECOVERY = "recovery"
TOOL_VALIDITY = "tool_validity"

# name -> (extractor, higher_is_better)
_DIMENSIONS: dict[str, tuple[str, bool]] = {
    EVIDENCE: ("evidence", True),
    TOOL_CALLS: ("tool_calls", False),
    COST: ("cost", False),
    WALL_TIME: ("wall_time_s", False),
    RECOVERY: ("recovered", True),
    TOOL_VALIDITY: ("tool_call_validity", True),
}

DIMENSIONS = tuple(_DIMENSIONS)


class SelectionError(ValueError):
    """A selection policy is malformed."""


def _value(candidate: Any, dimension: str, *, primary_evidence: str) -> float | None:
    """The candidate's value on a dimension, or None when it cannot be measured."""
    if dimension == EVIDENCE:
        if not primary_evidence:
            return None
        return candidate.verdict.score(primary_evidence)
    field, _ = _DIMENSIONS[dimension]
    value = getattr(candidate, field, None)
    if value is None:
        return None
    return float(value)


@dataclass(frozen=True)
class SelectionPolicy:
    """A declared tie-break order, fixed before the results are seen.

    `order` is applied after the Pareto frontier is computed, so it only ever chooses among
    attempts that no other attempt beat outright. An empty order is legitimate and means
    "let the evidence decide, and break a genuine tie on model id" -- which is the most
    conservative policy available and the right default for a task where nobody has
    justified a priority between cost and thoroughness.
    """

    order: tuple[str, ...] = ()
    # Dimensions the frontier is computed over. Deliberately separate from `order`: a
    # dimension can be worth *considering* for dominance without anyone having decided
    # where it sits in a priority list.
    compare: tuple[str, ...] = (EVIDENCE, TOOL_CALLS, COST, RECOVERY)
    primary_evidence: str = ""

    def __post_init__(self) -> None:
        for name, dims in (("order", self.order), ("compare", self.compare)):
            unknown = [d for d in dims if d not in _DIMENSIONS]
            if unknown:
                raise SelectionError(f"unknown selection {name} dimensions {unknown}; expected from {list(DIMENSIONS)}")
            if len(set(dims)) != len(dims):
                raise SelectionError(f"{name} names a dimension twice; the duplicate can never fire")
        if EVIDENCE in self.compare and not self.primary_evidence:
            raise SelectionError(
                "policy compares on evidence but names no primary_evidence key; every candidate "
                "would score 0 and the dimension would silently do nothing"
            )

    @property
    def digest(self) -> str:
        """A content address for the rules.

        Belongs in the harness digest. A tournament run under a reordered policy is not
        the same measurement as one run under the old order, and saying so is more honest
        than re-ranking old results under new rules.
        """
        payload = json.dumps(
            {"order": list(self.order), "compare": list(self.compare), "primary_evidence": self.primary_evidence},
            sort_keys=True,
            separators=(",", ":"),
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_record(self) -> dict[str, Any]:
        return {
            "order": list(self.order),
            "compare": list(self.compare),
            "primary_evidence": self.primary_evidence,
            "digest": self.digest,
        }


def dominates(a: Any, b: Any, *, policy: SelectionPolicy) -> bool:
    """Whether `a` beats `b` outright: at least as good everywhere, better somewhere.

    Only dimensions both candidates can be measured on are considered. A run with no
    recorded cost does not lose the cost dimension by default -- it simply is not compared
    on it, and if that leaves nothing to compare, neither dominates.
    """
    better_somewhere = False
    compared = 0
    for dimension in policy.compare:
        av = _value(a, dimension, primary_evidence=policy.primary_evidence)
        bv = _value(b, dimension, primary_evidence=policy.primary_evidence)
        if av is None or bv is None:
            continue
        compared += 1
        higher_is_better = _DIMENSIONS[dimension][1]
        if (av < bv) if higher_is_better else (av > bv):
            return False
        if av != bv:
            better_somewhere = True
    return compared > 0 and better_somewhere


@dataclass(frozen=True)
class Frontier:
    """Who survived comparison, and who beat whom."""

    surviving: tuple[Any, ...]
    dominated: tuple[tuple[str, str], ...] = ()

    @property
    def decided_by_evidence(self) -> bool:
        """True when one attempt beat every other outright.

        The distinction the win reason alone cannot make: a winner that dominated the field
        and a winner the policy picked from four incomparable attempts are different claims,
        and only the first is a claim about the teacher.
        """
        return len(self.surviving) == 1

    def to_record(self) -> dict[str, Any]:
        return {
            "surviving": [c.model for c in self.surviving],
            "dominated": [{"model": m, "by": d} for m, d in self.dominated],
            "decided_by_evidence": self.decided_by_evidence,
        }


def frontier(candidates: list[Any], *, policy: SelectionPolicy) -> Frontier:
    """Remove every attempt that another attempt beat outright."""
    if not candidates:
        return Frontier(surviving=())
    surviving: list[Any] = []
    beaten: list[tuple[str, str]] = []
    for candidate in candidates:
        winner = next((o for o in candidates if o is not candidate and dominates(o, candidate, policy=policy)), None)
        if winner is None:
            surviving.append(candidate)
        else:
            beaten.append((candidate.model, winner.model))
    return Frontier(surviving=tuple(surviving), dominated=tuple(beaten))


def apply_policy(candidates: list[Any], *, policy: SelectionPolicy) -> tuple[Any | None, tuple[str, ...]]:
    """Narrow a frontier by the declared order, then break any remaining tie on model id.

    Returns the choice and the dimensions that actually fired. A dimension that did not
    separate anything is not reported: naming it would suggest it decided something.
    """
    if not candidates:
        return None, ()
    pool = list(candidates)
    fired: list[str] = []
    for dimension in policy.order:
        values = [(c, _value(c, dimension, primary_evidence=policy.primary_evidence)) for c in pool]
        measurable = [(c, v) for c, v in values if v is not None]
        # A dimension only decides when *every* candidate can be measured on it. Letting a
        # partially-measured dimension narrow the pool would drop the candidates whose
        # value was missing, which is elimination by absence of data.
        if len(measurable) != len(pool) or not measurable:
            continue
        higher_is_better = _DIMENSIONS[dimension][1]
        best = max(v for _, v in measurable) if higher_is_better else min(v for _, v in measurable)
        narrowed = [c for c, v in measurable if v == best]
        if len(narrowed) != len(pool):
            pool = narrowed
            fired.append(dimension)
        if len(pool) == 1:
            break
    if len(pool) > 1:
        pool = [min(pool, key=lambda c: c.model)]
        fired.append("model_id")
    return pool[0], tuple(fired)


@dataclass(frozen=True)
class Selection:
    """A winner, and an honest account of what chose it."""

    winner: Any | None
    frontier: Frontier
    policy_dimensions: tuple[str, ...]
    policy_digest: str

    @property
    def decided_by_evidence(self) -> bool:
        return self.frontier.decided_by_evidence

    def to_record(self) -> dict[str, Any]:
        return {
            "winner": self.winner.model if self.winner else None,
            "decided_by_evidence": self.decided_by_evidence,
            "policy_dimensions": list(self.policy_dimensions),
            "policy_digest": self.policy_digest,
            "frontier": self.frontier.to_record(),
        }


def select(candidates: list[Any], *, policy: SelectionPolicy) -> Selection:
    """Pareto first, declared policy second.

    The order matters. Applying the policy first lets a low-priority dimension eliminate a
    candidate that was better on everything the policy ranks below it; taking the frontier
    first means the policy is only ever asked to choose between attempts that genuinely
    could not be separated on the measurements.
    """
    surviving = frontier(candidates, policy=policy)
    winner, fired = apply_policy(list(surviving.surviving), policy=policy)
    return Selection(
        winner=winner,
        frontier=surviving,
        policy_dimensions=fired,
        policy_digest=policy.digest,
    )
