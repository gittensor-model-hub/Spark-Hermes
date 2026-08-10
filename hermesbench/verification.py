"""How much a trajectory actually verified, computed from what was observed.

The acceptance rule the design needs is "a candidate may not verify less than the baseline",
because the cheapest way to use fewer tokens and fewer tool calls is to stop checking. That
rule needs a signal, and the two that exist cannot carry it:

`self_check_rate`'s `self_checked` is true when *any* successful call follows the last
mutating one. Its own docstring says it "cannot tell a real test run from an incidental
`ls`". `check_verification_ran` is a position-independent set intersection -- it passes if
the agent called a verification tool anywhere, including before doing any work -- and it is
vacuous on tasks that declare no `verification_tools`. `verification_steps` counts calls to
those same tools, which on most tasks *are* the work tools, so it counts nearly everything.

So a capsule whose policy is "after the last edit, read back the file you wrote, then stop"
sets both booleans, spends almost nothing, and wins on tokens **because it verified less**.
The guard at `hermes.tournament` only fires when the winner is False and the loser True, so
two Trues sail through.

## What is actually observable

Not "did it verify" -- that is a judgement about intent. What the trace can support is
narrower and checkable:

    observed_after_mutation   successful observations after the last state change
    rechecked                 commands run both before and after a mutation
    demonstrated_repairs      commands observed FAILING, then observed PASSING after a change

The third is the one that matters, and it is the one an `ls` cannot manufacture. It requires
an observed failure, an intervening change, and an observed pass of *the same command* -- a
transition the agent had to actually produce rather than assert. Ordering is load-bearing,
which is why this reads the step sequence rather than a set of tool names.

## What it still cannot do

`demonstrated_repairs` is gameable by an agent that manufactures its own trivial failure --
`test -f x` failing, `touch x`, `test -f x` passing -- and there is no cheap way to tell
that from repairing the task. It is strictly harder to fake than a read-back and still not
proof. So this returns a **vector and never a score**: a single number would invite exactly
the false confidence the design's `verification_strength: 0.92` implies, and comparing two
of those would be comparing two guesses. `at_least_as_strong` refuses an incomparable pair
rather than resolving it with a weighting nobody agreed to.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from hermes.trajectory import FINAL, TOOL_CALL, TOOL_RESULT, AgentTrajectory
from hermesbench.tasks import DEFAULT_MUTATING_TOOLS


def _key(tool: str, args: dict[str, Any]) -> str:
    """Identity of a command, so the same check run twice is recognisable as the same.

    Arguments are part of it: `pytest tests/a.py` and `pytest tests/b.py` are different
    checks, and treating them as one would let an agent claim a repair it never observed.
    """
    return tool + " " + json.dumps(args, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class VerificationEvidence:
    """What a trajectory demonstrably observed about its own work."""

    observed_after_mutation: int
    rechecked: int
    demonstrated_repairs: int
    mutated: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "observed_after_mutation": self.observed_after_mutation,
            "rechecked": self.rechecked,
            "demonstrated_repairs": self.demonstrated_repairs,
            "mutated": self.mutated,
            # Stated on every record rather than left to a reader. The strongest component
            # here is a fail-then-pass transition of one command, which is much harder to
            # fake than a read-back and is still not proof that the task was verified.
            "is_a_score": False,
            "demonstrated_repairs_can_be_self_manufactured": True,
        }


def evidence(
    trajectory: AgentTrajectory, *, mutating_tools: tuple[str, ...] = DEFAULT_MUTATING_TOOLS
) -> VerificationEvidence:
    """Read verification evidence out of the step sequence.

    Order matters throughout: a check that ran before the work proves nothing about the
    work, which is the specific hole in `check_verification_ran`.
    """
    mutating = set(mutating_tools)
    steps = list(trajectory.steps)

    # Result lookup first: a call and its observation are separate steps, and agents may
    # batch several calls before any of them come back.
    outcome: dict[str, bool] = {}
    for step in steps:
        if step.kind == TOOL_RESULT and step.call_id:
            outcome[step.call_id] = bool(step.ok)

    last_mutation = -1
    for index, step in enumerate(steps):
        if step.kind == TOOL_CALL and step.tool in mutating:
            last_mutation = index

    observed_after = 0
    seen_before: dict[str, bool] = {}  # command -> did it fail at least once before a change
    rechecked: set[str] = set()
    repaired: set[str] = set()
    mutations_so_far = 0

    for index, step in enumerate(steps):
        if step.kind != TOOL_CALL:
            continue
        if step.tool in mutating:
            mutations_so_far += 1
            continue
        key = _key(step.tool or "", step.args or {})
        ok = outcome.get(step.call_id or "")
        if ok is None:
            # A call the agent never saw the result of. It cannot be evidence of anything;
            # the runner appends real results, so this only happens on a truncated episode.
            continue

        if key in seen_before:
            rechecked.add(key)
            # A repair needs the earlier run to have FAILED and something to have changed
            # between them. Without the mutation requirement a flaky command that failed
            # and then passed on its own would read as a repair the agent performed.
            if seen_before[key] and ok and mutations_so_far > 0:
                repaired.add(key)
        seen_before[key] = seen_before.get(key, False) or not ok

        if last_mutation >= 0 and index > last_mutation and ok:
            observed_after += 1

    return VerificationEvidence(
        observed_after_mutation=observed_after,
        rechecked=len(rechecked),
        demonstrated_repairs=len(repaired),
        mutated=last_mutation >= 0,
    )


def at_least_as_strong(candidate: VerificationEvidence, baseline: VerificationEvidence) -> tuple[bool, str]:
    """Whether `candidate` verified no less than `baseline`. Returns (verdict, reason).

    Dominance over every component, not a weighted total. There is no defensible exchange
    rate between "ran the failing test again and watched it pass" and "looked at the file
    twice", so a candidate that is stronger on one and weaker on another is **refused as
    incomparable** rather than resolved by a coefficient nobody agreed to. Refusing is the
    conservative direction: the rule exists to stop a candidate winning on efficiency by
    checking less, and an unresolved comparison is not evidence that it did not.
    """
    if not baseline.mutated:
        # Nothing changed state, so there was nothing to verify and no bar to clear.
        return True, ""
    weaker = [
        name
        for name, c, b in (
            ("demonstrated_repairs", candidate.demonstrated_repairs, baseline.demonstrated_repairs),
            ("rechecked", candidate.rechecked, baseline.rechecked),
            ("observed_after_mutation", candidate.observed_after_mutation, baseline.observed_after_mutation),
        )
        if c < b
    ]
    if not weaker:
        return True, ""
    return False, (
        "verified less than the baseline on " + ", ".join(weaker) + "; the cheapest way to spend fewer "
        "tokens is to stop checking, so an efficiency win that costs verification is refused"
    )


def unmeasured_claim_count(trajectory: AgentTrajectory) -> int:
    """Numbers in the final answer that appear in no observed tool result.

    The one existing check built the right way round -- a claim anchored to validator-observed
    output rather than to the agent's say-so. Reused here rather than reimplemented; it is
    WARNING-only in `hermesbench.integrity` and is a real component of how much a trajectory
    is standing behind.
    """
    from hermesbench.integrity import check_unmeasured_claims

    final = next((s.content for s in reversed(trajectory.steps) if s.kind == FINAL), "")
    if not final.strip():
        return 0
    return len(check_unmeasured_claims(trajectory))


__all__ = ["VerificationEvidence", "at_least_as_strong", "evidence", "unmeasured_claim_count"]
