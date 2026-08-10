"""The acceptance bar: correctness first, then efficiency, and only above the noise.

Three gates in order, and the order is the whole design. A candidate that is cheaper and
wrong is not a candidate, so nothing about tokens is computed until correctness has passed.

    1. correctness   the withheld check passes, every attempt, with enough attempts
    2. verification  the candidate did not get cheaper by checking less
    3. efficiency    a margin that exceeds the measured run-to-run spread

## "100% success" is not a bar until it says how many attempts

Observed n-out-of-n, and what it bounds the true rate to at 95%:

     1/1  ->  20.7%     a model that truly fails four times in five can show 100%
     3/3  ->  43.9%
     5/5  ->  56.6%
    10/10 ->  72.2%
    20/20 ->  83.9%

So a one-shot 100% is close to no information. `MIN_ATTEMPTS` is the smallest count where
the bound clears a half-decent rate, and the gate reports the interval alongside the verdict
so nobody reads "100%" as certainty.

## A 20% token gate on single runs is mostly noise

Two draws from the *same* distribution -- no real improvement whatsoever -- clear a 20%
reduction this often:

    run-to-run CV     P(apparent 20% win)     paired repeats to get under 5%
        5%                    0.1%                        1
       10%                    6.2%                        3
       15%                   14.7%                        5
       20%                   21.8%                        7
       30%                   30.1%                       21

**The spread is unknown until a baseline run exists**, so a fixed threshold cannot be
calibrated on its own. This gate therefore does not trust the constant: it asks whether the
margin survives the noise in the data it was measured from.

## How it asks that, and the version of this that was wrong

The first implementation compared the margin against `2.0 x observed_spread`. The baseline run
of 2026-08-10 showed why that is not merely conservative but broken. Measured per-task token
spreads ranged from 7.3% to 98.3%, and at the top of that range the rule demanded a **196.6%**
token reduction. A reduction cannot exceed 100%, so the gate was unsatisfiable: verified by
running it, a candidate that cut tokens by 99.9% with a perfectly tight distribution was still
refused. The noisiest tasks were permanently unwinnable, which is worse than a mis-set
threshold -- it silently removed them from the competition.

The statistical error is specific. Raw spread does not shrink with more repeats; the
*uncertainty in the median* does, roughly as one over root n. So comparing a margin against
raw spread conflates "this task is noisy" with "we cannot tell whether this margin is real",
and the table above already says they are different things: at 30% variation, twenty-one
paired repeats bring the false-positive rate under 5%. The code contradicted its own
docstring.

So the gate now bootstraps a confidence interval on the reduction itself and requires its
**lower bound** to clear the bar. That has the properties the multiple was reaching for and
the one it lacked:

    high spread          -> wide interval -> low lower bound -> refused
    more paired repeats  -> tighter interval -> lower bound rises -> satisfiable
    reduction <= 100%    -> the requirement can never exceed what is achievable

A refusal now means "not yet distinguishable from noise, run more repeats", which is
actionable, rather than "impossible", which was not.

## Tool calls: in the vector, never a bounty

A separate bounty on tool-call reduction pays for the one metric that is trivially gameable.
A `SKILL.md` saying "write a helper that greps, tests and summarises in one pass, then run
it" turns thirty operations into one call, and a bounty makes that the highest-value move on
the board. Tokens and wall time are not fooled the same way -- the hidden work still costs
seconds and its output still enters context -- so tool calls stay a reported dimension with
no separate purse.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import median
from typing import Any

# Smallest attempt count where n-of-n bounds the true rate above 70% at 95%. Below this,
# "100% success" is a statement about luck.
MIN_ATTEMPTS = 10

# The margins the design asks for. Starting points, not calibrated -- `Gate.decide` refuses
# a margin the observed spread cannot support regardless of what these say.
MIN_TOKEN_REDUCTION = 0.20
MIN_TOOL_CALL_REDUCTION = 2

# Bootstrap settings for the interval on the reduction. Seeded, because an acceptance decision
# that differs between two runs over identical data is not an acceptance decision -- and the
# whole point of this module is that the same evidence yields the same verdict for everyone.
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260810
CONFIDENCE = 0.95


class AcceptanceError(ValueError):
    """A submission cannot be judged from what was supplied."""


@dataclass(frozen=True)
class Arm:
    """One side of the comparison: repeated runs of one strategy on one task."""

    passes: int
    attempts: int
    tokens: tuple[int, ...]
    tool_calls: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.attempts <= 0:
            raise AcceptanceError("an arm with no attempts cannot be compared")
        if self.passes > self.attempts:
            raise AcceptanceError(f"{self.passes} passes out of {self.attempts} attempts")

    @property
    def all_passed(self) -> bool:
        return self.passes == self.attempts

    @property
    def token_spread(self) -> float:
        """Observed relative spread of token counts, as a fraction of the median.

        Half the interquartile-ish range over the median: robust to one outlier and
        computable from three runs. This is the number a margin has to beat, and it is
        measured rather than assumed because nobody has run the baseline yet.
        """
        if len(self.tokens) < 2:
            return float("inf")
        mid = median(self.tokens)
        if mid <= 0:
            return float("inf")
        return (max(self.tokens) - min(self.tokens)) / (2.0 * mid)


def reduction_interval(
    baseline_tokens: tuple[int, ...],
    candidate_tokens: tuple[int, ...],
    *,
    confidence: float = CONFIDENCE,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile-bootstrap interval on the relative reduction in median tokens.

    Resamples both arms with replacement and recomputes the reduction each time, so the
    interval reflects the uncertainty in both medians rather than in the baseline alone.

    Unlike a multiple of the raw spread, this tightens as attempts accumulate -- which is the
    property that makes a refusal actionable. Bounded above by 1.0 by construction, because a
    reduction is `(base - cand) / base` and no resample can make the candidate negative.

    Seeded. Two people judging the same submission must reach the same verdict, and an
    unseeded bootstrap would make acceptance a coin flip in the fourth decimal place.
    """
    if len(baseline_tokens) < 2 or len(candidate_tokens) < 2:
        # One sample says nothing about its own variability. Returning a degenerate interval
        # would let any margin through; returning this refuses until there are repeats.
        return (-float("inf"), float("inf"))

    rng = random.Random(seed)
    base_n, cand_n = len(baseline_tokens), len(candidate_tokens)
    draws: list[float] = []
    for _ in range(resamples):
        base = median([baseline_tokens[rng.randrange(base_n)] for _ in range(base_n)])
        cand = median([candidate_tokens[rng.randrange(cand_n)] for _ in range(cand_n)])
        if base <= 0:
            continue
        draws.append((base - cand) / base)
    if not draws:
        return (-float("inf"), float("inf"))
    draws.sort()
    tail = (1.0 - confidence) / 2.0
    lo = draws[min(len(draws) - 1, int(tail * len(draws)))]
    hi = draws[min(len(draws) - 1, int((1.0 - tail) * len(draws)))]
    return (round(lo, 4), round(hi, 4))


@dataclass(frozen=True)
class Decision:
    accepted: bool
    reasons: tuple[str, ...]
    token_reduction: float | None = None
    tool_call_reduction: int | None = None
    true_rate_lower_bound: float | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "token_reduction": self.token_reduction,
            "tool_call_reduction": self.tool_call_reduction,
            "candidate_true_success_rate_lower_bound": self.true_rate_lower_bound,
            # Stated so a reader does not turn a reported reduction into a claim the
            # measurement cannot support.
            "tool_calls_are_a_reported_dimension_not_a_bounty": True,
        }


def decide(
    *,
    candidate: Arm,
    baseline: Arm,
    verification_ok: bool = True,
    verification_reason: str = "",
    min_attempts: int = MIN_ATTEMPTS,
    min_token_reduction: float = MIN_TOKEN_REDUCTION,
) -> Decision:
    """Judge one candidate against one baseline. Every failing reason, not the first.

    Correctness is evaluated before anything else is computed, and a failure there returns
    immediately: reporting a token reduction beside a wrong answer invites someone to quote
    the number.
    """
    from hermesbench.repeats import wilson

    reasons: list[str] = []
    bound = wilson(candidate.passes, candidate.attempts)

    # --- gate 1: correctness -------------------------------------------------------------
    if not candidate.all_passed:
        return Decision(
            accepted=False,
            reasons=(
                f"candidate passed {candidate.passes} of {candidate.attempts}; the withheld check must "
                "pass on every attempt before efficiency is considered at all",
            ),
            true_rate_lower_bound=round(bound.low, 4),
        )
    if candidate.attempts < min_attempts:
        return Decision(
            accepted=False,
            reasons=(
                f"{candidate.passes}/{candidate.attempts} is 100% of too few attempts: it bounds the true "
                f"success rate only to {bound.low:.1%} at 95%, so it is a statement about luck. "
                f"{min_attempts} attempts are needed for the bound to mean something.",
            ),
            true_rate_lower_bound=round(bound.low, 4),
        )

    # --- gate 2: verification did not regress --------------------------------------------
    if not verification_ok:
        return Decision(
            accepted=False,
            reasons=(verification_reason or "the candidate verified less than the baseline",),
            true_rate_lower_bound=round(bound.low, 4),
        )

    # --- gate 3: efficiency, above the noise ---------------------------------------------
    base_tokens, cand_tokens = median(baseline.tokens), median(candidate.tokens)
    if base_tokens <= 0:
        raise AcceptanceError("baseline reported no tokens; there is nothing to improve on")
    reduction = (base_tokens - cand_tokens) / base_tokens
    tool_delta = int(median(baseline.tool_calls) - median(candidate.tool_calls))

    spread = max(baseline.token_spread, candidate.token_spread)
    if len(baseline.tokens) < 2 or len(candidate.tokens) < 2:
        # Named separately from a wide interval. "One of these arms has a single measurement"
        # and "the margin is inside the noise" call for different actions, and printing an
        # infinite interval to describe the first invites a reader to think the data was noisy
        # when in fact there was no second observation to be noisy about.
        return Decision(
            accepted=False,
            reasons=(
                f"one arm has too few token measurements to bound a reduction "
                f"(baseline {len(baseline.tokens)}, candidate {len(candidate.tokens)}); a single "
                "measurement carries no information about its own variability, so no interval "
                "can be computed and any margin would be accepted on faith",
            ),
            token_reduction=round(reduction, 4),
            tool_call_reduction=tool_delta,
            true_rate_lower_bound=round(bound.low, 4),
        )

    low, high = reduction_interval(baseline.tokens, candidate.tokens)
    if low < min_token_reduction:
        if reduction >= min_token_reduction:
            # The interesting refusal: the point estimate cleared the bar and the interval did
            # not. Says what to DO about it, because at high spread the answer is more repeats
            # rather than a bigger margin -- the interval narrows with n, the spread does not.
            reasons.append(
                f"token reduction {reduction:.1%} clears the {min_token_reduction:.0%} bar but its 95% "
                f"interval is [{low:.1%}, {high:.1%}], whose lower bound does not (run-to-run spread "
                f"{spread:.1%}). On this evidence the margin cannot be distinguished from noise. The "
                f"interval narrows as one over root n, so more paired repeats resolve it -- "
                f"{len(candidate.tokens)} attempts here."
            )
        else:
            reasons.append(
                f"token reduction {reduction:.1%} is below the {min_token_reduction:.0%} bar "
                f"(95% interval [{low:.1%}, {high:.1%}])"
            )

    if not reasons:
        return Decision(
            accepted=True,
            reasons=(),
            token_reduction=round(reduction, 4),
            tool_call_reduction=tool_delta,
            true_rate_lower_bound=round(bound.low, 4),
        )
    return Decision(
        accepted=False,
        reasons=tuple(reasons),
        token_reduction=round(reduction, 4),
        tool_call_reduction=tool_delta,
        true_rate_lower_bound=round(bound.low, 4),
    )


def dominates(candidate: Arm, incumbent: Arm, *, min_attempts: int = MIN_ATTEMPTS) -> tuple[bool, str]:
    """Whether a challenger may take the crown, on the deterministic metrics only.

    Tokens and tool calls are exact and recomputable from the trace. Wall time is not, and a
    king-of-the-hill bar that only ever rises would lock in whichever run got favourable
    scheduling -- permanently, because no later run could legitimately beat it. Latency is
    reported beside the crown and never gates it.

    ## The attempt floor is the same floor `decide` applies, and for a worse reason

    This function used to check `all_passed` and nothing else about sample size, which meant
    a single passing attempt could take the crown -- while `decide` refused that same arm,
    quoting the bound it implies: 1/1 pins the true success rate no higher than 20.7%. Two
    gates disagreeing about what counts as evidence is bad on its own, but the asymmetry ran
    the wrong way. `decide` awards a one-off acceptance; the crown *persists* and every later
    challenger has to beat it. A king crowned on one lucky sample sets a bar that no honest
    strategy can clear, and it does not decay.

    A round shape that gives each miner one attempt at one task therefore cannot crown
    anybody, which is the intended reading rather than an obstacle: it needs `min_attempts`
    repeats per miner per task, and those repeats are also the only way the run-to-run spread
    `decide` needs ever gets measured.
    """
    from hermesbench.repeats import wilson

    for label, arm in (("challenger", candidate), ("incumbent", incumbent)):
        if arm.attempts < min_attempts:
            bound = wilson(arm.passes, arm.attempts)
            return False, (
                f"the {label} has {arm.passes}/{arm.attempts} attempts, which bounds its true success "
                f"rate only to {bound.low:.1%} at 95%; the crown persists and every later challenger "
                f"must beat it, so it cannot be set from that. {min_attempts} attempts are the floor, "
                "the same one decide() applies."
            )
    if not candidate.all_passed:
        return False, "a challenger that does not pass every attempt cannot hold the crown"
    tokens = median(candidate.tokens) <= median(incumbent.tokens)
    calls = median(candidate.tool_calls) <= median(incumbent.tool_calls)
    if (
        tokens
        and calls
        and (
            median(candidate.tokens) < median(incumbent.tokens)
            or median(candidate.tool_calls) < median(incumbent.tool_calls)
        )
    ):
        return True, ""
    return False, "does not dominate the incumbent on tokens and tool calls"


__all__ = [
    "MIN_ATTEMPTS",
    "MIN_TOKEN_REDUCTION",
    "MIN_TOOL_CALL_REDUCTION",
    "BOOTSTRAP_RESAMPLES",
    "CONFIDENCE",
    "reduction_interval",
    "AcceptanceError",
    "Arm",
    "Decision",
    "decide",
    "dominates",
]
