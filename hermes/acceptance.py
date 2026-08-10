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

The right-hand column is history, not the rule. It comes from a simulation of the fixed-multiple
threshold described below, which was replaced -- so it is here to show why a constant cannot work
and must not be read as what this gate now requires. `repeats_needed` answers that question by
asking the current gate, and its answers are larger: at 30% spread a 25% win projects past 200
paired repeats rather than 21.

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


# Ladder for `repeats_needed`, as multiples of the repeats already run rather than absolute
# counts. A ladder because each rung is a full bootstrap and the answer is advice about scale:
# "about 40" and "about 45" lead a miner to the same decision, so the resolution is not worth
# paying for. Multiples because a rung must replicate the observed sample a whole number of
# times -- see `_replicated`.
REPEAT_MULTIPLES: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20)


def repeats_needed(
    baseline_tokens: tuple[int, ...],
    candidate_tokens: tuple[int, ...],
    *,
    margin: float = MIN_TOKEN_REDUCTION,
    multiples: tuple[int, ...] = REPEAT_MULTIPLES,
    resamples: int = 600,
    seed: int = BOOTSTRAP_SEED,
) -> int | None:
    """Paired repeats at which *this* rule would accept *this* margin. None if beyond the ladder.

    Gate 3's refusal used to end with "more paired repeats resolve it", which is true and useless:
    at 7% spread the answer is a couple and at 98% it is over a hundred, and a miner given no
    number has to guess how much GPU time to buy. The design note in this module's docstring lists
    counts for a few spreads, but those come from a simulation of the rule this gate no longer
    uses, so quoting them beside a bootstrap refusal would point a miner at a table that does not
    describe the gate that refused them.

    So this asks the gate itself. Each rung replicates both observed samples k times and runs the
    real `reduction_interval` over them, returning the first count whose lower bound clears the
    margin.

    ## What it assumes

    Replication holds the measured distribution and the measured margin exactly where they were
    and lets only the interval move, which is the one thing more repeats actually change. The first
    version resampled instead, and that was wrong twice over: a resampled arm is a single draw
    whose median wanders off the observed one, so the projected margin wandered too and a candidate
    sitting on the bar was told "about 25 paired repeats" when no count could have helped. And
    resampling to a length that was not a multiple of the sample size over-weighted whichever
    observations came first, which made the answer non-monotone in n -- the projection got *worse*
    at some larger counts.

    So a count that comes back is conditional on the observed distribution and margin persisting.
    A candidate whose real advantage is smaller than these runs showed will still be refused there.

    ## Why None does not mean "impossible", and the plateau that causes it

    Replication cannot invent information the sample does not have. A bootstrap median can only
    land on a value that was observed, so the interval converges on the gap between the middle
    order statistics rather than on zero, and past that point more replication changes nothing.
    Measured on ten draws at 30% spread with a 27.3% observed win:

        n=10   [ 2.2%, 55.4%]
        n=50   [10.5%, 49.5%]
        n=200  [16.2%, 37.0%]
        n=500  [16.2%, 37.0%]     <- plateau: the 5th and 6th order statistics are 14.8% apart
        n=2000 [16.2%, 37.0%]

    So a real 27.3% win against a 20% bar projects to None here, and that is a limit of the
    projection rather than a fact about the candidate. None therefore means "not projectable from
    this sample" -- more *distinct* runs would move it and this method cannot say how many -- and
    the refusal says exactly that instead of offering advice the number cannot support.
    """
    if len(baseline_tokens) < 2 or len(candidate_tokens) < 2:
        return None
    have = min(len(baseline_tokens), len(candidate_tokens))
    for k in multiples:
        low, _ = reduction_interval(
            _replicated(baseline_tokens, k), _replicated(candidate_tokens, k), resamples=resamples, seed=seed
        )
        if low >= margin:
            return have * k
    return None


def _replicated(values: tuple[int, ...], k: int) -> tuple[int, ...]:
    """Each observation k times over: the same distribution, k times as much of it.

    Element-wise rather than tiling the sequence and truncating. Truncation to a length that is
    not a multiple of the sample size keeps an extra copy of whichever values sit at the front,
    which for an unsorted sample is an arbitrary reweighting -- it moved the projected median by
    7% at some rungs and made the ladder non-monotone.
    """
    return tuple(v for v in values for _ in range(k))


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
            # Says how many, not just "more". At 7% spread the answer is a couple and at 98% it
            # is over a hundred; a miner given only "more" has to guess how much GPU time to buy.
            needed = repeats_needed(baseline.tokens, candidate.tokens, margin=min_token_reduction)
            advice = (
                f"about {needed} paired repeats would settle evidence like this"
                if needed is not None
                else (
                    f"how many repeats would settle it cannot be projected from {len(candidate.tokens)} "
                    "measurements: a bootstrap median can only land on a value that was observed, so the "
                    "interval stops narrowing once it reaches the gap between the middle ones. More "
                    "distinct runs would move it"
                )
            )
            reasons.append(
                f"token reduction {reduction:.1%} clears the {min_token_reduction:.0%} bar but its 95% "
                f"interval is [{low:.1%}, {high:.1%}], whose lower bound does not (run-to-run spread "
                f"{spread:.1%}). On this evidence the margin cannot be distinguished from noise. The "
                f"interval narrows as one over root n, so paired repeats resolve it: "
                f"{len(candidate.tokens)} attempts here, {advice}."
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

    # The token advantage has to survive the noise, not merely show up in the medians.
    #
    # This compared raw medians until an end-to-end run over the first real baseline split each
    # task's ten attempts into two arms of five and asked whether a strategy dominates ITSELF.
    # The true difference is zero by construction. `decide` refused all fifteen; `dominates`
    # crowned six, one of them on a spurious 39.2% "reduction" -- whichever half drew the
    # luckier episodes won. No fixture-based test could catch that, because the fixtures used
    # arms with zero variance and real episodes have plenty.
    #
    # `decide` and `dominates` were fixed in the wrong order. The attempt floor arrived first,
    # which stopped a crown being set from ONE lucky sample and did nothing about a crown set
    # from ten noisy ones. The crown persists and every later challenger must beat it, so a bar
    # placed by chance is worse here than a one-off acceptance made by chance.
    low, _ = reduction_interval(incumbent.tokens, candidate.tokens)
    if low <= 0.0:
        return False, (
            f"the challenger's token advantage does not survive the noise: the 95% interval on "
            f"the reduction has a lower bound of {low:.1%}, so a difference of zero is consistent "
            "with this evidence. The crown persists once taken, so it needs a demonstrated "
            "advantage rather than a favourable median."
        )
    if median(candidate.tool_calls) > median(incumbent.tool_calls):
        # Tool calls remain a non-regression constraint rather than a second purse -- a helper
        # script collapses them trivially, which is why `MIN_TOOL_CALL_REDUCTION` never gates.
        return False, (
            f"the challenger spends more tool calls than the incumbent "
            f"({median(candidate.tool_calls):.0f} vs {median(incumbent.tool_calls):.0f}); the crown "
            "does not move on a trade"
        )
    return True, ""


__all__ = [
    "MIN_ATTEMPTS",
    "MIN_TOKEN_REDUCTION",
    "MIN_TOOL_CALL_REDUCTION",
    "BOOTSTRAP_RESAMPLES",
    "CONFIDENCE",
    "REPEAT_MULTIPLES",
    "_replicated",
    "reduction_interval",
    "repeats_needed",
    "AcceptanceError",
    "Arm",
    "Decision",
    "decide",
    "dominates",
]
