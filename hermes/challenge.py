"""Turning a baseline failure into something a miner can compete on.

A challenge is what the flywheel starts from: the frozen model attempted a task and failed,
looped, or succeeded far outside its resource envelope. Until something packages that, a
baseline run is a JSON file somebody reads by hand and the market has no input.

Three properties, and each exists because its absence has a specific cost.

**Confirmed over repeats, never opened on one run.** A single failure is often the sampling
noise of a stochastic decoder, and a challenge opened on it sends every miner in the subnet
to work on a task the baseline solves four times in five. `Baseline.confirmed` reports the
Wilson bound on the true pass rate rather than a count, because "failed once" and "fails
reliably" are the same integer and very different facts. The bound is the same arithmetic
`hermes.acceptance` uses for the other direction: at one attempt, observed 0/1 leaves the
true pass rate anywhere up to 79%.

**Classified from the trace, not asserted.** The failure class decides what a miner
optimises, so a mislabelled challenge wastes the whole round. Every class here is a
predicate over `EpisodeMetrics` and the step sequence -- nothing is passed in by a caller
who already believes the answer.

**Carries the commitment, never the withheld check.** A challenge is published to miners.
The withheld check is the only reason `overfit_rate` measures anything, so the packet
carries `hidden_verify_commitment` from the task and the body stays in the private tree. A
challenge that shipped the check would hand every miner the answer key on the way in.

## What a challenge deliberately does not contain

No acceptance thresholds. `MIN_TOKEN_REDUCTION` and the reduction interval in
`hermes.acceptance` are uncalibrated until a real baseline exists -- the run that produces
challenges is the same run that measures the spread. Baking a threshold into the packet
would freeze a guess at exactly the moment the data to replace it arrives. The packet
records the observed resource envelope and the gate reads the spread from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any

from hermes.harness import digest_mapping

# Why a task became a challenge. Ordered by what a miner would work on first: a task the
# baseline cannot do at all is worth more than one it does expensively.
PUBLIC_VERIFY_FAILED = "public_verify_failed"
HIDDEN_VERIFY_FAILED = "hidden_verify_failed"
OVERFIT = "overfit"
MALFORMED_PROTOCOL = "malformed_protocol"
NO_PROGRESS_LOOP = "no_progress_loop"
STEP_BUDGET_EXHAUSTED = "step_budget_exhausted"
SETUP_FAILED = "setup_failed"
EXPENSIVE_SUCCESS = "expensive_success"

# Not a challenge. Named so the classifier can return something rather than None, and so a
# caller cannot mistake "nothing wrong" for "unclassified".
HEALTHY = "healthy"


class ChallengeError(ValueError):
    """A challenge cannot be built from what was supplied."""


@dataclass(frozen=True)
class Attempt:
    """One baseline episode, reduced to what a challenge needs.

    A projection of `EpisodeMetrics` rather than the thing itself, so packaging does not
    drag the whole bench into anything that reads a challenge.
    """

    public_passed: bool
    hidden_passed: bool | None
    tokens: int
    tool_calls: int
    wall_time_s: float
    steps: int
    max_steps_hit: bool = False
    setup_failed: bool = False
    malformed_turns: int = 0
    repeated_actions: int = 0

    @property
    def verified(self) -> bool:
        """Passed what is published AND was not caught by what is withheld."""
        return self.public_passed and self.hidden_passed is not False

    @classmethod
    def from_metrics(cls, metrics: Any, *, repeated_actions: int = 0) -> Attempt:
        return cls(
            public_passed=bool(metrics.public_passed),
            hidden_passed=metrics.hidden_passed,
            tokens=int(metrics.tokens_used),
            tool_calls=int(metrics.tool_calls),
            wall_time_s=float(metrics.wall_time_s),
            steps=int(metrics.steps),
            max_steps_hit=bool(metrics.max_steps_hit),
            setup_failed=bool(metrics.setup_failed),
            malformed_turns=int(getattr(metrics, "malformed_turns", 0)),
            repeated_actions=repeated_actions,
        )


def classify(attempt: Attempt, *, envelope_tokens: int | None = None, envelope_multiple: float = 2.0) -> str:
    """Why this attempt is a challenge, or HEALTHY.

    Ordered deliberately. `setup_failed` first because it is infrastructure breakage
    masquerading as an agent failure -- a challenge opened on it sends miners to fix a
    broken workspace. `overfit` before `hidden_verify_failed` because the two are the same
    exit status and completely different findings: one is a model that could not do the
    task, the other is a model that learned the published check.
    """
    if attempt.setup_failed:
        return SETUP_FAILED
    if attempt.malformed_turns:
        # Before correctness: a run the protocol could not read was never really scored, and
        # the fix is the wire format rather than the strategy.
        return MALFORMED_PROTOCOL
    if attempt.public_passed and attempt.hidden_passed is False:
        return OVERFIT
    if attempt.hidden_passed is False:
        return HIDDEN_VERIFY_FAILED
    if not attempt.public_passed:
        if attempt.max_steps_hit:
            return STEP_BUDGET_EXHAUSTED
        if attempt.repeated_actions >= 3:
            return NO_PROGRESS_LOOP
        return PUBLIC_VERIFY_FAILED
    if envelope_tokens and attempt.tokens > envelope_multiple * envelope_tokens:
        return EXPENSIVE_SUCCESS
    return HEALTHY


@dataclass(frozen=True)
class Baseline:
    """Repeated attempts by the frozen model under the canonical strategy."""

    task_id: str
    attempts: tuple[Attempt, ...]

    def __post_init__(self) -> None:
        if not self.attempts:
            raise ChallengeError(f"{self.task_id}: a baseline with no attempts establishes nothing")

    @property
    def passes(self) -> int:
        return sum(1 for a in self.attempts if a.verified)

    @property
    def pass_rate(self) -> float:
        return self.passes / len(self.attempts)

    @property
    def bound(self) -> tuple[float, float]:
        """95% interval on the true pass rate, from the repo's own Wilson implementation."""
        from hermesbench.repeats import wilson

        i = wilson(self.passes, len(self.attempts))
        return round(i.low, 4), round(i.high, 4)

    @property
    def median_tokens(self) -> int:
        return int(median([a.tokens for a in self.attempts]))

    @property
    def token_spread(self) -> float:
        """Observed relative spread, which is what an acceptance margin has to beat.

        Recorded here because the run that produces challenges is the only run that can
        measure it, and `hermes.acceptance` refuses a margin the spread swamps.
        """
        toks = [a.tokens for a in self.attempts]
        mid = median(toks)
        if len(toks) < 2 or mid <= 0:
            return float("inf")
        return round((max(toks) - min(toks)) / (2.0 * mid), 4)

    def confirmed(self, *, max_pass_rate: float = 0.5, min_attempts: int = 5) -> tuple[bool, str]:
        """Whether this is reliably a failure, or one unlucky run. Returns (verdict, reason).

        A single failure is often the sampling noise of a stochastic decoder. Opening a
        challenge on it sends every miner in the subnet to work on a task the baseline
        solves most of the time, and the round produces nothing anyone can learn from.
        """
        low, high = self.bound
        if len(self.attempts) < min_attempts:
            return False, (
                f"{self.passes}/{len(self.attempts)} attempts leaves the true pass rate anywhere in "
                f"[{low:.0%}, {high:.0%}]; a challenge opened on that may be one unlucky sample. "
                f"{min_attempts} attempts is the floor."
            )
        if self.pass_rate > max_pass_rate:
            return False, (
                f"the baseline passes {self.pass_rate:.0%} of attempts, above the {max_pass_rate:.0%} bar; "
                "this is a flaky task rather than a capability gap, and miners cannot tell the difference "
                "from inside a round"
            )
        return True, ""

    def dominant_class(self, *, envelope_tokens: int | None = None) -> str:
        """The class most attempts fell into. Ties resolve toward the earlier, worse class."""
        order = [
            SETUP_FAILED,
            MALFORMED_PROTOCOL,
            OVERFIT,
            HIDDEN_VERIFY_FAILED,
            STEP_BUDGET_EXHAUSTED,
            NO_PROGRESS_LOOP,
            PUBLIC_VERIFY_FAILED,
            EXPENSIVE_SUCCESS,
            HEALTHY,
        ]
        seen = [classify(a, envelope_tokens=envelope_tokens) for a in self.attempts]
        counts = {c: seen.count(c) for c in set(seen)}
        best = max(counts.values())
        return next(c for c in order if counts.get(c) == best)


# What a challenge may say about its task. An allowlist, for the same reason
# `hermes.miner_contract` is one: a denylist that filtered `hidden_verify` out would keep
# working right up until somebody added a second withheld field, and the failure mode is
# silent publication of the answer key.
#
# A caller handing in a whole task record is the expected case rather than an abuse -- the
# packet is the thing that gets published, so the packet does the stripping. Relying on
# every caller to pre-filter is the arrangement `redact_for_release` already exists because
# nobody reliably does.
PUBLISHABLE_TASK_KEYS = frozenset(
    {
        "task_id",
        "category",
        "prompt",
        "setup",
        "verify",
        "max_steps",
        "timeout_s",
        "env",
        "fingerprint",
        "lineage",
        "split",
        "hidden_verify_commitment",
        "has_hidden_tests",
        "declares_hidden_tests",
    }
)


@dataclass(frozen=True)
class Challenge:
    """One published, immutable challenge."""

    task_id: str
    failure_class: str
    baseline: Baseline
    epoch: dict[str, Any]
    task_pins: dict[str, Any] = field(default_factory=dict)

    @property
    def published_pins(self) -> dict[str, Any]:
        """Task fields this packet may carry. Anything unrecognised is dropped."""
        return {k: v for k, v in sorted(self.task_pins.items()) if k in PUBLISHABLE_TASK_KEYS}

    @property
    def dropped_task_keys(self) -> tuple[str, ...]:
        """Names of the fields stripped on the way out. Names only, never values.

        Reported so a maintainer who passed a full task record can see that the strip
        happened, instead of having to trust that it did.
        """
        return tuple(sorted(k for k in self.task_pins if k not in PUBLISHABLE_TASK_KEYS))

    @property
    def digest(self) -> str:
        """Content address. Excludes nothing, so two identical packets are one challenge."""
        return digest_mapping(self.to_record(with_digest=False))

    def to_record(self, *, with_digest: bool = True) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema_version": "spark-challenge-v1",
            "task_id": self.task_id,
            "failure_class": self.failure_class,
            "epoch": dict(sorted(self.epoch.items())),
            "task": self.published_pins,
            "baseline": {
                "attempts": len(self.baseline.attempts),
                "verified_passes": self.baseline.passes,
                "pass_rate": round(self.baseline.pass_rate, 4),
                "true_pass_rate_interval": list(self.baseline.bound),
                "median_tokens": self.baseline.median_tokens,
                "median_tool_calls": int(median([a.tool_calls for a in self.baseline.attempts])),
                "median_steps": int(median([a.steps for a in self.baseline.attempts])),
                # Recorded, never scored across nodes. Wall time is not reproducible, and a
                # bar that only rises would lock in whichever run got favourable scheduling.
                "median_wall_time_s": round(median([a.wall_time_s for a in self.baseline.attempts]), 3),
                "token_spread": self.baseline.token_spread,
            },
            # What a miner may NOT have, stated in the packet so nobody has to infer it from
            # an absence. The commitment proves which check will be used without revealing it.
            "withheld": {
                "hidden_verify_commitment": str(self.task_pins.get("hidden_verify_commitment") or ""),
                "body_included": False,
                "dropped_task_keys": list(self.dropped_task_keys),
            },
            "acceptance_thresholds_included": False,
            "why_no_thresholds": (
                "the run that produces challenges is the run that measures the token spread, so a "
                "threshold baked in here would freeze a guess at the moment the data to replace it "
                "arrives. hermes.acceptance reads the spread from baseline.token_spread instead."
            ),
        }
        if with_digest:
            record["challenge_digest"] = self.digest
        return record


def open_challenge(
    baseline: Baseline,
    *,
    epoch: dict[str, Any],
    task_pins: dict[str, Any] | None = None,
    envelope_tokens: int | None = None,
    max_pass_rate: float = 0.5,
    min_attempts: int = 5,
) -> Challenge:
    """Package a confirmed baseline failure. Refuses an unconfirmed one.

    Raising rather than returning a flag: an unconfirmed challenge that reaches miners costs
    a whole round, and the caller has no better information than this function does.
    """
    failure_class = baseline.dominant_class(envelope_tokens=envelope_tokens)

    # Classification-driven refusals come before the confirmation gate, because the gate
    # would otherwise misdiagnose them. A baseline that passes every attempt inside its
    # envelope trips the pass-rate bar and gets reported as "a flaky task" -- which is
    # exactly backwards, and the maintainer reading it goes looking for nondeterminism that
    # isn't there.
    if failure_class == SETUP_FAILED:
        raise ChallengeError(
            f"{baseline.task_id}: the task's own setup failed, which is infrastructure breakage rather "
            "than an agent failure. A challenge here sends miners to fix a broken workspace."
        )
    # Both remaining special cases require the baseline to pass CONSISTENTLY, and that
    # condition is doing real work rather than being belt-and-braces. `dominant_class` is a
    # majority vote, so a baseline that passes 7 of 10 returns HEALTHY -- and refusing that
    # as "nothing to improve" would bury a task that fails almost a third of the time.
    # Mixed baselines fall through to the confirmation gate, which names the reliability
    # problem instead.
    if baseline.pass_rate == 1.0:
        if failure_class == HEALTHY:
            raise ChallengeError(
                f"{baseline.task_id}: the baseline passes all {len(baseline.attempts)} attempts inside "
                "its resource envelope; there is nothing for a miner to improve and a challenge opened "
                "on it wastes a round"
            )
        if failure_class == EXPENSIVE_SUCCESS:
            # A resource challenge, so the pass-rate bar cannot apply -- it would refuse the
            # packet for the very thing that defines it. The attempt-count floor still does:
            # the envelope is a median, and one sample has no median worth publishing.
            if len(baseline.attempts) < min_attempts:
                raise ChallengeError(
                    f"{baseline.task_id}: "
                    f"{baseline.confirmed(max_pass_rate=max_pass_rate, min_attempts=min_attempts)[1]}"
                )
            return Challenge(
                task_id=baseline.task_id,
                failure_class=failure_class,
                baseline=baseline,
                epoch=epoch,
                task_pins=dict(task_pins or {}),
            )

    ok, reason = baseline.confirmed(max_pass_rate=max_pass_rate, min_attempts=min_attempts)
    if not ok:
        raise ChallengeError(f"{baseline.task_id}: {reason}")

    return Challenge(
        task_id=baseline.task_id,
        failure_class=failure_class,
        baseline=baseline,
        epoch=epoch,
        task_pins=dict(task_pins or {}),
    )


__all__ = [
    "EXPENSIVE_SUCCESS",
    "HEALTHY",
    "HIDDEN_VERIFY_FAILED",
    "MALFORMED_PROTOCOL",
    "NO_PROGRESS_LOOP",
    "OVERFIT",
    "PUBLIC_VERIFY_FAILED",
    "SETUP_FAILED",
    "STEP_BUDGET_EXHAUSTED",
    "PUBLISHABLE_TASK_KEYS",
    "Attempt",
    "Baseline",
    "Challenge",
    "ChallengeError",
    "classify",
    "open_challenge",
]
