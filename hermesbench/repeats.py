"""Repeated runs, and the interval a single-shot number does not have.

`run_suite` runs each task once. On a fifteen-task suite one task flipping moves the
success rate by nearly seven points, and on v0 alone it moves it by thirty-three. A single
number off a suite that small is not reproducible even when everything upstream is pinned:
the error bar swamps the signal, and reporting the point estimate alone invites a
comparison the data cannot support.

Two things are needed and they are different. **Repeats** measure how much a model's own
sampling moves the result -- run the same task five times and count how often it passes.
**An interval** says how precisely the suite locates the rate, given how few tasks there
are; that one is a property of the suite size and does not shrink by rerunning.

Both are reported, because they answer different objections. "Would this number come out
the same tomorrow" is answered by repeats. "Is this model better than that one" is answered
by whether the intervals overlap, and on a fifteen-task suite they almost always will --
which is the honest finding and the reason to grow the suite rather than to rerun it.

The interval is Wilson rather than normal-approximation. At n=15 the normal interval is
simply wrong: it is symmetric around the estimate, so at a success rate of 1.0 it produces
a zero-width interval claiming perfect certainty from fifteen observations.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# 95% two-sided.
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class Interval:
    """A proportion with the range the observations actually support."""

    estimate: float
    low: float
    high: float
    n: int
    confidence: float = 0.95

    @property
    def width(self) -> float:
        return self.high - self.low

    def overlaps(self, other: Interval) -> bool:
        """Whether two rates are distinguishable at this confidence.

        Non-overlapping intervals are sufficient for a difference, not necessary -- two
        overlapping intervals can still differ significantly. Reported this way round on
        purpose: the useful direction is refusing to claim a difference, and this is the
        conservative test for that.
        """
        return self.low <= other.high and other.low <= self.high

    def to_record(self) -> dict[str, Any]:
        return {
            "estimate": round(self.estimate, 4),
            "low": round(self.low, 4),
            "high": round(self.high, 4),
            "width": round(self.width, 4),
            "n": self.n,
            "confidence": self.confidence,
        }


def wilson(successes: int, n: int, *, z: float = Z_95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Chosen over the normal approximation because this suite is small and the rates are
    near the boundaries, which is exactly where the normal interval fails: at 15/15 it
    reports a width of zero, claiming certainty from fifteen observations.
    """
    if n <= 0:
        raise ValueError("an interval over no observations describes nothing")
    if not 0 <= successes <= n:
        raise ValueError(f"{successes} successes out of {n} is not a proportion")
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return Interval(estimate=p, low=max(0.0, centre - spread), high=min(1.0, centre + spread), n=n)


@dataclass(frozen=True)
class TaskRepeats:
    """One task, run several times."""

    task_id: str
    passes: int
    attempts: int

    @property
    def rate(self) -> float:
        return self.passes / self.attempts if self.attempts else 0.0

    @property
    def flaky(self) -> bool:
        """Passed sometimes and failed sometimes.

        Worth naming rather than averaging away: a task that flips between runs is
        measuring sampling noise, and a suite where many do cannot separate two models
        however many times it is rerun.
        """
        return 0 < self.passes < self.attempts

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "passes": self.passes,
            "attempts": self.attempts,
            "rate": round(self.rate, 4),
            "flaky": self.flaky,
        }


@dataclass(frozen=True)
class RepeatedSuite:
    """A suite run several times, with what that does and does not establish."""

    tasks: tuple[TaskRepeats, ...]
    repeats: int

    @property
    def flaky_tasks(self) -> tuple[TaskRepeats, ...]:
        return tuple(t for t in self.tasks if t.flaky)

    @property
    def mean_rate(self) -> float:
        return sum(t.rate for t in self.tasks) / len(self.tasks) if self.tasks else 0.0

    @property
    def interval(self) -> Interval:
        """Interval over *tasks*, not over attempts.

        Counting every attempt as an independent observation would divide the width by the
        square root of the repeat count, which is a claim that rerunning the same fifteen
        tasks tells you more about the population of tasks. It does not: repeats measure
        sampling noise, and the suite's size is what bounds the precision.
        """
        solved = sum(1 for t in self.tasks if t.rate >= 0.5)
        return wilson(solved, len(self.tasks))

    def to_record(self) -> dict[str, Any]:
        return {
            "repeats": self.repeats,
            "tasks": len(self.tasks),
            "mean_rate": round(self.mean_rate, 4),
            "flaky_tasks": [t.task_id for t in self.flaky_tasks],
            "interval": self.interval.to_record(),
            "per_task": [t.to_record() for t in self.tasks],
        }


def repeat_suite(task_ids: list[str], run: Callable[[str, int], bool], *, repeats: int) -> RepeatedSuite:
    """Run each task `repeats` times and record how often it passed.

    `run` receives the task id and the attempt index, so a caller can vary a seed without
    this module knowing what a seed is.
    """
    if repeats < 1:
        raise ValueError("a suite must be run at least once")
    results = []
    for task_id in task_ids:
        passes = sum(1 for attempt in range(repeats) if run(task_id, attempt))
        results.append(TaskRepeats(task_id=task_id, passes=passes, attempts=repeats))
    return RepeatedSuite(tasks=tuple(results), repeats=repeats)


def distinguishable(a: Interval, b: Interval) -> tuple[bool, str]:
    """Whether two suite results support a claim that one model is better.

    Returns a reason when they do not, because "these two numbers are not distinguishable
    at this suite size" is the finding, and publishing the difference anyway is how a
    fifteen-task benchmark becomes a marketing chart.
    """
    if a.overlaps(b):
        return False, (
            f"intervals overlap ({a.low:.3f}-{a.high:.3f} vs {b.low:.3f}-{b.high:.3f}) at n={a.n} and n={b.n}; "
            "the suite is too small to separate these"
        )
    return True, ""


@dataclass(frozen=True)
class PairedComparison:
    """Two models on the SAME tasks, compared on the tasks where they differed.

    `wilson` and `distinguishable` compare two rates as if the two runs were independent
    samples. They are not. Every teacher runs every task -- that is the fair-fight invariant
    `Tournament` enforces -- so the two results are paired, and the pairing carries most of
    the information. Discarding it is what makes a fifteen-task suite look hopeless: the
    unpaired intervals reflect uncertainty about *the suite's difficulty*, which is shared by
    both models and therefore cancels.

    Concretely: if two models pass exactly the same eleven tasks and fail the same four, the
    unpaired intervals are identical and overlapping, and the honest answer is "no evidence
    of a difference". If one passes eleven and the other passes a *different* eleven, the
    unpaired intervals are still identical -- but eight tasks disagree, and that is real
    evidence they are not the same model. Only the paired view can tell those apart.
    """

    both_passed: int
    both_failed: int
    only_a: int
    only_b: int

    @property
    def n(self) -> int:
        return self.both_passed + self.both_failed + self.only_a + self.only_b

    @property
    def discordant(self) -> int:
        """The tasks that carry the signal. Agreements tell you nothing about the ordering."""
        return self.only_a + self.only_b

    def to_record(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "both_passed": self.both_passed,
            "both_failed": self.both_failed,
            "only_a": self.only_a,
            "only_b": self.only_b,
            "discordant": self.discordant,
            "p_value": round(self.p_value, 6),
        }

    @property
    def p_value(self) -> float:
        """Exact two-sided McNemar, computed as a binomial sign test on the discordant pairs.

        Exact rather than the chi-square approximation, which needs the discordant count to
        be comfortably above about 25. Ours will be single digits for a long time, and the
        approximation is anti-conservative exactly there -- it would report significance a
        fifteen-task suite cannot support, which is the error this module exists to prevent.

        With no discordant pairs the models made identical decisions everywhere and p is 1.0:
        no evidence of a difference, which is not the same as evidence of no difference.
        """
        k, n = min(self.only_a, self.only_b), self.discordant
        if n == 0:
            return 1.0
        # P(X <= k) + P(X >= n-k) under X ~ Binomial(n, 0.5), which for the symmetric null
        # is twice the lower tail. Clamped because at k == n/2 the two tails overlap and the
        # doubled sum exceeds one.
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
        return min(1.0, 2 * tail)


def paired(a: dict[str, bool], b: dict[str, bool]) -> PairedComparison:
    """Build the comparison from two {task_id: passed} maps.

    Refuses a task present in one and not the other. A pair is only a pair if both models
    were asked the same question, and silently intersecting the keys would compute a real
    number over a set the caller never chose -- which is the failure mode that makes a
    paired test look stronger than the data it ran on.
    """
    missing = set(a) ^ set(b)
    if missing:
        raise ValueError(
            f"tasks {sorted(missing)} appear in one result and not the other; a paired test needs "
            "both models on the same tasks, and dropping the difference measures a suite nobody ran"
        )
    both_passed = sum(1 for t in a if a[t] and b[t])
    both_failed = sum(1 for t in a if not a[t] and not b[t])
    only_a = sum(1 for t in a if a[t] and not b[t])
    only_b = sum(1 for t in a if not a[t] and b[t])
    return PairedComparison(both_passed=both_passed, both_failed=both_failed, only_a=only_a, only_b=only_b)


def paired_distinguishable(comparison: PairedComparison, *, alpha: float = 0.05) -> tuple[bool, str]:
    """Whether the paired data support a claim that one model is better.

    The paired counterpart of `distinguishable`, and strictly more powerful on the same
    runs -- but it can still only report what the discordant count supports. On a suite this
    size that is usually "not yet", and saying so is the point.
    """
    if comparison.discordant == 0:
        return False, (
            f"the two models made identical decisions on all {comparison.n} tasks; there is no "
            "evidence of a difference, which is not evidence that there is none"
        )
    if comparison.p_value > alpha:
        # The floor is worth naming: below six discordant pairs no split can reach p<=0.05,
        # so the suite forecloses the claim before any data is collected.
        floor = (
            ""
            if comparison.discordant >= 6
            else (
                f"; with only {comparison.discordant} discordant tasks no split can reach p<={alpha:g}, "
                "so this suite cannot separate them however they perform"
            )
        )
        return False, (
            f"exact McNemar p={comparison.p_value:.3f} on {comparison.discordant} discordant tasks "
            f"({comparison.only_a} vs {comparison.only_b}){floor}"
        )
    return True, ""
