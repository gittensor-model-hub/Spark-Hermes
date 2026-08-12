"""HermesBench scoring.

Every metric here is computed from what the agent *did* -- the trajectory -- plus the
objective verification result. None of them read the agent's final message, because a
metric that trusts the summary measures the model's confidence rather than its work.

Score-unit convention matches `eval.benchmarks`: every rate is a fraction in [0, 1],
never a 0-100 percentage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from hermes.trajectory import TOOL_CALL, TOOL_RESULT, AgentTrajectory
from hermesbench.tasks import DEFAULT_MUTATING_TOOLS


@dataclass(frozen=True)
class EpisodeMetrics:
    """Per-episode measurements. `success` is the verified outcome, not the claim."""

    task_id: str
    success: bool
    tool_calls: int
    failed_calls: int
    # True only when a tool *reported* failure. An agent that appends a diagnostic to a
    # failing command -- `wc -l *.log; echo "exit=$?"` -- makes the shell exit 0, so the
    # harness sees success and this stays False even though the agent visibly recovered.
    # Observed on the first live run, on the task named `recover-from-bad-command`. The
    # limit is real and not closable in general: the harness can only see the status of
    # what the agent chose to run. `recovery_eligible` is reported beside `recovery_rate`
    # so a suite where recovery tasks produced no eligible episodes says so rather than
    # reporting a rate over nothing.
    hit_failure: bool
    recovered: bool
    mutated: bool
    self_checked: bool
    tokens_used: int
    wall_time_s: float
    steps: int
    max_steps_hit: bool = False
    # What the episode cost, when a price book was supplied. `None` means unpriced, which
    # is deliberately not 0.0: an unpriced run reported as free makes the model nobody
    # priced the cheapest one in every comparison. See hermes/cost.py.
    cost: float | None = None
    # Normalised token counts, kept apart from the total so a cache-heavy run is
    # distinguishable from a short one. Both can report the same `tokens_used`.
    usage: dict[str, Any] | None = None
    # The task's own setup command failed, so the agent never ran. Infrastructure
    # breakage, not an agent failure -- scored separately so it cannot masquerade as one.
    setup_failed: bool = False
    # Long-horizon only. Fractions in [0, 1]; -1.0 means "not a long-horizon task", so a
    # short task cannot be averaged in as if it had scored zero on objectives it never had.
    objective_completion: float = -1.0
    objectives_total: int = 0
    objectives_met: int = 0
    # Objectives the agent satisfied and then broke again while working on something
    # else. This is goal drift / context corruption made concrete: not "the agent seemed
    # to lose the plot" but "checkpoint B passed at step 12 and failed at step 30".
    objectives_regressed: int = 0
    # Which Hermes capability this episode exercised, from the task's first tag. Empty
    # for tasks that predate the taxonomy.
    category: str = ""
    # Published checks vs withheld ones. `hidden_passed` is None when the task declares
    # no hidden tests, which is different from having failed them.
    public_passed: bool = False
    hidden_passed: bool | None = None
    # Assistant turns the Hermes parser could not read: a `<tool_call>` holding unparseable
    # JSON, cut off mid-call, or spelled some other way. Counted rather than inferred,
    # because `steps_from_turn` records the failure as a THINKING step and nothing
    # downstream can tell that apart from deliberation.
    #
    # This is the one measurement that protects the wire format. Hermes is upstream and
    # fixed; the product is a better model *for* it, not a variant of it. But the format
    # contract lives inside the system prompt -- `hermes.protocol` spells out `<tool_call>`
    # and the reasoning block in prose the model reads -- so any guidance appended after it
    # can degrade conformance without touching a line of code, and degrading it is
    # *cheaper*: a well-formed call and a planning block both cost tokens. An efficiency
    # score with no conformance term rewards drifting off-protocol.
    malformed_turns: int = 0
    # sha256 of the task's published verify script, stamped at run time.
    #
    # An episode says whether the grader passed and, until this field existed, nothing said
    # WHICH grader. That gap cost two rounds: `fix-failing-test` and `verify-speedup-claim`
    # invoked a bare `python`, failed all ten attempts for a reason the model never caused, and
    # were fixed in a later commit -- leaving a log that reads as a 0/10 capability gap against
    # a grader that now scores 10/10. `hermes.challenge` opened challenges on both.
    #
    # The published script only, so no salt is needed and nothing withheld is digested. That
    # limits what this detects to changes in the public verifier, which is where the observed
    # failure was. Empty means unstamped -- a log written before this field, which callers must
    # treat as unverifiable rather than as matching.
    verify_digest: str = ""
    # Which Hermes wire dialect drove the episode.
    #
    # Beside `verify_digest` for the same reason: a log that does not say how it was produced
    # gets re-read later as though it were comparable. The dialect decides which blocks the
    # model is told to emit, and a run under the wrong one is a run of prose answers that still
    # reports `protocol_clean: true` -- so "0 tool calls" means something entirely different
    # depending on this field, and without it nothing can tell which.
    dialect: str = ""

    @property
    def overfit(self) -> bool:
        """Passed what is published, failed what is withheld: learned the benchmark."""
        return self.public_passed and self.hidden_passed is False

    @property
    def protocol_clean(self) -> bool:
        """Every assistant turn parsed. The floor for a trajectory that may be trained on."""
        return self.malformed_turns == 0

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "success": self.success,
            "public_passed": self.public_passed,
            "hidden_passed": self.hidden_passed,
            "overfit": self.overfit,
            "malformed_turns": self.malformed_turns,
            "protocol_clean": self.protocol_clean,
            "setup_failed": self.setup_failed,
            "objective_completion": round(self.objective_completion, 4),
            "objectives_total": self.objectives_total,
            "objectives_met": self.objectives_met,
            "objectives_regressed": self.objectives_regressed,
            "tool_calls": self.tool_calls,
            "failed_calls": self.failed_calls,
            "hit_failure": self.hit_failure,
            "recovered": self.recovered,
            "mutated": self.mutated,
            "self_checked": self.self_checked,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
            "usage": self.usage,
            "wall_time_s": round(self.wall_time_s, 3),
            "steps": self.steps,
            "max_steps_hit": self.max_steps_hit,
            "verify_digest": self.verify_digest,
            "dialect": self.dialect,
        }


@dataclass(frozen=True)
class SuiteMetrics:
    """Aggregate scores over a bench run.

    `recovery_eligible` and `mutation_eligible` are reported alongside their rates
    because both are conditional: a suite where nothing ever failed has an undefined
    recovery rate, and printing 0.0 without the denominator reads as "never recovers"
    rather than "never had to".

    `episodes` counts episodes that actually ran. Ones whose setup broke are counted in
    `setup_failures` and excluded from every rate.
    """

    success_rate: float
    tool_efficiency: float
    recovery_rate: float
    self_check_rate: float
    mean_tokens: float
    mean_wall_time_s: float
    mean_tool_calls: float
    episodes: int
    recovery_eligible: int
    mutation_eligible: int
    setup_failures: int = 0
    # Long-horizon aggregates, over episodes that declared checkpoints. Reported with
    # their denominator because a suite of only short tasks has no objectives at all --
    # 0.0 there would read as total failure rather than "not applicable".
    objective_completion: float = 0.0
    objective_regression_rate: float = 0.0
    long_horizon_episodes: int = 0
    # success_rate per capability category, with its denominator. Pooling categories
    # hides the case that matters: strong tool selection and hopeless objective
    # maintenance average out to a mediocre single number that describes neither.
    category_success: dict[str, float] = field(default_factory=dict)
    category_support: dict[str, int] = field(default_factory=dict)
    # Of episodes that passed the published checks, the share that failed the withheld
    # ones. This is the saturation signal: a benchmark whose overfit rate climbs over
    # releases is being trained against rather than solved.
    overfit_rate: float = 0.0
    hidden_test_episodes: int = 0
    # Episodes the harness cut off at the step budget, and how many of those also failed.
    #
    # `max_steps_hit` has always been on every episode and was never aggregated, so the headline
    # reported `success_rate` and `mean_tokens` with no sign that most of the run was censored.
    # Measured on two real 19-task runs: 53% and 56% of episodes ended with the harness cutting in
    # rather than the model finishing. That breaks two things at once.
    #
    # `mean_tokens` and `mean_tool_calls` are *upper-censored* for those episodes -- an agent that
    # would have used more is recorded at the cap -- so comparing two runs' efficiency compares
    # truncated distributions, and efficiency is what the promotion gate scores.
    #
    # And a task that ran out of budget is not a task the model cannot do. `truncated_failures` is
    # the number that says how much of `1 - success_rate` might be budget rather than capability.
    # Same principle as `recovery_eligible`: report the denominator rather than let a rate stand in
    # for a measurement it does not describe.
    truncated_episodes: int = 0
    truncated_failures: int = 0
    per_episode: tuple[EpisodeMetrics, ...] = field(default=())
    # Summed over the episodes that had a price, with the count reported alongside. A
    # total whose denominator is hidden reads as the cost of the whole suite when it may
    # be the cost of a third of it.
    total_cost: float = 0.0
    priced_episodes: int = 0

    @property
    def fully_priced(self) -> bool:
        """Whether every episode that ran contributed to `total_cost`.

        Reported beside the total because a sum whose denominator is hidden reads as the
        cost of the whole suite when it may be the cost of a third of it.
        """
        return self.priced_episodes == self.episodes - self.setup_failures

    def to_record(self) -> dict[str, Any]:
        return {
            "success_rate": round(self.success_rate, 4),
            "tool_efficiency": round(self.tool_efficiency, 4),
            "recovery_rate": round(self.recovery_rate, 4),
            "self_check_rate": round(self.self_check_rate, 4),
            "mean_tokens": round(self.mean_tokens, 1),
            "total_cost": round(self.total_cost, 6),
            "priced_episodes": self.priced_episodes,
            "fully_priced": self.fully_priced,
            "mean_wall_time_s": round(self.mean_wall_time_s, 3),
            "mean_tool_calls": round(self.mean_tool_calls, 2),
            "episodes": self.episodes,
            "recovery_eligible": self.recovery_eligible,
            "mutation_eligible": self.mutation_eligible,
            "setup_failures": self.setup_failures,
            "objective_completion": round(self.objective_completion, 4),
            "objective_regression_rate": round(self.objective_regression_rate, 4),
            "long_horizon_episodes": self.long_horizon_episodes,
            "category_success": {k: round(v, 4) for k, v in sorted(self.category_success.items())},
            "category_support": dict(sorted(self.category_support.items())),
            "overfit_rate": round(self.overfit_rate, 4),
            "hidden_test_episodes": self.hidden_test_episodes,
            "truncated_episodes": self.truncated_episodes,
            "truncated_failures": self.truncated_failures,
            "per_episode": [e.to_record() for e in self.per_episode],
        }


def _recovery_check(trajectory: AgentTrajectory) -> tuple[bool, bool]:
    """Did a real tool fail, and did the agent then act? Returns `(hit_failure, acted_after)`.

    Both halves close a hole that `bool(failed_steps)` left open, and the first one is
    cheap enough to be worth spelling out.

    **Not every `ok=False` came from a tool.** The runner synthesises one for a call to a
    tool the task never offered -- "tool 'noop' is not available for this task" -- so that
    the trajectory's `tools_available` claim stays true. Counting those as failures meant
    naming a nonexistent tool once, for about twenty tokens, manufactured `hit_failure`;
    succeed afterwards and the episode also scored `recovered`. A refusal by the harness
    is not an observed failure of the agent's work, so a failure is only counted when its
    matching call named a tool the task actually advertised.

    **Recovery requires a recovery.** The old rule was `bool(failed) and verified_success`,
    which asked for no action at all: fail, declare done, pass, score as having recovered.
    `AgentTrajectory.recovery_steps` -- the calls issued after an observed failure -- has
    existed and been tested since the trajectory type was written, and nothing consulted it.

    What this still cannot see is whether the action *addressed* the failure. Requiring the
    same check to be re-run and now pass would be stricter, and wrong: a genuine recovery
    often changes approach rather than retrying. `recovery_rate` is reported beside
    `recovery_eligible` so the denominator stays visible either way.
    """
    available = set(trajectory.tools_available)
    called_tool = {s.call_id: (s.tool or "") for s in trajectory.steps if s.kind == TOOL_CALL and s.call_id}
    seen_real_failure = False
    acted_after = False
    for step in trajectory.steps:
        if step.kind == TOOL_RESULT and not step.ok:
            # No recorded tool set means an older trajectory that predates the field;
            # fall back to counting the failure rather than silently dropping it.
            tool = called_tool.get(step.call_id or "", "")
            if not available or tool in available:
                seen_real_failure = True
        elif step.kind == TOOL_CALL and seen_real_failure:
            acted_after = True
    return seen_real_failure, acted_after


def _mutation_check(trajectory: AgentTrajectory, mutating_tools: tuple[str, ...]) -> tuple[bool, bool]:
    """Did the agent observe anything after its last state-changing call?

    Returns `(mutated, self_checked)`. An episode that never mutated has nothing to
    check; scoring it either way would turn the metric into a proxy for task type rather
    than behavior, so it is excluded from the denominator entirely.

    This is deliberately a weak, cheap proxy with two known blind spots. It cannot tell a
    real test run from an incidental `ls`, and it only sees mutation performed through a
    *named* mutating tool -- an agent that rewrites a file via `terminal` or `python`
    reads as never having mutated, and drops out of the denominator instead of scoring
    badly. Tasks whose expected solution edits through a shell should say so in
    `mutating_tools`. What it does catch is the common and expensive failure: edit, then
    declare done, having never looked.
    """
    mutating = set(mutating_tools)
    last_mutation = -1
    for index, step in enumerate(trajectory.steps):
        if step.kind == TOOL_CALL and step.tool in mutating:
            last_mutation = index
    if last_mutation < 0:
        return False, False

    # Keyed off where each *call* was issued, not where its result landed. Agents may
    # batch calls before observing any of them, so a result sitting after the edit can
    # belong to a call made before it -- which proves nothing about the edit. Only a
    # call issued after the last mutation, and observed to succeed, counts as a check.
    # The mutation's own result is excluded for the same reason: it confirms the write
    # landed, not that the work is right.
    later_calls = {
        step.call_id for step in trajectory.steps[last_mutation + 1 :] if step.kind == TOOL_CALL and step.call_id
    }
    if not later_calls:
        return True, False
    observed_after = any(
        step.kind == TOOL_RESULT and step.ok and step.call_id in later_calls for step in trajectory.steps
    )
    return True, observed_after


def analyse_checkpoints(timeline: Sequence[dict[str, bool]]) -> tuple[int, int, int]:
    """Reduce a checkpoint pass/fail timeline to (total, met_at_end, regressed).

    "Regressed" means an objective passed at some sample and was failing at the end.
    Sampling is periodic, so this under-reports: an objective broken and repaired between
    two samples is invisible. It never over-reports, which is the direction that matters
    for a metric meant to catch drift -- a false accusation of regression would be worse
    than a missed one.
    """
    if not timeline:
        return 0, 0, 0
    final = timeline[-1]
    ever_passed = {key for sample in timeline for key, ok in sample.items() if ok}
    met = sum(1 for ok in final.values() if ok)
    regressed = sum(1 for key in ever_passed if not final.get(key, False))
    return len(final), met, regressed


def episode_metrics(
    trajectory: AgentTrajectory,
    *,
    task_id: str,
    verified_success: bool,
    tokens_used: int = 0,
    cost: float | None = None,
    usage: dict[str, Any] | None = None,
    wall_time_s: float = 0.0,
    mutating_tools: tuple[str, ...] = DEFAULT_MUTATING_TOOLS,
    max_steps_hit: bool = False,
    setup_failed: bool = False,
    category: str = "",
    public_passed: bool = False,
    hidden_passed: bool | None = None,
    malformed_turns: int = 0,
    verify_digest: str = "",
    dialect: str = "",
    checkpoint_timeline: Sequence[dict[str, bool]] = (),
) -> EpisodeMetrics:
    failed = trajectory.failed_steps
    hit_failure, acted_after_failure = _recovery_check(trajectory)
    mutated, self_checked = _mutation_check(trajectory, mutating_tools)
    total_obj, met_obj, regressed_obj = analyse_checkpoints(checkpoint_timeline)
    completion = _rate(met_obj, total_obj) if total_obj else -1.0
    return EpisodeMetrics(
        task_id=task_id,
        success=verified_success,
        tool_calls=len(trajectory.tool_calls),
        failed_calls=len(failed),
        hit_failure=hit_failure,
        # Recovery requires that something broke, that the agent then *did* something,
        # and that the episode still reached a verified success. See `_recovery_check`
        # for why the first two are not the same as `bool(failed)`.
        recovered=hit_failure and acted_after_failure and verified_success,
        mutated=mutated,
        self_checked=self_checked,
        tokens_used=tokens_used,
        cost=cost,
        usage=usage,
        wall_time_s=wall_time_s,
        steps=len(trajectory.steps),
        max_steps_hit=max_steps_hit,
        setup_failed=setup_failed,
        objective_completion=completion,
        objectives_total=total_obj,
        objectives_met=met_obj,
        objectives_regressed=regressed_obj,
        category=category,
        public_passed=public_passed,
        hidden_passed=hidden_passed,
        malformed_turns=malformed_turns,
        verify_digest=verify_digest,
        dialect=dialect,
    )


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def suite_metrics(episodes: list[EpisodeMetrics]) -> SuiteMetrics:
    """Aggregate per-episode metrics into suite scores.

    `tool_efficiency` is ok-calls / total-calls pooled across the suite rather than
    averaged per episode, so a two-call episode does not outweigh a forty-call one. It
    is a flailing proxy -- it measures whether calls worked, not whether they were the
    right calls to make.
    """
    if not episodes:
        return SuiteMetrics(
            success_rate=0.0,
            tool_efficiency=0.0,
            recovery_rate=0.0,
            self_check_rate=0.0,
            mean_tokens=0.0,
            mean_wall_time_s=0.0,
            mean_tool_calls=0.0,
            episodes=0,
            recovery_eligible=0,
            mutation_eligible=0,
        )

    # Episodes whose setup broke never reached the agent. Counting them as failures
    # would let infrastructure breakage read as a worse model, so they are reported
    # separately and excluded from every behavioral rate.
    setup_failures = [e for e in episodes if e.setup_failed]
    ran = [e for e in episodes if not e.setup_failed]
    if not ran:
        # episodes=0: nothing reached the agent, so there is no behavior to report.
        return SuiteMetrics(
            success_rate=0.0,
            tool_efficiency=0.0,
            recovery_rate=0.0,
            self_check_rate=0.0,
            mean_tokens=0.0,
            mean_wall_time_s=0.0,
            mean_tool_calls=0.0,
            episodes=0,
            recovery_eligible=0,
            mutation_eligible=0,
            setup_failures=len(setup_failures),
            per_episode=tuple(episodes),
        )

    total_calls = sum(e.tool_calls for e in ran)
    total_failed = sum(e.failed_calls for e in ran)
    recovery_pool = [e for e in ran if e.hit_failure]
    mutation_pool = [e for e in ran if e.mutated]
    # objective_completion is -1.0 on short tasks, so filtering on it keeps a suite that
    # mixes horizons from averaging a 5-step task in as a failed long-horizon one.
    long_horizon = [e for e in ran if e.objectives_total > 0]

    with_hidden = [e for e in ran if e.hidden_passed is not None]
    public_passers = [e for e in with_hidden if e.public_passed]

    categories: dict[str, list[EpisodeMetrics]] = {}
    for episode in ran:
        if episode.category:
            categories.setdefault(episode.category, []).append(episode)
    total_objectives = sum(e.objectives_total for e in long_horizon)

    return SuiteMetrics(
        success_rate=_rate(sum(1 for e in ran if e.success), len(ran)),
        tool_efficiency=_rate(total_calls - total_failed, total_calls),
        recovery_rate=_rate(sum(1 for e in recovery_pool if e.recovered), len(recovery_pool)),
        self_check_rate=_rate(sum(1 for e in mutation_pool if e.self_checked), len(mutation_pool)),
        mean_tokens=mean(e.tokens_used for e in ran),
        total_cost=sum(e.cost for e in ran if e.cost is not None),
        priced_episodes=sum(1 for e in ran if e.cost is not None),
        mean_wall_time_s=mean(e.wall_time_s for e in ran),
        mean_tool_calls=mean(e.tool_calls for e in ran),
        episodes=len(ran),
        recovery_eligible=len(recovery_pool),
        mutation_eligible=len(mutation_pool),
        setup_failures=len(setup_failures),
        objective_completion=_rate(sum(e.objectives_met for e in long_horizon), total_objectives),
        objective_regression_rate=_rate(sum(e.objectives_regressed for e in long_horizon), total_objectives),
        long_horizon_episodes=len(long_horizon),
        category_success={
            name: _rate(sum(1 for e in group if e.success), len(group)) for name, group in categories.items()
        },
        category_support={name: len(group) for name, group in categories.items()},
        overfit_rate=_rate(sum(1 for e in public_passers if e.overfit), len(public_passers)),
        hidden_test_episodes=len(with_hidden),
        truncated_episodes=sum(1 for e in ran if e.max_steps_hit),
        truncated_failures=sum(1 for e in ran if e.max_steps_hit and not e.success),
        per_episode=tuple(episodes),
    )
