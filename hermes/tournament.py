"""Teacher tournaments: run every approved teacher on one task, let verification decide.

Rather than assuming "Claude reasons, Kimi codes, Qwen agents", the same task is given to
every eligible teacher under an identical harness and the *verifier* picks the winner.
One tournament yields four different artifacts:

    winner trajectory          -> SFT
    winner vs each failure     -> DPO
    per-teacher outcome vector -> router training (CapabilityDB)
    split decisions            -> disagreement corpus

The last two are the point. Dataset generation is not just a way to make training data;
its outcomes *are* the evidence a routing layer is built from, which is why losing
trajectories are kept rather than discarded.

Three rules are enforced here, not merely documented:

**The fight must be fair.** Candidates run under different harness pins measure
`model × harness`, not model capability. `Tournament` refuses to score a field whose
members disagree on the pin — a silently unfair comparison would poison the capability
matrix and every route derived from it.

**Verification decides, not a judge.** A deterministic verdict outranks a judged one
regardless of how confident the judge was.

**Never decide by majority.** One uniquely correct trajectory beats two agreeing but
wrong ones. Consensus is not evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hermes.router.capability import CapabilityDB, CapabilityRecord
from hermes.router.manifest import RIGHTS_APPROVED
from hermes.router.spec import TaskSpec

# Why a candidate won, in the order the tie-breaks were applied.
WON_ONLY_PASS = "only_passing_candidate"
WON_DETERMINISTIC = "deterministic_verdict_beats_judged"
WON_EVIDENCE = "stronger_measured_evidence"
WON_FEWER_CALLS = "fewer_tool_calls"
WON_CHEAPER = "lower_cost"
WON_RECOVERY = "better_recovery"
WON_STABLE_ORDER = "deterministic_tiebreak_on_model_id"

# How a preference pair was derived.
PAIR_SUCCESS_VS_FAILURE = "success_vs_failure"
PAIR_EFFICIENCY = "efficiency"

# Why a success-vs-success pair was refused.
REFUSED_NARROW = "margin_below_threshold"
REFUSED_LESS_VERIFICATION = "chosen_verified_less"
REFUSED_WORSE_EVIDENCE = "chosen_measured_worse"


class TournamentError(ValueError):
    """A tournament is malformed or its field was not comparable."""


@dataclass(frozen=True)
class Verdict:
    """What the verifier concluded about one candidate's work.

    `deterministic` records whether this came from running something (tests, a benchmark,
    a numerical comparison) or from asking a model. The distinction survives into
    selection: a judged pass never outranks a deterministic one.
    """

    passed: bool
    verifier: str
    deterministic: bool = True
    # Measured quantities the verifier produced -- speedup, tests passed, max abs error.
    # Used to separate two passing candidates; absent for a judged verdict.
    evidence: dict[str, float] = field(default_factory=dict)
    detail: str = ""

    def score(self, key: str) -> float:
        return float(self.evidence.get(key, 0.0))

    def to_record(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "verifier": self.verifier,
            "deterministic": self.deterministic,
            "evidence": self.evidence,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CandidateRun:
    """One teacher's attempt at the task, plus how it was judged and what it cost."""

    model: str
    trajectory_sha256: str
    verdict: Verdict
    harness_digest: str
    model_version: str = ""
    tool_calls: int = 0
    invalid_tool_calls: int = 0
    recovered: bool = False
    hit_failure: bool = False
    tokens: int = 0
    wall_time_s: float = 0.0
    # Whether the agent checked its own work before reporting. Carried at this level
    # because it is the thing an efficiency preference can silently destroy: the cheapest
    # way to use fewer tool calls is to stop verifying.
    self_checked: bool = False
    # What the run cost, when it is known. `None` means unpriced, which is deliberately
    # not the same as 0.0: the cost tie-break below refuses to run at all unless every
    # candidate in the pool has a price, because the alternative is that the model nobody
    # priced wins on cost against every model somebody did. See hermes/cost.py.
    cost: float | None = None
    training_rights: str = RIGHTS_APPROVED
    # Set when anti-cheat disqualified the run (hermesbench.integrity). Kept separate
    # from the verdict on purpose: the verifier answers "did the checks go green", this
    # answers "were the checks still measuring anything".
    disqualified: bool = False
    disqualification_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def may_train(self) -> bool:
        """Fails closed: unknown rights are not approved rights."""
        return self.training_rights == RIGHTS_APPROVED

    @property
    def tool_call_validity(self) -> float:
        if not self.tool_calls:
            return 0.0
        return (self.tool_calls - self.invalid_tool_calls) / self.tool_calls

    def to_record(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "model_version": self.model_version,
            "trajectory_sha256": self.trajectory_sha256,
            "verdict": self.verdict.to_record(),
            "harness_digest": self.harness_digest,
            "tool_calls": self.tool_calls,
            "tool_call_validity": round(self.tool_call_validity, 4),
            "recovered": self.recovered,
            "tokens": self.tokens,
            "wall_time_s": round(self.wall_time_s, 2),
            "cost": None if self.cost is None else round(self.cost, 4),
            "training_rights": self.training_rights,
            "disqualified": self.disqualified,
            "disqualification_reason": self.disqualification_reason,
        }


@dataclass(frozen=True)
class Tournament:
    """One task, every teacher's attempt at it, and the verdicts."""

    task: TaskSpec
    candidates: tuple[CandidateRun, ...]
    # Key in `Verdict.evidence` that decides between two passing candidates -- `speedup`
    # for a kernel, `tests_passed` for a repo fix. Higher is better.
    primary_evidence: str = ""

    def __post_init__(self) -> None:
        if not self.candidates:
            raise TournamentError(f"{self.task.task_id}: tournament has no candidates")

        models = [c.model for c in self.candidates]
        if len(set(models)) != len(models):
            raise TournamentError(f"{self.task.task_id}: the same model entered twice")

        # The fair-fight invariant. Comparing runs from different harnesses measures the
        # harness as much as the model, and the resulting capability numbers would look
        # exactly as credible as real ones.
        digests = {c.harness_digest for c in self.candidates}
        if len(digests) != 1:
            raise TournamentError(
                f"{self.task.task_id}: candidates ran under different harnesses {sorted(digests)}; "
                "a tournament across harness pins measures model x harness, not model capability"
            )
        if not next(iter(digests)):
            raise TournamentError(f"{self.task.task_id}: candidates have no harness digest; the fight is unverifiable")

    @property
    def harness_digest(self) -> str:
        return self.candidates[0].harness_digest

    @property
    def passing(self) -> tuple[CandidateRun, ...]:
        """Candidates that passed *and* were not disqualified.

        A cheated pass is not a weak pass -- it is not a pass. Letting it into the field
        on points would invite it to win on speed, and worse, its trajectory would become
        SFT data teaching the student the exploit.
        """
        return tuple(c for c in self.candidates if c.verdict.passed and not c.disqualified)

    @property
    def failing(self) -> tuple[CandidateRun, ...]:
        return tuple(c for c in self.candidates if not c.verdict.passed or c.disqualified)

    @property
    def disqualified(self) -> tuple[CandidateRun, ...]:
        return tuple(c for c in self.candidates if c.disqualified)

    @property
    def is_split_decision(self) -> bool:
        """Some passed and some failed -- the most informative kind of tournament.

        Unanimous outcomes mostly tell you the task was easy or impossible. A split says
        the task discriminates between teachers, which is exactly what a router needs.
        """
        return bool(self.passing) and bool(self.failing)

    @property
    def unanimous_failure(self) -> bool:
        return not self.passing

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "bucket": self.task.bucket,
            "harness_digest": self.harness_digest,
            "candidates": [c.to_record() for c in self.candidates],
            "split_decision": self.is_split_decision,
        }


def select_winner(tournament: Tournament) -> tuple[CandidateRun | None, tuple[str, ...]]:
    """Pick the winner and report the tie-breaks that got there.

    Ordering, strongest first:

    1. Passing beats failing.
    2. A deterministic verdict beats a judged one -- proof beats opinion.
    3. Stronger measured evidence on the task's primary metric.
    4. Fewer tool calls (less flailing for the same result).
    5. Lower cost.
    6. Recovered from a failure (it demonstrated the behaviour we want to train).
    7. Model id, purely so the result is reproducible.

    Returns `(None, reasons)` when nothing passed. Majority agreement is never consulted:
    two teachers agreeing on a wrong answer is not evidence, and a router trained on
    consensus would learn to prefer the popular failure mode.
    """
    passing = tournament.passing
    if not passing:
        return None, ("no_candidate_passed",)

    if len(passing) == 1:
        return passing[0], (WON_ONLY_PASS,)

    reasons: list[str] = []
    pool = list(passing)

    deterministic = [c for c in pool if c.verdict.deterministic]
    if deterministic and len(deterministic) != len(pool):
        pool = deterministic
        reasons.append(WON_DETERMINISTIC)

    if tournament.primary_evidence:
        best = max(c.verdict.score(tournament.primary_evidence) for c in pool)
        narrowed = [c for c in pool if c.verdict.score(tournament.primary_evidence) == best]
        if len(narrowed) != len(pool):
            pool = narrowed
            reasons.append(WON_EVIDENCE)

    best_calls = min(c.tool_calls for c in pool)
    narrowed = [c for c in pool if c.tool_calls == best_calls]
    if len(narrowed) != len(pool):
        pool = narrowed
        reasons.append(WON_FEWER_CALLS)

    # Cost only decides when every candidate has one. An unpriced run used to arrive here
    # as 0.0 and beat every priced rival, taking the SFT slot and pushing the model it
    # "beat" into the DPO rejections -- on the strength of being unknown rather than cheap.
    # A partially-priced field falls through to the later tie-breaks instead.
    if all(c.cost is not None for c in pool):
        best_cost = min(c.cost for c in pool if c.cost is not None)
        narrowed = [c for c in pool if c.cost == best_cost]
        if len(narrowed) != len(pool):
            pool = narrowed
            reasons.append(WON_CHEAPER)

    recovered = [c for c in pool if c.recovered]
    if recovered and len(recovered) != len(pool):
        pool = recovered
        reasons.append(WON_RECOVERY)

    if len(pool) > 1:
        pool.sort(key=lambda c: c.model)
        reasons.append(WON_STABLE_ORDER)

    return pool[0], tuple(reasons)


@dataclass(frozen=True)
class PreferenceMargin:
    """Why one passing trajectory is preferred over another, component by component.

    Kept apart rather than summed into one number, because the components are not
    interchangeable and the aggregate hides which one is driving the preference. A pair
    won on measured evidence is teaching the model to produce better work; a pair won on
    wall time is teaching it to produce work faster; those are different lessons and a
    single scalar makes them look like the same one.
    """

    tool_efficiency: float = 0.0
    wall_time: float = 0.0
    # None when the component could not be measured -- the task declared no primary
    # evidence, or one of the runs was unpriced. Distinct from 0.0, which means measured
    # and equal: counting an unmeasurable component as zero would drag every margin toward
    # a tie and make an unpriced comparison look even when it was not.
    evidence: float | None = None
    cost: float | None = None

    @property
    def components(self) -> dict[str, float]:
        present = {"tool_efficiency": self.tool_efficiency, "wall_time": self.wall_time}
        if self.evidence is not None:
            present["evidence"] = self.evidence
        if self.cost is not None:
            present["cost"] = self.cost
        return present

    @property
    def overall(self) -> float:
        """Mean of the components that could be measured.

        Averaged over what was measurable rather than over a fixed denominator: a pair
        whose runs were unpriced should not be diluted toward zero for lacking a component
        nobody could compute, which would silently make every unpriced comparison look
        like a tie.
        """
        values = list(self.components.values())
        return sum(values) / len(values) if values else 0.0

    def to_record(self) -> dict[str, Any]:
        return {
            "components": {k: round(v, 4) for k, v in sorted(self.components.items())},
            "overall": round(self.overall, 4),
        }


@dataclass(frozen=True)
class DPOPair:
    """One preference, and what it was derived from.

    Success-vs-failure is the safe case: the verifier decided it, and the lesson is
    "do the thing that worked".

    Success-vs-success is the case that needs guarding, and it is off by default. Two
    passing trajectories can differ in something real -- thirty-one tool calls against
    sixty-seven is not taste -- but the cheapest way for a model to use fewer tool calls
    is to stop checking its own work, and a preference built on efficiency alone points
    exactly that way. So the margin is recorded, near-ties are dropped as noise, and a
    pair whose winner verified less than its loser is refused outright.
    """

    prompt: str
    chosen_model: str
    chosen_sha256: str
    rejected_model: str
    rejected_sha256: str
    pair_type: str = PAIR_SUCCESS_VS_FAILURE
    margin: PreferenceMargin | None = None

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "prompt": self.prompt,
            "chosen": {"model": self.chosen_model, "trajectory_sha256": self.chosen_sha256},
            "rejected": {"model": self.rejected_model, "trajectory_sha256": self.rejected_sha256},
            "pair_type": self.pair_type,
        }
        if self.margin is not None:
            record["margin"] = self.margin.to_record()
        return record


def _relative_gain(better: float, worse: float) -> float:
    """Signed relative advantage of `better` over `worse`, where lower is better.

    Normalised by the larger value so the number is comparable across tasks: saving
    thirty calls on a sixty-call task is a different achievement from saving thirty on a
    six-hundred-call one, and an absolute delta calls them equal.
    """
    scale = max(better, worse)
    if scale <= 0:
        return 0.0
    return (worse - better) / scale


def efficiency_margin(
    chosen: CandidateRun,
    rejected: CandidateRun,
    *,
    primary_evidence: str = "",
) -> PreferenceMargin:
    """Measure how much better one passing run was than another."""
    evidence: float | None = None
    if primary_evidence:
        chosen_score = chosen.verdict.score(primary_evidence)
        rejected_score = rejected.verdict.score(primary_evidence)
        scale = max(abs(chosen_score), abs(rejected_score))
        evidence = (chosen_score - rejected_score) / scale if scale > 0 else 0.0
    # Cost only when both runs have one. Comparing a priced run against an unpriced one
    # would score the unmeasured as free, which is the failure `PriceBook` refuses.
    cost = None
    if chosen.cost is not None and rejected.cost is not None:
        cost = _relative_gain(chosen.cost, rejected.cost)
    return PreferenceMargin(
        evidence=evidence,
        tool_efficiency=_relative_gain(chosen.tool_calls, rejected.tool_calls),
        wall_time=_relative_gain(chosen.wall_time_s, rejected.wall_time_s),
        cost=cost,
    )


def efficiency_pair(
    tournament: Tournament,
    chosen: CandidateRun,
    rejected: CandidateRun,
    *,
    min_margin: float,
) -> tuple[DPOPair | None, str]:
    """Build a success-vs-success preference, or say why it was refused.

    Three refusals, in the order that matters most:

    **The winner must not have verified less.** Otherwise "used fewer tools" and "skipped
    the check" are the same gradient direction, and the pair teaches a model that already
    knows how to verify to stop bothering. This is the one that makes the whole idea safe.

    **The winner must not have measured worse.** On a task with primary evidence -- a
    speedup, a test count -- a solution that was quicker to produce but performs worse is
    not the better solution, and preferring it optimises the wrong end of the task.
    `select_winner` already narrows on evidence, so a winner reaching here should never
    have measured worse; the check is for direct callers pairing two candidates the
    selector did not choose between, where nothing else would catch it.

    **Near-ties are dropped.** Two runs separated by noise carry no signal, and a
    preference pair asserting one is better trains confidence in a distinction that was
    not there.
    """
    if not chosen.self_checked and rejected.self_checked:
        return None, REFUSED_LESS_VERIFICATION
    margin = efficiency_margin(chosen, rejected, primary_evidence=tournament.primary_evidence)
    if margin.evidence is not None and margin.evidence < 0:
        return None, REFUSED_WORSE_EVIDENCE
    if margin.overall < min_margin:
        return None, REFUSED_NARROW
    return (
        DPOPair(
            prompt=tournament.task.prompt,
            chosen_model=chosen.model,
            chosen_sha256=chosen.trajectory_sha256,
            rejected_model=rejected.model,
            rejected_sha256=rejected.trajectory_sha256,
            pair_type=PAIR_EFFICIENCY,
            margin=margin,
        ),
        "",
    )


@dataclass(frozen=True)
class TournamentArtifacts:
    """Everything one tournament produces. Losers are kept, not discarded."""

    task_id: str
    bucket: str
    winner: CandidateRun | None
    win_reasons: tuple[str, ...]
    sft_trajectory_sha256: str | None
    dpo_pairs: tuple[DPOPair, ...]
    router_example: dict[str, Any]
    is_split_decision: bool
    withheld_for_rights: tuple[str, ...] = ()
    # Success-vs-success pairs that were considered and dropped, with the reason. Kept
    # because "no efficiency pairs were produced" and "every candidate was a near-tie" are
    # different facts about a tournament, and only one of them means the field was even.
    refused_pairs: tuple[tuple[str, str], ...] = ()

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "bucket": self.bucket,
            "winner": self.winner.model if self.winner else None,
            "win_reasons": list(self.win_reasons),
            "sft_trajectory_sha256": self.sft_trajectory_sha256,
            "dpo_pairs": [p.to_record() for p in self.dpo_pairs],
            "router_example": self.router_example,
            "split_decision": self.is_split_decision,
            "withheld_for_rights": list(self.withheld_for_rights),
            "refused_pairs": [{"model": m, "reason": r} for m, r in self.refused_pairs],
        }


def build_artifacts(
    tournament: Tournament,
    *,
    prefer_on_efficiency: bool = False,
    min_margin: float = 0.15,
    winner: CandidateRun | None = None,
    win_reasons: tuple[str, ...] = (),
) -> TournamentArtifacts:
    """Turn one tournament into SFT, DPO, and router-training data.

    Rights are enforced per candidate rather than per tournament: a teacher we may not
    train on can still *inform the router* — knowing it succeeded is a fact about the
    world — but its trajectory never becomes SFT or a DPO side. That split is why the
    router example includes every candidate while the training artifacts do not.

    `prefer_on_efficiency` additionally pairs the winner against the *other passing*
    candidates. It is off by default and gated when on, because the lesson it teaches is
    one step removed from correctness: see `efficiency_pair` for the three refusals.
    """
    # A caller that already ran a selector passes its answer in. Running `select_winner`
    # here regardless would let two selectors disagree -- the Pareto-and-policy one the
    # arena uses to report `decided_by_evidence`, and this lexicographic one -- and the
    # artifacts would then credit a different teacher than the report names.
    if winner is None:
        winner, reasons = select_winner(tournament)
    else:
        reasons = tuple(win_reasons)

    withheld: list[str] = []
    sft_sha: str | None = None
    if winner is not None:
        if winner.may_train:
            sft_sha = winner.trajectory_sha256
        else:
            withheld.append(winner.model)

    pairs: list[DPOPair] = []
    if winner is not None and winner.may_train:
        for loser in tournament.failing:
            if not loser.may_train:
                withheld.append(loser.model)
                continue
            pairs.append(
                DPOPair(
                    prompt=tournament.task.prompt,
                    chosen_model=winner.model,
                    chosen_sha256=winner.trajectory_sha256,
                    rejected_model=loser.model,
                    rejected_sha256=loser.trajectory_sha256,
                )
            )

    refused: list[tuple[str, str]] = []
    if prefer_on_efficiency and winner is not None and winner.may_train:
        for other in tournament.passing:
            if other.model == winner.model:
                continue
            if not other.may_train:
                withheld.append(other.model)
                continue
            pair, reason = efficiency_pair(tournament, winner, other, min_margin=min_margin)
            if pair is not None:
                pairs.append(pair)
            else:
                refused.append((other.model, reason))

    router_example = {
        "task": tournament.task.to_record(),
        "harness_digest": tournament.harness_digest,
        "outcomes": [
            {
                "model": c.model,
                "model_version": c.model_version,
                "passed": c.verdict.passed,
                "deterministic": c.verdict.deterministic,
                "evidence": c.verdict.evidence,
                "tool_call_validity": round(c.tool_call_validity, 4),
                "recovered": c.recovered,
                "disqualified": c.disqualified,
                "tokens": c.tokens,
                "wall_time_s": round(c.wall_time_s, 2),
                # None rather than 0.0 when unpriced: a router example that reports an
                # unpriced run as free would teach the router the same lie the tie-break
                # was corrected for.
                "cost": None if c.cost is None else round(c.cost, 4),
            }
            for c in tournament.candidates
        ],
        "winner": winner.model if winner else None,
        "split_decision": tournament.is_split_decision,
    }

    return TournamentArtifacts(
        task_id=tournament.task.task_id,
        bucket=tournament.task.bucket,
        winner=winner,
        win_reasons=reasons,
        sft_trajectory_sha256=sft_sha,
        dpo_pairs=tuple(pairs),
        router_example=router_example,
        is_split_decision=tournament.is_split_decision,
        refused_pairs=tuple(refused),
        withheld_for_rights=tuple(dict.fromkeys(withheld)),
    )


def update_capabilities(db: CapabilityDB, tournaments: list[Tournament], *, effort: str = "default") -> CapabilityDB:
    """Fold tournament outcomes into the capability matrix the router ranks on.

    Accumulates attempts and verified successes per (model, bucket, harness). Records are
    keyed by harness digest so numbers measured under different pins never silently merge
    — the same reason the tournament refuses a mixed field in the first place.
    """
    tallies: dict[tuple[str, str, str], dict[str, float]] = {}
    for tournament in tournaments:
        bucket = tournament.task.bucket
        for candidate in tournament.candidates:
            key = (candidate.model, bucket, tournament.harness_digest)
            entry = tallies.setdefault(
                key,
                {
                    "attempts": 0,
                    "successes": 0,
                    "validity": 0.0,
                    "recoveries": 0,
                    "eligible": 0,
                    "cost": 0.0,
                    "priced": 0,
                },
            )
            entry["attempts"] += 1
            # A disqualified run counts as an attempt but never as a success -- otherwise
            # gaming the verifier would raise the teacher's capability score.
            entry["successes"] += 1 if (candidate.verdict.passed and not candidate.disqualified) else 0
            entry["validity"] += candidate.tool_call_validity
            entry["cost"] += candidate.cost or 0.0
            entry["priced"] += 1 if candidate.cost is not None else 0
            # Recovery is only meaningful where something failed first, so it gets its
            # own denominator rather than being averaged over every attempt.
            if candidate.hit_failure:
                entry["eligible"] += 1
                entry["recoveries"] += 1 if candidate.recovered else 0

    for (model, bucket, harness), entry in tallies.items():
        existing = db.get(model, bucket, harness=harness, effort=effort)
        attempts = int(entry["attempts"]) + (existing.attempts if existing else 0)
        successes = int(entry["successes"]) + (existing.verified_successes if existing else 0)
        eligible = int(entry["eligible"])
        db.add(
            CapabilityRecord(
                model=model,
                bucket=bucket,
                harness=harness,
                effort=effort,
                attempts=attempts,
                verified_successes=successes,
                tool_call_validity=entry["validity"] / entry["attempts"],
                recovery_rate=(entry["recoveries"] / eligible) if eligible else 0.0,
                # Averaged over the runs that had a price, not over all attempts: dividing
                # by attempts would report a model whose runs were half unpriced as half
                # as expensive as it is.
                estimated_cost=(entry["cost"] / entry["priced"]) if entry["priced"] else 0.0,
            )
        )
    return db
