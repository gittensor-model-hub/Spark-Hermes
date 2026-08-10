"""Agent evolution: improving the system around the model, without touching weights.

A worker is `model + skills + tool descriptions + prompts + verifier`. Only one of those
needs a GPU to change. Rewriting a skill so it says "run the tests before reporting
success" can move task completion more than a training run, costs nothing, and ships in a
pull request.

The loop is propose → evaluate → gate → select. What makes it dangerous is that it is
*search against a scorer*, which is the definition of overfitting when the scorer is
fixed. Left alone, evolutionary search on a benchmark produces variants that fit the
benchmark, and the resulting numbers rise while the worker does not improve.

Three defences, all of which are refusals rather than warnings:

**A development split and a holdout.** Variants are proposed and ranked on the dev tasks
and confirmed on tasks the search never saw. A candidate that gains on dev and does not
hold on holdout is recorded as `overfit` and refused, no matter how large the dev gain.

**Constraint gates, evaluated before scoring.** A variant that drops a required behaviour
is not a low-scoring candidate, it is not a candidate. Scoring first and filtering later
lets a large dev gain argue its way past a constraint, which is exactly what a search
process is good at.

**Parent comparison, not absolute score.** A candidate must beat the thing it replaces.
Absolute thresholds reward an easy task set.

Every accepted change carries an `EvolutionTrace` -- what failed, what was changed, what
happened next. That record is the training signal the model layer wants: a concrete
failure paired with the repair that fixed it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

# What a candidate changes. Weights are deliberately absent: this layer does not train.
SKILL = "skill"
PROMPT = "prompt"
TOOL_DESCRIPTION = "tool_description"
TOOL_CODE = "tool_code"

EVOLVABLE = (SKILL, PROMPT, TOOL_DESCRIPTION, TOOL_CODE)

# Why a candidate was refused.
GATE_FAILED = "gate_failed"
NO_DEV_GAIN = "no_dev_gain"
OVERFIT = "overfit_to_dev_split"


class EvolutionError(ValueError):
    """A candidate or population is malformed."""


@dataclass(frozen=True)
class Candidate:
    """One proposed variant of a skill, prompt, tool description or tool implementation."""

    candidate_id: str
    kind: str
    content: str
    parent_id: str | None = None
    mutation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise EvolutionError("candidate needs an id")
        if self.kind not in EVOLVABLE:
            raise EvolutionError(f"unknown candidate kind {self.kind!r}; expected one of {list(EVOLVABLE)}")
        if not self.content.strip():
            raise EvolutionError(f"{self.candidate_id}: empty content")

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "parent_id": self.parent_id,
            "mutation": self.mutation,
            "content_chars": len(self.content),
        }


@dataclass(frozen=True)
class Gate:
    """A hard constraint on a variant, checked before it is ever scored.

    `check` returns a reason string to refuse, or None to allow. Gates exist for the
    behaviours that must survive optimisation -- an agent that stopped verifying its work
    would score well on any benchmark that only counts completions.
    """

    name: str
    check: Callable[[Candidate], str | None]


def requires_phrase(name: str, phrase: str) -> Gate:
    """Gate: the variant must still mention `phrase` (case-insensitive)."""

    def _check(candidate: Candidate) -> str | None:
        if phrase.lower() in candidate.content.lower():
            return None
        return f"dropped required phrase {phrase!r}"

    return Gate(name=name, check=_check)


def max_length(name: str, limit: int) -> Gate:
    """Gate: variants may not grow without bound.

    Search will happily discover that appending more instructions helps a little and
    keeps doing it, until the skill costs more context than it saves.
    """

    def _check(candidate: Candidate) -> str | None:
        if len(candidate.content) <= limit:
            return None
        return f"content is {len(candidate.content)} chars, limit {limit}"

    return Gate(name=name, check=_check)


@dataclass(frozen=True)
class Evaluation:
    """A candidate's scores on both splits."""

    candidate_id: str
    dev_score: float
    holdout_score: float
    parent_dev_score: float
    parent_holdout_score: float

    @property
    def dev_gain(self) -> float:
        return self.dev_score - self.parent_dev_score

    @property
    def holdout_gain(self) -> float:
        return self.holdout_score - self.parent_holdout_score

    @property
    def overfit(self) -> bool:
        """Gained on the split it was searched against, lost ground on the one it was not.

        The characteristic signature of search against a fixed scorer, and the reason the
        holdout exists at all.
        """
        return self.dev_gain > 0 and self.holdout_gain < 0

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "dev_score": round(self.dev_score, 4),
            "holdout_score": round(self.holdout_score, 4),
            "dev_gain": round(self.dev_gain, 4),
            "holdout_gain": round(self.holdout_gain, 4),
            "overfit": self.overfit,
        }


@dataclass(frozen=True)
class Rejection:
    candidate_id: str
    reason: str
    detail: str = ""

    def to_record(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, "reason": self.reason, "detail": self.detail}


@dataclass(frozen=True)
class EvolutionTrace:
    """What failed, what was changed, and what happened next.

    The point of recording this rather than only the winning variant: a concrete failure
    paired with the repair that fixed it is exactly the shape the model layer wants for
    preference data. The winning text alone teaches what to say, not what it was for.
    """

    candidate_id: str
    kind: str
    observed_failure: str
    mutation: str
    dev_gain: float
    holdout_gain: float
    accepted: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "observed_failure": self.observed_failure,
            "mutation": self.mutation,
            "dev_gain": round(self.dev_gain, 4),
            "holdout_gain": round(self.holdout_gain, 4),
            "accepted": self.accepted,
        }


@dataclass(frozen=True)
class EvolutionResult:
    winner: Candidate | None
    evaluations: tuple[Evaluation, ...]
    rejected: tuple[Rejection, ...]
    traces: tuple[EvolutionTrace, ...]

    @property
    def improved(self) -> bool:
        return self.winner is not None

    @property
    def overfit_rejections(self) -> tuple[Rejection, ...]:
        return tuple(r for r in self.rejected if r.reason == OVERFIT)

    def to_record(self) -> dict[str, Any]:
        return {
            "winner": self.winner.candidate_id if self.winner else None,
            "improved": self.improved,
            "evaluations": [e.to_record() for e in self.evaluations],
            "rejected": [r.to_record() for r in self.rejected],
            "traces": [t.to_record() for t in self.traces],
        }


ScoreFn = Callable[[Candidate, Sequence[Any]], float]


def evolve(
    parent: Candidate,
    candidates: Iterable[Candidate],
    *,
    score: ScoreFn,
    dev_tasks: Sequence[Any],
    holdout_tasks: Sequence[Any],
    gates: Sequence[Gate] = (),
    observed_failure: str = "",
    min_dev_gain: float = 0.0,
) -> EvolutionResult:
    """Run one generation: gate, score on both splits, and select a confirmed winner.

    `holdout_tasks` must be non-empty. Running without one turns this into a search that
    optimises the scorer, and the resulting number would look exactly like an improvement.
    """
    if not dev_tasks:
        raise EvolutionError("evolution needs development tasks to search against")
    if not holdout_tasks:
        raise EvolutionError(
            "evolution needs a holdout split; searching and confirming on the same tasks "
            "measures fit to those tasks rather than improvement"
        )

    parent_dev = score(parent, dev_tasks)
    parent_holdout = score(parent, holdout_tasks)

    evaluations: list[Evaluation] = []
    rejected: list[Rejection] = []
    traces: list[EvolutionTrace] = []
    viable: list[tuple[Candidate, Evaluation]] = []

    for candidate in candidates:
        # Gates first. A variant that dropped a required behaviour is not a low-scoring
        # candidate; scoring it first would let a big dev gain argue past the constraint.
        gate_failure = next((reason for g in gates if (reason := g.check(candidate))), None)
        if gate_failure is not None:
            rejected.append(Rejection(candidate.candidate_id, GATE_FAILED, gate_failure))
            traces.append(
                EvolutionTrace(
                    candidate_id=candidate.candidate_id,
                    kind=candidate.kind,
                    observed_failure=observed_failure,
                    mutation=candidate.mutation,
                    dev_gain=0.0,
                    holdout_gain=0.0,
                    accepted=False,
                )
            )
            continue

        evaluation = Evaluation(
            candidate_id=candidate.candidate_id,
            dev_score=score(candidate, dev_tasks),
            holdout_score=score(candidate, holdout_tasks),
            parent_dev_score=parent_dev,
            parent_holdout_score=parent_holdout,
        )
        evaluations.append(evaluation)
        traces.append(
            EvolutionTrace(
                candidate_id=candidate.candidate_id,
                kind=candidate.kind,
                observed_failure=observed_failure,
                mutation=candidate.mutation,
                dev_gain=evaluation.dev_gain,
                holdout_gain=evaluation.holdout_gain,
                accepted=False,
            )
        )

        if evaluation.dev_gain <= min_dev_gain:
            rejected.append(Rejection(candidate.candidate_id, NO_DEV_GAIN, f"dev gain {evaluation.dev_gain:+.4f}"))
            continue
        if evaluation.overfit:
            rejected.append(
                Rejection(
                    candidate.candidate_id,
                    OVERFIT,
                    f"dev {evaluation.dev_gain:+.4f} but holdout {evaluation.holdout_gain:+.4f}",
                )
            )
            continue
        viable.append((candidate, evaluation))

    if not viable:
        return EvolutionResult(None, tuple(evaluations), tuple(rejected), tuple(traces))

    # Rank on holdout, because that is the split nothing was fitted to. Ties break on dev
    # and then on id, so a rerun selects the same winner.
    winner, winning_eval = max(viable, key=lambda pair: (pair[1].holdout_gain, pair[1].dev_gain, pair[0].candidate_id))
    traces = [
        EvolutionTrace(
            candidate_id=t.candidate_id,
            kind=t.kind,
            observed_failure=t.observed_failure,
            mutation=t.mutation,
            dev_gain=t.dev_gain,
            holdout_gain=t.holdout_gain,
            accepted=t.candidate_id == winner.candidate_id,
        )
        for t in traces
    ]
    _ = winning_eval
    return EvolutionResult(winner, tuple(evaluations), tuple(rejected), tuple(traces))


def preference_pairs(result: EvolutionResult) -> list[dict[str, Any]]:
    """Accepted variant against each refused one, as preference data.

    Only emitted when a winner was confirmed on holdout. Preferring a variant that merely
    won a dev split would teach the model the benchmark's shape rather than the behaviour,
    which is the same mistake one layer up.
    """
    if result.winner is None:
        return []
    accepted = next(t for t in result.traces if t.accepted)
    return [
        {
            "context": accepted.observed_failure,
            "kind": accepted.kind,
            "chosen": {"candidate_id": accepted.candidate_id, "mutation": accepted.mutation},
            "rejected": {"candidate_id": t.candidate_id, "mutation": t.mutation},
        }
        for t in result.traces
        if not t.accepted
    ]
