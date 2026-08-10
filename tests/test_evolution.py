"""Agent evolution: search against a scorer, with the overfitting guarded."""

import json

import pytest

from hermes.evolution import (
    GATE_FAILED,
    NO_DEV_GAIN,
    OVERFIT,
    PROMPT,
    SKILL,
    Candidate,
    EvolutionError,
    evolve,
    max_length,
    preference_pairs,
    requires_phrase,
)

PARENT = Candidate(candidate_id="parent", kind=SKILL, content="Review the code. Run the tests before reporting.")
DEV = ["d1", "d2", "d3"]
HOLDOUT = ["h1", "h2"]


def _candidate(cid, content="Review the code. Run the tests before reporting. Also check imports.", **kw):
    return Candidate(candidate_id=cid, kind=SKILL, content=content, parent_id="parent", **kw)


def _scorer(table):
    """Score from a {candidate_id: (dev, holdout)} table."""

    def score(candidate, tasks):
        dev, holdout = table[candidate.candidate_id]
        return dev if list(tasks) == DEV else holdout

    return score


# --- candidates --------------------------------------------------------------------


def test_unknown_kind_is_rejected():
    with pytest.raises(EvolutionError, match="unknown candidate kind"):
        Candidate(candidate_id="c", kind="weights", content="x")


def test_empty_content_is_rejected():
    with pytest.raises(EvolutionError, match="empty content"):
        Candidate(candidate_id="c", kind=SKILL, content="   ")


def test_weights_are_not_an_evolvable_kind():
    """This layer improves the system around the model; it does not train."""
    from hermes.evolution import EVOLVABLE

    assert "weights" not in EVOLVABLE
    assert set(EVOLVABLE) == {"skill", "prompt", "tool_description", "tool_code"}


# --- the holdout is mandatory ------------------------------------------------------


def test_evolution_without_a_holdout_is_refused():
    """Searching and confirming on the same tasks measures fit, not improvement."""
    with pytest.raises(EvolutionError, match="holdout"):
        evolve(
            PARENT,
            [_candidate("c1")],
            score=_scorer({"parent": (0.5, 0.5), "c1": (0.9, 0.9)}),
            dev_tasks=DEV,
            holdout_tasks=[],
        )


def test_evolution_without_dev_tasks_is_refused():
    with pytest.raises(EvolutionError, match="development tasks"):
        evolve(PARENT, [], score=_scorer({"parent": (0, 0)}), dev_tasks=[], holdout_tasks=HOLDOUT)


# --- overfitting is a refusal, not a warning ---------------------------------------


def test_a_variant_that_gains_on_dev_and_loses_on_holdout_is_refused():
    """The characteristic signature of search against a fixed scorer."""
    result = evolve(
        PARENT,
        [_candidate("overfit")],
        score=_scorer({"parent": (0.50, 0.50), "overfit": (0.95, 0.40)}),
        dev_tasks=DEV,
        holdout_tasks=HOLDOUT,
    )
    assert result.winner is None
    assert [r.reason for r in result.rejected] == [OVERFIT]
    assert result.overfit_rejections


def test_a_large_dev_gain_does_not_buy_its_way_past_the_holdout():
    result = evolve(
        PARENT,
        [_candidate("huge")],
        score=_scorer({"parent": (0.10, 0.50), "huge": (0.99, 0.49)}),
        dev_tasks=DEV,
        holdout_tasks=HOLDOUT,
    )
    assert result.winner is None


def test_a_genuine_improvement_is_accepted():
    result = evolve(
        PARENT,
        [_candidate("real")],
        score=_scorer({"parent": (0.50, 0.50), "real": (0.70, 0.65)}),
        dev_tasks=DEV,
        holdout_tasks=HOLDOUT,
    )
    assert result.winner is not None and result.winner.candidate_id == "real"
    assert result.improved


def test_holding_steady_on_holdout_is_enough():
    """Only a *loss* on holdout is overfitting; flat is not a regression."""
    result = evolve(
        PARENT,
        [_candidate("flat")],
        score=_scorer({"parent": (0.50, 0.50), "flat": (0.70, 0.50)}),
        dev_tasks=DEV,
        holdout_tasks=HOLDOUT,
    )
    assert result.winner is not None


# --- parent comparison -------------------------------------------------------------


def test_a_candidate_must_beat_the_thing_it_replaces():
    """Absolute thresholds reward an easy task set."""
    result = evolve(
        PARENT,
        [_candidate("worse")],
        score=_scorer({"parent": (0.80, 0.80), "worse": (0.70, 0.90)}),
        dev_tasks=DEV,
        holdout_tasks=HOLDOUT,
    )
    assert result.winner is None
    assert [r.reason for r in result.rejected] == [NO_DEV_GAIN]


def test_ranking_uses_the_split_nothing_was_fitted_to():
    result = evolve(
        PARENT,
        [_candidate("dev_star"), _candidate("holdout_star")],
        score=_scorer({"parent": (0.50, 0.50), "dev_star": (0.95, 0.55), "holdout_star": (0.60, 0.80)}),
        dev_tasks=DEV,
        holdout_tasks=HOLDOUT,
    )
    assert result.winner.candidate_id == "holdout_star"


def test_selection_is_deterministic_on_a_tie():
    table = {"parent": (0.5, 0.5), "alpha": (0.7, 0.7), "zeta": (0.7, 0.7)}
    winners = {
        evolve(
            PARENT,
            [_candidate(a), _candidate(b)],
            score=_scorer(table),
            dev_tasks=DEV,
            holdout_tasks=HOLDOUT,
        ).winner.candidate_id
        for a, b in (("alpha", "zeta"), ("zeta", "alpha"))
    }
    assert winners == {"zeta"}


# --- gates -------------------------------------------------------------------------


def test_a_gate_refuses_before_scoring():
    """An agent that stopped verifying would score well on anything counting completions."""
    scored = []

    def score(candidate, tasks):
        scored.append(candidate.candidate_id)
        return 1.0

    result = evolve(
        PARENT,
        [_candidate("dropped", content="Review the code. Report when done.")],
        score=score,
        dev_tasks=DEV,
        holdout_tasks=HOLDOUT,
        gates=[requires_phrase("keeps_verification", "run the tests")],
    )
    assert result.winner is None
    assert [r.reason for r in result.rejected] == [GATE_FAILED]
    # Only the parent was scored; the gated candidate never was.
    assert "dropped" not in scored


def test_a_gate_failure_names_what_was_dropped():
    result = evolve(
        PARENT,
        [_candidate("dropped", content="Review the code.")],
        score=_scorer({"parent": (0.5, 0.5)}),
        dev_tasks=DEV,
        holdout_tasks=HOLDOUT,
        gates=[requires_phrase("keeps_verification", "run the tests")],
    )
    assert "run the tests" in result.rejected[0].detail


def test_a_length_gate_stops_unbounded_growth():
    """Search discovers that appending more instructions helps a little, and repeats."""
    result = evolve(
        PARENT,
        [_candidate("bloated", content="Run the tests. " + "extra guidance. " * 200)],
        score=_scorer({"parent": (0.5, 0.5)}),
        dev_tasks=DEV,
        holdout_tasks=HOLDOUT,
        gates=[max_length("bounded", 500)],
    )
    assert result.winner is None
    assert result.rejected[0].reason == GATE_FAILED


def test_a_passing_gate_lets_the_candidate_through():
    result = evolve(
        PARENT,
        [_candidate("ok")],
        score=_scorer({"parent": (0.5, 0.5), "ok": (0.7, 0.7)}),
        dev_tasks=DEV,
        holdout_tasks=HOLDOUT,
        gates=[requires_phrase("keeps_verification", "run the tests"), max_length("bounded", 10_000)],
    )
    assert result.winner is not None


# --- traces and preference data ----------------------------------------------------


def test_every_candidate_leaves_a_trace_including_the_refused_ones():
    result = evolve(
        PARENT,
        [_candidate("good"), _candidate("bad")],
        score=_scorer({"parent": (0.5, 0.5), "good": (0.7, 0.7), "bad": (0.4, 0.4)}),
        dev_tasks=DEV,
        holdout_tasks=HOLDOUT,
        observed_failure="claimed success without running the tests",
    )
    assert {t.candidate_id for t in result.traces} == {"good", "bad"}
    assert sum(1 for t in result.traces if t.accepted) == 1


def test_a_trace_pairs_the_failure_with_the_repair():
    """The winning text alone teaches what to say, not what it was for."""
    result = evolve(
        PARENT,
        [_candidate("fix", mutation="add an explicit verification step")],
        score=_scorer({"parent": (0.5, 0.5), "fix": (0.8, 0.7)}),
        dev_tasks=DEV,
        holdout_tasks=HOLDOUT,
        observed_failure="claimed success without running the tests",
    )
    trace = next(t for t in result.traces if t.accepted)
    assert trace.observed_failure == "claimed success without running the tests"
    assert trace.mutation == "add an explicit verification step"


def test_preference_pairs_need_a_confirmed_winner():
    """Preferring a dev-split winner teaches the benchmark's shape, not the behaviour."""
    result = evolve(
        PARENT,
        [_candidate("overfit")],
        score=_scorer({"parent": (0.5, 0.5), "overfit": (0.9, 0.3)}),
        dev_tasks=DEV,
        holdout_tasks=HOLDOUT,
    )
    assert preference_pairs(result) == []


def test_preference_pairs_prefer_the_confirmed_variant_over_each_refusal():
    result = evolve(
        PARENT,
        [_candidate("good", mutation="verify first"), _candidate("bad", mutation="skip checks")],
        score=_scorer({"parent": (0.5, 0.5), "good": (0.8, 0.8), "bad": (0.4, 0.4)}),
        dev_tasks=DEV,
        holdout_tasks=HOLDOUT,
        observed_failure="claimed success without running the tests",
    )
    pairs = preference_pairs(result)
    assert len(pairs) == 1
    assert pairs[0]["chosen"]["candidate_id"] == "good"
    assert pairs[0]["rejected"]["candidate_id"] == "bad"
    assert pairs[0]["context"] == "claimed success without running the tests"


def test_result_is_json_safe():
    result = evolve(
        PARENT,
        [_candidate("c1")],
        score=_scorer({"parent": (0.5, 0.5), "c1": (0.7, 0.7)}),
        dev_tasks=DEV,
        holdout_tasks=HOLDOUT,
    )
    assert json.loads(json.dumps(result.to_record()))["improved"] is True


def test_prompt_kind_is_evolvable_too():
    candidate = Candidate(candidate_id="p", kind=PROMPT, content="You are a careful agent.")
    assert candidate.kind == PROMPT
