"""Selection: separating what the evidence decided from what a policy decided."""

import json

import pytest

from hermes.selection import (
    COST,
    EVIDENCE,
    RECOVERY,
    TOOL_CALLS,
    WALL_TIME,
    SelectionError,
    SelectionPolicy,
    dominates,
    frontier,
    select,
)
from hermes.tournament import CandidateRun, Verdict

PIN = "sha256:harness-abc"


def _run(model, *, calls=30, cost=None, wall=100.0, recovered=False, speedup=None, validity_calls=None):
    return CandidateRun(
        model=model,
        trajectory_sha256=f"sha-{model}",
        verdict=Verdict(passed=True, verifier="tests", evidence={"speedup": speedup} if speedup else {}),
        harness_digest=PIN,
        tool_calls=validity_calls or calls,
        wall_time_s=wall,
        cost=cost,
        recovered=recovered,
    )


PLAIN = SelectionPolicy(compare=(TOOL_CALLS, COST, RECOVERY))


# --- the policy is data, and it has a digest ---------------------------------------


def test_an_unknown_dimension_is_refused():
    with pytest.raises(SelectionError, match="unknown selection"):
        SelectionPolicy(order=("vibes",))


def test_a_repeated_dimension_is_refused():
    with pytest.raises(SelectionError, match="never fire"):
        SelectionPolicy(order=(TOOL_CALLS, COST, TOOL_CALLS), compare=(TOOL_CALLS,))


def test_comparing_on_evidence_without_naming_the_metric_is_refused():
    """Every candidate would score 0 and the dimension would silently do nothing."""
    with pytest.raises(SelectionError, match="silently do nothing"):
        SelectionPolicy(compare=(EVIDENCE, TOOL_CALLS))


def test_reordering_the_policy_changes_its_digest():
    """A rule change after the results are known must not be silent."""
    a = SelectionPolicy(order=(TOOL_CALLS, COST), compare=(TOOL_CALLS, COST))
    b = SelectionPolicy(order=(COST, TOOL_CALLS), compare=(TOOL_CALLS, COST))
    assert a.digest != b.digest


def test_the_same_policy_digests_the_same():
    assert (
        SelectionPolicy(order=(COST,), compare=(COST,)).digest == SelectionPolicy(order=(COST,), compare=(COST,)).digest
    )


def test_policy_record_is_json_safe():
    record = json.loads(json.dumps(PLAIN.to_record()))
    assert record["digest"].startswith("sha256:")


# --- dominance ----------------------------------------------------------------------


def test_better_on_everything_dominates():
    assert dominates(_run("a", calls=10, cost=1.0), _run("b", calls=40, cost=4.0), policy=PLAIN)


def test_equal_everywhere_dominates_nothing():
    assert not dominates(_run("a", calls=10, cost=1.0), _run("b", calls=10, cost=1.0), policy=PLAIN)


def test_a_trade_off_is_not_domination():
    """Cheaper but chattier beats nothing; the two are incomparable."""
    a = _run("a", calls=40, cost=1.0)
    b = _run("b", calls=10, cost=4.0)
    assert not dominates(a, b, policy=PLAIN) and not dominates(b, a, policy=PLAIN)


def test_an_unmeasurable_dimension_does_not_decide_domination():
    """An unpriced run must not dominate a priced one on the strength of a missing number."""
    unpriced = _run("unpriced", calls=10, cost=None)
    priced = _run("priced", calls=10, cost=5.0)
    assert not dominates(unpriced, priced, policy=PLAIN)
    assert not dominates(priced, unpriced, policy=PLAIN)


def test_with_nothing_comparable_neither_dominates():
    policy = SelectionPolicy(compare=(COST,))
    assert not dominates(_run("a", cost=None), _run("b", cost=None), policy=policy)


# --- the frontier says whether a policy was needed at all ----------------------------


def test_a_dominant_attempt_leaves_a_frontier_of_one():
    result = frontier([_run("best", calls=5, cost=1.0), _run("worse", calls=50, cost=9.0)], policy=PLAIN)
    assert result.decided_by_evidence
    assert [c.model for c in result.surviving] == ["best"]
    assert result.dominated == (("worse", "best"),)


def test_incomparable_attempts_all_survive():
    """A frontier of four means the policy decides, not the evidence."""
    result = frontier(
        [_run("cheap", calls=50, cost=1.0), _run("lean", calls=5, cost=9.0), _run("mid", calls=20, cost=5.0)],
        policy=PLAIN,
    )
    assert not result.decided_by_evidence
    assert len(result.surviving) == 3


def test_the_frontier_records_who_beat_whom():
    result = frontier([_run("a", calls=5, cost=1.0), _run("b", calls=50, cost=9.0)], policy=PLAIN)
    assert json.loads(json.dumps(result.to_record()))["dominated"][0]["by"] == "a"


def test_an_empty_field_has_an_empty_frontier():
    assert frontier([], policy=PLAIN).surviving == ()


# --- select: Pareto first, policy second ---------------------------------------------


def test_a_dominant_winner_is_reported_as_decided_by_evidence():
    policy = SelectionPolicy(order=(TOOL_CALLS,), compare=(TOOL_CALLS, COST))
    result = select([_run("best", calls=5, cost=1.0), _run("worse", calls=50, cost=9.0)], policy=policy)
    assert result.winner.model == "best"
    assert result.decided_by_evidence
    assert result.policy_dimensions == ()  # nothing to break; the frontier was one


def test_a_policy_choice_is_reported_as_such():
    """The distinction a win reason alone cannot make."""
    policy = SelectionPolicy(order=(TOOL_CALLS,), compare=(TOOL_CALLS, COST))
    result = select([_run("cheap", calls=50, cost=1.0), _run("lean", calls=5, cost=9.0)], policy=policy)
    assert result.winner.model == "lean"
    assert not result.decided_by_evidence
    assert result.policy_dimensions == (TOOL_CALLS,)


def test_taking_the_frontier_first_protects_a_broadly_better_candidate():
    """Applying the policy first lets a low-priority dimension eliminate a candidate that
    was better on everything ranked below it."""
    policy = SelectionPolicy(order=(WALL_TIME,), compare=(TOOL_CALLS, COST))
    strong = _run("strong", calls=5, cost=1.0, wall=100.0)
    shallow = _run("shallow", calls=50, cost=9.0, wall=1.0)
    assert select([strong, shallow], policy=policy).winner.model == "strong"


def test_a_dimension_that_separated_nothing_is_not_reported():
    """Naming it would suggest it decided something."""
    # Compared on cost alone, so the two are incomparable and both reach the policy;
    # cost then separates nothing and tool_calls does the work.
    policy = SelectionPolicy(order=(COST, TOOL_CALLS), compare=(COST,))
    result = select([_run("a", calls=5, cost=3.0), _run("b", calls=50, cost=3.0)], policy=policy)
    assert not result.decided_by_evidence
    assert result.policy_dimensions == (TOOL_CALLS,)


def test_a_partially_measurable_dimension_does_not_narrow():
    """Otherwise a candidate is eliminated by absence of data rather than by being worse."""
    policy = SelectionPolicy(order=(COST,), compare=(TOOL_CALLS,))
    result = select([_run("priced", calls=10, cost=1.0), _run("unpriced", calls=10, cost=None)], policy=policy)
    assert COST not in result.policy_dimensions
    assert result.policy_dimensions == ("model_id",)


def test_a_genuine_tie_breaks_on_model_id_and_says_so():
    policy = SelectionPolicy(order=(TOOL_CALLS,), compare=(TOOL_CALLS,))
    result = select([_run("zeta", calls=10), _run("alpha", calls=10)], policy=policy)
    assert result.winner.model == "alpha"
    assert result.policy_dimensions == ("model_id",)


def test_selection_is_deterministic_regardless_of_input_order():
    policy = SelectionPolicy(order=(TOOL_CALLS,), compare=(TOOL_CALLS,))
    a, b = _run("zeta", calls=10), _run("alpha", calls=10)
    assert select([a, b], policy=policy).winner.model == select([b, a], policy=policy).winner.model


def test_an_empty_policy_lets_the_evidence_decide_alone():
    """The most conservative policy: Pareto, then a deterministic tie-break."""
    policy = SelectionPolicy(compare=(TOOL_CALLS, COST))
    result = select([_run("a", calls=5, cost=1.0), _run("b", calls=50, cost=9.0)], policy=policy)
    assert result.winner.model == "a" and result.decided_by_evidence


def test_recovery_participates_as_higher_is_better():
    policy = SelectionPolicy(order=(RECOVERY,), compare=(RECOVERY,))
    result = select([_run("gave_up", recovered=False), _run("recovered", recovered=True)], policy=policy)
    assert result.winner.model == "recovered"


def test_evidence_participates_as_higher_is_better():
    policy = SelectionPolicy(compare=(EVIDENCE,), primary_evidence="speedup")
    result = select([_run("slow", speedup=1.1), _run("fast", speedup=2.4)], policy=policy)
    assert result.winner.model == "fast" and result.decided_by_evidence


def test_the_selection_record_carries_the_policy_digest():
    policy = SelectionPolicy(order=(TOOL_CALLS,), compare=(TOOL_CALLS,))
    record = json.loads(json.dumps(select([_run("a"), _run("b", calls=90)], policy=policy).to_record()))
    assert record["policy_digest"] == policy.digest
    assert record["decided_by_evidence"] is True


def test_selecting_from_nothing_returns_nothing():
    assert select([], policy=PLAIN).winner is None
