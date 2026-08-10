"""Multi-session tasks: does the agent carry anything forward, measured against a control."""

import json

import pytest

from hermesbench.session import (
    CarryoverReport,
    MemoryStore,
    SessionError,
    SessionResult,
    SessionSpec,
    carryover,
    summarize,
)


def _result(session_id="s2", passed=True, steps=10) -> SessionResult:
    return SessionResult(session_id=session_id, passed=passed, steps=steps)


# --- session specs -----------------------------------------------------------------


def test_a_session_without_a_check_is_rejected():
    """A session with no verify contributes nothing measurable."""
    with pytest.raises(SessionError, match="measure nothing"):
        SessionSpec(session_id="s1", prompt="do something", verify="  ")


def test_a_session_needs_an_id_and_a_prompt():
    with pytest.raises(SessionError, match="needs an id and a prompt"):
        SessionSpec(session_id="", prompt="p", verify="true")


def test_session_round_trips_through_a_record():
    spec = SessionSpec(session_id="s1", prompt="p", verify="true", max_steps=5)
    assert SessionSpec.from_record(spec.to_record()) == spec


def test_missing_fields_are_named_in_the_error():
    with pytest.raises(SessionError, match="verify"):
        SessionSpec.from_record({"session_id": "s", "prompt": "p"})


# --- memory store ------------------------------------------------------------------


def test_memory_round_trips_a_note():
    store = MemoryStore()
    store.write("baseline latency was 120us")
    assert store.search("latency") == ["baseline latency was 120us"]


def test_search_matches_any_term():
    store = MemoryStore()
    store.write("kernel A uses shared memory")
    assert store.search("kernel occupancy") == ["kernel A uses shared memory"]


def test_search_is_case_insensitive():
    store = MemoryStore()
    store.write("Baseline Latency 120us")
    assert store.search("baseline")


def test_an_empty_query_returns_nothing_rather_than_everything():
    store = MemoryStore()
    store.write("something")
    assert store.search("   ") == []


def test_blank_notes_are_not_stored():
    store = MemoryStore()
    store.write("   ")
    assert store.empty


def test_clear_empties_the_store():
    store = MemoryStore()
    store.write("a")
    store.clear()
    assert store.empty


def test_memory_record_is_json_safe():
    store = MemoryStore()
    store.write("note")
    assert json.loads(json.dumps(store.to_record()))["count"] == 1


# --- carryover, against the control ------------------------------------------------


def test_carryover_is_success_warm_and_failure_cold():
    """The only part attributable to carrying something forward."""
    report = carryover("t", _result(passed=True), _result(passed=False))
    assert report.carried is True
    assert report.regressed is False


def test_succeeding_in_both_arms_is_inconclusive_not_carryover():
    """A task both arms solve measures ease, not recall."""
    report = carryover("t", _result(passed=True, steps=5), _result(passed=True, steps=5))
    assert report.carried is False
    assert report.inconclusive is True


def test_failing_in_both_arms_is_inconclusive():
    report = carryover("t", _result(passed=False), _result(passed=False))
    assert report.inconclusive is True
    assert report.carried is False


def test_failing_warm_while_passing_cold_is_a_regression():
    """Worse for having the earlier context: context corruption, seen from outside."""
    report = carryover("t", _result(passed=False), _result(passed=True))
    assert report.regressed is True
    assert report.carried is False


def test_steps_saved_is_the_skill_reuse_signal():
    """Twenty steps becoming three."""
    report = carryover("t", _result(passed=True, steps=3), _result(passed=True, steps=20))
    assert report.steps_saved == 17
    assert not report.inconclusive


def test_finishing_faster_while_failing_is_not_an_improvement():
    report = carryover("t", _result(passed=False, steps=2), _result(passed=True, steps=20))
    assert report.steps_saved == 0


def test_comparing_different_sessions_is_refused():
    """It would produce a number that looks like carryover and measures something else."""
    with pytest.raises(SessionError, match="different sessions"):
        carryover("t", _result(session_id="s2"), _result(session_id="s3"))


def test_report_is_json_safe():
    report = carryover("t", _result(passed=True), _result(passed=False))
    assert json.loads(json.dumps(report.to_record()))["carried"] is True


# --- aggregation -------------------------------------------------------------------


def _report(carried=False, regressed=False, inconclusive=False, saved=0) -> CarryoverReport:
    if inconclusive:
        return CarryoverReport("t", warm_passed=True, cold_passed=True, warm_steps=5, cold_steps=5)
    if regressed:
        return CarryoverReport("t", warm_passed=False, cold_passed=True, warm_steps=5, cold_steps=5)
    if carried:
        return CarryoverReport("t", warm_passed=True, cold_passed=False, warm_steps=5, cold_steps=9)
    return CarryoverReport("t", warm_passed=True, cold_passed=True, warm_steps=5, cold_steps=5 + saved)


def test_inconclusive_tasks_leave_the_denominator():
    """Letting them drag the rate down makes an easy suite look like a forgetful agent."""
    metrics = summarize([_report(carried=True), _report(inconclusive=True), _report(inconclusive=True)])
    assert metrics.tasks == 3
    assert metrics.inconclusive == 2
    assert metrics.carryover_rate == 1.0  # 1 of 1 conclusive


def test_regressions_are_counted_separately_from_misses():
    metrics = summarize([_report(carried=True), _report(regressed=True)])
    assert metrics.carryover_rate == 0.5
    assert metrics.regression_rate == 0.5


def test_mean_steps_saved_ignores_tasks_that_saved_nothing():
    metrics = summarize([_report(saved=10), _report(saved=20), _report(inconclusive=True)])
    assert metrics.mean_steps_saved == 15.0


def test_an_all_inconclusive_suite_does_not_divide_by_zero():
    metrics = summarize([_report(inconclusive=True)])
    assert metrics.carryover_rate == 0.0
    assert metrics.tasks == 1


def test_summarize_of_nothing():
    metrics = summarize([])
    assert metrics.tasks == 0 and metrics.carryover_rate == 0.0


def test_metrics_record_is_json_safe():
    record = summarize([_report(carried=True)]).to_record()
    assert json.loads(json.dumps(record))["carryover_rate"] == 1.0
