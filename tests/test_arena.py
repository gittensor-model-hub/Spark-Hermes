"""The arena: one task, every teacher, a verified winner, rows you can trace back."""

import json

import pytest

from hermes.arena import (
    ERROR_PROVIDER,
    ArenaBatch,
    check_batch,
    export_digests,
    run_task,
    write_exports,
)
from hermes.selection import COST, TOOL_CALLS, SelectionPolicy
from hermes.teachers import TEACHER_FIELD_V1
from hermes.tournament import CandidateRun, Verdict
from hermes.trajectory import FINAL, TOOL_CALL, TOOL_RESULT, AgentTrajectory, Step
from hermesbench.tasks import Task

PIN = "sha256:harness-under-test"
POLICY = SelectionPolicy(order=(TOOL_CALLS, COST), compare=(TOOL_CALLS, COST))


def _task(task_id="t1") -> Task:
    return Task.from_record(
        {"task_id": task_id, "prompt": "do it", "verify": "true", "tools": ["terminal"], "max_steps": 10}
    )


def _trajectory(task_id="t1") -> AgentTrajectory:
    return AgentTrajectory(
        task=task_id,
        success=True,
        tools_available=("terminal",),
        steps=(
            Step(kind=TOOL_CALL, tool="terminal", args={"command": "ls"}, call_id="c0"),
            Step(kind=TOOL_RESULT, call_id="c0", content="ok"),
            Step(kind=FINAL, content="done"),
        ),
    )


def _candidate(model, *, passed=True, calls=10, cost=1.0, self_checked=True, disqualified=False) -> CandidateRun:
    return CandidateRun(
        model=model,
        # Left empty so the arena computes it from the trajectory it was handed, which is
        # what candidate_from does in production. A literal here would now be refused as a
        # candidate that does not address its own trajectory.
        trajectory_sha256="",
        verdict=Verdict(passed=passed, verifier="hermesbench"),
        harness_digest="placeholder",
        tool_calls=calls,
        cost=cost,
        self_checked=self_checked,
        disqualified=disqualified,
    )


def _runner(behaviour):
    """behaviour: teacher_id -> CandidateRun, or an Exception to raise."""

    def run_one(task, teacher):
        outcome = behaviour[teacher.teacher_id]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome, _trajectory(task.task_id)

    return run_one


def _run(behaviour, task_id="t1", **kwargs):
    # No real sleeping: retry backoff is production behaviour, and a suite that waits it
    # out measures patience.
    kwargs.setdefault("sleep", lambda _: None)
    return run_task(
        _task(task_id),
        TEACHER_FIELD_V1,
        run_one=_runner(behaviour),
        harness_digest=PIN,
        policy=POLICY,
        **kwargs,
    )


# --- a provider outage is not a model failure -------------------------------------------


def test_a_teacher_that_errors_is_recorded_not_scored_as_a_loss():
    """A capability matrix built from timeouts measures uptime."""
    outcome = _run({"qwen3.8-max": _candidate("q"), "deepseek-v4-pro": TimeoutError("gateway timeout")})
    errored = outcome.errored
    assert [r.teacher_id for r in errored] == ["deepseek-v4-pro"]
    assert errored[0].error == ERROR_PROVIDER and "TimeoutError" in errored[0].detail


def test_an_errored_teacher_never_becomes_a_candidate():
    outcome = _run({"qwen3.8-max": _candidate("q"), "deepseek-v4-pro": RuntimeError("500")})
    assert outcome.tournament is None  # only one candidate, so no tournament


def test_the_batch_survives_every_teacher_failing():
    outcome = _run({"qwen3.8-max": RuntimeError("a"), "deepseek-v4-pro": RuntimeError("b")})
    assert not outcome.comparable and len(outcome.errored) == 2


# --- a field of one is not a tournament ---------------------------------------------------


def test_one_candidate_is_not_a_comparison():
    """There is a trajectory but no comparison; 'won' is not a fact about it."""
    outcome = _run({"qwen3.8-max": _candidate("q"), "deepseek-v4-pro": RuntimeError("down")})
    assert not outcome.comparable
    assert outcome.artifacts is None


def test_two_candidates_make_a_tournament():
    outcome = _run({"qwen3.8-max": _candidate("q", calls=5), "deepseek-v4-pro": _candidate("k", calls=40)})
    assert outcome.comparable and outcome.artifacts is not None
    assert outcome.artifacts.winner is not None


# --- the fair fight is satisfied by construction --------------------------------------------


def test_every_candidate_is_stamped_with_the_batch_harness():
    """Tournament refuses a mixed field; one place stamping them all is how that holds."""
    outcome = _run({"qwen3.8-max": _candidate("q"), "deepseek-v4-pro": _candidate("k")})
    assert all(r.candidate.harness_digest == PIN for r in outcome.runs if r.ran)


def test_the_candidate_model_is_taken_from_the_teacher_not_the_caller():
    """A caller mislabelling a run would put one teacher's work under another's name."""
    outcome = _run({"qwen3.8-max": _candidate("whatever"), "deepseek-v4-pro": _candidate("nonsense")})
    assert sorted(r.candidate.model for r in outcome.runs if r.ran) == ["deepseek-v4-pro", "qwen3.8-max"]


def test_rights_come_from_the_teacher_registry():
    outcome = _run({"qwen3.8-max": _candidate("q"), "deepseek-v4-pro": _candidate("k")})
    assert all(r.candidate.training_rights == "approved" for r in outcome.runs if r.ran)


# --- the batch --------------------------------------------------------------------------------


def _batch(*outcomes) -> ArenaBatch:
    return ArenaBatch(
        round_id="r1", miner_id="miner-alpha", harness_digest=PIN, teachers=TEACHER_FIELD_V1, outcomes=list(outcomes)
    )


def test_a_batch_reports_its_incomparable_tasks_rather_than_dropping_them():
    """Silently shrinking presents a provider outage as a smaller round."""
    good = _run({"qwen3.8-max": _candidate("q", calls=5), "deepseek-v4-pro": _candidate("k", calls=40)}, "t1")
    bad = _run({"qwen3.8-max": _candidate("q"), "deepseek-v4-pro": RuntimeError("down")}, "t2")
    batch = _batch(good, bad)
    assert [o.task_id for o in batch.incomparable] == ["t2"]
    assert len(batch.comparable) == 1


def test_sft_rows_carry_the_winning_trajectory_and_its_digest():
    outcome = _run({"qwen3.8-max": _candidate("q", calls=5), "deepseek-v4-pro": _candidate("k", calls=40)})
    rows = _batch(outcome).sft_rows()
    assert len(rows) == 1
    assert rows[0]["teacher"] == "qwen3.8-max"
    assert rows[0]["trajectory"]["task"] == "t1"
    assert rows[0]["trajectory_sha256"].startswith("sha256:")


def test_a_candidate_that_does_not_address_its_own_trajectory_is_refused():
    """Otherwise a row lands in the corpus under an address resolving to different work,
    and every downstream check agrees with itself."""
    from hermes.tournament import CandidateRun, Verdict

    lying = CandidateRun(
        model="q",
        trajectory_sha256="sha256:" + "0" * 64,
        verdict=Verdict(passed=True, verifier="hermesbench"),
        harness_digest="placeholder",
    )
    outcome = _run({"qwen3.8-max": lying, "deepseek-v4-pro": _candidate("k")})
    errored = [r for r in outcome.runs if not r.ran]
    assert errored and "digests to" in errored[0].detail


def test_duplicate_task_ids_are_refused_before_digesting():
    """sorted() is stable, so two outcomes sharing an id make the digest depend on run order."""
    from hermes.arena import ArenaError

    a = _run({"qwen3.8-max": _candidate("q"), "deepseek-v4-pro": _candidate("k")}, "same")
    b = _run({"qwen3.8-max": _candidate("q"), "deepseek-v4-pro": _candidate("k")}, "same")
    with pytest.raises(ArenaError, match="duplicate task ids"):
        _batch(a, b).manifest()


def test_export_digests_can_be_folded_into_the_manifest():
    """Outside it, the digests are a claim the sealed check never saw."""
    outcome = _run({"qwen3.8-max": _candidate("q"), "deepseek-v4-pro": _candidate("k")})
    plain = _batch(outcome).manifest()
    with_exports = _batch(outcome).manifest(export_digests={"sft": "sha256:" + "e" * 64})
    assert plain["manifest_digest"] != with_exports["manifest_digest"]
    assert with_exports["exports"]["sft"].startswith("sha256:")


def test_dpo_rows_come_from_the_tournament():
    outcome = _run({"qwen3.8-max": _candidate("q"), "deepseek-v4-pro": _candidate("k", passed=False)})
    rows = _batch(outcome).dpo_rows()
    assert rows and rows[0]["chosen"]["model"] == "qwen3.8-max"


def test_router_rows_include_every_candidate_even_the_losers():
    outcome = _run({"qwen3.8-max": _candidate("q"), "deepseek-v4-pro": _candidate("k", passed=False)})
    rows = _batch(outcome).router_rows()
    assert {o["model"] for o in rows[0]["outcomes"]} == {"qwen3.8-max", "deepseek-v4-pro"}


def test_a_disqualified_winner_yields_no_sft_row():
    """A cheated pass is not a pass, and must not become the trained trajectory."""
    outcome = _run(
        {"qwen3.8-max": _candidate("q", disqualified=True), "deepseek-v4-pro": _candidate("k", passed=False)}
    )
    assert _batch(outcome).sft_rows() == []


# --- the manifest is digests only ---------------------------------------------------------------


def test_the_manifest_carries_no_trajectory_content():
    """It is handed to a sealed worker with no egress; it only needs addresses."""
    outcome = _run({"qwen3.8-max": _candidate("q"), "deepseek-v4-pro": _candidate("k")})
    blob = json.dumps(_batch(outcome).manifest())
    assert "tool_call" not in blob and "steps" not in blob
    assert "trajectory_sha256" in blob


def test_the_manifest_digest_is_stable_across_run_order():
    a = _run({"qwen3.8-max": _candidate("q"), "deepseek-v4-pro": _candidate("k")}, "t1")
    b = _run({"qwen3.8-max": _candidate("q"), "deepseek-v4-pro": _candidate("k")}, "t2")
    assert _batch(a, b).manifest()["manifest_digest"] == _batch(b, a).manifest()["manifest_digest"]


def test_changing_a_verdict_changes_the_manifest_digest():
    """Otherwise a receipt would bind a batch whose verdicts could be edited afterwards."""
    a = _run({"qwen3.8-max": _candidate("q"), "deepseek-v4-pro": _candidate("k")})
    b = _run({"qwen3.8-max": _candidate("q"), "deepseek-v4-pro": _candidate("k", passed=False)})
    assert _batch(a).manifest()["manifest_digest"] != _batch(b).manifest()["manifest_digest"]


def test_the_manifest_names_the_harness_so_two_batches_are_comparable_or_not():
    manifest = _batch(_run({"qwen3.8-max": _candidate("q"), "deepseek-v4-pro": _candidate("k")})).manifest()
    assert manifest["harness_digest"] == PIN
    assert [t["teacher_id"] for t in manifest["teachers"]] == ["deepseek-v4-pro", "qwen3.8-max"]


# --- export ----------------------------------------------------------------------------------------


def test_exports_are_written_as_three_separate_views(tmp_path):
    outcome = _run({"qwen3.8-max": _candidate("q"), "deepseek-v4-pro": _candidate("k", passed=False)})
    written = write_exports(_batch(outcome), tmp_path)
    assert written["sft"] == 1 and written["dpo"] >= 1
    assert (tmp_path / "sft.jsonl").is_file() and (tmp_path / "manifest.json").is_file()


def test_exported_rows_are_json_lines(tmp_path):
    outcome = _run({"qwen3.8-max": _candidate("q"), "deepseek-v4-pro": _candidate("k", passed=False)})
    write_exports(_batch(outcome), tmp_path)
    for line in (tmp_path / "sft.jsonl").read_text().splitlines():
        assert json.loads(line)["task_id"] == "t1"


def test_export_digests_change_when_a_row_changes(tmp_path, tmp_path_factory):
    a = _run({"qwen3.8-max": _candidate("q", calls=5), "deepseek-v4-pro": _candidate("k", calls=40)})
    b = _run({"qwen3.8-max": _candidate("q", calls=40), "deepseek-v4-pro": _candidate("k", calls=5)})
    write_exports(_batch(a), tmp_path)
    first = export_digests(tmp_path)
    other = tmp_path_factory.mktemp("other")
    write_exports(_batch(b), other)
    assert first["sft"] != export_digests(other)["sft"]


# --- batch checks ------------------------------------------------------------------------------------


def test_an_empty_batch_is_reported():
    assert "ran no tasks" in check_batch(_batch())[0]


def test_a_batch_with_no_harness_digest_cannot_be_compared():
    batch = ArenaBatch(round_id="r", miner_id="m", harness_digest="", teachers=TEACHER_FIELD_V1)
    batch.outcomes = [_run({"qwen3.8-max": _candidate("q"), "deepseek-v4-pro": _candidate("k")})]
    assert any("no harness digest" in p for p in check_batch(batch))


def test_a_batch_where_nothing_was_comparable_is_reported():
    outcome = _run({"qwen3.8-max": _candidate("q"), "deepseek-v4-pro": RuntimeError("down")})
    assert any("nothing in it is a tournament result" in p for p in check_batch(_batch(outcome)))


def test_a_mostly_incomparable_batch_is_flagged_as_a_biased_sample():
    good = _run({"qwen3.8-max": _candidate("q"), "deepseek-v4-pro": _candidate("k")}, "t1")
    bad = [_run({"qwen3.8-max": _candidate("q"), "deepseek-v4-pro": RuntimeError("x")}, f"t{i}") for i in range(2, 5)]
    assert any("biased sample" in p for p in check_batch(_batch(good, *bad)))


def test_a_healthy_batch_has_no_problems():
    outcomes = [
        _run({"qwen3.8-max": _candidate("q", calls=5), "deepseek-v4-pro": _candidate("k", calls=40)}, f"t{i}")
        for i in range(3)
    ]
    assert check_batch(_batch(*outcomes)) == []


def test_batch_record_is_json_safe():
    outcome = _run({"qwen3.8-max": _candidate("q"), "deepseek-v4-pro": _candidate("k", passed=False)})
    record = json.loads(json.dumps(_batch(outcome).to_record()))
    assert record["comparable"] == 1 and record["sft_rows"] == 1


def test_a_transient_provider_error_is_retried():
    """One flaky call costs the whole task's comparison, not just one teacher's run."""
    calls = {"n": 0}

    def flaky(task, teacher):
        if teacher.teacher_id == "qwen3.8-max":
            calls["n"] += 1
            if calls["n"] < 2:
                raise RuntimeError("502 from the gateway")
        return _candidate(teacher.teacher_id), _trajectory(task.task_id)

    outcome = run_task(
        _task(),
        TEACHER_FIELD_V1,
        run_one=flaky,
        harness_digest=PIN,
        policy=POLICY,
        attempts=3,
        sleep=lambda _: None,
    )
    assert outcome.comparable
    assert all(r.ran for r in outcome.runs)


def test_a_persistent_failure_still_reports_every_attempt():
    def always(task, teacher):
        raise RuntimeError("gateway down")

    outcome = run_task(
        _task(),
        TEACHER_FIELD_V1,
        run_one=always,
        harness_digest=PIN,
        policy=POLICY,
        attempts=2,
        sleep=lambda _: None,
    )
    detail = outcome.errored[0].detail
    assert "attempt 1" in detail and "attempt 2" in detail


def test_retries_can_be_switched_off():
    calls = {"n": 0}

    def once(task, teacher):
        calls["n"] += 1
        raise RuntimeError("no")

    run_task(
        _task(),
        TEACHER_FIELD_V1,
        run_one=once,
        harness_digest=PIN,
        policy=POLICY,
        attempts=1,
        sleep=lambda _: None,
    )
    assert calls["n"] == len(TEACHER_FIELD_V1)


def test_the_backoff_grows_between_attempts():
    """Three immediate retries against a rate limiter are one retry that took longer."""
    waits: list[float] = []

    def always(task, teacher):
        raise RuntimeError("429")

    run_task(
        _task(),
        TEACHER_FIELD_V1,
        run_one=always,
        harness_digest=PIN,
        policy=POLICY,
        attempts=4,
        backoff_s=2.0,
        sleep=waits.append,
    )
    assert waits[:3] == [2.0, 4.0, 8.0]


def test_a_permanent_failure_is_not_retried():
    """A wrong key retried with backoff looks exactly like a busy gateway, and the operator
    tunes the backoff instead of fixing the key."""

    calls = {"n": 0}

    def unauthorised(task, teacher):
        calls["n"] += 1
        raise RuntimeError("Error code: 401 - invalid_api_key")

    outcome = run_task(
        _task(),
        TEACHER_FIELD_V1,
        run_one=unauthorised,
        harness_digest=PIN,
        policy=POLICY,
        attempts=5,
        sleep=lambda _: None,
    )
    assert calls["n"] == len(TEACHER_FIELD_V1)  # one attempt each, not five
    assert outcome.errored[0].error == "permanent_error"


def test_a_transient_failure_is_still_retried():
    calls = {"n": 0}

    def flaky(task, teacher):
        calls["n"] += 1
        raise RuntimeError("502 Bad Gateway")

    run_task(
        _task(),
        TEACHER_FIELD_V1,
        run_one=flaky,
        harness_digest=PIN,
        policy=POLICY,
        attempts=3,
        sleep=lambda _: None,
    )
    assert calls["n"] == 3 * len(TEACHER_FIELD_V1)
