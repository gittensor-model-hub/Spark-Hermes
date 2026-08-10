"""The arena driver: the thing that makes the pipeline runnable rather than designed."""

import json

import pytest

from hermes.generate_batch import (
    DEFAULT_POLICY,
    POLICIES,
    GenerateError,
    check_policy_is_usable,
    check_scope,
    describe_outcome,
    generate,
    load_price_book,
    resolve_teachers,
    select_tasks,
)
from hermes.selection import COST
from hermes.tournament import CandidateRun, Verdict
from hermes.trajectory import FINAL, TOOL_CALL, TOOL_RESULT, AgentTrajectory, Step

FIELD = "qwen3.8-max,deepseek-v4-pro"

# Supplied rather than built from the repo: these tests drive a stub runner, so the shipped
# harness is not what ran, and `build_pin` would refuse the dirty tree of a dev checkout.
STUB_DIGEST = "sha256:" + "b1" * 32


# --- the teacher field is resolved from the registry, not from a flag ---------------------


def test_the_declared_field_resolves():
    teachers = resolve_teachers(FIELD)
    assert [t.teacher_id for t in teachers] == ["qwen3.8-max", "deepseek-v4-pro"]


def test_a_field_of_one_is_refused_before_anything_is_paid_for():
    """One teacher yields trajectories and no comparison, so the batch would produce no rows."""
    with pytest.raises(GenerateError, match="at least two teachers"):
        resolve_teachers("qwen3.8-max")


def test_an_unknown_teacher_is_refused_here_because_the_gate_refuses_it_at_merge():
    with pytest.raises(GenerateError, match="unknown teacher ids"):
        resolve_teachers("qwen3.8-max,gpt-nonexistent")


def test_a_teacher_entered_against_itself_is_refused():
    with pytest.raises(GenerateError, match="named twice"):
        resolve_teachers("qwen3.8-max,qwen3.8-max")


def test_the_default_field_is_resolvable():
    """The CLI default has to work, or the first command anyone runs fails."""
    from hermes.teachers import TEACHER_FIELD_V1

    assert resolve_teachers(",".join(t.teacher_id for t in TEACHER_FIELD_V1))


# --- a policy that cannot fire is refused rather than silently skipped ---------------------


def test_a_cost_ordered_policy_without_prices_is_refused():
    """Otherwise the manifest records that `thrift` ran while nothing about it was thrifty:
    every candidate's cost is None, so the dimension falls through on every task."""
    with pytest.raises(GenerateError, match="orders on cost but no --prices"):
        check_policy_is_usable("thrift", POLICIES["thrift"], None)


def test_a_cost_ordered_policy_with_prices_is_allowed():
    check_policy_is_usable("thrift", POLICIES["thrift"], object())


def test_the_default_policy_needs_no_prices():
    """The default has to be usable with no configuration, and evidence-only is the
    conservative choice: let dominance decide and break a true tie on model id."""
    check_policy_is_usable(DEFAULT_POLICY, POLICIES[DEFAULT_POLICY], None)


def test_every_declared_policy_is_a_valid_selection_policy():
    """`SelectionPolicy.__post_init__` refuses unknown or repeated dimensions, so this
    catches a typo in the table at import time rather than mid-batch."""
    for name, policy in POLICIES.items():
        assert policy.digest, name


def test_a_policy_ordering_on_cost_also_compares_on_it():
    """Ordering on a dimension the frontier never considers is a rule that cannot fire."""
    for name, policy in POLICIES.items():
        assert COST not in policy.order or COST in policy.compare, name


# --- task selection ------------------------------------------------------------------------


def test_the_suite_loads():
    assert select_tasks(suite="all")


def test_named_tasks_are_selected_in_the_order_given():
    tasks = select_tasks(suite="all", task_ids="recover-from-bad-command,fix-failing-test")
    assert [t.task_id for t in tasks] == ["recover-from-bad-command", "fix-failing-test"]


def test_an_unknown_task_id_is_refused():
    with pytest.raises(GenerateError, match="are not in suite"):
        select_tasks(suite="all", task_ids="no-such-task")


def test_a_tag_filter_that_selects_nothing_is_refused_rather_than_running_empty():
    """An empty batch reports success over zero tasks, which is a measurement of nothing."""
    with pytest.raises(GenerateError, match="selected no tasks"):
        select_tasks(suite="all", tags="tag-that-matches-nothing")


# --- scope is checked before the money is spent ----------------------------------------------


def _round_record(tmp_path, miner_ids=("miner-a", "miner-b")):
    from hermes.seed import Round

    tasks = [t.task_id for t in select_tasks(suite="all")]
    round_ = Round(round_id="r1", seed="a" * 64, task_ids=tuple(tasks), miner_ids=tuple(miner_ids))
    path = tmp_path / "round.json"
    path.write_text(json.dumps(round_.to_record()), encoding="utf-8")
    return path, round_


def test_tasks_assigned_to_this_miner_pass_scope(tmp_path):
    path, round_ = _round_record(tmp_path)
    mine = list(round_.assigned_to("miner-a"))
    tasks = select_tasks(suite="all", task_ids=",".join(mine))
    check_scope(round_record=path, miner_id="miner-a", tasks=tasks)


def test_someone_elses_tasks_are_refused_before_they_are_run(tmp_path):
    """The gate refuses out-of-scope rows at merge; refusing here turns a rejected
    submission into a refused run, which costs one file read."""
    path, round_ = _round_record(tmp_path)
    theirs = [t for t in round_.task_ids if not round_.owns("miner-a", t)]
    tasks = select_tasks(suite="all", task_ids=",".join(theirs))
    with pytest.raises(GenerateError, match="not assigned to"):
        check_scope(round_record=path, miner_id="miner-a", tasks=tasks)


def test_no_round_record_means_no_scope_check(tmp_path):
    check_scope(round_record=None, miner_id="anyone", tasks=select_tasks(suite="all"))


# --- price book -------------------------------------------------------------------------------


def test_no_price_file_means_no_price_book():
    assert load_price_book(None) is None


def test_a_price_book_round_trips(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text(
        json.dumps(
            {
                "revision": "2026-08",
                "prices": [{"model": "qwen3.8-max", "input": 1.2, "output": 6.0}],
            }
        ),
        encoding="utf-8",
    )
    book = load_price_book(path)
    assert book.knows("qwen3.8-max")


# --- the batch runs, exports, and reports what it produced ---------------------------------------


def _trajectory(task_id: str) -> AgentTrajectory:
    return AgentTrajectory(
        task=task_id,
        success=True,
        tools_available=("terminal",),
        steps=(
            Step(kind=TOOL_CALL, tool="terminal", args={"command": "pytest -q"}, call_id="c0"),
            Step(kind=TOOL_RESULT, call_id="c0", content="ok"),
            Step(kind=FINAL, content="done"),
        ),
    )


def _candidate(model: str, *, passed=True, calls=10) -> CandidateRun:
    return CandidateRun(
        model=model,
        # Left empty so the arena derives it from the trajectory it was handed, the way
        # candidate_from does. A literal would be refused as a candidate that does not
        # address its own trajectory.
        trajectory_sha256="",
        verdict=Verdict(passed=passed, verifier="hermesbench", deterministic=True),
        harness_digest="placeholder",
        tool_calls=calls,
        self_checked=True,
    )


def _stub_runner(behaviour):
    """behaviour: teacher_id -> (passed, tool_calls)."""

    def make_runner(_digest):
        def run_one(task, teacher):
            passed, calls = behaviour[teacher.teacher_id]
            return _candidate(teacher.teacher_id, passed=passed, calls=calls), _trajectory(task.task_id)

        return run_one

    return make_runner


def _generate(tmp_path, behaviour, task_ids="fix-failing-test"):
    tasks = select_tasks(suite="all", task_ids=task_ids)
    return generate(
        round_id="r1",
        miner_id="miner-a",
        tasks=tasks,
        teachers=resolve_teachers(FIELD),
        suite_name="all",
        workspace_root=tmp_path / "ws",
        out_dir=tmp_path / "out",
        policy=POLICIES[DEFAULT_POLICY],
        harness_digest=STUB_DIGEST,
        make_runner=_stub_runner(behaviour),
    )


def test_a_batch_writes_the_three_exports_and_a_manifest(tmp_path):
    behaviour = {"qwen3.8-max": (True, 8), "deepseek-v4-pro": (False, 20)}
    batch, written, problems = _generate(tmp_path, behaviour)
    out = tmp_path / "out"
    assert (out / "sft.jsonl").is_file()
    assert (out / "dpo.jsonl").is_file()
    assert (out / "router.jsonl").is_file()
    assert (out / "manifest.json").is_file()
    assert written["sft"] == 1
    assert not problems
    assert batch.harness_digest


def test_the_written_manifest_covers_the_exports_it_shipped_with(tmp_path):
    """A manifest written before the rows exist describes a claim the sealed check never saw.
    The digests have to be folded in after the files are on disk."""
    behaviour = {"qwen3.8-max": (True, 8), "deepseek-v4-pro": (False, 20)}
    _generate(tmp_path, behaviour)
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["exports"]) == {"sft", "dpo", "router"}

    from hermes.arena import export_digests

    assert manifest["exports"] == export_digests(tmp_path / "out")


def test_the_manifest_digest_covers_the_export_digests(tmp_path):
    """Otherwise a miner could publish honest rows, attest the manifest, then submit
    different ones -- the receipt would still verify."""
    from hermes.harness import digest_mapping

    behaviour = {"qwen3.8-max": (True, 8), "deepseek-v4-pro": (False, 20)}
    _generate(tmp_path, behaviour)
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    claimed = manifest.pop("manifest_digest")
    assert digest_mapping(manifest) == claimed

    manifest["exports"]["sft"] = "sha256:" + "0" * 64
    assert digest_mapping(manifest) != claimed


def test_every_candidate_carries_the_batch_harness_digest(tmp_path):
    """`Tournament` refuses a field that disagrees on the harness. One place stamps them
    all, so the fair-fight invariant holds by construction rather than by everyone
    remembering to pass the same value."""
    behaviour = {"qwen3.8-max": (True, 8), "deepseek-v4-pro": (True, 20)}
    batch, _written, _problems = _generate(tmp_path, behaviour)
    for outcome in batch.outcomes:
        for run in outcome.runs:
            if run.candidate is not None:
                assert run.candidate.harness_digest == batch.harness_digest


def test_a_batch_where_nothing_was_comparable_is_reported_as_unfit(tmp_path):
    """A provider outage must not present as a smaller round."""

    def make_runner(_digest):
        def run_one(task, teacher):
            raise RuntimeError("gateway is down")

        return run_one

    tasks = select_tasks(suite="all", task_ids="fix-failing-test")
    batch, written, problems = generate(
        round_id="r1",
        miner_id="miner-a",
        tasks=tasks,
        teachers=resolve_teachers(FIELD),
        suite_name="all",
        workspace_root=tmp_path / "ws",
        out_dir=tmp_path / "out",
        policy=POLICIES[DEFAULT_POLICY],
        attempts=1,
        harness_digest=STUB_DIGEST,
        make_runner=make_runner,
    )
    assert problems
    assert any("no task had two teachers" in p for p in problems)
    # Written anyway: the round is evidence about the round.
    assert (tmp_path / "out" / "manifest.json").is_file()
    assert written["sft"] == 0


def test_the_progress_line_says_whether_evidence_or_policy_decided(tmp_path):
    """A tie broken by the declared order is not the same claim as a dominant win, and a
    line that reads the same for both is how the distinction gets lost."""
    behaviour = {"qwen3.8-max": (True, 8), "deepseek-v4-pro": (False, 20)}
    batch, _w, _p = _generate(tmp_path, behaviour)
    line = describe_outcome(batch.outcomes[0])
    assert "winner=" in line
    assert "by evidence" in line or "by policy" in line


def test_an_incomparable_task_reports_why(tmp_path):
    """`incomparable` with no reason sends the operator to the wrong problem."""

    def make_runner(_digest):
        def run_one(task, teacher):
            if teacher.teacher_id == "qwen3.8-max":
                raise RuntimeError("boom")
            return _candidate(teacher.teacher_id), _trajectory(task.task_id)

        return run_one

    tasks = select_tasks(suite="all", task_ids="fix-failing-test")
    batch, _w, _p = generate(
        round_id="r1",
        miner_id="miner-a",
        tasks=tasks,
        teachers=resolve_teachers(FIELD),
        suite_name="all",
        workspace_root=tmp_path / "ws",
        out_dir=tmp_path / "out",
        policy=POLICIES[DEFAULT_POLICY],
        attempts=1,
        harness_digest=STUB_DIGEST,
        make_runner=make_runner,
    )
    line = describe_outcome(batch.outcomes[0])
    assert "incomparable" in line
    assert "qwen3.8-max" in line


def test_the_batch_is_generated_under_one_digest_for_every_task(tmp_path):
    """Two tasks in one batch must not be measured against two harnesses."""
    behaviour = {"qwen3.8-max": (True, 8), "deepseek-v4-pro": (False, 20)}
    batch, _w, _p = _generate(tmp_path, behaviour, task_ids="fix-failing-test,recover-from-bad-command")
    digests = {
        run.candidate.harness_digest for outcome in batch.outcomes for run in outcome.runs if run.candidate is not None
    }
    assert len(digests) == 1


# --- the endpoint profile travels with the batch ------------------------------------------


def _profile(overheads):
    """A PreflightReport built from stub endpoints billing the given fixed overheads."""
    from hermes.preflight import PROBE_LONG, PROBE_SHORT, PreflightReport, count_tokens, probe_overhead

    profiles = []
    for teacher_id, extra in overheads.items():
        billed = {PROBE_SHORT: count_tokens(PROBE_SHORT) + extra, PROBE_LONG: count_tokens(PROBE_LONG) + extra}
        profiles.append(
            probe_overhead(
                teacher_id=teacher_id,
                model=teacher_id,
                complete=lambda messages, b=billed: b[messages[0]["content"]],
                usage_of=lambda tokens: tokens,
            )
        )
    return PreflightReport(profiles=profiles)


def _generate_with(tmp_path, behaviour, **kwargs):
    tasks = select_tasks(suite="all", task_ids="fix-failing-test")
    return generate(
        round_id="r1",
        miner_id="miner-a",
        tasks=tasks,
        teachers=resolve_teachers(FIELD),
        suite_name="all",
        workspace_root=tmp_path / "ws",
        out_dir=tmp_path / "out",
        policy=POLICIES[DEFAULT_POLICY],
        harness_digest=STUB_DIGEST,
        make_runner=_stub_runner(behaviour),
        **kwargs,
    )


BOTH_PASS = {"qwen3.8-max": (True, 8), "deepseek-v4-pro": (False, 20)}


def test_the_endpoint_profile_is_written_beside_the_exports(tmp_path):
    """Beside, not inside the manifest. The manifest is what the sealed worker is handed and
    it is deliberately digests-only -- a token count is operator evidence, not a digest, and
    the worker has no egress to check one against anything."""
    profile = _profile({"qwen3.8-max": 4, "deepseek-v4-pro": 4})
    _batch, written, _problems = _generate_with(tmp_path, BOTH_PASS, endpoints=profile)
    assert (tmp_path / "out" / "endpoints.json").is_file()
    assert written["endpoints"]["endpoints"][0]["teacher_id"]
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert "endpoints" not in manifest


def test_a_batch_with_no_profile_writes_no_endpoints_file(tmp_path):
    """--skip-preflight has to leave no trace, or an absent measurement reads as a zero one."""
    _batch, written, _problems = _generate_with(tmp_path, BOTH_PASS)
    assert not (tmp_path / "out" / "endpoints.json").exists()
    assert "endpoints" not in written


def test_an_injected_prompt_is_reported_among_the_batch_problems(tmp_path):
    """The measured live figure: ~69 tokens of prompt nobody sent."""
    profile = _profile({"qwen3.8-max": 69, "deepseek-v4-pro": 69})
    _batch, _written, problems = _generate_with(tmp_path, BOTH_PASS, endpoints=profile)
    assert any("neither our messages nor a chat template account for" in p for p in problems)


def test_a_field_whose_endpoints_differ_is_reported(tmp_path):
    """Tournament asserts the fair fight by requiring one harness digest, and that check
    cannot see two endpoints prepending different amounts of hidden context."""
    profile = _profile({"qwen3.8-max": 61, "deepseek-v4-pro": 69})
    _batch, _written, problems = _generate_with(tmp_path, BOTH_PASS, endpoints=profile)
    assert any("different amounts of hidden context" in p for p in problems)


def test_a_moved_overhead_is_reported_against_the_previous_batch(tmp_path):
    """The reason to record it. An injected prompt is tolerable if it is stable and
    disclosed; the failure is a corpus whose halves were conditioned differently."""
    before = _profile({"qwen3.8-max": 4, "deepseek-v4-pro": 4}).to_record()
    after = _profile({"qwen3.8-max": 54, "deepseek-v4-pro": 4})
    _batch, _written, problems = _generate_with(tmp_path, BOTH_PASS, endpoints=after, previous_profile=before)
    assert any("prompt overhead moved" in p for p in problems)


def test_an_unchanged_overhead_adds_no_problem(tmp_path):
    steady = {"qwen3.8-max": 4, "deepseek-v4-pro": 4}
    _batch, _written, problems = _generate_with(
        tmp_path, BOTH_PASS, endpoints=_profile(steady), previous_profile=_profile(steady).to_record()
    )
    assert not any("prompt overhead moved" in p for p in problems)


def test_a_clean_profile_does_not_make_a_good_batch_unfit(tmp_path):
    """A batch that was fit before must stay fit; the profile adds evidence, not a hurdle."""
    profile = _profile({"qwen3.8-max": 4, "deepseek-v4-pro": 4})
    _batch, _written, problems = _generate_with(tmp_path, BOTH_PASS, endpoints=profile)
    assert problems == []
