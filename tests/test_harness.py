"""Harness digests: an invariant defended by a field nobody populates is a comment."""

import json

import pytest

from hermes.harness import (
    PROVES_CONSISTENCY,
    PROVES_CONTENT,
    HarnessError,
    RunManifest,
    TaskFingerprint,
    TaskResult,
    comparable,
    digest_mapping,
    digest_suite,
    fingerprint_task,
    harness_digest,
    salted_digest,
)
from hermes.router.manifest import HarnessPin
from hermesbench.tasks import Task

SALT = "a-suite-salt-long-enough"


def _task(**overrides) -> Task:
    record = {
        "task_id": "t1",
        "prompt": "do the thing",
        "verify": "test -f done.txt",
        "tools": ["terminal"],
        "max_steps": 20,
        "timeout_s": 30,
    }
    record.update(overrides)
    return Task.from_record(record)


def _pin(**overrides) -> HarnessPin:
    defaults = {
        "commit": "abc123",
        "system_prompt_digest": "sha256:sys",
        "tool_schema_digest": "sha256:tools",
    }
    defaults.update(overrides)
    return HarnessPin(**defaults)


def _suite(*tasks):
    return digest_suite("bench-v0", [fingerprint_task(t, salt=SALT) for t in tasks])


# --- canonicalisation ----------------------------------------------------------------


def test_key_order_does_not_change_a_digest():
    """Otherwise the digest fingerprints dict insertion order, which nothing depends on."""
    assert digest_mapping({"a": 1, "b": 2}) == digest_mapping({"b": 2, "a": 1})


def test_tool_order_does_not_change_a_task_fingerprint():
    a = fingerprint_task(_task(tools=["terminal", "file_read"]))
    b = fingerprint_task(_task(tools=["file_read", "terminal"]))
    assert a.digest == b.digest


def test_checkpoint_order_does_change_a_fingerprint():
    """Checkpoints run in sequence, so their order is part of the task."""
    a = TaskFingerprint(task_id="t", prompt="p", verify="v", checkpoints=(("a", "x"), ("b", "y")))
    b = TaskFingerprint(task_id="t", prompt="p", verify="v", checkpoints=(("b", "y"), ("a", "x")))
    assert a.digest != b.digest


def test_tags_and_metadata_do_not_change_a_fingerprint():
    """Re-labelling a task must not invalidate every prior result."""
    a = fingerprint_task(_task(tags=["cuda"], metadata={"author": "x"}))
    b = fingerprint_task(_task(tags=["swe", "hard"], metadata={"author": "y"}))
    assert a.digest == b.digest


def test_changing_the_verifier_changes_the_fingerprint():
    assert fingerprint_task(_task()).digest != fingerprint_task(_task(verify="true")).digest


def test_changing_a_step_budget_changes_the_fingerprint():
    """A task solvable in 20 steps and the same task in 3 are different measurements."""
    assert fingerprint_task(_task()).digest != fingerprint_task(_task(max_steps=3)).digest


# --- withheld checks -------------------------------------------------------------------


def test_a_withheld_check_needs_a_salt():
    """Unsalted, a short shell command is a crossword rather than a commitment."""
    with pytest.raises(HarnessError, match="unsalted would publish it"):
        fingerprint_task(_task(hidden_verify="pytest tests/test_hidden.py -q"))


def test_a_short_salt_is_refused():
    with pytest.raises(HarnessError, match="at least 16 characters"):
        salted_digest("pytest -q", "short")


def test_the_withheld_content_never_appears_in_the_fingerprint():
    secret = "pytest tests/test_withheld_edge_cases.py -q"
    payload = json.dumps(fingerprint_task(_task(hidden_verify=secret), salt=SALT).to_payload())
    assert secret not in payload
    assert "test_withheld_edge_cases" not in payload


def test_two_different_withheld_checks_digest_differently():
    """Otherwise substituting the withheld test would be undetectable."""
    a = fingerprint_task(_task(hidden_verify="pytest a.py"), salt=SALT)
    b = fingerprint_task(_task(hidden_verify="pytest b.py"), salt=SALT)
    assert a.digest != b.digest


def test_a_different_salt_hides_the_same_check_differently():
    a = salted_digest("pytest a.py", "salt-number-one-here")
    b = salted_digest("pytest a.py", "salt-number-two-here")
    assert a != b


# --- suite digests ---------------------------------------------------------------------


def test_a_suite_over_no_tasks_is_refused():
    with pytest.raises(HarnessError, match="certify nothing"):
        digest_suite("empty", [])


def test_duplicate_task_ids_are_refused():
    with pytest.raises(HarnessError, match="duplicate task ids"):
        _suite(_task(), _task())


def test_load_order_does_not_change_a_suite_digest():
    a = _suite(_task(task_id="a"), _task(task_id="b"))
    b = _suite(_task(task_id="b"), _task(task_id="a"))
    assert a.digest == b.digest


def test_adding_a_task_changes_the_suite_digest():
    assert _suite(_task(task_id="a")).digest != _suite(_task(task_id="a"), _task(task_id="b")).digest


def test_the_suite_reports_how_many_tasks_withhold_checks():
    suite = _suite(_task(task_id="a"), _task(task_id="b", hidden_verify="pytest b.py"))
    assert suite.withheld_tasks == 1


# --- harness digests ---------------------------------------------------------------------


def test_an_unpinned_harness_is_refused():
    """Digesting blanks yields a stable value that certifies nothing and passes the check."""
    with pytest.raises(HarnessError, match="not pinned"):
        harness_digest(HarnessPin(), suite=_suite(_task()), executor="local")


def test_a_harness_digest_needs_an_executor():
    with pytest.raises(HarnessError, match="executor"):
        harness_digest(_pin(), suite=_suite(_task()), executor="")


def test_the_same_pin_and_suite_give_the_same_digest():
    suite = _suite(_task())
    assert harness_digest(_pin(), suite=suite, executor="local") == harness_digest(
        _pin(), suite=suite, executor="local"
    )


def test_a_different_system_prompt_is_a_different_harness():
    suite = _suite(_task())
    a = harness_digest(_pin(), suite=suite, executor="local")
    b = harness_digest(_pin(system_prompt_digest="sha256:other"), suite=suite, executor="local")
    assert a != b


def test_a_different_container_is_a_different_harness():
    suite = _suite(_task())
    a = harness_digest(_pin(container_image_digest="sha256:img-a"), suite=suite, executor="local")
    b = harness_digest(_pin(container_image_digest="sha256:img-b"), suite=suite, executor="local")
    assert a != b


def test_a_different_executor_is_a_different_harness():
    suite = _suite(_task())
    a = harness_digest(_pin(), suite=suite, executor="local")
    b = harness_digest(_pin(), suite=suite, executor="docker")
    assert a != b


# --- run manifests -----------------------------------------------------------------------


def _manifest(**overrides) -> RunManifest:
    suite = overrides.pop("suite", None) or _suite(_task(task_id="a"), _task(task_id="b"))
    defaults = {
        "model": "spark-hermes-agent-3.8-27b",
        "suite": suite,
        "harness": harness_digest(_pin(), suite=suite, executor="local"),
        "results": (TaskResult("a", passed=True), TaskResult("b", passed=False)),
    }
    defaults.update(overrides)
    return RunManifest(**defaults)


def test_a_manifest_without_per_task_results_is_refused():
    """An aggregate cannot be spot-checked; a doubter has nothing to examine."""
    with pytest.raises(HarnessError, match="cannot be spot-checked"):
        _manifest(results=())


def test_a_manifest_without_a_model_names_no_subject():
    with pytest.raises(HarnessError, match="names no subject"):
        _manifest(model="")


def test_a_manifest_without_a_harness_cannot_be_compared():
    with pytest.raises(HarnessError, match="harness digest"):
        _manifest(harness="")


def test_reporting_a_subset_under_the_suites_name_is_refused():
    """The denominator says 1000 and the numerator saw 40."""
    with pytest.raises(HarnessError, match="misstates its own denominator"):
        _manifest(results=(TaskResult("a", passed=True),))


def test_reporting_a_task_the_suite_does_not_contain_is_refused():
    with pytest.raises(HarnessError, match="unexpected"):
        _manifest(results=(TaskResult("a", passed=True), TaskResult("b", passed=True), TaskResult("c", passed=True)))


def test_a_disqualified_pass_is_not_counted():
    manifest = _manifest(results=(TaskResult("a", passed=True, disqualified=True), TaskResult("b", passed=True)))
    assert manifest.passed == 1
    assert manifest.success_rate == 0.5


def test_the_manifest_says_what_its_withheld_digests_prove():
    suite = digest_suite("s", [fingerprint_task(_task(task_id="a", hidden_verify="pytest a.py"), salt=SALT)])
    manifest = _manifest(
        suite=suite,
        results=(TaskResult("a", passed=True),),
        harness=harness_digest(_pin(), suite=suite, executor="local"),
    )
    assert "same withheld checks" in manifest.withheld_claim
    assert "not that those checks are the ones" in manifest.withheld_claim


def test_revealing_the_salt_upgrades_the_claim():
    suite = digest_suite("s", [fingerprint_task(_task(task_id="a", hidden_verify="pytest a.py"), salt=SALT)])
    manifest = _manifest(
        suite=suite,
        results=(TaskResult("a", passed=True),),
        harness=harness_digest(_pin(), suite=suite, executor="local"),
        proves=PROVES_CONTENT,
    )
    assert "can be confirmed" in manifest.withheld_claim


def test_a_suite_with_no_withheld_checks_says_so():
    assert _manifest().withheld_claim == "no withheld checks in this suite"


def test_an_unknown_claim_strength_is_refused():
    with pytest.raises(HarnessError, match="claim strength"):
        _manifest(proves="trust-me")


def test_manifest_record_is_json_safe():
    record = json.loads(json.dumps(_manifest().to_record()))
    assert record["success_rate"] == 0.5
    assert len(record["results"]) == 2
    assert record["proves"] == PROVES_CONSISTENCY


# --- comparison --------------------------------------------------------------------------


def test_two_runs_on_the_same_suite_and_harness_are_comparable():
    ok, reason = comparable(_manifest(model="a"), _manifest(model="b"))
    assert ok and reason == ""


def test_different_suites_are_not_comparable():
    other = _suite(_task(task_id="a"), _task(task_id="b", verify="true"))
    ok, reason = comparable(
        _manifest(model="a"),
        _manifest(
            model="b",
            suite=other,
            harness=harness_digest(_pin(), suite=other, executor="local"),
            results=(TaskResult("a", passed=True), TaskResult("b", passed=True)),
        ),
    )
    assert not ok and "different suites" in reason


def test_different_harnesses_are_not_comparable():
    """Silently ranking them is how a harness change becomes a model improvement."""
    suite = _suite(_task(task_id="a"), _task(task_id="b"))
    ok, reason = comparable(
        _manifest(model="a", suite=suite),
        _manifest(model="b", suite=suite, harness=harness_digest(_pin(), suite=suite, executor="docker")),
    )
    assert not ok and "the gap includes the harness" in reason


def test_comparing_a_model_with_itself_says_so_rather_than_returning_a_winner():
    ok, reason = comparable(_manifest(model="same"), _manifest(model="same"))
    assert not ok and "nothing to compare" in reason


def test_the_observation_window_is_part_of_the_harness():
    """Double it and the agent sees a different world, with no diff in any task file."""
    suite = _suite(_task())
    a = harness_digest(_pin(), suite=suite, executor="local", observation_limit=8000)
    b = harness_digest(_pin(), suite=suite, executor="local", observation_limit=16000)
    assert a != b


def test_the_tool_timeout_is_part_of_the_harness():
    suite = _suite(_task())
    a = harness_digest(_pin(), suite=suite, executor="local", tool_timeout_s=120)
    b = harness_digest(_pin(), suite=suite, executor="local", tool_timeout_s=300)
    assert a != b


def test_the_selection_policy_is_part_of_the_harness():
    """Reordering the tie-breaks decides which teacher wins every incomparable field, so a
    reordered policy must make prior results incomparable rather than re-rankable."""
    from hermes.selection import COST, TOOL_CALLS, SelectionPolicy

    suite = _suite(_task())
    a = SelectionPolicy(order=(TOOL_CALLS, COST), compare=(TOOL_CALLS, COST))
    b = SelectionPolicy(order=(COST, TOOL_CALLS), compare=(TOOL_CALLS, COST))
    assert harness_digest(_pin(), suite=suite, executor="local", selection_policy=a) != harness_digest(
        _pin(), suite=suite, executor="local", selection_policy=b
    )
