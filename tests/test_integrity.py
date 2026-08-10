"""Anti-cheat: an agent that satisfied the verifier without doing the work."""

import json

from hermes.router.capability import CapabilityDB
from hermes.router.spec import TaskSpec
from hermes.tournament import CandidateRun, Tournament, Verdict, build_artifacts, select_winner, update_capabilities
from hermes.trajectory import FINAL, TOOL_CALL, TOOL_RESULT, AgentTrajectory, Step
from hermesbench.integrity import (
    PROTECTED_PATH_CREATED,
    PROTECTED_PATH_DELETED,
    PROTECTED_PATH_MODIFIED,
    UNMEASURED_CLAIM,
    VERIFICATION_SKIPPED,
    IntegrityReport,
    check_integrity,
    check_protected_paths,
    check_unmeasured_claims,
    check_verification_ran,
    digest_paths,
    enforce,
)


def _traj(final="done", results=(), tools=("terminal",)) -> AgentTrajectory:
    steps = []
    for i, (tool, content) in enumerate(results):
        steps.append(Step(kind=TOOL_CALL, tool=tool, args={}, call_id=f"c{i}"))
        steps.append(Step(kind=TOOL_RESULT, call_id=f"c{i}", content=content))
    steps.append(Step(kind=FINAL, content=final))
    return AgentTrajectory(task="t", steps=tuple(steps), success=True, tools_available=tools)


# --- protected paths ---------------------------------------------------------------


def test_modifying_a_protected_file_disqualifies(tmp_path):
    (tmp_path / "tests").mkdir()
    target = tmp_path / "tests" / "test_x.py"
    target.write_text("assert 1 == 1")
    before = digest_paths(tmp_path, ("tests/test_x.py",))
    target.write_text("pass  # neutered")
    signals = check_protected_paths(before, digest_paths(tmp_path, ("tests/test_x.py",)))
    assert [s.code for s in signals] == [PROTECTED_PATH_MODIFIED]
    assert signals[0].disqualifying


def test_deleting_a_protected_file_disqualifies(tmp_path):
    target = tmp_path / "baseline.txt"
    target.write_text("x")
    before = digest_paths(tmp_path, ("baseline.txt",))
    target.unlink()
    assert [s.code for s in check_protected_paths(before, digest_paths(tmp_path, ("baseline.txt",)))] == [
        PROTECTED_PATH_DELETED
    ]


def test_creating_a_protected_file_disqualifies(tmp_path):
    """e.g. dropping in a permissive conftest.py that neuters the suite."""
    before = digest_paths(tmp_path, ("conftest.py",))
    (tmp_path / "conftest.py").write_text("collect_ignore_glob = ['*']")
    assert [s.code for s in check_protected_paths(before, digest_paths(tmp_path, ("conftest.py",)))] == [
        PROTECTED_PATH_CREATED
    ]


def test_untouched_protected_files_produce_no_signal(tmp_path):
    (tmp_path / "a.py").write_text("x")
    before = digest_paths(tmp_path, ("a.py",))
    assert check_protected_paths(before, digest_paths(tmp_path, ("a.py",))) == []


def test_a_task_declaring_no_protected_paths_is_unaffected(tmp_path):
    assert check_protected_paths(digest_paths(tmp_path, ()), digest_paths(tmp_path, ())) == []


# --- verification skipped ----------------------------------------------------------


def test_declaring_success_without_verifying_disqualifies():
    trajectory = _traj(final="Fixed it, all good", results=[("file_write", "wrote a.py")])
    signals = check_verification_ran(trajectory, ("terminal", "python"))
    assert [s.code for s in signals] == [VERIFICATION_SKIPPED]
    assert signals[0].disqualifying


def test_calling_a_verification_tool_clears_the_check():
    trajectory = _traj(final="fixed", results=[("terminal", "2 passed")])
    assert check_verification_ran(trajectory, ("terminal",)) == []


def test_no_verification_tools_declared_means_no_check():
    """Backwards compatible: tasks that never declared any are unaffected."""
    assert check_verification_ran(_traj(results=[("file_write", "x")]), ()) == []


# --- unmeasured claims -------------------------------------------------------------


def test_a_fabricated_number_is_flagged_as_a_warning():
    """'Do not merely make a verifiable claim -- generate the evidence.'"""
    trajectory = _traj(final="The optimization is 35% faster.", results=[("terminal", "build ok")])
    signals = check_unmeasured_claims(trajectory)
    assert [s.code for s in signals] == [UNMEASURED_CLAIM]
    # A heuristic, so it warns rather than disqualifies.
    assert not signals[0].disqualifying


def test_a_measured_number_is_not_flagged():
    trajectory = _traj(final="Speedup is 1.52x.", results=[("terminal", "before 2.14 ms after 1.41 ms => 1.52x")])
    assert check_unmeasured_claims(trajectory) == []


def test_bare_small_integers_are_not_treated_as_claims():
    """Flagging '3 tests' would bury the real signal in noise."""
    trajectory = _traj(final="Fixed 3 tests across 2 files.", results=[("terminal", "ok")])
    assert check_unmeasured_claims(trajectory) == []


def test_claims_are_only_checked_against_successful_observations():
    """Output from a failed call is not evidence for a claim."""
    steps = (
        Step(kind=TOOL_CALL, tool="terminal", args={}, call_id="c1"),
        Step(kind=TOOL_RESULT, call_id="c1", content="speedup 9.9x", ok=False),
        Step(kind=FINAL, content="Achieved 9.9x speedup."),
    )
    trajectory = AgentTrajectory(task="t", steps=steps, success=True)
    assert [s.code for s in check_unmeasured_claims(trajectory)] == [UNMEASURED_CLAIM]


# --- combined report ---------------------------------------------------------------


def test_clean_run_produces_no_signals():
    trajectory = _traj(final="fixed", results=[("terminal", "2 passed")])
    report = check_integrity(trajectory, verification_tools=("terminal",))
    assert report.clean and not report.disqualified


def test_enforce_overrides_a_green_verifier():
    """A cheated pass is not a weak pass; it is not a pass."""
    from hermesbench.integrity import DISQUALIFYING, IntegritySignal

    dirty = IntegrityReport((IntegritySignal(PROTECTED_PATH_MODIFIED, DISQUALIFYING, "tests changed"),))
    assert enforce(True, dirty) is False
    assert enforce(True, IntegrityReport()) is True


def test_a_warning_alone_does_not_override_the_verifier():
    trajectory = _traj(final="35% faster", results=[("terminal", "ok")])
    report = check_integrity(trajectory, verification_tools=("terminal",))
    assert report.warnings
    assert not report.disqualified
    assert enforce(True, report) is True


def test_report_is_json_safe():
    trajectory = _traj(final="99% faster", results=[("terminal", "ok")])
    report = check_integrity(trajectory, verification_tools=("terminal",))
    assert json.loads(json.dumps(report.to_record()))["disqualified"] is False


# --- tournament consequences -------------------------------------------------------

PIN = "sha256:h"


def _cand(model, passed=True, disqualified=False) -> CandidateRun:
    return CandidateRun(
        model=model,
        trajectory_sha256=f"sha256:{model}",
        verdict=Verdict(passed=passed, verifier="v"),
        harness_digest=PIN,
        disqualified=disqualified,
        disqualification_reason="protected path modified" if disqualified else "",
    )


def _task() -> TaskSpec:
    return TaskSpec.from_record(
        {"task_id": "t", "prompt": "p", "domain": ["swe"], "action": ["debug"], "verification": "unit_tests"}
    )


def test_a_disqualified_candidate_cannot_win():
    t = Tournament(task=_task(), candidates=(_cand("cheater", disqualified=True), _cand("honest")))
    winner, _ = select_winner(t)
    assert winner.model == "honest"


def test_a_disqualified_candidate_alone_yields_no_winner():
    """Better no training data than data that teaches the exploit."""
    t = Tournament(task=_task(), candidates=(_cand("cheater", disqualified=True),))
    winner, reasons = select_winner(t)
    assert winner is None
    assert "no_candidate_passed" in reasons


def test_a_disqualified_run_becomes_a_dpo_rejection():
    t = Tournament(task=_task(), candidates=(_cand("honest"), _cand("cheater", disqualified=True)))
    artifacts = build_artifacts(t)
    assert artifacts.sft_trajectory_sha256 == "sha256:honest"
    assert [p.rejected_model for p in artifacts.dpo_pairs] == ["cheater"]


def test_gaming_the_verifier_does_not_raise_the_capability_score():
    t = Tournament(task=_task(), candidates=(_cand("cheater", disqualified=True),))
    db = update_capabilities(CapabilityDB(), [t])
    record = db.get("cheater", t.task.bucket, harness=PIN)
    assert record.attempts == 1
    assert record.verified_successes == 0
