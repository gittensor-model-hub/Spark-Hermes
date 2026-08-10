"""Verification evidence, read from what the trajectory observed rather than from tool names."""

from hermes.trajectory import FINAL, TOOL_CALL, TOOL_RESULT, AgentTrajectory, Step
from hermesbench.metrics import episode_metrics
from hermesbench.verification import VerificationEvidence, at_least_as_strong, evidence


def _call(cid, tool="terminal", **args):
    return Step(kind=TOOL_CALL, tool=tool, args=args, call_id=cid)


def _res(cid, ok=True):
    return Step(kind=TOOL_RESULT, call_id=cid, content="out", ok=ok)


def _traj(steps, **kw):
    return AgentTrajectory(task="t", steps=tuple(steps), success=True, **kw)


# --- the exploit these signals exist to catch ---------------------------------------------

READ_BACK = _traj(
    [
        _call("c1", tool="edit", path="a.py"),
        _res("c1"),
        _call("c2", tool="file_read", path="a.py"),  # read back what you just wrote
        _res("c2"),
        Step(kind=FINAL, content="done"),
    ]
)

REAL_VERIFY = _traj(
    [
        _call("c1", command="pytest -q"),
        _res("c1", ok=False),  # observed failing
        _call("c2", tool="edit", path="a.py"),
        _res("c2"),
        _call("c3", command="pytest -q"),  # same command again
        _res("c3", ok=True),  # observed passing
        Step(kind=FINAL, content="done"),
    ]
)


def test_the_existing_signals_cannot_separate_them():
    """Both set `self_checked`. That is the hole: a read-back is indistinguishable from a
    test run, so a capsule that reads back wins on tokens *because it verified less*."""
    read_back = episode_metrics(READ_BACK, task_id="t", verified_success=True)
    real = episode_metrics(REAL_VERIFY, task_id="t", verified_success=True)
    assert read_back.self_checked is True
    assert real.self_checked is True


def test_demonstrated_repairs_separates_them():
    """An `ls` cannot manufacture a fail-then-pass transition of the same command."""
    assert evidence(READ_BACK).demonstrated_repairs == 0
    assert evidence(REAL_VERIFY).demonstrated_repairs == 1


def test_the_read_back_is_refused_as_weaker():
    ok, reason = at_least_as_strong(evidence(READ_BACK), evidence(REAL_VERIFY))
    assert ok is False
    assert "demonstrated_repairs" in reason
    assert "stop checking" in reason


# --- what makes a repair count -------------------------------------------------------------


def test_a_pass_without_a_prior_failure_is_not_a_repair():
    """Running a check twice and seeing it pass twice demonstrates nothing was broken."""
    t = _traj(
        [
            _call("c1", command="pytest -q"),
            _res("c1", ok=True),
            _call("c2", tool="edit", path="a.py"),
            _res("c2"),
            _call("c3", command="pytest -q"),
            _res("c3", ok=True),
            Step(kind=FINAL, content="done"),
        ]
    )
    e = evidence(t)
    assert e.rechecked == 1
    assert e.demonstrated_repairs == 0


def test_a_recovery_with_no_intervening_change_is_not_a_repair():
    """A flaky command that failed and then passed on its own is not work the agent did."""
    t = _traj(
        [
            _call("c1", command="pytest -q"),
            _res("c1", ok=False),
            _call("c2", command="pytest -q"),
            _res("c2", ok=True),
            Step(kind=FINAL, content="done"),
        ]
    )
    assert evidence(t).demonstrated_repairs == 0


def test_a_different_command_is_not_the_same_check():
    """`pytest tests/a.py` and `pytest tests/b.py` are different checks; conflating them
    would let an agent claim a repair it never observed."""
    t = _traj(
        [
            _call("c1", command="pytest tests/a.py"),
            _res("c1", ok=False),
            _call("c2", tool="edit", path="a.py"),
            _res("c2"),
            _call("c3", command="pytest tests/b.py"),
            _res("c3", ok=True),
            Step(kind=FINAL, content="done"),
        ]
    )
    assert evidence(t).demonstrated_repairs == 0


def test_a_call_whose_result_was_never_observed_is_not_evidence():
    t = _traj(
        [
            _call("c1", command="pytest -q"),
            _res("c1", ok=False),
            _call("c2", tool="edit", path="a.py"),
            _res("c2"),
            _call("c3", command="pytest -q"),  # no result step
            Step(kind=FINAL, content="done"),
        ]
    )
    assert evidence(t).demonstrated_repairs == 0


# --- the comparison refuses rather than resolves --------------------------------------------


def test_an_incomparable_pair_is_refused_not_weighted():
    """There is no defensible exchange rate between watching a failing test pass and looking
    at a file twice. Refusing is the conservative direction: the rule exists to stop an
    efficiency win that cost verification."""
    stronger_repairs = VerificationEvidence(
        observed_after_mutation=1, rechecked=1, demonstrated_repairs=2, mutated=True
    )
    stronger_reads = VerificationEvidence(observed_after_mutation=9, rechecked=1, demonstrated_repairs=0, mutated=True)
    ok, reason = at_least_as_strong(stronger_reads, stronger_repairs)
    assert ok is False
    assert "demonstrated_repairs" in reason


def test_equal_evidence_passes():
    e = evidence(REAL_VERIFY)
    assert at_least_as_strong(e, e) == (True, "")


def test_strictly_stronger_passes():
    weak = VerificationEvidence(observed_after_mutation=1, rechecked=1, demonstrated_repairs=1, mutated=True)
    strong = VerificationEvidence(observed_after_mutation=2, rechecked=2, demonstrated_repairs=2, mutated=True)
    assert at_least_as_strong(strong, weak)[0] is True


def test_a_baseline_that_changed_nothing_sets_no_bar():
    """No mutation means there was nothing to verify, so there is no bar to clear."""
    nothing = VerificationEvidence(observed_after_mutation=0, rechecked=0, demonstrated_repairs=0, mutated=False)
    empty = VerificationEvidence(observed_after_mutation=0, rechecked=0, demonstrated_repairs=0, mutated=True)
    assert at_least_as_strong(empty, nothing) == (True, "")


# --- the record states its own limits --------------------------------------------------------


def test_the_record_refuses_to_look_like_a_score():
    """A single number would invite the false confidence `verification_strength: 0.92`
    implies, and comparing two of those is comparing two guesses."""
    record = evidence(REAL_VERIFY).to_record()
    assert record["is_a_score"] is False
    assert record["demonstrated_repairs_can_be_self_manufactured"] is True


def test_a_self_manufactured_repair_is_counted_and_labelled():
    """The residual hole, stated rather than hidden: an agent can break something trivial and
    fix it. Strictly harder than a read-back, still not proof."""
    t = _traj(
        [
            _call("c1", command="test -f x"),
            _res("c1", ok=False),
            _call("c2", tool="edit", path="x"),
            _res("c2"),
            _call("c3", command="test -f x"),
            _res("c3", ok=True),
            Step(kind=FINAL, content="done"),
        ]
    )
    assert evidence(t).demonstrated_repairs == 1
    assert evidence(t).to_record()["demonstrated_repairs_can_be_self_manufactured"] is True
