"""Executing the lazy strategy: the gate that catches a trap which is merely decorative."""

import pytest

from hermesbench.shortcut_sweep import BROKEN, CAUGHT, ESCAPED, UNRESOLVED, run_shortcut, sweep, undeclared
from hermesbench.tasks import Shortcut, Task, TaskError, load_suite


def _task(**kw):
    base = dict(
        task_id="t",
        prompt="do the thing",
        # Passes only when the agent wrote the real answer into result.txt.
        verify="grep -qx 42 result.txt",
        tools=("terminal",),
        setup="printf 'the answer is 42\\n' > notes.txt",
        timeout_s=60,
    )
    base.update(kw)
    return Task(**base)


def _shortcut(apply, expectation="fails_public", shortcut_id="s"):
    return Shortcut(shortcut_id=shortcut_id, apply=apply, expectation=expectation)


# --- the finding this module exists for ------------------------------------------------------


def test_a_decorative_trap_is_reported_as_escaped(tmp_path):
    """The whole point. A task whose lazy strategy PASSES the published check is easier than it
    claims, and no amount of "fresh workspace fails, correct solution passes, deterministic"
    reveals that -- all three stay true. Only running the shortcut does."""
    task = _task()
    # The giveaway: setup left the answer lying in notes.txt, so copying it out is enough.
    lazy = _shortcut("grep -o '[0-9]*' notes.txt > result.txt")
    outcome = run_shortcut(task, lazy, tmp_path / "ws")
    assert outcome.verdict == ESCAPED
    assert outcome.public_passed is True
    assert "decorative" in outcome.detail
    assert outcome.is_failure is True


def test_a_real_trap_is_reported_as_caught(tmp_path):
    task = _task(setup="printf 'no answer here\\n' > notes.txt")
    lazy = _shortcut("grep -o '[0-9]*' notes.txt > result.txt || true")
    outcome = run_shortcut(task, lazy, tmp_path / "ws")
    assert outcome.verdict == CAUGHT
    assert outcome.public_passed is False
    assert outcome.is_failure is False


# --- the overfit expectation ------------------------------------------------------------------


def test_an_overfit_shortcut_needs_the_withheld_body_and_says_so_when_absent(tmp_path):
    """A sweep that reported success for a check it could not run would certify the exact
    property it failed to test, which is worse than reporting nothing."""
    task = _task()  # no hidden_verify in this checkout
    lazy = _shortcut("printf '42\\n' > result.txt", expectation="passes_public_fails_hidden")
    outcome = run_shortcut(task, lazy, tmp_path / "ws")
    assert outcome.verdict == UNRESOLVED
    assert outcome.public_passed is True
    assert outcome.is_failure is False  # a public clone must still be able to go green


def test_an_overfit_shortcut_caught_by_the_withheld_check(tmp_path):
    """Passes what is published, rejected by what is withheld -- the overfit signal itself."""
    task = _task(hidden_verify="grep -qx 42 result.txt && test -f worked_it_out")
    lazy = _shortcut("printf '42\\n' > result.txt", expectation="passes_public_fails_hidden")
    outcome = run_shortcut(task, lazy, tmp_path / "ws")
    assert outcome.verdict == CAUGHT
    assert (outcome.public_passed, outcome.hidden_passed) == (True, False)


def test_an_overfit_shortcut_that_passes_both_checks_is_a_failure(tmp_path):
    """Nothing distinguishes it from a correct solution, so overfit_rate cannot fire on it."""
    task = _task(hidden_verify="grep -qx 42 result.txt")
    lazy = _shortcut("printf '42\\n' > result.txt", expectation="passes_public_fails_hidden")
    outcome = run_shortcut(task, lazy, tmp_path / "ws")
    assert outcome.verdict == ESCAPED
    assert outcome.hidden_passed is True


def test_an_overfit_declaration_that_the_public_check_already_rejects_is_a_failure(tmp_path):
    """It measures nothing about overfitting, so either the expectation or the script is wrong,
    and silently accepting it would leave a task believing it has overfit coverage."""
    task = _task()
    lazy = _shortcut("printf 'wrong\\n' > result.txt", expectation="passes_public_fails_hidden")
    outcome = run_shortcut(task, lazy, tmp_path / "ws")
    assert outcome.verdict == ESCAPED
    assert "PUBLISHED check already rejects it" in outcome.detail


# --- the shortcut stands in for an agent, not for a grader -------------------------------------


def test_the_shortcut_runs_with_the_agents_environment_not_the_graders(tmp_path):
    """`for_verification=True` sets PYTHONSAFEPATH and strips agent-writable PATH entries.
    Handing those to a shortcut would give the lazy strategy protections the real agent never
    has, and a shortcut that only fails under them is not actually caught."""
    task = _task(
        setup="printf 'no answer here\\n' > notes.txt",
        verify="test -f saw_unsafe_path",
    )
    lazy = _shortcut('test -z "$PYTHONSAFEPATH" && touch saw_unsafe_path')
    outcome = run_shortcut(task, lazy, tmp_path / "ws")
    # The shortcut observed no PYTHONSAFEPATH, so it created the file and `verify` passed --
    # reported as ESCAPED here, which is exactly the signal proving the agent env was used.
    assert outcome.verdict == ESCAPED


# --- a shortcut that cannot run is not a verdict about the task --------------------------------


def test_a_shortcut_script_that_fails_is_reported_as_broken_not_as_a_caught_trap(tmp_path):
    """Otherwise a typo in the script reads as a strong trap -- wrong in the dangerous
    direction, because it certifies a task nobody actually probed."""
    outcome = run_shortcut(_task(), _shortcut("exit 3"), tmp_path / "ws")
    assert outcome.verdict == BROKEN
    assert outcome.is_failure is True
    assert "tested nothing" in outcome.detail


def test_broken_task_setup_is_reported_as_broken(tmp_path):
    outcome = run_shortcut(_task(setup="exit 1"), _shortcut("true"), tmp_path / "ws")
    assert outcome.verdict == BROKEN
    assert "task setup failed" in outcome.detail


def test_each_shortcut_gets_a_fresh_workspace(tmp_path):
    """A shortcut that mutates the tree would otherwise contaminate the next one, and the
    contamination reads as a stronger trap."""
    task = _task(setup="printf 'no answer here\\n' > notes.txt")
    first = _shortcut("printf '42\\n' > result.txt", shortcut_id="a")
    second = _shortcut("true", shortcut_id="b")
    ws = tmp_path / "ws"
    assert run_shortcut(task, first, ws).verdict == ESCAPED
    # Same directory, rebuilt: `b` writes nothing, so result.txt must not survive from `a`.
    assert run_shortcut(task, second, ws).verdict == CAUGHT


# --- parsing ----------------------------------------------------------------------------------


def test_shortcuts_parse_from_a_task_record():
    task = Task.from_record(
        {
            "task_id": "t",
            "prompt": "p",
            "verify": "true",
            "tools": ["terminal"],
            "shortcuts": [{"shortcut_id": "lazy", "apply": "true", "description": "d"}],
        }
    )
    assert task.shortcuts[0].shortcut_id == "lazy"
    assert task.shortcuts[0].expectation == "fails_public"


def test_an_unknown_expectation_is_a_load_error_not_an_assertion_that_does_nothing():
    with pytest.raises(TaskError, match="is not one of"):
        Task.from_record(
            {
                "task_id": "t",
                "prompt": "p",
                "verify": "true",
                "tools": ["terminal"],
                "shortcuts": [{"shortcut_id": "s", "apply": "true", "expectation": "fails_hidden"}],
            }
        )


def test_a_shortcut_missing_its_script_is_refused():
    with pytest.raises(TaskError, match="missing required field"):
        Task.from_record(
            {"task_id": "t", "prompt": "p", "verify": "true", "tools": ["t"], "shortcuts": [{"shortcut_id": "s"}]}
        )


def test_duplicate_shortcut_ids_are_refused():
    """They would collapse in the report, so one that started winning could hide behind a
    same-named one that still loses."""
    with pytest.raises(TaskError, match="duplicate shortcut_id"):
        Task.from_record(
            {
                "task_id": "t",
                "prompt": "p",
                "verify": "true",
                "tools": ["t"],
                "shortcuts": [
                    {"shortcut_id": "s", "apply": "true"},
                    {"shortcut_id": "s", "apply": "false"},
                ],
            }
        )


def test_a_task_with_no_shortcuts_still_loads():
    """The field is additive: the corpus predates it and most tasks have no declared trap."""
    assert Task.from_record({"task_id": "t", "prompt": "p", "verify": "true", "tools": ["t"]}).shortcuts == ()


# --- an untested trap claim is visible ---------------------------------------------------------


def test_a_prose_trap_claim_with_no_shortcut_is_flagged():
    """How both known defects shipped: a trap described in prose and never executed."""
    claimed = _task(prompt="Careful, this one has a trap in it.")
    assert undeclared([claimed]) == ["t"]
    assert undeclared([_task(prompt="Careful, this one has a trap in it.", shortcuts=(_shortcut("true"),))]) == []


# --- the real corpus --------------------------------------------------------------------------


def test_every_declared_shortcut_in_the_corpus_is_caught(tmp_path):
    """The regression guard. Any task whose declared lazy strategy starts passing the published
    check fails here, which is the check that was missing when tc-log-rotation-order and
    lh-i18n-catalog-parity shipped."""
    declared = [t for t in load_suite("all") if t.shortcuts]
    assert declared, "no task declares a shortcut; the sweep would be vacuous"
    outcomes = sweep(declared, tmp_path)
    failures = [f"{o.task_id}#{o.shortcut_id}: {o.verdict} -- {o.detail}" for o in outcomes if o.is_failure]
    assert not failures, "\n".join(failures)
