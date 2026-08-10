"""The check that a verifier still measures something."""

from hermesbench.suitecheck import check_task
from hermesbench.tasks import Task, load_suite


def _task(**overrides) -> Task:
    record = {"task_id": "t", "prompt": "p", "verify": "test -f done.txt", "tools": ["terminal"]}
    record.update(overrides)
    return Task.from_record(record)


def test_a_verifier_that_passes_an_untouched_workspace_is_reported(tmp_path):
    """The most damaging defect a task can have: every model scores a free point."""
    problems = check_task(_task(verify="true"), tmp_path)
    assert problems and "PASSES an untouched workspace" in problems[0]


def test_a_verifier_that_needs_work_done_is_fine(tmp_path):
    assert check_task(_task(), tmp_path) == []


def test_a_task_whose_setup_fails_is_reported_rather_than_scored(tmp_path):
    """Every episode on it would be scored as a failure the model never caused."""
    problems = check_task(_task(setup="exit 3"), tmp_path)
    assert problems and "setup failed" in problems[0]


def test_a_withheld_check_that_passes_an_untouched_workspace_is_reported(tmp_path):
    problems = check_task(_task(hidden_verify="true"), tmp_path)
    assert any("withheld verifier passes" in p for p in problems)


def test_every_shipped_task_still_measures_something(tmp_path):
    """The regression guard for the whole suite, run without a model."""
    for task in load_suite("all"):
        assert check_task(task, tmp_path) == [], f"{task.task_id} no longer measures the agent"


# --- a verifier that cannot run is invisible to the fresh-workspace assertion ------------------


def _no_bare_python(name, *a, **k):
    """A machine with python3 and no `python` alias -- i.e. the GPU box, and any stock host."""
    import shutil

    return None if name == "python" else shutil.which(name, *a, **k)


def test_the_lint_catches_the_grader_the_short_circuit_hid():
    """`verify-speedup-claim` read `test -f winner.txt && python - <<EOF`. On a clean workspace
    the `test` failed first, so the command exited 1 -- which is exactly what suitecheck asserts
    a verifier must do. The short-circuit satisfied the check while the grader could never PASS:
    once an agent created winner.txt the interpreter lookup ran and exited 127. It scored 0/10 on
    the first real baseline and read as a capability gap in the model."""
    from unittest.mock import patch

    from hermesbench.suitecheck import missing_commands

    with patch("hermesbench.suitecheck.shutil.which", _no_bare_python):
        assert missing_commands("test -f winner.txt && python - <<'EOF'\nimport sys\nEOF") == ["python"]


def test_the_resolved_form_is_not_flagged():
    """`command` is a shell builtin, so `shutil.which` cannot see it. An incomplete builtin list
    would have failed precisely the graders that had been repaired."""
    from unittest.mock import patch

    from hermesbench.suitecheck import missing_commands

    fixed = 'set -e\nPY="$(command -v python3 || command -v python)"\n"$PY" - <<\'EOF\'\nimport sys\nEOF'
    with patch("hermesbench.suitecheck.shutil.which", _no_bare_python):
        assert missing_commands(fixed) == []


def test_heredoc_bodies_are_not_scanned_for_commands():
    """Their contents are input to a program, not commands. Scanning them would flag every
    Python identifier at the start of a line."""
    from hermesbench.suitecheck import missing_commands

    script = "cat > f.py <<'PYEOF'\nnotacommand_at_all()\nimport os\nPYEOF"
    assert missing_commands(script) == []


def test_paths_and_variables_are_left_alone():
    """`./bin/tool` and `$PY` resolve elsewhere; flagging them would be noise."""
    from hermesbench.suitecheck import missing_commands

    assert missing_commands('"$PY" -c "pass"\n./bin/thing --flag\nVAR=1 make') == []


def test_only_interpreters_are_reported_and_that_is_deliberate():
    """The narrowing is the design, not an oversight. A general "find every command" version was
    written first and produced false positives immediately, because
    `PYTHONPATH=deps python3 -c "` opens a multi-line quoted Python string and a line-oriented
    reader treats `import` and `raise` as commands. A lint that cries wolf gets switched off,
    taking the real finding with it.

    The trade is defensible on the asymmetry: an absent `jq` fails a verifier loudly and
    immediately, while an absent interpreter alias fails identically to a model that cannot do
    the task -- which is what cost a full baseline run."""
    from hermesbench.suitecheck import INTERPRETERS, missing_commands

    assert missing_commands("definitely-not-installed-xyz --run") == []
    assert "python" in INTERPRETERS


def test_every_shipped_verifier_can_run_on_this_machine():
    """The standing guard. A task whose grader cannot run scores 0 for every model and reads as
    a capability gap, which is the most expensive way for a benchmark to be wrong."""
    from hermesbench.suitecheck import missing_commands
    from hermesbench.tasks import load_suite

    broken = {}
    for task in load_suite("all"):
        absent = missing_commands(task.verify)
        if absent:
            broken[task.task_id] = absent
    assert not broken, f"verifiers invoking absent commands: {broken}"
