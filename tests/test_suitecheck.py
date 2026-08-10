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
