"""Long-horizon scoring: sub-objectives, goal drift, and the verification budget."""

import tempfile
from pathlib import Path

import pytest

from hermes.trajectory import FINAL, TOOL_CALL, TOOL_RESULT, AgentTrajectory, Step
from hermesbench.metrics import analyse_checkpoints, episode_metrics, suite_metrics
from hermesbench.runner import run_episode
from hermesbench.tasks import Checkpoint, Task, TaskError, load_suite
from hermesbench.verify import run_checkpoints, setup_task, verify_task

V1 = load_suite("v1")
# Written when v1 held only long-horizon tasks. It now spans all four capability
# categories, so checkpoint-specific invariants must select the tasks they apply to
# rather than assuming the whole suite.
LONG_HORIZON = [t for t in V1 if t.category == "long_horizon"]


# --- checkpoint schema -------------------------------------------------------------


def test_checkpoint_requires_id_and_verify():
    with pytest.raises(TaskError, match="missing required field"):
        Checkpoint.from_record({"checkpoint_id": "a"})


def test_duplicate_checkpoint_ids_are_rejected():
    """Duplicates would collapse in the timeline and could mask a regression."""
    record = {
        "task_id": "t",
        "prompt": "p",
        "verify": "true",
        "tools": ["terminal"],
        "checkpoints": [
            {"checkpoint_id": "same", "verify": "true"},
            {"checkpoint_id": "same", "verify": "false"},
        ],
    }
    with pytest.raises(TaskError, match="duplicate checkpoint_id"):
        Task.from_record(record)


def test_short_tasks_are_not_long_horizon():
    assert all(not t.is_long_horizon for t in load_suite("v0"))


def test_long_horizon_tasks_declare_checkpoints():
    """Only the long_horizon category needs them; the other three legitimately do not."""
    assert LONG_HORIZON
    assert all(t.is_long_horizon for t in LONG_HORIZON)


def test_non_long_horizon_tasks_are_not_required_to_have_checkpoints():
    others = [t for t in V1 if t.category and t.category != "long_horizon"]
    assert others, "v1 should span more than one category"


def test_checkpoint_every_must_be_positive():
    record = {
        "task_id": "t",
        "prompt": "p",
        "verify": "true",
        "tools": ["terminal"],
        "checkpoints": [{"checkpoint_id": "a", "verify": "true"}],
        "checkpoint_every": 0,
    }
    with pytest.raises(TaskError, match="checkpoint_every"):
        Task.from_record(record)


# --- drift detection ---------------------------------------------------------------


def test_regression_is_detected_when_a_passing_objective_later_fails():
    timeline = [{"a": False}, {"a": True}, {"a": False}]
    total, met, regressed = analyse_checkpoints(timeline)
    assert (total, met, regressed) == (1, 0, 1)


def test_never_passing_objective_is_not_a_regression():
    """Failing throughout is incompetence, not drift; conflating them hides both."""
    total, met, regressed = analyse_checkpoints([{"a": False}, {"a": False}])
    assert regressed == 0


def test_objective_repaired_before_the_end_is_not_a_regression():
    total, met, regressed = analyse_checkpoints([{"a": True}, {"a": False}, {"a": True}])
    assert (met, regressed) == (1, 0)


def test_multiple_regressions_are_counted():
    timeline = [{"a": True, "b": True, "c": True}, {"a": False, "b": False, "c": True}]
    total, met, regressed = analyse_checkpoints(timeline)
    assert (total, met, regressed) == (3, 1, 2)


def test_empty_timeline_yields_zeros():
    assert analyse_checkpoints([]) == (0, 0, 0)


def test_short_task_reports_completion_as_not_applicable():
    """-1.0, not 0.0: a task with no objectives must not average in as a total failure."""
    trajectory = AgentTrajectory(
        task="t",
        steps=(
            Step(kind=TOOL_CALL, tool="terminal", args={}, call_id="c1"),
            Step(kind=TOOL_RESULT, call_id="c1", content="ok"),
            Step(kind=FINAL),
        ),
        success=True,
    )
    m = episode_metrics(trajectory, task_id="t", verified_success=True)
    assert m.objective_completion == -1.0
    assert m.objectives_total == 0


def test_suite_excludes_short_tasks_from_long_horizon_aggregates():
    def _ep(task_id, timeline):
        trajectory = AgentTrajectory(
            task="t",
            steps=(
                Step(kind=TOOL_CALL, tool="terminal", args={}, call_id="c1"),
                Step(kind=TOOL_RESULT, call_id="c1", content="ok"),
                Step(kind=FINAL),
            ),
            success=True,
        )
        return episode_metrics(trajectory, task_id=task_id, verified_success=True, checkpoint_timeline=timeline)

    short = _ep("short", [])
    long_ok = _ep("long", [{"a": True, "b": True}])
    metrics = suite_metrics([short, long_ok])

    assert metrics.long_horizon_episodes == 1
    assert metrics.objective_completion == 1.0
    assert metrics.objective_regression_rate == 0.0


def test_suite_reports_the_regression_rate():
    trajectory = AgentTrajectory(
        task="t",
        steps=(
            Step(kind=TOOL_CALL, tool="terminal", args={}, call_id="c1"),
            Step(kind=TOOL_RESULT, call_id="c1", content="ok"),
            Step(kind=FINAL),
        ),
        success=False,
    )
    drifted = episode_metrics(
        trajectory,
        task_id="d",
        verified_success=False,
        checkpoint_timeline=[{"a": True, "b": True}, {"a": False, "b": True}],
    )
    metrics = suite_metrics([drifted])
    assert metrics.objective_regression_rate == 0.5
    assert metrics.objective_completion == 0.5


# --- verification budget -----------------------------------------------------------


class _Executor:
    def __init__(self):
        self.calls = []

    def execute(self, tool, args, *, workspace, env=None):
        self.calls.append(tool)
        return True, "ok"


def _budget_task(**overrides):
    record = {
        "task_id": "budget",
        "prompt": "p",
        "verify": "true",
        "tools": ["file_write", "terminal"],
        "verification_tools": ["terminal"],
        "max_steps": 3,
        "max_verification_steps": 10,
        "timeout_s": 30,
    }
    record.update(overrides)
    return Task.from_record(record)


def test_verification_calls_do_not_consume_the_action_budget(tmp_path):
    """Otherwise the harness penalises exactly what self_check_rate rewards."""

    class Checker:
        def __init__(self):
            self.n = 0

        def next_steps(self, task, history):
            self.n += 1
            return [Step(kind=TOOL_CALL, tool="terminal", args={}, call_id=f"v{self.n}")]

        @property
        def tokens_used(self):
            return 0

    executor = _Executor()
    run_episode(_budget_task(), Checker(), executor, tmp_path / "ws")
    # max_steps is 3, but every call is a verification tool, so the far larger
    # verification allowance is what bounds the episode.
    assert len(executor.calls) == 10


def test_action_budget_still_bounds_non_verification_calls(tmp_path):
    class Writer:
        def __init__(self):
            self.n = 0

        def next_steps(self, task, history):
            self.n += 1
            return [Step(kind=TOOL_CALL, tool="file_write", args={"path": "a", "content": "x"}, call_id=f"w{self.n}")]

        @property
        def tokens_used(self):
            return 0

    executor = _Executor()
    run_episode(_budget_task(), Writer(), executor, tmp_path / "ws")
    assert len(executor.calls) == 3


def test_a_task_without_verification_tools_keeps_one_shared_budget(tmp_path):
    """Backwards compatible: v0 tasks declare none and behave exactly as before."""

    class Caller:
        def __init__(self):
            self.n = 0

        def next_steps(self, task, history):
            self.n += 1
            return [Step(kind=TOOL_CALL, tool="terminal", args={}, call_id=f"c{self.n}")]

        @property
        def tokens_used(self):
            return 0

    executor = _Executor()
    run_episode(_budget_task(verification_tools=[]), Caller(), executor, tmp_path / "ws")
    assert len(executor.calls) == 3


# --- the shipped v1 tasks ----------------------------------------------------------


@pytest.mark.parametrize("task", V1, ids=lambda t: t.task_id)
def test_v1_task_starts_with_every_objective_unmet(task):
    """If an objective passes before the agent acts, it measures nothing."""
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        setup = setup_task(task, workspace)
        assert setup is None or setup.passed, f"setup failed: {setup.stderr[:400]}"
        assert not any(run_checkpoints(task, workspace).values())
        assert not verify_task(task, workspace).passed


@pytest.mark.parametrize("task", LONG_HORIZON, ids=lambda t: t.task_id)
def test_long_horizon_task_declares_interacting_objectives(task):
    """Drift needs objectives that can break each other; isolated ones cannot regress."""
    assert len(task.checkpoints) >= 3
    assert task.checkpoint_every >= 1


@pytest.mark.parametrize("task", V1, ids=lambda t: t.task_id)
def test_v1_prompt_does_not_leak_the_grader(task):
    assert task.verify.strip() not in task.prompt
    for checkpoint in task.checkpoints:
        assert checkpoint.verify.strip() not in task.prompt


def test_v1_checkpoint_ids_are_unique_across_the_suite():
    for task in V1:
        ids = [c.checkpoint_id for c in task.checkpoints]
        assert len(ids) == len(set(ids))


# --- suite-wide invariants for v1 ---------------------------------------------------


@pytest.mark.parametrize("task", V1, ids=lambda t: t.task_id)
def test_every_v1_task_declares_a_capability_category(task):
    """A task with no category vanishes from the per-category breakdown."""
    from hermesbench import HERMES_CATEGORIES

    assert task.category in HERMES_CATEGORIES, f"{task.task_id} has tags {task.tags}"


@pytest.mark.parametrize("task", LONG_HORIZON, ids=lambda t: t.task_id)
def test_no_v1_checkpoint_passes_before_the_agent_acts(task):
    """An objective already satisfied at setup measures nothing.

    Stronger than the final-verify check: a task can fail overall while quietly handing
    out free checkpoints, which would inflate objective_completion for doing nothing.
    """
    from hermesbench.verify import run_checkpoints

    if not task.checkpoints:
        pytest.skip("no checkpoints")
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        setup = setup_task(task, workspace)
        assert setup is None or setup.passed, setup.stderr[:300]
        passing = [k for k, ok in run_checkpoints(task, workspace).items() if ok]
        assert not passing, f"{task.task_id}: {passing} pass before any work"
