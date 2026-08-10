"""Environments: task + harness + verifier + state, as one replaceable thing."""

import json
from pathlib import Path

import pytest

from hermes.trajectory import FINAL, TOOL_CALL, TOOL_RESULT, AgentTrajectory, Step
from hermesbench.environment import EnvState, Observation, Reward, WorkspaceEnvironment
from hermesbench.runner import LocalToolExecutor
from hermesbench.tasks import Task


class _Executor:
    def __init__(self, outcomes=None):
        self.calls = []
        self.outcomes = outcomes or {}

    def execute(self, tool, args, *, workspace, env=None):
        self.calls.append((tool, args))
        return self.outcomes.get(tool, (True, f"{tool} ok"))


def _task(**overrides) -> Task:
    record = {
        "task_id": "env-demo",
        "prompt": "do the thing",
        "verify": "test -f done.txt",
        "tools": ["terminal", "file_read"],
        "timeout_s": 30,
        "max_steps": 20,
    }
    record.update(overrides)
    return Task.from_record(record)


def _call(tool="terminal", **args) -> Step:
    return Step(kind=TOOL_CALL, tool=tool, args=args, call_id="c1")


# --- reset / state -----------------------------------------------------------------


def test_reset_builds_the_workspace_and_returns_state(tmp_path):
    env = WorkspaceEnvironment(_Executor(), tmp_path)
    state = env.reset(_task(setup="touch marker"))
    assert state.workspace is not None and state.workspace.exists()
    assert (state.workspace / "marker").exists()
    assert state.step_count == 0


def test_state_is_explicit_rather_than_implicit_in_a_directory(tmp_path):
    """State that only exists as 'whatever is on disk' cannot be compared to a claim."""
    env = WorkspaceEnvironment(_Executor(), tmp_path)
    state = env.reset(_task())
    assert isinstance(state, EnvState)
    assert json.loads(json.dumps(state.to_record()))["task_id"] == "env-demo"


def test_protected_digests_are_taken_after_setup_and_before_the_agent(tmp_path):
    """Earlier records an empty workspace; later records the agent's edits as baseline."""
    env = WorkspaceEnvironment(_Executor(), tmp_path)
    state = env.reset(_task(setup="echo original > guarded.txt", protected_paths=["guarded.txt"]))
    assert state.protected_digests["guarded.txt"] is not None


def test_setup_failure_is_reported_without_raising(tmp_path):
    env = WorkspaceEnvironment(_Executor(), tmp_path)
    env.reset(_task(setup="exit 3"))
    assert env.setup_failed is True
    assert env.setup_result is not None and not env.setup_result.passed


def test_step_before_reset_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="before reset"):
        WorkspaceEnvironment(_Executor(), tmp_path).step(_call())


def test_verify_before_reset_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="before reset"):
        WorkspaceEnvironment(_Executor(), tmp_path).verify()


# --- step --------------------------------------------------------------------------


def test_step_performs_the_action_and_reports_the_observation(tmp_path):
    executor = _Executor({"terminal": (False, "boom")})
    env = WorkspaceEnvironment(executor, tmp_path)
    env.reset(_task())
    observation = env.step(_call())

    assert isinstance(observation, Observation)
    assert (observation.ok, observation.content) == (False, "boom")
    assert executor.calls == [("terminal", {})]


def test_a_tool_outside_the_task_is_refused_without_executing(tmp_path):
    executor = _Executor()
    env = WorkspaceEnvironment(executor, tmp_path)
    env.reset(_task(tools=["file_read"]))
    observation = env.step(_call("terminal"))

    assert observation.ok is False
    assert "not available for this task" in observation.content
    assert executor.calls == []


def test_step_count_advances(tmp_path):
    env = WorkspaceEnvironment(_Executor(), tmp_path)
    state = env.reset(_task())
    env.step(_call())
    env.step(_call())
    assert state.step_count == 2


def test_checkpoints_are_sampled_over_time(tmp_path):
    task = _task(
        setup="touch a",
        checkpoint_every=1,
        checkpoints=[{"checkpoint_id": "exists", "verify": "test -f a"}],
    )
    env = WorkspaceEnvironment(_Executor(), tmp_path)
    state = env.reset(task)
    env.step(_call())
    env.step(_call())
    # one sample at reset, one per step
    assert len(state.checkpoint_timeline) == 3


# --- verify ------------------------------------------------------------------------


def test_verify_reflects_the_workspace_not_the_agent(tmp_path):
    env = WorkspaceEnvironment(LocalToolExecutor(allow_unsandboxed=True, timeout_s=30), tmp_path)
    env.reset(_task())
    assert env.verify().passed is False

    env.step(_call("terminal", command="touch done.txt"))
    assert env.verify().passed is True


def test_withheld_checks_are_separate_from_published_ones(tmp_path):
    task = _task(verify="true", hidden_verify="test -f secret.txt")
    env = WorkspaceEnvironment(LocalToolExecutor(allow_unsandboxed=True, timeout_s=30), tmp_path)
    env.reset(task)
    assert env.verify().passed is True
    assert env.verify_withheld().passed is False


def test_no_withheld_checks_reports_none(tmp_path):
    env = WorkspaceEnvironment(_Executor(), tmp_path)
    env.reset(_task())
    assert env.verify_withheld() is None


# --- reward ------------------------------------------------------------------------


def test_a_failed_run_earns_nothing(tmp_path):
    """A reward that pays for a tidy failure is one an agent can farm by failing tidily."""
    env = WorkspaceEnvironment(_Executor(), tmp_path)
    env.reset(_task())
    reward = env.reward(verified=False, tool_calls=3, failed_calls=0)
    assert reward.total == 0.0
    assert reward.verified is False


def test_success_dominates_the_reward(tmp_path):
    env = WorkspaceEnvironment(_Executor(), tmp_path)
    env.reset(_task())
    sloppy_success = env.reward(verified=True, tool_calls=80, failed_calls=40)
    clean_failure = env.reward(verified=False, tool_calls=3, failed_calls=0)
    assert sloppy_success.total > clean_failure.total


def test_efficiency_separates_two_successes(tmp_path):
    env = WorkspaceEnvironment(_Executor(), tmp_path)
    env.reset(_task())
    lean = env.reward(verified=True, tool_calls=10, failed_calls=0)
    flailing = env.reward(verified=True, tool_calls=80, failed_calls=0)
    assert lean.total > flailing.total


def test_cleanliness_separates_two_equally_long_successes(tmp_path):
    env = WorkspaceEnvironment(_Executor(), tmp_path)
    env.reset(_task())
    clean = env.reward(verified=True, tool_calls=20, failed_calls=0)
    messy = env.reward(verified=True, tool_calls=20, failed_calls=15)
    assert clean.total > messy.total


def test_reward_never_exceeds_one(tmp_path):
    env = WorkspaceEnvironment(_Executor(), tmp_path)
    env.reset(_task())
    assert env.reward(verified=True, tool_calls=0, failed_calls=0).total <= 1.0


def test_reward_components_are_visible(tmp_path):
    """Shaping is where gaming enters, so the parts have to be inspectable."""
    env = WorkspaceEnvironment(_Executor(), tmp_path)
    env.reset(_task())
    reward = env.reward(verified=True, tool_calls=10, failed_calls=1)
    assert set(reward.components) == {"success", "efficiency", "cleanliness"}
    assert json.loads(json.dumps(reward.to_record()))["verified"] is True


def test_reward_record_is_json_safe():
    assert json.loads(json.dumps(Reward(verified=False, total=0.0).to_record()))["total"] == 0.0


# --- the interface ------------------------------------------------------------------


def test_workspace_environment_satisfies_the_protocol(tmp_path):
    """A second environment (browser, emulator, hardware) implements this, not a rewrite."""
    from hermesbench.environment import Environment

    env: Environment = WorkspaceEnvironment(_Executor(), tmp_path)
    assert hasattr(env, "reset") and hasattr(env, "step")
    assert hasattr(env, "verify") and hasattr(env, "reward")


def test_snapshot_describes_the_environment_and_its_state(tmp_path):
    env = WorkspaceEnvironment(_Executor(), tmp_path)
    env.reset(_task())
    snapshot = env.snapshot()
    assert snapshot["environment"] == "workspace"
    assert snapshot["state"]["task_id"] == "env-demo"
    assert json.loads(json.dumps(snapshot))["setup_failed"] is False


# --- regressions found by adversarial review ---------------------------------------


def test_a_failed_setup_stops_the_episode_rather_than_scoring_it(tmp_path):
    """A half-built world scored a perfect run: setup broke, verify passed, reward 1.0."""
    env = WorkspaceEnvironment(LocalToolExecutor(allow_unsandboxed=True, timeout_s=30), tmp_path)
    env.reset(_task(setup="touch done.txt; exit 3"))

    assert env.setup_failed
    observation = env.step(_call("terminal", command="echo hi"))
    assert observation.ok is False
    assert "setup failed" in observation.content
    assert env.verify().passed is False


def test_reset_cleans_the_workspace_so_a_rerun_does_not_start_solved(tmp_path):
    """One env reused across a suite handed the second run a free pass at max reward."""
    env = WorkspaceEnvironment(LocalToolExecutor(allow_unsandboxed=True, timeout_s=30), tmp_path)
    task = _task()

    env.reset(task)
    env.step(_call("terminal", command="touch done.txt"))
    assert env.verify().passed is True

    env.reset(task)
    assert env.verify().passed is False, "second episode inherited the first one's work"


def test_a_refused_call_still_advances_the_clock(tmp_path):
    """Counting only executed calls let a disallowed-tool loop run forever."""
    env = WorkspaceEnvironment(_Executor(), tmp_path)
    state = env.reset(_task(tools=["file_read"]))
    for _ in range(5):
        env.step(_call("terminal"))
    assert state.step_count == 5


def test_outcome_enforces_integrity_where_verify_alone_does_not(tmp_path):
    """verify() is green for a run that overwrote a protected file; outcome() is not."""
    env = WorkspaceEnvironment(LocalToolExecutor(allow_unsandboxed=True, timeout_s=30), tmp_path)
    env.reset(
        _task(
            setup="echo original > guarded.txt; touch done.txt",
            protected_paths=["guarded.txt"],
            tools=["terminal"],
        )
    )
    env.step(_call("terminal", command="echo tampered > guarded.txt"))

    assert env.verify().passed is True
    outcome = env.outcome()
    assert outcome.integrity_disqualified is True
    assert outcome.verified is False


def test_outcome_requires_the_withheld_checks_too(tmp_path):
    env = WorkspaceEnvironment(LocalToolExecutor(allow_unsandboxed=True, timeout_s=30), tmp_path)
    env.reset(_task(verify="true", hidden_verify="test -f secret.txt"))
    outcome = env.outcome()
    assert outcome.public.passed is True
    assert outcome.verified is False
    assert outcome.overfit is True


def test_outcome_is_verified_when_everything_holds(tmp_path):
    env = WorkspaceEnvironment(LocalToolExecutor(allow_unsandboxed=True, timeout_s=30), tmp_path)
    env.reset(_task(verify="true", hidden_verify="true"))
    outcome = env.outcome()
    assert outcome.verified is True and outcome.overfit is False
    assert json.loads(json.dumps(outcome.to_record()))["verified"] is True


def test_verify_is_idempotent(tmp_path):
    """Each unguarded call lengthened the timeline and re-ran every checkpoint."""
    task = _task(setup="touch a", checkpoint_every=1, checkpoints=[{"checkpoint_id": "e", "verify": "test -f a"}])
    env = WorkspaceEnvironment(_Executor(), tmp_path)
    state = env.reset(task)
    for _ in range(3):
        env.verify()
    assert len(state.checkpoint_timeline) == 2  # one at reset, one final


def test_a_reset_that_raises_does_not_desynchronise_task_from_state(tmp_path):
    """Holding the new task with the old workspace verified work never attempted."""
    env = WorkspaceEnvironment(_Executor(), tmp_path)
    env.reset(_task(task_id="task-a"))
    first_state = env.state

    blocker = tmp_path / "task-b"
    blocker.write_text("not a directory")  # rmtree/mkdir will fail on a file
    with pytest.raises(Exception):
        env.reset(_task(task_id="task-b"))

    assert env.task is not None and env.task.task_id == "task-a"
    assert env.state is first_state


def test_reward_cannot_go_negative_on_malformed_counts():
    """failed_calls > tool_calls drove a verified run below an unverified one's floor.

    Reachable, not theoretical: load_jsonl does not call validate(), so a trajectory with
    more results than calls can be loaded and measured.
    """
    env = WorkspaceEnvironment(_Executor(), Path("/tmp"))
    reward = env.reward(verified=True, tool_calls=1, failed_calls=500)
    assert 0.0 <= reward.total <= 1.0
    assert reward.total > env.reward(verified=False, tool_calls=1, failed_calls=0).total


def test_reward_ignores_negative_inputs():
    env = WorkspaceEnvironment(_Executor(), Path("/tmp"))
    assert 0.0 <= env.reward(verified=True, tool_calls=-5, failed_calls=-5).total <= 1.0


def test_doing_nothing_is_not_the_highest_scoring_verified_run():
    """Otherwise 'do nothing' is optimal the moment any upstream bug lets a no-op verify."""
    env = WorkspaceEnvironment(_Executor(), Path("/tmp"))
    no_op = env.reward(verified=True, tool_calls=0, failed_calls=0)
    real_work = env.reward(verified=True, tool_calls=12, failed_calls=0)
    assert real_work.total > no_op.total


# --- the anti-cheat must say what it did not check ----------------------------------


def test_outcome_without_a_trajectory_reports_that_the_check_was_partial(tmp_path):
    """Two of the three detectors read the trajectory; without it they return clean
    without having looked, and a third of the anti-cheat is reported as all of it."""
    from hermesbench.integrity import INTEGRITY_PARTIAL

    env = WorkspaceEnvironment(executor=None, root=tmp_path)
    env.reset(_task(verification_tools=("terminal",)))
    outcome = env.outcome()
    assert outcome.fully_checked is False
    assert INTEGRITY_PARTIAL in [s.code for s in outcome.integrity.signals]


def test_outcome_with_a_trajectory_runs_every_detector(tmp_path):
    env = WorkspaceEnvironment(executor=None, root=tmp_path)
    env.reset(_task(verification_tools=("terminal",)))
    trajectory = AgentTrajectory(
        task="p",
        success=True,
        steps=(
            Step(kind=TOOL_CALL, tool="terminal", args={}, call_id="c1"),
            Step(kind=TOOL_RESULT, call_id="c1", content="ok"),
            Step(kind=FINAL, content="done"),
        ),
    )
    assert env.outcome(trajectory).fully_checked is True


def test_an_unmeasured_claim_is_now_reachable_on_the_environment_path(tmp_path):
    """It could never fire before: the trajectory passed in had no final answer."""
    env = WorkspaceEnvironment(executor=None, root=tmp_path)
    env.reset(_task(verification_tools=("terminal",)))
    trajectory = AgentTrajectory(
        task="p",
        success=True,
        steps=(
            Step(kind=TOOL_CALL, tool="terminal", args={}, call_id="c1"),
            Step(kind=TOOL_RESULT, call_id="c1", content="ok"),
            Step(kind=FINAL, content="Latency improved by 27.3%."),
        ),
    )
    codes = [s.code for s in env.outcome(trajectory).integrity.signals]
    assert "unmeasured_claim" in codes


def test_a_verifier_writing_a_protected_path_is_not_charged_to_the_agent(tmp_path):
    env = WorkspaceEnvironment(executor=None, root=tmp_path)
    env.reset(_task(verify="touch grader-made-this.txt", protected_paths=("grader-made-this.txt",)))
    assert not env.outcome().integrity_disqualified


def test_outcome_is_idempotent_when_a_grader_writes_a_protected_path(tmp_path):
    """Recomputing the digest per call made the verdict order-dependent: the verifier
    runs shell in this workspace, so the second call charged its file to the agent."""
    env = WorkspaceEnvironment(executor=None, root=tmp_path)
    env.reset(_task(verify="touch grader-made-this.txt", protected_paths=("grader-made-this.txt",)))
    first = env.outcome().integrity_disqualified
    second = env.outcome().integrity_disqualified
    assert first is False and second is False


def test_calling_verify_before_outcome_does_not_frame_the_agent(tmp_path):
    """This is the sequence outcome()'s own docstring warns callers about."""
    env = WorkspaceEnvironment(executor=None, root=tmp_path)
    env.reset(_task(verify="touch grader-made-this.txt", protected_paths=("grader-made-this.txt",)))
    env.verify()
    assert not env.outcome().integrity_disqualified


def test_a_real_agent_write_is_still_caught_on_every_call(tmp_path):
    """The fix must not blind the check to the thing it exists for."""
    env = WorkspaceEnvironment(executor=LocalToolExecutor(allow_unsandboxed=True), root=tmp_path)
    env.reset(_task(verify="true", protected_paths=("guarded.txt",)))
    env.step(Step(kind=TOOL_CALL, tool="terminal", args={"command": "touch guarded.txt"}, call_id="c1"))
    assert env.outcome().integrity_disqualified
    assert env.outcome().integrity_disqualified
