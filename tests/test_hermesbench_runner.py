import pytest

from hermes.trajectory import (
    FINAL,
    THINKING,
    TOOL_CALL,
    TOOL_RESULT,
    AgentTrajectory,
    Step,
    TrajectoryError,
    validate,
)
from hermesbench.runner import (
    LocalToolExecutor,
    ReplayPolicy,
    SandboxError,
    _agent_batches,
    run_episode,
    run_suite,
)
from hermesbench.tasks import Checkpoint, Task


class ScriptedPolicy:
    """Emits a fixed list of step batches, one per turn."""

    def __init__(self, batches):
        self.batches = list(batches)
        self.seen_histories = []

    def next_steps(self, task, history):
        self.seen_histories.append(list(history))
        return self.batches.pop(0) if self.batches else [Step(kind=FINAL, content="out of script")]

    @property
    def tokens_used(self):
        return 42


class RecordingExecutor:
    def __init__(self, outcomes=None):
        self.calls = []
        self.outcomes = outcomes or {}
        self.envs = []

    def execute(self, tool, args, *, workspace, env=None):
        self.calls.append((tool, args))
        self.envs.append(env)
        return self.outcomes.get(tool, (True, f"{tool} ok"))


def _task(**overrides) -> Task:
    record = {
        "task_id": "demo",
        "prompt": "do the thing",
        "verify": "test -f done.txt",
        "tools": ["terminal", "edit"],
        "max_steps": 20,
        "timeout_s": 30,
    }
    record.update(overrides)
    return Task.from_record(record)


def test_runner_appends_the_real_result_after_each_call(tmp_path):
    policy = ScriptedPolicy(
        [
            [Step(kind=THINKING, content="go"), Step(kind=TOOL_CALL, tool="terminal", args={}, call_id="c1")],
            [Step(kind=FINAL, content="done")],
        ]
    )
    executor = RecordingExecutor({"terminal": (False, "boom")})
    result = run_episode(_task(), policy, executor, tmp_path / "ws")

    kinds = [s.kind for s in result.trajectory.steps]
    assert kinds == [THINKING, TOOL_CALL, TOOL_RESULT, FINAL]
    observed = result.trajectory.steps[2]
    assert (observed.ok, observed.content, observed.call_id) == (False, "boom", "c1")


def test_policy_cannot_supply_its_own_tool_result(tmp_path):
    """A policy that fabricates a result must not have it believed."""
    policy = ScriptedPolicy(
        [
            [
                Step(kind=TOOL_CALL, tool="terminal", args={}, call_id="c1"),
                Step(kind=TOOL_RESULT, call_id="c1", content="I claim this passed", ok=True),
            ],
            [Step(kind=FINAL, content="done")],
        ]
    )
    executor = RecordingExecutor({"terminal": (False, "actually failed")})
    result = run_episode(_task(), policy, executor, tmp_path / "ws")

    results = [s for s in result.trajectory.steps if s.kind == TOOL_RESULT]
    assert [s.content for s in results] == ["actually failed"]
    assert [s.ok for s in results] == [False]
    assert executor.calls == [("terminal", {})]
    # The fabricated row is dropped entirely rather than sitting beside the real one,
    # which would also make the trajectory schema-invalid (two results for one call).
    validate(result.trajectory)


def test_success_comes_from_verification_not_the_final_message(tmp_path):
    policy = ScriptedPolicy(
        [
            [Step(kind=TOOL_CALL, tool="terminal", args={}, call_id="c1")],
            [Step(kind=FINAL, content="Everything passes, task complete!")],
        ]
    )
    result = run_episode(_task(), policy, RecordingExecutor(), tmp_path / "ws")
    assert result.verification.passed is False
    assert result.trajectory.success is False
    assert result.metrics.success is False


def test_verified_success_when_the_workspace_really_changed(tmp_path):
    policy = ScriptedPolicy(
        [
            [Step(kind=TOOL_CALL, tool="terminal", args={"command": "touch done.txt"}, call_id="c1")],
            [Step(kind=FINAL, content="done")],
        ]
    )
    executor = LocalToolExecutor(allow_unsandboxed=True, timeout_s=30)
    result = run_episode(_task(), policy, executor, tmp_path / "ws")
    assert result.verification.passed is True
    assert result.metrics.success is True


def test_executed_episodes_are_stamped_as_such(tmp_path):
    policy = ScriptedPolicy([[Step(kind=TOOL_CALL, tool="terminal", args={}, call_id="c1")], [Step(kind=FINAL)]])
    result = run_episode(_task(), policy, RecordingExecutor(), tmp_path / "ws")
    assert result.trajectory.metadata["executed"] is True


def test_produced_trajectory_is_schema_valid(tmp_path):
    policy = ScriptedPolicy(
        [
            [Step(kind=THINKING, content="a"), Step(kind=TOOL_CALL, tool="edit", args={}, call_id="c1")],
            [Step(kind=TOOL_CALL, tool="terminal", args={}, call_id="c2")],
            [Step(kind=FINAL, content="done")],
        ]
    )
    result = run_episode(_task(), policy, RecordingExecutor(), tmp_path / "ws")
    validate(result.trajectory)


def test_step_budget_terminates_a_looping_policy(tmp_path):
    class Forever:
        counter = 0

        def next_steps(self, task, history):
            Forever.counter += 1
            return [Step(kind=TOOL_CALL, tool="terminal", args={}, call_id=f"c{Forever.counter}")]

        @property
        def tokens_used(self):
            return 0

    result = run_episode(_task(max_steps=6), Forever(), RecordingExecutor(), tmp_path / "ws")
    assert result.metrics.max_steps_hit is True
    assert result.trajectory.steps[-1].kind == FINAL


def test_policy_returning_nothing_terminates_instead_of_hanging(tmp_path):
    """An empty model response must end the episode; the step budget can never trip."""

    class Idle:
        def next_steps(self, task, history):
            return []

        @property
        def tokens_used(self):
            return 0

    result = run_episode(_task(max_steps=5), Idle(), RecordingExecutor(), tmp_path / "ws")
    assert result.trajectory.steps[-1].kind == FINAL
    assert "no actionable steps" in result.trajectory.steps[-1].content
    assert result.metrics.success is False


def test_policy_emitting_only_fabricated_results_terminates(tmp_path):
    """Every step gets dropped as fabricated, so the turn adds nothing -- still must end."""

    class OnlyFakeResults:
        def next_steps(self, task, history):
            return [Step(kind=TOOL_RESULT, call_id="c1", content="I did it", ok=True)]

        @property
        def tokens_used(self):
            return 0

    result = run_episode(_task(max_steps=5), OnlyFakeResults(), RecordingExecutor(), tmp_path / "ws")
    assert result.trajectory.steps[-1].kind == FINAL
    assert result.metrics.tool_calls == 0


def test_max_steps_is_enforced_inside_a_single_batch(tmp_path):
    """A budget checked only between turns does not bound host execution at all."""

    class Flood:
        def next_steps(self, task, history):
            return [Step(kind=TOOL_CALL, tool="terminal", args={}, call_id=f"c{i}") for i in range(50)]

        @property
        def tokens_used(self):
            return 0

    executor = RecordingExecutor()
    result = run_episode(_task(max_steps=4), Flood(), executor, tmp_path / "ws")
    assert len(executor.calls) == 4
    assert result.metrics.max_steps_hit is True


def test_step_budget_counts_agent_steps_not_observations(tmp_path):
    """Runner-appended results must not eat half the agent's declared budget."""

    class Caller:
        def __init__(self):
            self.n = 0

        def next_steps(self, task, history):
            self.n += 1
            return [Step(kind=TOOL_CALL, tool="terminal", args={}, call_id=f"c{self.n}")]

        @property
        def tokens_used(self):
            return 0

    executor = RecordingExecutor()
    run_episode(_task(max_steps=6), Caller(), executor, tmp_path / "ws")
    assert len(executor.calls) == 6


def test_tool_outside_the_task_allowlist_is_refused_not_executed(tmp_path):
    policy = ScriptedPolicy(
        [
            [Step(kind=TOOL_CALL, tool="browser", args={"url": "http://x"}, call_id="c1")],
            [Step(kind=FINAL, content="done")],
        ]
    )
    executor = RecordingExecutor()
    result = run_episode(_task(tools=["file_read"]), policy, executor, tmp_path / "ws")

    assert executor.calls == []
    refusal = [s for s in result.trajectory.steps if s.kind == TOOL_RESULT][0]
    assert refusal.ok is False
    assert "not available for this task" in refusal.content


def test_episode_that_invented_a_tool_is_not_trainable(tmp_path):
    """The refusal is recorded, but the episode must not become training data.

    The call is still paired with its refusal result, so the trajectory is well-formed;
    it fails validation on the allowlist alone. That is the wanted outcome — training on
    it would teach the student that `browser` exists on a file_read-only task.
    """
    from hermes.format import convert

    policy = ScriptedPolicy(
        [
            [Step(kind=TOOL_CALL, tool="browser", args={}, call_id="c1")],
            [Step(kind=FINAL, content="done")],
        ]
    )
    result = run_episode(_task(tools=["file_read"]), policy, RecordingExecutor(), tmp_path / "ws")

    with pytest.raises(TrajectoryError, match="not in tools_available"):
        validate(result.trajectory)
    records, skipped = convert([result.trajectory])
    assert records == []
    assert "not in tools_available" in skipped[0]


def test_harness_authored_final_is_flagged_in_metadata(tmp_path):
    """'step budget exhausted' is the harness talking; it must be filterable."""

    class Forever:
        def next_steps(self, task, history):
            return [Step(kind=TOOL_CALL, tool="terminal", args={}, call_id=f"c{len(history)}")]

        @property
        def tokens_used(self):
            return 0

    result = run_episode(_task(max_steps=4), Forever(), RecordingExecutor(), tmp_path / "ws")
    assert result.trajectory.metadata["harness_final"] is True


def test_agent_authored_final_is_not_flagged(tmp_path):
    policy = ScriptedPolicy(
        [[Step(kind=TOOL_CALL, tool="terminal", args={}, call_id="c1")], [Step(kind=FINAL, content="done")]]
    )
    result = run_episode(_task(), policy, RecordingExecutor(), tmp_path / "ws")
    assert result.trajectory.metadata["harness_final"] is False


def test_setup_failure_is_flagged_on_the_metrics_and_the_trajectory(tmp_path):
    policy = ScriptedPolicy([[Step(kind=FINAL, content="never runs")]])
    result = run_episode(_task(setup="exit 3"), policy, RecordingExecutor(), tmp_path / "ws")
    assert result.metrics.setup_failed is True
    assert result.trajectory.metadata["harness_final"] is True


def test_setup_failures_do_not_deflate_the_suite_success_rate(tmp_path):
    """Infrastructure breakage must not read as a worse model."""
    tasks = [_task(task_id="ok"), _task(task_id="broken", setup="exit 3")]

    def factory(task):
        return ScriptedPolicy(
            [
                [Step(kind=TOOL_CALL, tool="terminal", args={"command": "touch done.txt"}, call_id="c1")],
                [Step(kind=FINAL, content="done")],
            ]
        )

    metrics, _ = run_suite(tasks, factory, LocalToolExecutor(allow_unsandboxed=True, timeout_s=30), tmp_path / "root")
    assert metrics.setup_failures == 1
    assert metrics.episodes == 1
    assert metrics.success_rate == 1.0


def test_task_env_reaches_the_executor(tmp_path):
    policy = ScriptedPolicy([[Step(kind=TOOL_CALL, tool="terminal", args={}, call_id="c1")], [Step(kind=FINAL)]])
    executor = RecordingExecutor()
    run_episode(_task(env={"HERMES_MARKER": "set"}), policy, executor, tmp_path / "ws")
    assert executor.envs[0]["HERMES_MARKER"] == "set"


def test_setup_failure_short_circuits_the_episode(tmp_path):
    policy = ScriptedPolicy([[Step(kind=FINAL, content="never runs")]])
    executor = RecordingExecutor()
    result = run_episode(_task(setup="exit 3"), policy, executor, tmp_path / "ws")
    assert result.setup_failed is True
    assert result.metrics.success is False
    assert executor.calls == []


def test_local_executor_refuses_without_the_sandbox_assertion():
    with pytest.raises(SandboxError, match="allow_unsandboxed"):
        LocalToolExecutor()


def test_local_executor_confines_reads_to_the_workspace(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (tmp_path / "secret.txt").write_text("do not read me")
    executor = LocalToolExecutor(allow_unsandboxed=True)
    ok, output = executor.execute("file_read", {"path": "../secret.txt"}, workspace=workspace)
    assert ok is False
    assert "escapes the task workspace" in output


def test_local_executor_confines_writes_to_the_workspace(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    executor = LocalToolExecutor(allow_unsandboxed=True)
    ok, output = executor.execute("file_write", {"path": "../escaped.txt", "content": "x"}, workspace=workspace)
    assert ok is False
    assert not (tmp_path / "escaped.txt").exists()


def test_local_executor_round_trips_a_file(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    executor = LocalToolExecutor(allow_unsandboxed=True)
    assert executor.execute("file_write", {"path": "a/b.txt", "content": "hi"}, workspace=workspace)[0]
    assert executor.execute("file_read", {"path": "a/b.txt"}, workspace=workspace) == (True, "hi")


def test_local_executor_reports_a_nonzero_exit_as_failure(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    executor = LocalToolExecutor(allow_unsandboxed=True, timeout_s=30)
    ok, _ = executor.execute("terminal", {"command": "exit 7"}, workspace=workspace)
    assert ok is False


def test_local_executor_rejects_an_unknown_tool(tmp_path):
    executor = LocalToolExecutor(allow_unsandboxed=True)
    ok, output = executor.execute("telepathy", {}, workspace=tmp_path)
    assert ok is False
    assert "unknown tool" in output


def test_local_executor_runs_python(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    executor = LocalToolExecutor(allow_unsandboxed=True, timeout_s=60)
    ok, output = executor.execute("python", {"code": "print(6 * 7)"}, workspace=workspace)
    assert ok is True
    assert "42" in output


def test_python_snippet_survives_shell_metacharacters(tmp_path):
    """Snippets go through a file, not a heredoc, so quoting cannot mangle them."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    executor = LocalToolExecutor(allow_unsandboxed=True, timeout_s=60)
    code = "print('it\\'s $HOME `date` \"quoted\" & done')"
    ok, output = executor.execute("python", {"code": code}, workspace=workspace)
    assert ok is True
    assert 'it\'s $HOME `date` "quoted" & done' in output


def test_python_snippet_containing_the_old_heredoc_delimiter_still_runs(tmp_path):
    """A snippet mentioning the delimiter must not break out of its own invocation."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    executor = LocalToolExecutor(allow_unsandboxed=True, timeout_s=60)
    ok, output = executor.execute("python", {"code": "x = 'HERMES_EOF'\nprint('survived', x)"}, workspace=workspace)
    assert ok is True
    assert "survived HERMES_EOF" in output


def test_python_snippet_runs_with_the_workspace_as_cwd(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    executor = LocalToolExecutor(allow_unsandboxed=True, timeout_s=60)
    ok, _ = executor.execute("python", {"code": "open('made_here.txt', 'w').write('x')"}, workspace=workspace)
    assert ok is True
    assert (workspace / "made_here.txt").exists()


def test_python_snippet_file_does_not_pollute_the_workspace(tmp_path):
    """A stray snippet file could show up in a task's own verification."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    executor = LocalToolExecutor(allow_unsandboxed=True, timeout_s=60)
    executor.execute("python", {"code": "print('hi')"}, workspace=workspace)
    assert list(workspace.iterdir()) == []


def test_empty_python_snippet_is_a_failure(tmp_path):
    executor = LocalToolExecutor(allow_unsandboxed=True)
    assert executor.execute("python", {"code": "   "}, workspace=tmp_path) == (False, "empty code")


def test_empty_terminal_command_is_a_failure(tmp_path):
    executor = LocalToolExecutor(allow_unsandboxed=True)
    assert executor.execute("terminal", {"command": ""}, workspace=tmp_path) == (False, "empty command")


def test_agent_batches_split_on_observations():
    steps = (
        Step(kind=THINKING, content="a"),
        Step(kind=TOOL_CALL, tool="t", args={}, call_id="c1"),
        Step(kind=TOOL_RESULT, call_id="c1", content="r"),
        Step(kind=FINAL, content="done"),
    )
    assert [[s.kind for s in b] for b in _agent_batches(steps)] == [[THINKING, TOOL_CALL], [FINAL]]


def test_replay_policy_reissues_the_recorded_calls(tmp_path):
    recorded = AgentTrajectory(
        task="t",
        success=True,
        tools_available=("terminal",),
        steps=(
            Step(kind=TOOL_CALL, tool="terminal", args={"command": "touch done.txt"}, call_id="c1"),
            Step(kind=TOOL_RESULT, call_id="c1", content="stale recorded output"),
            Step(kind=FINAL, content="done"),
        ),
    )
    executor = LocalToolExecutor(allow_unsandboxed=True, timeout_s=30)
    result = run_episode(_task(), ReplayPolicy(recorded), executor, tmp_path / "ws")
    # Re-executed for real, so the recorded output does not carry over.
    assert "stale recorded output" not in [s.content for s in result.trajectory.steps]
    assert result.verification.passed is True


def test_run_suite_aggregates_and_isolates_workspaces(tmp_path):
    tasks = [_task(task_id="a"), _task(task_id="b", verify="test -f nope.txt")]

    def factory(task):
        command = "touch done.txt" if task.task_id == "a" else "true"
        return ScriptedPolicy(
            [
                [Step(kind=TOOL_CALL, tool="terminal", args={"command": command}, call_id="c1")],
                [Step(kind=FINAL, content="done")],
            ]
        )

    metrics, results = run_suite(
        tasks,
        factory,
        LocalToolExecutor(allow_unsandboxed=True, timeout_s=30),
        tmp_path / "root",
    )
    assert metrics.episodes == 2
    assert metrics.success_rate == 0.5
    assert {r.task_id for r in results} == {"a", "b"}
    assert (tmp_path / "root" / "a").is_dir()
    assert (tmp_path / "root" / "b").is_dir()


def test_episode_result_record_is_json_safe(tmp_path):
    import json

    policy = ScriptedPolicy([[Step(kind=TOOL_CALL, tool="terminal", args={}, call_id="c1")], [Step(kind=FINAL)]])
    result = run_episode(_task(), policy, RecordingExecutor(), tmp_path / "ws")
    assert json.loads(json.dumps(result.to_record()))["task_id"] == "demo"


def test_the_grader_cannot_frame_the_agent_for_a_protected_path(tmp_path):
    """Verifiers run shell in the agent's workspace. A file the *grader* creates was
    being attributed to the agent -- and that signal is disqualifying, which is not a
    lower score but a removed candidate, a DPO rejection and a dent in a model's record."""
    task = Task(
        task_id="grader-writes-protected",
        prompt="do nothing",
        verify="touch grader-artifact.txt",
        protected_paths=("grader-artifact.txt",),
        tools=("terminal",),
        max_steps=2,
    )
    policy = ReplayPolicy(
        AgentTrajectory(
            task="t",
            success=True,
            steps=(
                Step(kind=TOOL_CALL, tool="terminal", args={"command": "echo hi"}, call_id="c1"),
                Step(kind=TOOL_RESULT, call_id="c1", content="hi"),
                Step(kind=FINAL, content="done"),
            ),
        )
    )
    result = run_episode(task, policy, executor=LocalToolExecutor(allow_unsandboxed=True), workspace=tmp_path)
    assert [s.code for s in result.integrity.signals] == []
    assert not result.integrity.disqualified


def test_a_hidden_verifier_writing_a_protected_path_is_not_the_agents_doing(tmp_path):
    task = Task(
        task_id="hidden-writes-protected",
        prompt="do nothing",
        verify="true",
        hidden_verify="touch withheld-artifact.txt",
        protected_paths=("withheld-artifact.txt",),
        tools=("terminal",),
        max_steps=2,
    )
    policy = ReplayPolicy(
        AgentTrajectory(
            task="t",
            success=True,
            steps=(
                Step(kind=TOOL_CALL, tool="terminal", args={"command": "echo hi"}, call_id="c1"),
                Step(kind=TOOL_RESULT, call_id="c1", content="hi"),
                Step(kind=FINAL, content="done"),
            ),
        )
    )
    result = run_episode(task, policy, executor=LocalToolExecutor(allow_unsandboxed=True), workspace=tmp_path)
    assert not result.integrity.disqualified
    assert result.metrics.success


def test_the_agent_is_still_caught_when_it_is_the_one_writing(tmp_path):
    """The fix must not blind the check to the thing it exists for."""
    task = Task(
        task_id="agent-writes-protected",
        prompt="do nothing",
        verify="true",
        protected_paths=("guarded.txt",),
        tools=("terminal",),
        max_steps=2,
    )
    policy = ReplayPolicy(
        AgentTrajectory(
            task="t",
            success=True,
            steps=(
                Step(kind=TOOL_CALL, tool="terminal", args={"command": "touch guarded.txt"}, call_id="c1"),
                Step(kind=TOOL_RESULT, call_id="c1", content=""),
                Step(kind=FINAL, content="done"),
            ),
        )
    )
    result = run_episode(task, policy, executor=LocalToolExecutor(allow_unsandboxed=True), workspace=tmp_path)
    assert result.integrity.disqualified


def test_a_checkpoint_writing_a_protected_path_is_not_charged_to_the_agent(tmp_path):
    """Checkpoints interleave with agent steps and run shell in the workspace, so they
    cannot be hoisted outside the digest bracket the way verify was."""
    task = Task(
        task_id="checkpoint-writes-protected",
        prompt="do nothing",
        verify="true",
        checkpoints=(Checkpoint(checkpoint_id="cp1", verify="touch cp-artifact.txt"),),
        checkpoint_every=1,
        protected_paths=("cp-artifact.txt",),
        tools=("terminal",),
        max_steps=3,
    )
    policy = ReplayPolicy(
        AgentTrajectory(
            task="t",
            success=True,
            steps=(
                Step(kind=TOOL_CALL, tool="terminal", args={"command": "echo hi"}, call_id="c1"),
                Step(kind=TOOL_RESULT, call_id="c1", content="hi"),
                Step(kind=FINAL, content="done"),
            ),
        )
    )
    result = run_episode(task, policy, executor=LocalToolExecutor(allow_unsandboxed=True), workspace=tmp_path)
    assert not result.integrity.disqualified


# --- repeats: the flag existed and did nothing ------------------------------------------


def _one_shot():
    """A policy that touches the verify file and stops, so the episode passes."""
    return ScriptedPolicy(
        [
            [Step(kind=TOOL_CALL, tool="terminal", args={"command": "touch done.txt"}, call_id="c1")],
            [Step(kind=FINAL, content="done")],
        ]
    )


def _repeat_task(**overrides):
    return _task(task_id="echo", verify="true", **overrides)


def test_repeats_runs_each_task_that_many_times(tmp_path):
    metrics, results = run_suite([_repeat_task()], lambda t: _one_shot(), RecordingExecutor(), tmp_path, repeats=3)
    assert len(results) == 3
    assert metrics.episodes == 3


def test_each_repeat_gets_its_own_workspace(tmp_path):
    """Sharing one across attempts lets the second start from the edits the first left
    behind, and an agent that inherits a finished repository looks both more capable and
    more efficient than it is."""
    run_suite([_repeat_task()], lambda t: _one_shot(), RecordingExecutor(), tmp_path, repeats=3)
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_dir()) == ["echo", "echo#1", "echo#2"]


def test_a_single_run_keeps_the_original_workspace_layout(tmp_path):
    """Existing callers and every written manifest expect the bare task id."""
    run_suite([_repeat_task()], lambda t: _one_shot(), RecordingExecutor(), tmp_path)
    assert [p.name for p in tmp_path.iterdir() if p.is_dir()] == ["echo"]


def test_zero_repeats_is_refused(tmp_path):
    with pytest.raises(ValueError, match="at least once"):
        run_suite([_repeat_task()], lambda t: _one_shot(), RecordingExecutor(), tmp_path, repeats=0)


def test_every_attempt_is_kept_not_just_the_best(tmp_path):
    """Keeping the best would make the pass rate a best-of-k order statistic -- 1-(1-p)^k,
    which converges to 1.0 for any p above zero. Measuring the spread is the point."""
    from hermesbench.runner import repeated_from

    _metrics, results = run_suite([_repeat_task()], lambda t: _one_shot(), RecordingExecutor(), tmp_path, repeats=4)
    assert repeated_from(results, repeats=4).tasks[0].attempts == 4


def test_flakiness_is_reported_per_task(tmp_path):
    """A task that does not agree with itself is named, rather than averaged away."""
    from hermesbench.runner import repeated_from

    attempt = {"n": 0}

    def alternating(_task):
        attempt["n"] += 1
        return _one_shot()

    tasks = [_task(task_id="steady", verify="true"), _task(task_id="doomed", verify="false")]
    _metrics, results = run_suite(tasks, alternating, RecordingExecutor(), tmp_path, repeats=3)
    repeated = repeated_from(results, repeats=3)
    by_id = {t.task_id: t for t in repeated.tasks}
    assert by_id["steady"].passes == 3
    assert by_id["doomed"].passes == 0
    # Neither is flaky: both agreed with themselves. Flakiness is disagreement, not failure.
    assert repeated.flaky_tasks == ()


# --- sharding a baseline across processes -----------------------------------------------------


def test_task_ids_selects_a_shard(capsys):
    """A repeated baseline is the only thing that measures the run-to-run spread, and it is
    the most expensive run here -- 19 tasks x 10 repeats was ~9 hours sequentially. Episodes
    are already independent, so the only thing blocking parallel shards was naming a task."""
    from hermesbench.runner import main

    assert (
        main(
            [
                "--suite",
                "all",
                "--workspace-root",
                "/tmp/shard-ls",
                "--list",
                "--task-ids",
                "fix-failing-test,migrate-and-keep-green",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "fix-failing-test" in out
    assert "migrate-and-keep-green" in out
    assert "lh-i18n-catalog-parity" not in out


def test_an_unknown_task_id_is_refused_rather_than_skipped(capsys):
    """A typo would otherwise shrink one shard silently, and the merged baseline would be
    short some attempts with nothing to show it."""
    from hermesbench.runner import main

    assert (
        main(
            [
                "--suite",
                "all",
                "--workspace-root",
                "/tmp/shard-ls",
                "--list",
                "--task-ids",
                "fix-failing-test,fix-failing-tests",
            ]
        )
        == 2
    )
    assert "no such task(s)" in capsys.readouterr().err


def test_shards_partition_the_suite_with_no_task_lost_or_duplicated():
    """The property that makes merging shard results a baseline rather than a sample."""
    from hermesbench.tasks import load_suite

    ids = [t.task_id for t in load_suite("all")]
    shards = [ids[i::4] for i in range(4)]
    flat = [i for s in shards for i in s]
    assert sorted(flat) == sorted(ids)
    assert len(flat) == len(set(flat)) == 19
