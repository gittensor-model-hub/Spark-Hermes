"""The episode log has to be useful to whoever reads it *after* the run died.

Every property here is about that reader: the bytes are on disk before the next episode
starts, a half-written last line does not poison the file, and a line that is complete but
wrong is reported rather than quietly dropped.
"""

import json

import pytest

from hermesbench.sink import JsonlEpisodeSink, SinkError, read_episode_prefix, read_episodes


class _Metrics:
    def __init__(self, record):
        self.record = record

    def to_record(self):
        return dict(self.record)


class _Integrity:
    def __init__(self, disqualified=False):
        self.disqualified = disqualified


class _Result:
    """The four attributes a sink reads off an `EpisodeResult`."""

    def __init__(self, task_id, *, metrics=None, setup_failed=False, disqualified=False):
        self.task_id = task_id
        self.setup_failed = setup_failed
        self.metrics = _Metrics(metrics if metrics is not None else {"task_id": task_id, "success": True})
        self.integrity = _Integrity(disqualified)


def test_each_episode_is_readable_before_the_next_one_starts(tmp_path):
    """The measured failure: shard logs at 0 bytes until the shard finished.

    Read through a second handle with the sink still open, which is exactly the position a
    post-mortem reader is in when the writer has been killed.
    """
    log = tmp_path / "episodes.jsonl"
    sink = JsonlEpisodeSink(log)
    sink.append(_Result("first"))
    assert [e["task_id"] for e in read_episodes(log)] == ["first"]
    sink.append(_Result("second"))
    assert [e["task_id"] for e in read_episodes(log)] == ["first", "second"]
    sink.close()


def test_a_torn_final_line_does_not_make_the_whole_log_unreadable(tmp_path):
    """A kill mid-write must cost the episode in flight and not the 188 before it."""
    log = tmp_path / "episodes.jsonl"
    with JsonlEpisodeSink(log) as sink:
        sink.append(_Result("done"))
    with log.open("a", encoding="utf-8") as handle:
        handle.write('{"episode": 1, "task_id": "half-writ')

    episodes = list(read_episode_prefix(log))
    assert [e["task_id"] for e in episodes] == ["done"]
    with pytest.raises(SinkError, match="truncated"):
        list(read_episodes(log))


def test_a_complete_line_that_does_not_parse_is_refused_not_skipped(tmp_path):
    """Corruption in the middle is not an interrupted write.

    Skipping it would silently shrink the episode count, and a success rate computed over
    a quietly shortened log is wrong in the direction nobody questions.
    """
    log = tmp_path / "episodes.jsonl"
    with JsonlEpisodeSink(log) as sink:
        sink.append(_Result("a"))
    with log.open("a", encoding="utf-8") as handle:
        handle.write("}not json at all{\n")
        handle.write(json.dumps({"episode": 2}) + "\n")

    with pytest.raises(SinkError, match="line 2"):
        list(read_episodes(log))


def test_appending_to_a_log_that_already_holds_a_run_is_refused(tmp_path):
    """Two runs in one file read as one suite, and truncating destroys the evidence."""
    log = tmp_path / "episodes.jsonl"
    with JsonlEpisodeSink(log) as sink:
        sink.append(_Result("from-the-first-run"))

    with pytest.raises(SinkError, match="already holds"):
        JsonlEpisodeSink(log)
    # The refusal must not have cost the previous run its log.
    assert [e["task_id"] for e in read_episodes(log)] == ["from-the-first-run"]


def test_a_log_killed_before_its_first_episode_reads_as_no_episodes(tmp_path):
    """An empty file is a run that got nowhere, not a parse error."""
    log = tmp_path / "episodes.jsonl"
    JsonlEpisodeSink(log).close()
    assert list(read_episodes(log)) == []


def test_a_field_containing_a_newline_stays_on_one_line(tmp_path):
    """One line, one episode -- even when a metric carries a shell transcript.

    A raw newline in the payload would split one episode into two lines, the second of
    which is not JSON, and the reader would call the whole log corrupt.
    """
    log = tmp_path / "episodes.jsonl"
    with JsonlEpisodeSink(log) as sink:
        sink.append(_Result("noisy", metrics={"stderr": "line one\nline two\n"}))

    assert len(log.read_text(encoding="utf-8").splitlines()) == 1
    assert list(read_episodes(log))[0]["metrics"]["stderr"] == "line one\nline two\n"


def test_the_sink_holds_no_collection_that_grows_with_the_run(tmp_path):
    """Buffering the run here would double the memory a suite already spends on
    trajectories, and a sink that makes a long run likelier to be OOM-killed is a sink
    that gets switched off -- which is the state it exists to end."""
    log = tmp_path / "episodes.jsonl"
    with JsonlEpisodeSink(log) as sink:
        for i in range(25):
            sink.append(_Result(f"task-{i}"))
        assert sink.episodes_written == 25
        assert [v for v in vars(sink).values() if isinstance(v, (list, dict, set, tuple))] == []


def test_the_line_records_a_disqualification_the_success_flag_cannot_show(tmp_path):
    """'failed' and 'cheated and was caught' are different post-mortems."""
    log = tmp_path / "episodes.jsonl"
    with JsonlEpisodeSink(log) as sink:
        sink.append(_Result("cheater", disqualified=True))
    assert list(read_episodes(log))[0]["disqualified"] is True


def test_lines_are_numbered_in_run_order(tmp_path):
    """The k-th line for a task id names the workspace it left behind (`id`, `id#1`, ...),
    which is how a reader walks from a log line to the directory to inspect."""
    log = tmp_path / "episodes.jsonl"
    with JsonlEpisodeSink(log) as sink:
        for task_id in ("a", "b", "b"):
            sink.append(_Result(task_id))
    episodes = list(read_episodes(log))
    assert [(e["episode"], e["task_id"]) for e in episodes] == [(0, "a"), (1, "b"), (2, "b")]


# --- wired into run_suite ---------------------------------------------------------------------


def test_run_suite_records_each_episode_as_it_finishes(tmp_path):
    """The property the sink exists for: a run that dies partway leaves the episodes that
    already completed. Asserted by reading the file DURING the run rather than after, because
    a check that only looks at the end cannot tell incremental writing from a final dump."""
    from hermes.trajectory import FINAL, AgentTrajectory, Step
    from hermesbench.runner import ReplayPolicy, run_suite
    from hermesbench.sink import JsonlEpisodeSink, read_episodes
    from hermesbench.tasks import Task

    out = tmp_path / "episodes.jsonl"
    sink = JsonlEpisodeSink(out)
    seen_midway: list[int] = []

    tasks = [Task(task_id=f"t{i}", prompt="p", verify="true", tools=("terminal",), setup=None) for i in range(3)]

    class Executor:
        def execute(self, tool, args, *, workspace, env=None):
            return True, "ok"

    def policy_factory(task):
        # Count the lines already on disk each time a new episode starts. If the sink only
        # wrote at the end, every reading would be zero.
        seen_midway.append(len(out.read_text().splitlines()) if out.exists() else 0)
        return ReplayPolicy(AgentTrajectory(task=task.prompt, steps=(Step(kind=FINAL, content="done"),), success=True))

    run_suite(tasks, policy_factory, Executor(), tmp_path / "ws", sink=sink)
    sink.close()

    assert seen_midway == [0, 1, 2], f"episodes were not written incrementally: {seen_midway}"
    assert [r["task_id"] for r in read_episodes(out)] == ["t0", "t1", "t2"]


def test_a_partial_log_from_a_killed_run_is_still_readable(tmp_path):
    """A single JSON array is unreadable until its closing bracket arrives, which is exactly
    what made the old behaviour useless -- the file existed and told you nothing."""
    from hermesbench.sink import read_episodes

    out = tmp_path / "episodes.jsonl"
    out.write_text('{"task_id": "a", "success": true}\n{"task_id": "b", "succ')

    recovered = list(read_episode_prefix(out))
    with pytest.raises(SinkError, match="truncated"):
        list(read_episodes(out))
    assert [r["task_id"] for r in recovered] == ["a"]


def test_run_suite_without_a_sink_is_unchanged(tmp_path):
    """Additive: every existing caller passes no sink and must behave exactly as before."""
    from hermes.trajectory import FINAL, AgentTrajectory, Step
    from hermesbench.runner import ReplayPolicy, run_suite
    from hermesbench.tasks import Task

    class Executor:
        def execute(self, tool, args, *, workspace, env=None):
            return True, "ok"

    tasks = [Task(task_id="t0", prompt="p", verify="true", tools=("terminal",), setup=None)]
    metrics, results = run_suite(
        tasks,
        lambda t: ReplayPolicy(AgentTrajectory(task=t.prompt, steps=(Step(kind=FINAL, content="done"),), success=True)),
        Executor(),
        tmp_path / "ws",
    )
    assert len(results) == 1
    assert metrics.episodes == 1
