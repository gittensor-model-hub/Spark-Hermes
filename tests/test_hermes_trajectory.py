import json

import pytest

from hermes.trajectory import (
    FINAL,
    THINKING,
    TOOL_CALL,
    TOOL_RESULT,
    AgentTrajectory,
    Step,
    TrajectoryError,
    is_valid,
    load_jsonl,
    validate,
    write_jsonl,
)


def _steps(*, fail_first: bool = False, end_final: bool = True) -> list[Step]:
    steps = [
        Step(kind=THINKING, content="inspect the repo"),
        Step(kind=TOOL_CALL, tool="terminal", args={"command": "pytest"}, call_id="c1"),
        Step(kind=TOOL_RESULT, call_id="c1", content="3 failed" if fail_first else "ok", ok=not fail_first),
        Step(kind=TOOL_CALL, tool="edit", args={"path": "a.py"}, call_id="c2"),
        Step(kind=TOOL_RESULT, call_id="c2", content="applied", ok=True),
    ]
    if end_final:
        steps.append(Step(kind=FINAL, content="done"))
    return steps


def _traj(steps: list[Step], **kwargs) -> AgentTrajectory:
    defaults = {"task": "fix the bug", "success": True, "tools_available": ("terminal", "edit")}
    defaults.update(kwargs)
    return AgentTrajectory(steps=tuple(steps), **defaults)


def test_valid_trajectory_passes():
    validate(_traj(_steps()))


def test_tool_call_without_result_is_rejected():
    steps = [
        Step(kind=TOOL_CALL, tool="terminal", args={"command": "pytest"}, call_id="c1"),
        Step(kind=FINAL, content="all good"),
    ]
    with pytest.raises(TrajectoryError, match="no observed result"):
        validate(_traj(steps))


def test_tool_result_without_call_is_rejected():
    steps = [
        Step(kind=TOOL_RESULT, call_id="ghost", content="output"),
        Step(kind=FINAL, content="done"),
    ]
    with pytest.raises(TrajectoryError, match="unknown or already-resolved"):
        validate(_traj(steps))


def test_duplicate_call_id_is_rejected():
    steps = [
        Step(kind=TOOL_CALL, tool="terminal", args={}, call_id="c1"),
        Step(kind=TOOL_RESULT, call_id="c1", content="ok"),
        Step(kind=TOOL_CALL, tool="terminal", args={}, call_id="c1"),
        Step(kind=TOOL_RESULT, call_id="c1", content="ok"),
        Step(kind=FINAL, content="done"),
    ]
    with pytest.raises(TrajectoryError, match="duplicate call_id"):
        validate(_traj(steps))


def test_undeclared_tool_is_rejected():
    steps = [
        Step(kind=TOOL_CALL, tool="browser", args={}, call_id="c1"),
        Step(kind=TOOL_RESULT, call_id="c1", content="ok"),
        Step(kind=FINAL, content="done"),
    ]
    with pytest.raises(TrajectoryError, match="not in tools_available"):
        validate(_traj(steps))


def test_chat_record_without_tool_calls_is_rejected():
    """A prompt/response pair is not an agent trajectory; training it teaches answering."""
    steps = [Step(kind=THINKING, content="thinking"), Step(kind=FINAL, content="the answer is 4")]
    with pytest.raises(TrajectoryError, match="chat record"):
        validate(_traj(steps))


def test_trajectory_must_end_with_final():
    with pytest.raises(TrajectoryError, match="does not end with a final"):
        validate(_traj(_steps(end_final=False)))


def test_empty_trajectory_is_rejected():
    with pytest.raises(TrajectoryError, match="no steps"):
        validate(_traj([]))


def test_call_without_call_id_is_rejected():
    steps = [Step(kind=TOOL_CALL, tool="terminal", args={}), Step(kind=FINAL, content="x")]
    with pytest.raises(TrajectoryError, match="no call_id"):
        validate(_traj(steps))


def test_is_valid_matches_validate():
    assert is_valid(_traj(_steps()))
    assert not is_valid(_traj([]))


def test_recovery_steps_are_calls_after_an_observed_failure():
    trajectory = _traj(_steps(fail_first=True))
    assert [s.call_id for s in trajectory.recovery_steps] == ["c2"]
    assert [s.call_id for s in trajectory.failed_steps] == ["c1"]


def test_no_recovery_steps_when_nothing_failed():
    assert _traj(_steps()).recovery_steps == ()


def test_final_answer_reads_the_last_final_step():
    assert _traj(_steps()).final_answer == "done"
    assert _traj(_steps(end_final=False)).final_answer == ""


def test_round_trip_through_record_preserves_everything():
    original = _traj(_steps(fail_first=True), task_id="t1", source="unit", metadata={"executed": True})
    restored = AgentTrajectory.from_record(original.to_record())
    assert restored == original


def test_unknown_step_kind_is_rejected():
    with pytest.raises(TrajectoryError, match="unknown step kind"):
        AgentTrajectory.from_record({"task": "x", "steps": [{"kind": "wat"}], "success": True})


def test_future_schema_version_is_rejected():
    with pytest.raises(TrajectoryError, match="newer than supported"):
        AgentTrajectory.from_record({"task": "x", "steps": [], "success": True, "schema_version": 99})


def test_record_without_task_is_rejected():
    with pytest.raises(TrajectoryError, match="no task"):
        AgentTrajectory.from_record({"steps": [], "success": True})


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "traj.jsonl"
    trajectories = [_traj(_steps()), _traj(_steps(fail_first=True))]
    assert write_jsonl(path, trajectories) == 2
    assert list(load_jsonl(path)) == trajectories


def test_non_integer_schema_version_raises_trajectory_error():
    """Must be TrajectoryError, not bare ValueError: only that type gets file:line."""
    with pytest.raises(TrajectoryError, match="schema_version must be an integer"):
        AgentTrajectory.from_record({"task": "x", "steps": [], "success": True, "schema_version": "1.0"})


def test_bad_schema_version_in_a_file_is_reported_with_its_line(tmp_path):
    path = tmp_path / "bad_version.jsonl"
    path.write_text(
        json.dumps({"task": "ok", "steps": [], "success": True})
        + "\n"
        + json.dumps({"task": "x", "steps": [], "success": True, "schema_version": "1.0"})
        + "\n"
    )
    with pytest.raises(TrajectoryError, match=r"bad_version\.jsonl:2"):
        list(load_jsonl(path))


def test_load_jsonl_reports_the_offending_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"task": "ok", "steps": [], "success": True}) + "\n{ not json\n")
    with pytest.raises(TrajectoryError, match=r"bad\.jsonl:2"):
        list(load_jsonl(path))


# --- abstention: declining to act is part of the protocol ---------------------------


def _abstention() -> AgentTrajectory:
    return AgentTrajectory(
        task="What is 2+2?",
        steps=(Step(kind=FINAL, content="4."),),
        success=True,
        tools_available=("python", "browser"),
        abstention=True,
    )


def test_an_abstention_is_a_valid_trajectory():
    """Hermes tells the model to answer in words when the offered tools are not relevant.
    A corpus that cannot express that trains a worker to always reach for a tool."""
    validate(_abstention())


def test_an_abstention_that_calls_a_tool_is_refused():
    trajectory = AgentTrajectory(
        task="t",
        steps=(
            Step(kind=TOOL_CALL, tool="python", call_id="c1"),
            Step(kind=TOOL_RESULT, call_id="c1", content="ok"),
            Step(kind=FINAL, content="done"),
        ),
        success=True,
        tools_available=("python",),
        abstention=True,
    )
    with pytest.raises(TrajectoryError, match="not an abstention"):
        validate(trajectory)


def test_an_abstention_with_no_tools_offered_is_still_a_chat_record():
    """There was no decision to get right."""
    trajectory = AgentTrajectory(task="t", steps=(Step(kind=FINAL, content="4."),), success=True, abstention=True)
    with pytest.raises(TrajectoryError, match="no choice to make"):
        validate(trajectory)


def test_an_ordinary_trajectory_with_no_tool_calls_is_still_refused():
    """Only an explicit abstention is exempt; silence is not consent."""
    trajectory = AgentTrajectory(task="t", steps=(Step(kind=FINAL, content="4."),), success=True)
    with pytest.raises(TrajectoryError, match="chat record"):
        validate(trajectory)


def test_the_abstention_flag_round_trips():
    assert AgentTrajectory.from_record(_abstention().to_record()).abstention is True


def test_an_abstention_with_nothing_said_is_refused():
    """Declining to act is answering in words; an empty final is a crashed episode."""
    trajectory = AgentTrajectory(
        task="What is 2+2?",
        steps=(Step(kind=FINAL, content="   "),),
        success=True,
        tools_available=("python",),
        abstention=True,
    )
    with pytest.raises(TrajectoryError, match="must say something"):
        validate(trajectory)
