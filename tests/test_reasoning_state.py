"""Structured reasoning traces: what a student learns instead of teacher prose."""

import json

import pytest

from hermes.format import convert, to_messages_record
from hermes.state import STATE_PROTOCOL, ReasoningState, StateError, measure_compression
from hermes.trajectory import FINAL, THINKING, TOOL_CALL, TOOL_RESULT, AgentTrajectory, Step, TrajectoryError

STATE = ReasoningState(
    goal="identify the CUDA bottleneck",
    hypothesis="memory bandwidth limited",
    action="run Nsight Compute",
    expected_signal="memory throughput near peak if the hypothesis holds",
    observed_signal="memory throughput 95%, occupancy low",
    decision="fuse the kernel operations",
)


def _traj(thinking_steps, **kwargs) -> AgentTrajectory:
    steps = [
        *thinking_steps,
        Step(kind=TOOL_CALL, tool="terminal", args={}, call_id="c1"),
        Step(kind=TOOL_RESULT, call_id="c1", content="ok"),
        Step(kind=FINAL, content="done"),
    ]
    defaults = {"task": "optimize the kernel", "success": True, "tools_available": ("terminal",)}
    defaults.update(kwargs)
    return AgentTrajectory(steps=tuple(steps), **defaults)


# --- the state itself --------------------------------------------------------------


def test_state_requires_a_goal():
    with pytest.raises(StateError, match="needs a goal"):
        ReasoningState(goal="  ", action="do something")


def test_state_requires_an_action():
    """A state with no next action is a sentiment, not a step."""
    with pytest.raises(StateError, match="sentiment"):
        ReasoningState(goal="find the bug", action="")


def test_render_is_labelled_and_terse():
    rendered = STATE.render()
    assert rendered.startswith("Goal: identify the CUDA bottleneck")
    assert "Action: run Nsight Compute" in rendered
    assert "Observed: memory throughput 95%, occupancy low" in rendered


def test_render_omits_empty_fields():
    rendered = ReasoningState(goal="g", action="a").render()
    assert rendered == "Goal: g\nAction: a"


def test_prediction_tracking():
    assert STATE.made_a_prediction and STATE.prediction_was_checked
    unchecked = ReasoningState(goal="g", action="a", expected_signal="x")
    assert unchecked.made_a_prediction and not unchecked.prediction_was_checked


def test_state_round_trips_through_a_record():
    assert ReasoningState.from_record(STATE.to_record()) == STATE


def test_state_protocol_asks_for_a_prediction_before_acting():
    assert "expected_signal" in STATE_PROTOCOL
    assert "before you act" in STATE_PROTOCOL


# --- schema integration ------------------------------------------------------------


def test_step_carries_state_through_a_round_trip():
    step = Step(kind=THINKING, state=STATE)
    assert Step.from_record(step.to_record()).state == STATE


def test_a_malformed_state_is_reported_as_a_trajectory_error():
    """So load_jsonl can annotate it with file:line instead of raising a bare StateError."""
    with pytest.raises(TrajectoryError, match="invalid reasoning state"):
        Step.from_record({"kind": THINKING, "state": {"goal": "g"}})


def test_steps_without_state_still_work():
    """Backwards compatible: existing prose trajectories are unaffected."""
    assert Step.from_record({"kind": THINKING, "content": "hmm"}).state is None


def test_fully_structured_requires_every_thinking_step():
    both = _traj([Step(kind=THINKING, state=STATE), Step(kind=THINKING, content="raw prose")])
    assert not both.fully_structured
    assert len(both.structured_thinking) == 1

    clean = _traj([Step(kind=THINKING, state=STATE)])
    assert clean.fully_structured


def test_a_trajectory_with_no_thinking_is_not_fully_structured():
    """Nothing to normalize is not the same as having normalized everything."""
    assert not _traj([]).fully_structured


# --- rendering ---------------------------------------------------------------------


def test_structured_state_replaces_prose_in_the_trained_text():
    """Keeping both would train the labelled form *and* the accent."""
    trajectory = _traj([Step(kind=THINKING, content="I am thinking that maybe...", state=STATE)])
    assistant = to_messages_record(trajectory)["messages"][2]
    assert "Goal: identify the CUDA bottleneck" in assistant["content"]
    assert "I am thinking that maybe" not in assistant["content"]


def test_prose_survives_when_there_is_no_state():
    trajectory = _traj([Step(kind=THINKING, content="raw teacher prose")])
    assert "raw teacher prose" in to_messages_record(trajectory)["messages"][2]["content"]


# --- the multi-teacher mixing problem ----------------------------------------------


def test_structured_only_drops_unnormalized_rows():
    """A partially-normalized row still trains prose on the steps that were missed."""
    normalized = _traj([Step(kind=THINKING, state=STATE)])
    prose = _traj([Step(kind=THINKING, content="long deliberation in one teacher's voice")])

    records, skipped = convert([normalized, prose], structured_only=True)
    assert len(records) == 1
    assert "unnormalized reasoning" in skipped[0]


def test_structured_only_is_off_by_default():
    prose = _traj([Step(kind=THINKING, content="prose")])
    assert len(convert([prose])[0]) == 1


def test_three_teacher_styles_normalize_to_one_shape():
    """The point of the whole module: same slots, three different voices in."""
    claude = _traj(
        [
            Step(
                kind=THINKING,
                content="Let me carefully consider the architecture before making any changes...",
                state=ReasoningState(goal="find the bottleneck", action="profile the kernel"),
            )
        ]
    )
    qwen = _traj(
        [
            Step(
                kind=THINKING,
                content="Run Nsight first.",
                state=ReasoningState(goal="find the bottleneck", action="profile the kernel"),
            )
        ]
    )
    kimi = _traj(
        [
            Step(
                kind=THINKING,
                content="Compare occupancy against memory throughput across several strategies...",
                state=ReasoningState(goal="find the bottleneck", action="profile the kernel"),
            )
        ]
    )

    rendered = {to_messages_record(t)["messages"][2]["content"] for t in (claude, qwen, kimi)}
    # Three teacher voices in, one trained form out.
    assert len(rendered) == 1


# --- compression reporting ---------------------------------------------------------


def test_compression_is_measured_not_asserted():
    verbose = "I am thinking about several possible approaches. " * 40
    trajectory = _traj([Step(kind=THINKING, content=verbose, state=STATE)])
    report = measure_compression(trajectory.steps)

    assert report.fully_structured
    assert report.structured_coverage == 1.0
    assert report.compression_ratio > 1.0  # structured form is smaller than the prose
    assert report.prose_chars > report.structured_chars


def test_partial_coverage_is_reported():
    trajectory = _traj([Step(kind=THINKING, state=STATE), Step(kind=THINKING, content="prose")])
    report = measure_compression(trajectory.steps)
    assert report.structured_coverage == 0.5
    assert not report.fully_structured


def test_compression_report_handles_no_thinking_steps():
    report = measure_compression(_traj([]).steps)
    assert report.structured_coverage == 0.0
    assert report.compression_ratio == 0.0


def test_compression_report_is_json_safe():
    report = measure_compression(_traj([Step(kind=THINKING, state=STATE)]).steps)
    assert json.loads(json.dumps(report.to_record()))["fully_structured"] is True
