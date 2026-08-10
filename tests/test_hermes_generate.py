import json

import pytest

from hermes.generate import AGENT_BRIEF, GenerationError, generate_trajectories, parse_response
from hermes.trajectory import TrajectoryError

_STEPS = [
    {"kind": "thinking", "content": "look first"},
    {"kind": "tool_call", "call_id": "c1", "tool": "terminal", "args": {"command": "pytest"}},
    {"kind": "tool_result", "call_id": "c1", "ok": False, "content": "3 failed"},
    {"kind": "tool_call", "call_id": "c2", "tool": "edit", "args": {"path": "a.py"}},
    {"kind": "tool_result", "call_id": "c2", "ok": True, "content": "applied"},
    {"kind": "final", "content": "fixed"},
]


def _fenced(payload: dict) -> str:
    return f"here you go:\n```json\n{json.dumps(payload)}\n```\nthanks"


def test_parses_fenced_json():
    trajectory = parse_response(_fenced({"steps": _STEPS, "success": True}), task="t", tools=("terminal", "edit"))
    assert trajectory.success is True
    assert len(trajectory.tool_calls) == 2


def test_parses_bare_json_without_a_fence():
    payload = json.dumps({"steps": _STEPS, "success": True})
    assert parse_response(payload, task="t", tools=("terminal", "edit")).final_answer == "fixed"


def test_generated_rows_are_stamped_unexecuted():
    """Teacher-imagined tool output is not evidence; the flag is what keeps that honest."""
    trajectory = parse_response(_fenced({"steps": _STEPS, "success": True}), task="t", tools=("terminal", "edit"))
    assert trajectory.metadata["executed"] is False


def test_extra_metadata_is_preserved_alongside_the_flag():
    trajectory = parse_response(
        _fenced({"steps": _STEPS, "success": True}),
        task="t",
        tools=("terminal", "edit"),
        metadata={"teacher_model": "claude-fable-5"},
    )
    assert trajectory.metadata == {"teacher_model": "claude-fable-5", "executed": False}


def test_non_json_response_raises():
    with pytest.raises(GenerationError, match="not valid JSON"):
        parse_response("I cannot do that", task="t", tools=("terminal",))


def test_json_array_response_raises():
    with pytest.raises(GenerationError, match="expected an object"):
        parse_response("[1, 2, 3]", task="t", tools=("terminal",))


def test_invalid_trajectory_is_rejected_at_parse_time():
    broken = [{"kind": "tool_call", "call_id": "c1", "tool": "terminal", "args": {}}, {"kind": "final", "content": "x"}]
    with pytest.raises(TrajectoryError, match="no observed result"):
        parse_response(_fenced({"steps": broken, "success": True}), task="t", tools=("terminal",))


def test_tool_outside_declared_set_is_rejected():
    steps = [
        {"kind": "tool_call", "call_id": "c1", "tool": "browser", "args": {}},
        {"kind": "tool_result", "call_id": "c1", "content": "ok"},
        {"kind": "final", "content": "x"},
    ]
    with pytest.raises(TrajectoryError, match="not in tools_available"):
        parse_response(_fenced({"steps": steps, "success": True}), task="t", tools=("terminal",))


def test_brief_names_the_tools_and_the_task():
    brief = AGENT_BRIEF.format(tools="terminal, edit", task="fix the bug")
    assert "terminal, edit" in brief
    assert "fix the bug" in brief
    assert "recover from failures" in brief


class _FakeTeacher:
    name = "fake"
    model = "fake-1"

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt, *, system=None, max_tokens=2048, temperature=0.7, thinking_budget=None):
        self.prompts.append(prompt)

        class _R:
            response = self.responses.pop(0)

        return _R()


def test_generate_trajectories_collects_failures(monkeypatch):
    teacher = _FakeTeacher([_fenced({"steps": _STEPS, "success": True}), "not json at all"])
    monkeypatch.setattr("hermes.generate.get_teacher", lambda provider, model: teacher)

    tasks = [
        {"task": "one", "tools": ["terminal", "edit"], "task_id": "t1"},
        {"task": "two", "tools": ["terminal"], "task_id": "t2"},
    ]
    trajectories, failures = generate_trajectories(tasks, provider="fake")

    assert len(trajectories) == 1
    assert trajectories[0].task_id == "t1"
    assert trajectories[0].source == "synthetic:fake:fake-1"
    assert len(failures) == 1
    assert "t2" in failures[0]


def test_task_without_tools_records_the_defaults_it_advertised(monkeypatch):
    """Advertising tools but recording none makes the schema's allowlist a no-op."""
    from hermes.generate import DEFAULT_TOOLS

    steps = [
        {"kind": "tool_call", "call_id": "c1", "tool": "terminal", "args": {}},
        {"kind": "tool_result", "call_id": "c1", "content": "ok"},
        {"kind": "final", "content": "done"},
    ]
    teacher = _FakeTeacher([_fenced({"steps": steps, "success": True})])
    monkeypatch.setattr("hermes.generate.get_teacher", lambda provider, model: teacher)

    trajectories, failures = generate_trajectories([{"task": "no tools declared"}], provider="fake")
    assert failures == []
    assert trajectories[0].tools_available == DEFAULT_TOOLS
    assert ", ".join(DEFAULT_TOOLS) in teacher.prompts[0]


def test_invented_tool_is_rejected_even_when_the_task_declared_no_tools(monkeypatch):
    steps = [
        {"kind": "tool_call", "call_id": "c1", "tool": "quantum_oracle", "args": {}},
        {"kind": "tool_result", "call_id": "c1", "content": "42"},
        {"kind": "final", "content": "done"},
    ]
    teacher = _FakeTeacher([_fenced({"steps": steps, "success": True})])
    monkeypatch.setattr("hermes.generate.get_teacher", lambda provider, model: teacher)

    trajectories, failures = generate_trajectories([{"task": "t", "task_id": "t1"}], provider="fake")
    assert trajectories == []
    assert "not in tools_available" in failures[0]


def test_generate_trajectories_skips_tasks_with_no_prompt(monkeypatch):
    monkeypatch.setattr("hermes.generate.get_teacher", lambda provider, model: _FakeTeacher([]))
    trajectories, failures = generate_trajectories([{"task_id": "empty"}], provider="fake")
    assert trajectories == []
    assert "no 'task' field" in failures[0]
