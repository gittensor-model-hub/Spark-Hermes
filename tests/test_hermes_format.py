import json

import pytest

from hermes.format import DEFAULT_SYSTEM, convert, main, to_messages_record
from hermes.trajectory import FINAL, THINKING, TOOL_CALL, TOOL_RESULT, AgentTrajectory, Step, write_jsonl


def _traj(**kwargs) -> AgentTrajectory:
    steps = (
        Step(kind=THINKING, content="run the tests first"),
        Step(kind=TOOL_CALL, tool="terminal", args={"command": "pytest"}, call_id="c1"),
        Step(kind=TOOL_RESULT, call_id="c1", content="3 failed", ok=False),
        Step(kind=THINKING, content="inspect the file"),
        Step(kind=TOOL_CALL, tool="edit", args={"path": "a.py"}, call_id="c2"),
        Step(kind=TOOL_RESULT, call_id="c2", content="applied", ok=True),
        Step(kind=FINAL, content="fixed"),
    )
    defaults = {"task": "fix it", "success": True, "tools_available": ("terminal", "edit")}
    defaults.update(kwargs)
    return AgentTrajectory(steps=steps, **defaults)


def test_messages_shape_and_order():
    record = to_messages_record(_traj())
    assert [m["role"] for m in record["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert record["tools"] == ["terminal", "edit"]


def test_default_system_used_when_trajectory_has_none():
    """Hand-authored rows never had a system prompt, so the default is the honest filler."""
    assert to_messages_record(_traj())["messages"][0]["content"].startswith(DEFAULT_SYSTEM)


def test_an_executed_row_may_not_borrow_the_default_system_prompt():
    """Some specific text caused those exact tokens. Substituting a different one pairs the
    assistant turns with a prompt that did not produce them, and both prompts look plausible,
    so nothing downstream can notice. The harness prompt says "inspect before you change
    anything"; DEFAULT_SYSTEM does not -- a row built that way teaches inspect-first behaviour
    as though it were unprompted."""
    with pytest.raises(ValueError, match="recorded no system prompt"):
        to_messages_record(_traj(metadata={"executed": True}))


def test_an_executed_row_that_recorded_its_prompt_uses_it():
    system = to_messages_record(_traj(metadata={"executed": True}, system="Inspect before you change anything."))[
        "messages"
    ][0]["content"]
    assert system.startswith("Inspect before you change anything.")
    assert DEFAULT_SYSTEM not in system


def test_trajectory_system_overrides_default():
    system = to_messages_record(_traj(system="custom"))["messages"][0]["content"]
    assert system.startswith("custom")
    assert DEFAULT_SYSTEM not in system


def test_reasoning_becomes_a_think_block():
    assistant = to_messages_record(_traj())["messages"][2]
    assert assistant["content"] == "<think>\nrun the tests first\n</think>"


def test_tool_calls_serialize_arguments_as_json():
    call = to_messages_record(_traj())["messages"][2]["tool_calls"][0]
    assert call["id"] == "c1"
    assert call["function"]["name"] == "terminal"
    assert json.loads(call["function"]["arguments"]) == {"command": "pytest"}


def test_failed_tool_result_is_marked_as_an_error():
    """The model must be able to tell a failure from output; ok=False cannot be silent."""
    messages = to_messages_record(_traj())["messages"]
    assert messages[3]["content"] == "ERROR: 3 failed"
    assert messages[5]["content"] == "applied"


def test_final_turn_carries_the_answer():
    assert to_messages_record(_traj())["messages"][-1]["content"].endswith("fixed")


def test_consecutive_calls_share_one_assistant_turn():
    trajectory = AgentTrajectory(
        task="t",
        success=True,
        tools_available=("terminal",),
        steps=(
            Step(kind=TOOL_CALL, tool="terminal", args={"command": "a"}, call_id="c1"),
            Step(kind=TOOL_CALL, tool="terminal", args={"command": "b"}, call_id="c2"),
            Step(kind=TOOL_RESULT, call_id="c1", content="ra"),
            Step(kind=TOOL_RESULT, call_id="c2", content="rb"),
            Step(kind=FINAL, content="done"),
        ),
    )
    messages = to_messages_record(trajectory)["messages"]
    assistant = messages[2]
    assert len(assistant["tool_calls"]) == 2
    assert [m["role"] for m in messages[3:5]] == ["tool", "tool"]


def test_convert_drops_invalid_trajectories():
    bad = AgentTrajectory(task="t", success=True, steps=(Step(kind=FINAL, content="x"),))
    records, skipped = convert([_traj(), bad])
    assert len(records) == 1
    assert len(skipped) == 1


def test_convert_keep_invalid_retains_them():
    bad = AgentTrajectory(task="t", success=True, steps=(Step(kind=FINAL, content="x"),))
    records, skipped = convert([bad], keep_invalid=True)
    assert len(records) == 1
    assert skipped == []


def test_require_success_drops_unverified_rows():
    records, skipped = convert([_traj(success=False)], require_success=True)
    assert records == []
    assert "unverified outcome" in skipped[0]


def test_require_success_off_by_default_keeps_failures():
    """Failure/recovery rows are the valuable ones; they must survive the default path."""
    records, _ = convert([_traj(success=False)])
    assert len(records) == 1


def test_tool_names_appear_in_the_trained_system_turn():
    """Metadata the recipes never map cannot teach the model which tools exist."""
    system = to_messages_record(_traj())["messages"][0]["content"]
    assert "Available tools: terminal, edit" in system


def test_no_tool_line_when_the_trajectory_declares_none():
    system = to_messages_record(_traj(tools_available=()))["messages"][0]["content"]
    assert "Available tools:" not in system


def test_executed_only_drops_simulated_rows():
    simulated = _traj(metadata={"executed": False})
    # An executed row carries the prompt that produced it; without one the export refuses.
    executed = _traj(metadata={"executed": True}, system="You are a Hermes agent. Inspect first.")
    records, skipped = convert([simulated, executed], executed_only=True)
    assert len(records) == 1
    assert "simulated tool results" in skipped[0]


def test_executed_only_off_by_default():
    records, _ = convert([_traj(metadata={"executed": False})])
    assert len(records) == 1


def test_harness_authored_finals_are_dropped_by_default():
    """'step budget exhausted' must not become the assistant's trained answer."""
    records, skipped = convert([_traj(metadata={"executed": True, "harness_final": True})])
    assert records == []
    assert "harness-authored final" in skipped[0]


def test_harness_finals_can_be_kept_explicitly():
    records, _ = convert([_traj(metadata={"harness_final": True})], keep_harness_finals=True)
    assert len(records) == 1


def test_cli_executed_only_flag(tmp_path):
    src, dst = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_jsonl(
        src,
        [
            _traj(metadata={"executed": False}),
            _traj(metadata={"executed": True}, system="You are a Hermes agent. Inspect first."),
        ],
    )
    assert main(["--in", str(src), "--out", str(dst), "--executed-only"]) == 0
    assert len(dst.read_text().splitlines()) == 1


def test_cli_writes_messages_jsonl(tmp_path):
    src, dst = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_jsonl(src, [_traj()])
    assert main(["--in", str(src), "--out", str(dst)]) == 0
    rows = [json.loads(line) for line in dst.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["messages"][1] == {"role": "user", "content": "fix it"}


# --- the Hermes wire format ---------------------------------------------------------


def _schema_meta():
    return {
        "tool_schemas": [
            {"name": "terminal", "description": "Run a shell command", "parameters": {"type": "object"}},
            {"name": "edit", "description": "Edit a file", "parameters": {"type": "object"}},
        ]
    }


def _hermes_trajectory(**kw):
    kw.setdefault("metadata", _schema_meta())
    return _traj(**kw)


def test_hermes_rendering_emits_tool_call_tags_not_a_tool_calls_array():
    """A worker trained on the OpenAI shape emits nothing a Hermes runtime recognises."""
    from hermes.format import to_hermes_record
    from hermes.protocol import HERMES_3

    record = to_hermes_record(_hermes_trajectory(), dialect=HERMES_3)
    assistant = [m for m in record["messages"] if m["role"] == "assistant"]
    assert any("<tool_call>" in m["content"] for m in assistant)
    assert all("tool_calls" not in m for m in record["messages"])


def test_hermes_3_puts_results_in_a_tool_role_and_hermes_4_in_a_user_role():
    from hermes.format import to_hermes_record
    from hermes.protocol import HERMES_3, HERMES_4

    roles3 = {m["role"] for m in to_hermes_record(_hermes_trajectory(), dialect=HERMES_3)["messages"]}
    roles4 = {m["role"] for m in to_hermes_record(_hermes_trajectory(), dialect=HERMES_4)["messages"]}
    assert "tool" in roles3 and "tool" not in roles4


def test_results_carry_the_name_of_the_tool_that_produced_them():
    """The protocol has no call ids, so the name is the only identity a result has."""
    from hermes.format import to_hermes_record
    from hermes.protocol import HERMES_3

    record = to_hermes_record(_hermes_trajectory(), dialect=HERMES_3)
    results = [m for m in record["messages"] if m["role"] == "tool"]
    assert results and all('"name": "' in m["content"] for m in results)
    assert any("terminal" in m["content"] for m in results)


def test_rendering_without_recorded_schemas_is_refused():
    """Synthesizing signatures from bare names moves the invented-tool failure down a level."""
    from hermes.format import to_hermes_record
    from hermes.protocol import HERMES_3, ProtocolError

    with pytest.raises(ProtocolError, match="tool_schemas"):
        to_hermes_record(_hermes_trajectory(metadata={}), dialect=HERMES_3)


def test_schemas_that_disagree_with_the_allowlist_are_refused():
    """The model would be shown one tool set and graded against another."""
    from hermes.format import to_hermes_record
    from hermes.protocol import HERMES_3, ProtocolError

    meta = {"tool_schemas": [{"name": "terminal", "description": "", "parameters": {}}]}
    with pytest.raises(ProtocolError, match="do not match tools_available"):
        to_hermes_record(_hermes_trajectory(metadata=meta), dialect=HERMES_3)


def test_a_row_that_cannot_be_rendered_faithfully_is_skipped_not_downgraded():
    """A corpus half in each format teaches the model that both are acceptable."""
    from hermes.format import convert
    from hermes.protocol import HERMES_3

    records, skipped = convert([_hermes_trajectory(metadata={})], dialect=HERMES_3)
    assert records == []
    assert skipped and "tool_schemas" in skipped[0]


def test_the_default_target_is_still_the_openai_shape():
    from hermes.format import convert

    records, _ = convert([_hermes_trajectory()])
    assert "tool_calls" in records[0]["messages"][1] or any("tool_calls" in m for m in records[0]["messages"])


def test_a_turn_issuing_calls_records_no_observation():
    """Writing the result before the tool has run is a prediction dressed as evidence."""
    from hermes.format import to_hermes_record
    from hermes.protocol import HERMES_3
    from hermes.state import ReasoningState
    from hermes.trajectory import AgentTrajectory

    trajectory = AgentTrajectory(
        task="Optimize the kernel",
        steps=(
            Step(
                kind=THINKING,
                state=ReasoningState(goal="Improve latency", action="benchmark it", observed_signal="120us"),
            ),
            Step(kind=TOOL_CALL, tool="terminal", args={"command": "bench"}, call_id="c1"),
            Step(kind=TOOL_RESULT, call_id="c1", content="120us", ok=True),
            Step(kind=FINAL, content="done"),
        ),
        success=True,
        tools_available=("terminal",),
        metadata={"tool_schemas": [{"name": "terminal", "description": "run", "parameters": {}}]},
    )
    record = to_hermes_record(trajectory, dialect=HERMES_3)
    calling_turn = next(m for m in record["messages"] if m["role"] == "assistant" and "<tool_call>" in m["content"])
    assert "Observation: None" in calling_turn["content"]
    assert "120us" not in calling_turn["content"]


# --- the system policy: a choice with a measured price ------------------------------------


def test_keep_is_the_default_and_changes_nothing():
    record = to_messages_record(_traj(system="Inspect first."))
    assert record["messages"][0]["role"] == "system"
    assert "system_policy" not in record


def test_strip_removes_the_turn_rather_than_emptying_it():
    """An empty system message is not the serving condition it imitates -- a model served
    without a prompt sees no system turn at all. Training on "" teaches the shape of a
    blank instruction rather than the absence of one."""
    from hermes.format import STRIP

    record = to_messages_record(_traj(system="Inspect first."), system_policy=STRIP)
    assert all(m["role"] != "system" for m in record["messages"])
    assert record["messages"][0]["role"] == "user"
    assert record["system_policy"] == "strip"


def test_replace_substitutes_a_canonical_prompt():
    """Not a Hermes fork: the rule forbids forking the agent, not a model shipping its own
    recommended prompt in its model card and chat template."""
    from hermes.format import REPLACE

    record = to_messages_record(
        _traj(system="Miner guidance: keep a hypothesis ledger."),
        system_policy=REPLACE,
        system_replacement="You are a Spark-Hermes worker.",
    )
    assert record["messages"][0]["content"] == "You are a Spark-Hermes worker."
    assert "hypothesis ledger" not in json.dumps(record)
    assert record["system_policy"] == "replace"


def test_replace_without_a_replacement_is_refused():
    """An empty replacement is 'strip' by another name, and the two should not be reachable
    by the same flag -- they are different decisions with different costs."""
    from hermes.format import REPLACE

    with pytest.raises(ValueError, match="strip by another name|'strip' by another name"):
        to_messages_record(_traj(system="x"), system_policy=REPLACE, system_replacement="   ")


def test_an_unknown_policy_is_refused():
    with pytest.raises(ValueError, match="unknown system policy"):
        to_messages_record(_traj(system="x"), system_policy="delete_everything")


def test_the_policy_is_recorded_on_the_row():
    """A stripped row and a row that never had a prompt look identical in the messages.
    A corpus that cannot say how it was built cannot be reweighted by anyone who did not
    build it."""
    from hermes.format import STRIP

    assert to_messages_record(_traj(system="x"), system_policy=STRIP)["system_policy"] == "strip"


def test_convert_applies_the_policy_to_every_row():
    from hermes.format import STRIP

    records, _ = convert([_traj(system="a"), _traj(system="b")], system_policy=STRIP)
    assert all(m["role"] != "system" for r in records for m in r["messages"])


def test_the_cli_defaults_to_keep(tmp_path):
    src, dst = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_jsonl(src, [_traj(system="Inspect first.")])
    assert main(["--in", str(src), "--out", str(dst)]) == 0
    row = json.loads(dst.read_text().splitlines()[0])
    assert row["messages"][0]["role"] == "system"


def test_the_cli_can_strip(tmp_path):
    src, dst = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_jsonl(src, [_traj(system="Inspect first.")])
    assert main(["--in", str(src), "--out", str(dst), "--system-policy", "strip"]) == 0
    row = json.loads(dst.read_text().splitlines()[0])
    assert all(m["role"] != "system" for m in row["messages"])
    assert row["system_policy"] == "strip"
