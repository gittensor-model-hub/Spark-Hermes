"""Does a training row survive the chat template that will render it?

Every other test in this repository checks the *row*: the roles are right, the reasoning is present,
the tool call carries the arguments it should. All of those passed on an ATEM corpus that could not
be trained at all. The template is where the row is finally read, and it was the only thing that
disagreed:

  1. `tools` held bare names, and the template calls `.name` on each entry
     -> `'str object' has no attribute 'name'`
  2. `function.arguments` was a JSON string, and the template refuses one outright
     -> `a JSON string cannot be parsed in the HF jinja sandbox`
  3. reasoning was `<think>...</think>` inside `content`, and the template ignores `content`
     entirely on any message carrying `tool_calls`
     -> every reasoning block silently dropped from exactly the turns that reason toward a call,
        and trained as literal visible prose on the turns that do not

The first two raise, which is survivable -- somebody notices. The third does not, and a corpus that
trains 0 reasoning tokens while every row visibly contains reasoning is the failure this file exists
to make impossible.

So these tests render, with the same jinja sandbox `transformers` uses, and assert on the *rendered
text*. `jinja2` is a declared dev dependency rather than a transitive one so this cannot degrade into
a skip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

from hermes.format import to_messages_record
from hermes.pin import load_tool_schemas
from hermes.protocol import DIALECTS, ProtocolError
from hermes.trajectory import FINAL, THINKING, TOOL_CALL, TOOL_RESULT, AgentTrajectory, Step

ATEM = DIALECTS["atem"]
TEMPLATE = Path("hermes/templates/chat-template-atem.jinja")
SCHEMAS = Path("hermesbench/harness/tools.json")


def _render(record: dict[str, Any]) -> str:
    """The pinned template, in the sandbox `transformers` renders chat templates in."""
    environment = ImmutableSandboxedEnvironment()

    def raise_exception(message: str) -> None:
        raise TemplateError(message)

    environment.globals["raise_exception"] = raise_exception
    template = environment.from_string(TEMPLATE.read_text(encoding="utf-8"))
    return template.render(
        messages=record["messages"],
        tools=record.get("tools"),
        bos_token="<|bos|>",
        add_generation_prompt=False,
    )


@pytest.fixture
def trajectory() -> AgentTrajectory:
    """The shape the defects hid in: reasoning, then a call, then a result, then an answer.

    Reasoning on a *tool-calling* turn is the case that failed silently -- the template drops
    `content` when `tool_calls` are present -- so the fixture has to have one.
    """
    return AgentTrajectory(
        task="Count the lines in logs/one.log.",
        success=True,
        system="You work from evidence the workspace can produce.",
        tools_available=("terminal", "file_read"),
        steps=(
            Step(kind=THINKING, content="wc may not exist here. Read the file instead."),
            Step(kind=TOOL_CALL, tool="file_read", args={"path": "logs/one.log"}, call_id="c0"),
            Step(kind=TOOL_RESULT, content="a\nb\nc\n", call_id="c0", ok=True),
            Step(kind=THINKING, content="Three lines."),
            Step(kind=FINAL, content="3"),
        ),
    )


@pytest.fixture
def schemas() -> dict[str, dict[str, Any]]:
    return load_tool_schemas(SCHEMAS)


def test_an_atem_row_renders_at_all(trajectory, schemas):
    """The whole point. This raised on its first line before the row was dialect-aware."""
    text = _render(to_messages_record(trajectory, dialect=ATEM, tool_schemas=schemas))
    assert "<|bos|>" in text
    assert '<atem:invoke name="file_read">' in text, "the call has to reach the wire format"
    assert '<tool_output name="file_read">' in text, "and its result has to come back in one"


def test_the_reasoning_survives_on_a_tool_calling_turn(trajectory, schemas):
    """The silent one.

    The template renders `reasoning_content` as `assistant to=self` and reads `content` only on a
    message with no `tool_calls`. With reasoning in `content`, this assertion fails while the row
    itself still visibly contains the reasoning -- a corpus that looks right and teaches nothing
    about deliberating before acting.
    """
    text = _render(to_messages_record(trajectory, dialect=ATEM, tool_schemas=schemas))
    assert "to=self" in text
    assert "wc may not exist here" in text, "the reasoning that led to the call must be trained"
    assert "<think>" not in text, "and not as a tag this model's template has no notion of"


def test_the_reasoning_is_not_trained_as_visible_answer_prose(trajectory, schemas):
    """The other half of the same defect. On a turn with no calls the template *does* render
    `content`, so a `<think>` block there is trained as prose the user sees."""
    text = _render(to_messages_record(trajectory, dialect=ATEM, tool_schemas=schemas))
    answer = text.rsplit("<|start|>assistant", 1)[-1]
    assert "3" in answer
    assert "<think>" not in answer and "Three lines" not in answer.split("<|message|>")[-1].split("\n")[0]


def test_a_hermes_row_still_uses_the_hermes_shape(trajectory):
    """The default is unchanged, so nothing that was written against the Hermes shape moves.

    Asserted on the row rather than through a template because this is the shape the Hermes
    templates already consume, and it is what every existing caller of this function expects.
    """
    record = to_messages_record(trajectory)
    assistant = next(m for m in record["messages"] if m.get("tool_calls"))
    assert "<think>" in assistant["content"], "Hermes trains reasoning inside the content"
    assert "reasoning_content" not in assistant
    assert isinstance(assistant["tool_calls"][0]["function"]["arguments"], str), "a JSON string"
    assert record["tools"] == ["terminal", "file_read"], "names; the definitions are in the prompt"


def test_atem_arguments_are_a_mapping_not_a_string(trajectory, schemas):
    """The template says why: it cannot parse a JSON string in the HF jinja sandbox. It raises
    rather than rendering something wrong, so this was a hard failure on every tool-calling row."""
    record = to_messages_record(trajectory, dialect=ATEM, tool_schemas=schemas)
    arguments = next(m for m in record["messages"] if m.get("tool_calls"))["tool_calls"][0]["function"]["arguments"]
    assert arguments == {"path": "logs/one.log"}


def test_an_atem_row_without_schemas_is_refused_not_written(trajectory):
    """Refused at the renderer, because the alternative is a corpus file whose rows raise one by one
    inside somebody's training run, hours in, with the traceback pointing at a jinja template."""
    with pytest.raises(ProtocolError, match="renders tool definitions from the row"):
        to_messages_record(trajectory, dialect=ATEM)


def test_a_tool_this_harness_has_no_schema_for_is_refused(trajectory, schemas):
    """A stub signature would show the model parameters that do not exist. Same refusal
    `to_hermes_record` already makes, for the same reason."""
    invented = AgentTrajectory(
        task=trajectory.task,
        success=True,
        system=trajectory.system,
        tools_available=("terminal", "telepathy"),
        steps=trajectory.steps,
    )
    with pytest.raises(ProtocolError, match="no schema for"):
        to_messages_record(invented, dialect=ATEM, tool_schemas=schemas)


def test_the_pinned_template_still_refuses_a_json_string(schemas):
    """A guard on the guard.

    `tool_arguments_json=False` is only correct while the template actually refuses a string. If
    upstream starts accepting one, this test fails and the flag becomes a choice rather than a
    requirement -- which is worth knowing, because the reason for the flag would have changed.
    """
    record = {
        "messages": [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c0", "type": "function", "function": {"name": "terminal", "arguments": '{"command": "ls"}'}}
                ],
            },
        ],
        "tools": [],
    }
    with pytest.raises(TemplateError, match="requires tool_call.function.arguments to be a dict"):
        _render(record)


def test_the_pinned_template_still_ignores_content_beside_tool_calls():
    """The reason reasoning cannot live in `content` for this dialect. If this ever stops being true
    the `reasoning_in_content` flag is no longer load-bearing, and that should be visible."""
    text = _render(
        {
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": "PROSE-THAT-MUST-VANISH",
                    "tool_calls": [
                        {
                            "id": "c0",
                            "type": "function",
                            "function": {"name": "terminal", "arguments": {"command": "ls"}},
                        }
                    ],
                },
            ],
            "tools": [],
        }
    )
    assert "PROSE-THAT-MUST-VANISH" not in text, (
        "the template drops content on a tool-calling message; that is why reasoning goes in "
        "reasoning_content for this dialect"
    )


def test_wire_markup_in_a_pre_fix_trajectory_is_stripped_before_training(schemas):
    """The harness stopped recording call markup inside reasoning. Logs written before it did are on
    disk and will be aggregated, and this row would put that markup on the channel the template
    renders private deliberation to -- training the model to address a tool from inside its own head.

    Stripped rather than refused, because refusing makes every pre-fix log unaggregatable. Counted on
    the message, because a row that was silently repaired is a row nobody can audit.
    """
    dirty = AgentTrajectory(
        task="Count the lines in logs/one.log.",
        success=True,
        system="s",
        tools_available=("terminal",),
        steps=(
            Step(
                kind=THINKING,
                content=(
                    "We have logs directory. Let's list logs.\n"
                    "<atem:function_calls>\n"
                    '<atem:invoke name="terminal">\n'
                    '<atem:parameter name="command">ls -la logs</atem:parameter>\n'
                    "</atem:invoke>\n"
                    "</atem:function_calls>"
                ),
            ),
            Step(kind=TOOL_CALL, tool="terminal", args={"command": "ls -la logs"}, call_id="c0"),
            Step(kind=TOOL_RESULT, content="one.log\n", call_id="c0", ok=True),
            Step(kind=FINAL, content="3"),
        ),
    )
    record = to_messages_record(dirty, dialect=ATEM, tool_schemas=schemas)
    assistant = next(m for m in record["messages"] if m.get("tool_calls"))
    assert "We have logs directory" in assistant["reasoning_content"], "the prose survives"
    assert "<atem:" not in assistant["reasoning_content"], "the markup does not"
    assert assistant["reasoning_markup_stripped"] == 1, "and the repair is on the record"
    assert "<atem:invoke" not in _render(record).split("to=self<|message|>")[1].split("<|eom|>")[0]


def test_a_clean_trajectory_carries_no_repair_marker(trajectory, schemas):
    """The marker means something only if it is absent when nothing was repaired."""
    record = to_messages_record(trajectory, dialect=ATEM, tool_schemas=schemas)
    assert all("reasoning_markup_stripped" not in m for m in record["messages"])
