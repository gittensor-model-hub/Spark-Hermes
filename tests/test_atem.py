"""The ATEM wire format, and the failures it has that JSON does not.

Muse-Glimmer-30B emits one element per parameter with a text value, where Hermes emits one JSON
object per call. That removes a whole class of malformed turn -- there is no arguments object, so
there is no JSON syntax error to make -- and introduces two others: a call block can be opened and
truncated (an ATEM call is several times longer than the JSON equivalent, so there is more of it to
cut off), and a parameter can be left unterminated so its value swallows the rest of the call.

Both of those resolve to "no calls found" unless something looks for them, which is the reading
`ParsedTurn` exists to prevent -- a worker whose every generation truncates would otherwise post a
perfect abstention rate.

The round-trip tests go through `render_*` and back rather than asserting against a hand-written
string, except where the exact bytes matter: what the model's own template writes is the thing this
has to agree with, and a test that only checks self-consistency would pass on a format the serving
layer never sends.
"""

import json

import pytest

from hermes.atem import (
    CALLS_CLOSE,
    CALLS_OPEN,
    coerce,
    parse_turn,
    render_tool_call,
    render_tool_calls,
    render_tool_response,
)
from hermes.protocol import ProtocolError

SCHEMA = {
    "terminal": {"properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}},
    "edit": {"properties": {"path": {"type": "string"}, "lines": {"type": "array"}, "force": {"type": "boolean"}}},
}


# --- what the model's template actually writes ---------------------------------------------------


def test_a_call_renders_the_shape_the_template_writes():
    """Asserted against the literal bytes. The template is upstream and fixed; a renderer that
    only agreed with its own parser would train the model on a shape nothing sends it."""
    rendered = render_tool_call("terminal", {"command": "ls logs/"})
    assert rendered == (
        "<atem:function_calls>\n"
        '<atem:invoke name="terminal">\n'
        '<atem:parameter name="command">ls logs/</atem:parameter>\n'
        "</atem:invoke>\n"
        "</atem:function_calls>"
    )


def test_parameter_order_follows_the_mapping_not_a_sort():
    """The template iterates the arguments as given. Sorting here would produce rows the model
    never generated itself, teaching a normalisation nothing else applies."""
    rendered = render_tool_call("edit", {"path": "a.py", "force": True, "lines": [1, 2]})
    assert rendered.index('name="path"') < rendered.index('name="force"') < rendered.index('name="lines"')


def test_values_are_written_the_way_the_template_writes_them():
    rendered = render_tool_call("edit", {"force": True, "lines": [1, 2], "path": None, "timeout": 30})
    assert ">true<" in rendered
    assert ">[1, 2]<" in rendered, "list- and dict-shaped values go through tojson"
    assert ">null<" in rendered
    assert ">30<" in rendered


def test_several_calls_are_several_blocks():
    """The template emits one `<|start|>assistant to=<name>` turn per call, so one block per call.
    Nesting two invokes in one block parses here and is not what the model produces."""
    rendered = render_tool_calls([("terminal", {"command": "ls"}), ("terminal", {"command": "pwd"})])
    assert rendered.count(CALLS_OPEN) == 2 and rendered.count(CALLS_CLOSE) == 2
    assert len(parse_turn(rendered).calls) == 2


def test_a_tool_response_uses_the_output_tag():
    assert render_tool_response("terminal", "seg1.log") == '<tool_output name="terminal">\nseg1.log\n</tool_output>'


def test_a_call_without_a_name_is_refused():
    with pytest.raises(ProtocolError, match="needs a name"):
        render_tool_call("", {"command": "ls"})
    with pytest.raises(ProtocolError, match="needs the name"):
        render_tool_response("", "output")


# --- names are structural ---------------------------------------------------------------------------


def test_a_quote_in_a_name_cannot_close_the_attribute():
    """A name is a place where a crafted string changes the structure of the call rather than its
    content: an unescaped quote closes `name="` early and the remainder parses as more attributes."""
    rendered = render_tool_call('terminal" evil="yes', {"command": "ls"})
    assert 'name="terminal&quot; evil=&quot;yes"' in rendered
    parsed = parse_turn(rendered)
    # One call, and the injected `evil="yes"` is inert text inside the quoted value rather than a
    # second attribute. The name is not unescaped on the way back, deliberately: a tool name that
    # needed escaping is not a real tool name, so it fails as unknown at the executor -- which is
    # the direction that fails loudly instead of dispatching something crafted.
    assert len(parsed.calls) == 1
    assert parsed.calls[0].name == "terminal&quot; evil=&quot;yes"
    assert parsed.calls[0].arguments == {"command": "ls"}


def test_a_value_cannot_close_its_own_parameter():
    rendered = render_tool_call("terminal", {"command": "echo </atem:parameter> hi"})
    parsed = parse_turn(rendered)
    assert len(parsed.calls) == 1
    assert parsed.calls[0].arguments["command"].startswith("echo &lt;/atem:parameter>")


# --- the failures this format has that JSON does not -------------------------------------------------


def test_a_truncated_call_is_malformed_not_an_abstention():
    """The likeliest break under a token limit here. Folding it into `abstained` would score the
    protocol's hardest failure as its most disciplined behaviour."""
    cut = '<atem:function_calls>\n<atem:invoke name="terminal">\n<atem:parameter name="command">ls'
    parsed = parse_turn(cut)
    assert parsed.malformed and "never closed" in parsed.malformed[0]
    assert not parsed.abstained and not parsed.well_formed
    assert parsed.safe_calls == ()


def test_an_unterminated_parameter_is_reported_even_though_the_others_parsed():
    """The parameters that did parse look like a complete call, which is exactly why silence here
    would be wrong."""
    body = (
        "<atem:function_calls>\n"
        '<atem:invoke name="edit">\n'
        '<atem:parameter name="path">a.py</atem:parameter>\n'
        '<atem:parameter name="lines">[1, 2'
        "</atem:invoke>\n"
        "</atem:function_calls>"
    )
    parsed = parse_turn(body)
    assert parsed.malformed and "not closed" in parsed.malformed[0]
    assert parsed.safe_calls == ()


def test_an_empty_call_block_is_malformed():
    """Not an abstention and not a call. Unreported it resolves to "no calls found"."""
    parsed = parse_turn(f"{CALLS_OPEN}\n{CALLS_CLOSE}")
    assert parsed.malformed and "nothing callable" in parsed.malformed[0]


def test_an_empty_invoke_name_is_malformed():
    parsed = parse_turn(f'{CALLS_OPEN}\n<atem:invoke name="">\n</atem:invoke>\n{CALLS_CLOSE}')
    assert parsed.malformed and "empty name" in parsed.malformed[0]


@pytest.mark.parametrize(
    "near",
    [
        "<function_calls><invoke name='terminal'></invoke></function_calls>",
        '<atem :invoke name="terminal">',
        "<ATEM:FUNCTION_CALLS>",
        "</atem:invoke>",
    ],
)
def test_a_call_shaped_near_miss_is_reported_rather_than_read_as_prose(near):
    """The namespace prefix is part of the tag. Dropping it, or spacing it, produces text that
    looks like a call to a reader and is not one to the parser -- and would otherwise be counted
    as a well-formed natural-language answer."""
    parsed = parse_turn(f"I will run this.\n{near}")
    assert parsed.malformed, f"{near!r} passed as prose"
    assert not parsed.abstained


def test_a_near_miss_is_only_looked_for_outside_real_blocks():
    """A real call contains `<atem:invoke>`, which is loose-tag-shaped. Scanning the whole turn
    would report every correct call as a near miss."""
    assert parse_turn(render_tool_call("terminal", {"command": "ls"})).malformed == ()


def test_prose_with_no_call_is_an_abstention():
    parsed = parse_turn("No tool fits this question, so here is the answer directly.")
    assert parsed.abstained and parsed.calls == () and parsed.malformed == ()


def test_an_empty_completion_is_not_an_abstention():
    """A crashed worker, not a decision."""
    assert not parse_turn("   ").abstained


# --- reasoning arrives on its own channel --------------------------------------------------------------


def test_reasoning_is_taken_as_an_argument_not_found_in_the_text():
    """ATEM puts deliberation on a separate `assistant to=self` turn, surfaced as
    `reasoning_content`. Searching the text for it would find nothing and report every turn as
    having skipped deliberation."""
    parsed = parse_turn("Running it now.", reasoning="I should read the logs first.")
    assert parsed.scratch_pad == "I should read the logs first."


def test_a_think_tag_in_the_text_is_not_treated_as_reasoning():
    """This is not Hermes. A `<think>` block here is content the model wrote, not a channel."""
    parsed = parse_turn("<think>hmm</think> answer")
    assert parsed.scratch_pad == ""
    assert "hmm" in parsed.text


# --- recovering types the format threw away ------------------------------------------------------------


def test_declared_types_are_recovered_from_text():
    values = coerce({"command": "ls", "timeout": "30"}, SCHEMA["terminal"])
    assert values == {"command": "ls", "timeout": 30}


def test_a_value_with_no_declared_type_stays_a_string():
    """Sniffing turns `{"path": "123"}` into `{"path": 123}` and hands a tool an integer where it
    declared a filename. Under-recovering fails at the tool boundary, where the reason is visible."""
    assert coerce({"path": "123"}, None) == {"path": "123"}
    assert coerce({"path": "123"}, {"properties": {}}) == {"path": "123"}


def test_a_string_that_looks_like_a_bool_stays_a_string_when_declared_string():
    """The format cannot distinguish `true` from the string "true", so the schema decides."""
    assert coerce({"command": "true"}, SCHEMA["terminal"]) == {"command": "true"}


def test_objects_and_arrays_come_back_through_json():
    values = coerce({"lines": "[1, 2, 3]", "force": "true"}, SCHEMA["edit"])
    assert values["lines"] == [1, 2, 3] and values["force"] is True


def test_a_payload_disagreeing_with_its_declared_type_is_left_as_text():
    """A schema saying `array` and a payload holding an object is a disagreement. Passing it
    through moves the failure into the tool, where the reason is no longer visible."""
    assert coerce({"lines": '{"a": 1}'}, SCHEMA["edit"])["lines"] == '{"a": 1}'


def test_an_unparseable_value_is_returned_as_written():
    """So the tool refuses it with the real value in the message rather than a coerced one."""
    assert coerce({"timeout": "soon"}, SCHEMA["terminal"])["timeout"] == "soon"
    assert coerce({"lines": "[1, 2"}, SCHEMA["edit"])["lines"] == "[1, 2"


def test_parse_turn_applies_the_schema_when_given_one():
    rendered = render_tool_call("terminal", {"command": "ls", "timeout": 30})
    assert parse_turn(rendered).calls[0].arguments["timeout"] == "30", "no schema, no recovery"
    assert parse_turn(rendered, schemas=SCHEMA).calls[0].arguments["timeout"] == 30


# --- the round trip -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    [
        {"command": "ls -la"},
        {"command": "grep -r 'x' . | head -5"},
        {"command": "echo 'multi\nline\nvalue'"},
        {"path": "a.py", "lines": [1, 2, 3], "force": False},
        {"command": ""},
        {"command": "unicode: ⚡ ünïcode 日本語"},
    ],
)
def test_a_rendered_call_parses_back_to_what_went_in(arguments):
    parsed = parse_turn(render_tool_call("terminal", arguments), schemas={"terminal": {"properties": {}}})
    assert len(parsed.calls) == 1
    assert parsed.malformed == ()
    got = parsed.calls[0].arguments
    for key, value in arguments.items():
        expected = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        if isinstance(value, bool):
            expected = "true" if value else "false"
        assert got[key] == expected, key


def test_a_call_surrounded_by_prose_keeps_both():
    content = f"First I will look at the logs.\n{render_tool_call('terminal', {'command': 'ls'})}\nThen decide."
    parsed = parse_turn(content)
    assert len(parsed.calls) == 1
    assert "First I will look" in parsed.text and "Then decide" in parsed.text


# --- driving an episode in ATEM through the policy -------------------------------------------------
#
# The wiring, and the one part of it that is not obvious: the tool definitions must be passed with
# the request rather than embedded in the system prompt. That template appends
# `render_system_meta(tools)` to EVERY system message, and with no native tools it emits
# `# Valid recipients: "self", "user".` -- while a tool call is an assistant turn addressed
# `to=<tool namespace>`. Embedding the definitions therefore advertises tools and forbids calling
# them in the same prompt, and the symptom is a model that never calls one.


def _policy(reply, *, dialect_name="atem", system=""):
    from hermes.protocol import DIALECTS
    from hermesbench.policy import ServedModelPolicy

    seen = {}

    def complete(messages, *, tools=None):
        seen["messages"] = messages
        seen["tools"] = tools
        return reply, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    policy = ServedModelPolicy(
        complete=complete,
        dialect=DIALECTS[dialect_name],
        tool_schemas={
            "terminal": {
                "description": "Run a command.",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
            }
        },
        system=system,
    )
    return policy, seen


def _task():
    from hermesbench.tasks import Task

    return Task.from_record({"task_id": "t1", "prompt": "count the log lines", "verify": "true", "tools": ["terminal"]})


def test_the_tools_go_with_the_request_not_into_the_prompt():
    policy, seen = _policy("done")
    policy.next_steps(_task(), [])
    assert seen["tools"], "the definitions must reach the request, or the template renders none"
    assert seen["tools"][0]["function"]["name"] == "terminal"
    prompt = "".join(m["content"] for m in seen["messages"])
    assert "<atem:function_calls>" not in prompt, "the template writes the call format, not us"
    assert "terminal" not in prompt or "count the log lines" in prompt


def test_hermes_still_embeds_its_tools_and_sends_none_natively():
    """The other half of the dispatch. Hermes advertises inside `<tools>` and the request carries
    no tool field, which is what every existing run did."""
    policy, seen = _policy("done", dialect_name="hermes-4")
    policy.next_steps(_task(), [])
    assert seen["tools"] is None
    assert "<tools>" in seen["messages"][0]["content"]


def test_an_empty_system_turn_is_omitted_rather_than_sent_blank():
    """The template injects its own default system message when none is present -- reasoning
    strength, tool definitions, valid recipients. A blank system message suppresses that and leaves
    the model with no tool definitions at all."""
    policy, seen = _policy("done")
    policy.next_steps(_task(), [])
    assert [m["role"] for m in seen["messages"]] == ["user"]


def test_operator_framing_still_reaches_the_system_turn():
    policy, seen = _policy("done", system="One call per turn.")
    policy.next_steps(_task(), [])
    assert seen["messages"][0]["role"] == "system"
    assert seen["messages"][0]["content"] == "One call per turn."


def test_a_call_the_model_emits_is_parsed_and_becomes_steps():
    from hermes.trajectory import TOOL_CALL

    reply = render_tool_call("terminal", {"command": "wc -l logs/*.log"})
    policy, _ = _policy(reply)
    steps = policy.next_steps(_task(), [])
    calls = [s for s in steps if s.kind == TOOL_CALL]
    assert len(calls) == 1 and calls[0].tool == "terminal"
    assert calls[0].args == {"command": "wc -l logs/*.log"}
    assert policy.parse_failures == 0


def test_a_truncated_call_counts_as_a_parse_failure():
    """`malformed_turns` is what protects the wire format, and it has to mean the same thing in
    both dialects or a conformance comparison across them is meaningless."""
    policy, _ = _policy('<atem:function_calls>\n<atem:invoke name="terminal">\n<atem:parameter name="command">ls')
    policy.next_steps(_task(), [])
    # Exactly one. The truncated text still contains three call-shaped tags, and the first version
    # of the parser reported all of them -- four malformed entries for one broken turn, in the
    # metric the promotion gate bounds.
    assert policy.parse_failures == 1


def test_reasoning_content_is_read_off_the_response():
    """It arrives beside the content, not inside it. Looking for a tag in the text would find
    nothing and report every turn as having skipped deliberation."""
    from hermes.protocol import DIALECTS
    from hermesbench.policy import ServedModelPolicy

    def complete(messages, *, tools=None):
        # Realistic usage plus the reasoning key. `hermes.cost` refuses a usage object with none
        # of the OpenAI keys, which is the guard that caught the first version of this stub.
        return "Running it.", {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "reasoning_content": "I should list the files first.",
        }

    policy = ServedModelPolicy(
        complete=complete,
        dialect=DIALECTS["atem"],
        tool_schemas={"terminal": {"description": "d", "parameters": {}}},
    )
    steps = policy.next_steps(_task(), [])
    assert any("list the files" in (s.content or "") for s in steps)


def test_history_is_replayed_in_the_dialect_the_model_speaks():
    """A model reading its own prior turns in a foreign format is being taught mid-episode that
    the format is negotiable."""
    from hermes.trajectory import TOOL_CALL, TOOL_RESULT, Step

    history = [
        Step(kind=TOOL_CALL, tool="terminal", args={"command": "ls"}, call_id="c1"),
        Step(kind=TOOL_RESULT, call_id="c1", content="one.log", ok=True),
    ]
    policy, seen = _policy("done")
    policy.next_steps(_task(), history)
    replayed = "".join(m["content"] for m in seen["messages"])
    assert "<atem:invoke" in replayed and "<tool_output" in replayed
    assert "<tool_call>" not in replayed and "<tool_response>" not in replayed


def test_render_system_refuses_to_embed_tools_for_this_dialect():
    """Refused rather than ignored: dropping them silently produces a prompt that looks right and
    a model that cannot see its tools, and keeping them produces the recipient contradiction."""
    from hermes.protocol import DIALECTS, ProtocolError, render_system, tool_schema

    with pytest.raises(ProtocolError, match="serving template"):
        render_system([tool_schema("terminal", "d", {})], dialect=DIALECTS["atem"])
