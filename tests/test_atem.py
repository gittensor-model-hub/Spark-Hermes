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
from hermes.protocol import ParsedCall, ProtocolError

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


# --- the real model's real output -------------------------------------------------------------------
#
# Generated by meta-models/Muse-Glimmer-30B at revision 97c77dff on an RTX PRO 6000, through plain
# transformers with the tools passed natively, on 2026-08-11. Kept verbatim because every synthetic
# fixture in this file was written by the same person who wrote the parser, and this one was not.
#
# It also settles two things that had been reasoned about rather than observed:
#   - the model emits ATEM natively, so the dialect is real and not an instruction it may ignore
#   - with tools passed natively the template's recipient line reads
#     `# Valid recipients: "self", "terminal.*", "user".` -- the `terminal.*` entry appears only
#     because the tools went with the request, which is what `Dialect.tools_in_prompt` is about.

REAL_COMPLETION = (
    " to=self<|message|>The directory logs/ holds several .log files. Count the total lines across "
    "all of them. Start by listing the directory.\n\nWe need to list directory logs/. Then count "
    "total lines across all .log files.\n\nWe need to use terminal tool. First list "
    "directory.<|eom|><|start|>assistant to=terminal<|message|><atem:function_calls>\n"
    '<atem:invoke name="terminal">\n'
    '<atem:parameter name="command">ls -l logs</atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls><|eot|>"
)


def test_the_real_completion_parses_to_one_call():
    turn = parse_turn(REAL_COMPLETION, schemas=SCHEMA)
    assert turn.calls == (ParsedCall(name="terminal", arguments={"command": "ls -l logs"}),)
    assert turn.malformed == ()
    assert turn.well_formed and len(turn.safe_calls) == 1


def test_the_channel_framing_does_not_reach_the_answer():
    """`text` becomes the FINAL step, which is what a task is graded on. Left in, the answer would
    carry `<|eom|><|start|>assistant to=user<|message|>`."""
    turn = parse_turn(REAL_COMPLETION, schemas=SCHEMA)
    for marker in ("<|start|>", "<|message|>", "<|eom|>", "<|eot|>", "to=self", "to=terminal"):
        assert marker not in turn.text, marker


def test_inline_reasoning_is_recovered_when_the_server_does_not_split_it():
    """Two serving behaviours, both real: an endpoint that returns `reasoning_content` separately,
    and a raw generate() that leaves the `to=self` turn in the content. Handled either way rather
    than assuming the serving layer is tidy."""
    turn = parse_turn(REAL_COMPLETION, schemas=SCHEMA)
    assert "We need to use terminal tool" in turn.scratch_pad
    assert turn.text == "", "the deliberation is reasoning, not the answer"


def test_an_explicit_reasoning_argument_wins_over_the_inline_channel():
    """When the server splits it, that is the authoritative copy -- it has not been through a
    text-stripping heuristic."""
    turn = parse_turn(REAL_COMPLETION, reasoning="server-split copy", schemas=SCHEMA)
    assert turn.scratch_pad == "server-split copy"


def test_stripping_the_framing_does_not_eat_the_answer():
    """The first version used `to=\\S+`, which swallowed `to=self<|message|>The` in one bite and
    took the first word of the answer with it. Caught on this transcript."""
    turn = parse_turn(
        " to=self<|message|>thinking<|eom|><|start|>assistant to=user<|message|>The answer is 6.<|eot|>",
        schemas=SCHEMA,
    )
    assert turn.text == "The answer is 6."
    assert turn.scratch_pad == "thinking"
    assert turn.abstained, "a final answer with no call is an abstention, not a malformed turn"


# --- when the serving layer parses the wire format for us ------------------------------------------
#
# SGLang with `--tool-call-parser muse` returns OpenAI `tool_calls` and leaves `content` EMPTY.
# Measured on the real server: finish_reason=tool_calls, prompt_tokens=424, content=''. A policy
# that only read `content` would see nothing said and score an abstention -- reporting a model that
# called a tool correctly as one that declined to, on every single turn.


def _structured_policy(raw_extra, *, content=""):
    from hermes.protocol import DIALECTS
    from hermesbench.policy import ServedModelPolicy

    def complete(messages, *, tools=None):
        return content, {"prompt_tokens": 424, "completion_tokens": 231, "total_tokens": 655, **raw_extra}

    return ServedModelPolicy(
        complete=complete,
        dialect=DIALECTS["atem"],
        tool_schemas={"terminal": {"description": "d", "parameters": {}}},
    )


def test_server_parsed_calls_are_used_rather_than_the_empty_content():
    """The real response shape from SGLang."""
    from hermes.trajectory import TOOL_CALL

    policy = _structured_policy(
        {
            "tool_calls": [{"name": "terminal", "arguments": '{"command": "ls -l logs"}'}],
            "reasoning_content": "The directory logs/ holds several .log files.",
        }
    )
    steps = policy.next_steps(_task(), [])
    calls = [s for s in steps if s.kind == TOOL_CALL]
    assert len(calls) == 1 and calls[0].tool == "terminal"
    assert calls[0].args == {"command": "ls -l logs"}
    assert policy.parse_failures == 0


def test_an_empty_content_with_a_call_is_not_an_abstention():
    """The failure this exists to prevent: `abstained` requires something said AND no calls, so a
    turn whose calls were invisible would qualify."""
    from hermes.protocol import ParsedTurn
    from hermesbench.policy import _turn_from_tool_calls

    turn = _turn_from_tool_calls(
        [{"name": "terminal", "arguments": '{"command": "ls"}'}], text="", reasoning="thinking"
    )
    assert isinstance(turn, ParsedTurn)
    assert not turn.abstained and len(turn.calls) == 1


def test_unreadable_server_arguments_are_malformed_not_dropped():
    """The same failure `parse_turn` reports for an unreadable call. Dropping it would score the
    protocol's hardest failure as its most disciplined behaviour."""
    from hermesbench.policy import _turn_from_tool_calls

    turn = _turn_from_tool_calls([{"name": "terminal", "arguments": '{"command": '}], text="", reasoning="")
    assert turn.calls == () and turn.malformed and "not readable JSON" in turn.malformed[0]


def test_arguments_that_decode_to_a_non_object_are_refused():
    from hermesbench.policy import _turn_from_tool_calls

    turn = _turn_from_tool_calls([{"name": "terminal", "arguments": "[1, 2]"}], text="", reasoning="")
    assert turn.calls == () and "not an object" in turn.malformed[0]


def test_a_call_with_no_name_is_malformed():
    from hermesbench.policy import _turn_from_tool_calls

    turn = _turn_from_tool_calls([{"arguments": "{}"}], text="", reasoning="")
    assert turn.calls == () and "no function name" in turn.malformed[0]


def test_a_dict_of_arguments_is_accepted_as_well_as_a_json_string():
    """Servers differ: some hand back the decoded object. Both shapes are the same call."""
    from hermesbench.policy import _turn_from_tool_calls

    turn = _turn_from_tool_calls([{"name": "terminal", "arguments": {"command": "ls"}}], text="", reasoning="")
    assert turn.calls[0].arguments == {"command": "ls"}


def test_raw_atem_in_the_content_is_still_parsed_when_the_server_did_not():
    """A server without the parser, or a raw generate() path. Both are real, so both work."""
    from hermes.trajectory import TOOL_CALL

    policy = _structured_policy({}, content=render_tool_call("terminal", {"command": "ls"}))
    steps = policy.next_steps(_task(), [])
    assert any(s.kind == TOOL_CALL for s in steps)


# --- calls the serving layer leaves behind ---------------------------------------------------------
#
# Measured against SGLang on 2026-08-11: the model emitted two complete calls in one turn, the
# server returned the first as a structured tool_call and left the second's markup in `content`.
# Reading only the structured list dropped it -- never executed, never counted in `tool_calls`, and
# not malformed either, because it was perfectly well formed. It became a THINKING step, which is
# the one place a failure is indistinguishable from real reasoning.


REAL_LEFTOVER = (
    "We have logs directory. Let's list logs.\n"
    "<atem:function_calls>\n"
    '<atem:invoke name="terminal">\n'
    '<atem:parameter name="command">ls -la logs</atem:parameter>\n'
    "</atem:invoke>\n"
    "</atem:function_calls>"
)


def test_a_call_left_in_the_reasoning_is_not_lost():
    """The transcript above, verbatim from the run that exposed this.

    The server returned one call structurally and left this one in `reasoning_content`, which is the
    channel it usually lands in: an ATEM turn is `assistant to=self` deliberation followed by
    `assistant to=<tool>` carrying the call, so a reasoning parser that does not stop cleanly at the
    end of the first swallows the second. Over one 19-task run this happened in 15 turns across 11
    episodes, and 8 of those calls matched nothing the server returned -- executed nowhere, counted
    nowhere, and not malformed either.
    """
    from hermes.trajectory import TOOL_CALL

    policy, _ = _policy("")
    policy.complete = lambda messages, *, tools=None: (
        "",
        {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "reasoning_content": REAL_LEFTOVER,
            "tool_calls": [{"name": "terminal", "arguments": '{"command": "ls -la"}'}],
        },
    )
    steps = policy.next_steps(_task(), [])
    calls = [s for s in steps if s.kind == TOOL_CALL]
    assert [c.args["command"] for c in calls] == ["ls -la", "ls -la logs"], (
        "both the structured call and the one left in the reasoning must be executed, in order"
    )


def test_the_recovered_call_is_not_left_in_the_recorded_reasoning():
    """Because the trajectory is the SFT corpus.

    A thinking step that still carries `<atem:function_calls>` is rendered into a training row, and
    the pinned template puts a thinking step on the `to=self` channel -- so the row would teach the
    model to emit a call where its own template renders private deliberation. Stripping it at the
    parser is what keeps that out of every corpus built downstream.
    """
    from hermes.trajectory import THINKING

    policy, _ = _policy("")
    policy.complete = lambda messages, *, tools=None: (
        "",
        {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "reasoning_content": REAL_LEFTOVER,
            "tool_calls": [{"name": "terminal", "arguments": '{"command": "ls -la"}'}],
        },
    )
    thinking = [s for s in policy.next_steps(_task(), []) if s.kind == THINKING]
    assert thinking, "the deliberation itself is still recorded"
    joined = "\n".join(s.content for s in thinking)
    assert "We have logs directory" in joined, "the prose the model actually reasoned in survives"
    assert "<atem:" not in joined, "the wire markup does not"


def test_a_call_echoed_in_the_reasoning_is_executed_once():
    """7 of the 15 measured turns were echoes: the same call, returned structurally AND left in the
    reasoning. Running those twice is worse than losing them -- a duplicated mutating command is not
    idempotent, and `rm -rf build` twice is a different episode than once."""
    from hermes.trajectory import TOOL_CALL

    policy, _ = _policy("")
    policy.complete = lambda messages, *, tools=None: (
        "",
        {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "reasoning_content": "Let's clear the build.\n" + render_tool_call("terminal", {"command": "rm -rf build"}),
            "tool_calls": [{"name": "terminal", "arguments": '{"command": "rm -rf build"}'}],
        },
    )
    calls = [s for s in policy.next_steps(_task(), []) if s.kind == TOOL_CALL]
    assert len(calls) == 1, "deduplicated on name and arguments"


def test_a_call_left_in_the_text_is_not_lost_either():
    """The `content` channel gets the same treatment. Not measured in the run above -- every observed
    leftover was in the reasoning -- but a tool parser is under no obligation about which channel it
    leaves a call in, and the failure is silent in both."""
    from hermes.trajectory import TOOL_CALL

    policy, _ = _policy(REAL_LEFTOVER)
    policy.complete = lambda messages, *, tools=None: (
        REAL_LEFTOVER,
        {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "tool_calls": [{"name": "terminal", "arguments": '{"command": "ls -la"}'}],
        },
    )
    calls = [s for s in policy.next_steps(_task(), []) if s.kind == TOOL_CALL]
    assert [c.args["command"] for c in calls] == ["ls -la", "ls -la logs"]


def test_prose_beside_a_leftover_call_survives_as_text():
    """A turn that is half answer and half call keeps the answer: `text` becomes the FINAL step."""
    from hermes.protocol import DIALECTS
    from hermesbench.policy import _merge_leftover_calls, _turn_from_tool_calls

    turn = _turn_from_tool_calls([{"name": "terminal", "arguments": "{}"}], text=REAL_LEFTOVER, reasoning="r")
    merged = _merge_leftover_calls(turn, text=REAL_LEFTOVER, reasoning="r", dialect=DIALECTS["atem"], schemas={})
    assert "We have logs directory" in merged.text
    assert "<atem:invoke" not in merged.text
    assert merged.scratch_pad == "r"


def test_a_malformed_leftover_is_still_counted():
    """A truncated call left behind is the reason `malformed_turns` exists, and it must survive the
    merge rather than being replaced by the structured call's clean verdict."""
    from hermes.protocol import DIALECTS
    from hermesbench.policy import _merge_leftover_calls, _turn_from_tool_calls

    truncated = '<atem:function_calls>\n<atem:invoke name="terminal">\n<atem:parameter name="command">ls'
    turn = _turn_from_tool_calls([{"name": "terminal", "arguments": "{}"}], text="", reasoning=truncated)
    assert turn.malformed == ()
    merged = _merge_leftover_calls(turn, text="", reasoning=truncated, dialect=DIALECTS["atem"], schemas={})
    assert merged.malformed, "the truncation is the measurement"
    assert merged.safe_calls == (), "and nothing from a turn that confused the parser executes"


def test_the_parser_reads_the_reasoning_channel_without_a_structured_call():
    """The hole the first version of this fix left.

    `_merge_leftover_calls` runs only when the server returned structured `tool_calls`. When it
    returned none, `next_steps` reaches the dialect's parser directly -- and that parser recorded the
    supplied reasoning verbatim. A re-run of the suite measured 7 turns still carrying markup and 5
    calls still lost through exactly this path, so the recovery belongs in the parser, where both
    branches reach it.
    """
    from hermes.atem import parse_turn as parse_atem

    turn = parse_atem("", reasoning=REAL_LEFTOVER)
    assert [c.arguments["command"] for c in turn.calls] == ["ls -la logs"], "recovered, not lost"
    assert "We have logs directory" in turn.scratch_pad, "the deliberation survives"
    assert "<atem:" not in turn.scratch_pad, "the markup does not"


def test_a_call_in_the_reasoning_that_the_content_also_has_runs_once():
    """The echo case on the parser path: the server left the markup in the reasoning AND the model
    emitted the same call in the content. Dedupe is against what the content already yielded."""
    from hermes.atem import parse_turn as parse_atem

    same = render_tool_call("terminal", {"command": "ls pkg"})
    turn = parse_atem(same, reasoning="Let's look at pkg.\n" + same)
    assert len(turn.calls) == 1, "one call, not two"
    assert turn.scratch_pad == "Let's look at pkg."


def test_a_truncated_call_in_the_reasoning_is_counted_not_swallowed():
    """A leftover that broke mid-call is the reason `malformed_turns` exists. Recovering calls from
    this channel must not turn a truncation into silence."""
    from hermes.atem import parse_turn as parse_atem

    truncated = '<atem:function_calls>\n<atem:invoke name="terminal">\n<atem:parameter name="command">ls'
    turn = parse_atem("", reasoning="Let me look.\n" + truncated)
    assert turn.malformed, "the truncation is the measurement"
    assert turn.safe_calls == ()


def test_reasoning_with_no_markup_is_untouched():
    """The common case must not be reshaped by the recovery path -- a parser that rewrites ordinary
    reasoning is a parser that changes every episode to fix a few."""
    from hermes.atem import parse_turn as parse_atem

    prose = "The file may not exist.\nI should check before reading it."
    turn = parse_atem("", reasoning=prose)
    assert turn.scratch_pad == prose
    assert turn.calls == ()


def test_a_recovered_call_is_not_fed_back_to_the_model_as_its_own_prose():
    """The third consequence of the same defect, and the one that compounds.

    `_messages` rebuilds the conversation from the trajectory on every turn, so a thinking step
    holding raw call markup is re-sent to the model as something it said -- an unexecuted call with no
    result, re-sent again on every later turn. Measured on the pre-fix log: 2 of 2 rebuilt assistant
    messages carried it.

    So the cleaning that keeps markup out of the corpus keeps it out of the context too. Asserted here
    as well as in the corpus tests because these are two different consumers of one field, and a fix to
    either alone leaves the other broken.
    """
    from pathlib import Path as _Path

    from hermes.pin import load_tool_schemas
    from hermes.protocol import DIALECTS
    from hermes.trajectory import THINKING
    from hermesbench.policy import ServedModelPolicy, steps_from_turn
    from hermesbench.tasks import Task

    atem = DIALECTS["atem"]
    schemas = load_tool_schemas(_Path("hermesbench/harness/tools.json"))
    policy = ServedModelPolicy(
        complete=lambda messages, *, tools=None: ("", {}),
        dialect=atem,
        tool_schemas=schemas,
    )
    task = Task(task_id="t", prompt="p", tools=("terminal",), verify="true", tags=())

    history = steps_from_turn(parse_turn("", reasoning=REAL_LEFTOVER, schemas=schemas))
    assistant = [m for m in policy._messages(task, history) if m["role"] == "assistant"]
    assert assistant, "the turn is replayed at all"
    replayed = "\n".join(m["content"] for m in assistant)

    assert "We have logs directory" in replayed, "the model still sees what it was reasoning about"
    # Exactly one: the call the harness renders because it EXECUTED it. Two would mean the reasoning
    # kept its own copy, which is the state that fed the model an unexecuted call.
    assert replayed.count("<atem:function_calls>") == 1, replayed
    reasoning_step = next(s for s in history if s.kind == THINKING)
    assert "<atem:" not in reasoning_step.content, "and the recorded reasoning is clean at the source"
