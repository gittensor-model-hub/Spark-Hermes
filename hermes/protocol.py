"""The Hermes wire protocol: how a worker's tool use is actually spelled.

`hermes/format.py` renders trajectories into OpenAI-style `messages` with a structured
`tool_calls` array, which is what Axolotl and the Qwen3 chat template consume. That is a
fine training target for a Qwen worker and the wrong one for a *Hermes* worker. Hermes
does not emit a `tool_calls` array; it emits XML tags inside ordinary assistant text::

    <tool_call>
    {"name": "run_profiler", "arguments": {"kernel": "attention"}}
    </tool_call>

and reads results back as::

    <tool_response>
    {"name": "run_profiler", "content": {...}}
    </tool_response>

A model trained only on the OpenAI shape has learned tool use but not this protocol, and
the gap does not show up as a lower benchmark score -- it shows up as a worker that emits
nothing a Hermes runtime recognises. This module is the translation, so a trajectory can
be rendered into either target from one source of truth.

**Dialects are not interchangeable and there is no default.** Hermes 2 Pro and Hermes 3
return tool results in a `tool` role and reserve `<scratch_pad>` as a real token; Hermes 4
returns them in a **`user`** role -- for every base model, Llama and Qwen alike -- drops
`<scratch_pad>` entirely and reasons in `<think>`. Picking the wrong one produces a corpus
that trains the worker to expect observations in a role its runtime will never use, which
is invisible in the data and total at inference. So `dialect` is a required argument
everywhere rather than something with a sensible-looking fallback.

Two smaller facts that the format forces on any consumer:

**There are no call ids.** Nothing in the Hermes protocol correlates a call with its
result -- no `id`, no `tool_call_id`, in any version. Correlation is positional. The
internal `Step.call_id` is a device for validating trajectories and must not be rendered,
and anything reading results back has to match on order.

**Key order varies in Nous's own material.** The chat templates emit `name` first; the
Hermes 2 Pro and Hermes 3 model cards show `arguments` first, because the pydantic model
declares the fields in that order. Both orderings appear in training data, so parsing is
order-independent while emission is fixed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

TOOLS_OPEN, TOOLS_CLOSE = "<tools>", "</tools>"
CALL_OPEN, CALL_CLOSE = "<tool_call>", "</tool_call>"
RESPONSE_OPEN, RESPONSE_CLOSE = "<tool_response>", "</tool_response>"

# Nous's own agentic system prompt says `<tool_results>` (plural) in two places while the
# code only ever emits `<tool_response>`. The plural tag is never produced; recorded here
# so a reader who meets it in upstream material knows it is a typo and not a variant.
_NEVER_EMITTED = "<tool_results>"

# Tag-shaped but not the tag: wrong case, stray whitespace, attributes. Found only
# after real blocks are removed, so a near-miss is reported rather than read as silence.
_LOOSE_CALL_RE = re.compile(r"<\s*/?\s*tool_call[^>\n]*>", re.IGNORECASE)

# A `<` that opens or closes one of the protocol's own tags. Only these are escaped inside
# payloads; ordinary less-than signs are left alone.
_TAG_START_RE = re.compile(
    r"<(?=/?\s*(?:tools|tool_call|tool_response|tool_results|scratch_pad|think)\b)", re.IGNORECASE
)

# Whichever tag the dialect reasons in. The backreference stops `<think>...</scratch_pad>`
# from matching across two different blocks.
_REASONING_RE = re.compile(r"<(scratch_pad|think)>(.*?)</\1>", re.DOTALL)


class ProtocolError(ValueError):
    """A message cannot be rendered into, or read out of, the Hermes wire format."""


@dataclass(frozen=True)
class Dialect:
    """The parts of the protocol that changed between Hermes generations.

    Kept as data rather than branches so the differences are enumerable: a reader can see
    what actually varies instead of discovering it in a conditional.
    """

    name: str
    tool_result_role: str
    reasoning_tag: str
    supports_scratch_pad: bool
    # Hermes 2 Pro and 3 restate the FunctionCall pydantic schema in the system prompt.
    # Hermes 4 dropped it.
    pydantic_line: bool
    # The prompts themselves differ, not just their trimmings: Hermes 4 opens with an
    # identity line and a `# Tools` heading that Hermes 3 has no equivalent of. Sharing one
    # preamble between them would train a Hermes 4 worker on a system turn it never saw.
    preamble: str = ""
    call_instruction: str = ""

    def __post_init__(self) -> None:
        if self.tool_result_role not in ("tool", "user"):
            raise ProtocolError(f"{self.name}: unknown tool-result role {self.tool_result_role!r}")


_H3_PREAMBLE = (
    "You are a function calling AI model. You are provided with function signatures "
    f"within {TOOLS_OPEN} {TOOLS_CLOSE} XML tags. You may call one or more functions to "
    "assist with the user query. If available tools are not relevant in assisting with "
    "user query, just respond in natural conversational language. Don't make assumptions "
    "about what values to plug into functions. After calling & executing the functions, "
    f"you will be provided with function results within {RESPONSE_OPEN} {RESPONSE_CLOSE} XML tags."
)

_H4_PREAMBLE = (
    "You are Hermes, created by Nous Research.\n\n# Tools\n\n"
    "You are a function calling AI model. You may call one or more functions to assist "
    "with the user query.\n\n"
    f"You are provided with function signatures within {TOOLS_OPEN}{TOOLS_CLOSE} XML tags:"
)

_CALL_INSTRUCTION = (
    "For each function call return a json object with function name and arguments within "
    f"{CALL_OPEN} {CALL_CLOSE} XML tags as follows:\n{CALL_OPEN}\n"
    '{"name": <function-name>, "arguments": <args-dict>}\n' + CALL_CLOSE
)

_H4_CALL_INSTRUCTION = (
    "For each function call, return a json object with function name and arguments within "
    f"{CALL_OPEN}{CALL_CLOSE} XML tags:\n{CALL_OPEN}\n"
    '{"name": "<function-name>", "arguments": <args-json-object>}\n' + CALL_CLOSE
)

# Hermes 2 Pro and Hermes 3 ship byte-identical `tool_use` templates.
HERMES_3 = Dialect(
    name="hermes-3",
    tool_result_role="tool",
    reasoning_tag="scratch_pad",
    supports_scratch_pad=True,
    pydantic_line=True,
    preamble=_H3_PREAMBLE,
    call_instruction=_CALL_INSTRUCTION,
)

# Hermes 4 returns tool results as `user`, regardless of base model, and reasons in
# <think>. `<scratch_pad>` is not even a token any more, and the system prompt is a
# different prompt rather than the same one with pieces removed.
HERMES_4 = Dialect(
    name="hermes-4",
    tool_result_role="user",
    reasoning_tag="think",
    supports_scratch_pad=False,
    pydantic_line=False,
    preamble=_H4_PREAMBLE,
    call_instruction=_H4_CALL_INSTRUCTION,
)

DIALECTS = {d.name: d for d in (HERMES_3, HERMES_4)}

_PYDANTIC_SCHEMA = (
    '{"title": "FunctionCall", "type": "object", "properties": {"name": {"title": "Name", '
    '"type": "string"}, "arguments": {"title": "Arguments", "type": "object"}}, '
    '"required": ["name", "arguments"]}'
)

_PREAMBLE = (
    "You are a function calling AI model. You are provided with function signatures "
    f"within {TOOLS_OPEN} {TOOLS_CLOSE} XML tags. You may call one or more functions to "
    "assist with the user query. If available tools are not relevant in assisting with "
    "user query, just respond in natural conversational language. Don't make assumptions "
    "about what values to plug into functions. After calling & executing the functions, "
    f"you will be provided with function results within {RESPONSE_OPEN} {RESPONSE_CLOSE} XML tags."
)

# The GOAP scratchpad, from the Hermes-3 tool-use template. `Actions` is deliberately
# python-call syntax rather than JSON: it is a plan, and writing it in the same shape as a
# real <tool_call> would blur the line between intending to act and acting.
_GOAP_INSTRUCTION = (
    f"Each function call should be enclosed within {CALL_OPEN} {CALL_CLOSE} XML tags. "
    "You must use <scratch_pad> </scratch_pad> XML tags to record your reasoning and "
    "planning before you call the functions as follows.\nExample:\n<scratch_pad>\n"
    "Goal: <state task assigned by user>\nActions:\n"
    "- {result_var_name1} = functions.{function_name1}({param1}={value1},...)\n"
    "Observation: <set observation 'None' with tool calls; plan final tools results "
    "summary when provided>\nReflection: <evaluate query-tool relevance and required "
    "parameters when tools called; analyze overall task status when observations made>\n"
    "</scratch_pad>"
)


def tool_schema(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """One tool in the envelope the templates use.

    Nous material also contains the flat `{"name", "description", "parameters"}` shape in
    few-shot assets, but the chat template emits this one, so it is the authoritative
    form: what the model actually saw during training.
    """
    if not name.strip():
        raise ProtocolError("a tool needs a name")
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


def render_system(
    tools: list[dict[str, Any]],
    *,
    dialect: Dialect,
    scratch_pad: bool = False,
    extra: str = "",
) -> str:
    """The system turn advertising the tool set.

    An empty tool list is refused. A system prompt that describes how to call functions
    and then advertises none teaches the model to invent names, which is the failure
    `format.py` already guards on the other side.
    """
    if not tools:
        raise ProtocolError(
            "no tools to advertise; a prompt that explains the call format but offers "
            "nothing to call teaches the model to invent tool names"
        )
    if scratch_pad and not dialect.supports_scratch_pad:
        raise ProtocolError(
            f"{dialect.name} has no <scratch_pad> token -- it reasons in "
            f"<{dialect.reasoning_tag}>; asking for a scratchpad here would train a tag "
            "the tokenizer splits into pieces"
        )

    lines = [dialect.preamble, TOOLS_OPEN]
    # Through the same serializer as the assistant side: a tool description quoting
    # `</tools>` would otherwise close the block early and hide every tool after it, which
    # is the invented-tool failure arriving through the advertisement instead of the call.
    lines.extend(_payload(t, str(t.get("function", {}).get("name", i))) for i, t in enumerate(tools))
    lines.append(TOOLS_CLOSE)
    if dialect.pydantic_line:
        lead = (
            "For each function call return a JSON object, with the following pydantic model json schema:"
            if scratch_pad
            else "Use the following pydantic model json schema for each tool call you will make:"
        )
        lines.append(f"{lead}\n{_PYDANTIC_SCHEMA}")
    lines.append(_GOAP_INSTRUCTION if scratch_pad else dialect.call_instruction)
    if extra.strip():
        lines.append(extra.strip())
    return "\n".join(lines)


def _payload(obj: dict[str, Any], what: str) -> str:
    """Serialize, refusing what a downstream JSON reader could not take back.

    A `<` that *begins a protocol tag* is escaped to `\\u003c`. It is ordinary JSON
    escaping, fully reversible by any decoder, and it is the only way an argument may
    contain the literal text `</tool_call>` without truncating every reader at the injected
    tag -- an agent writing code about the protocol is not exotic on this repo.

    Only tag-initial `<` is touched. Escaping every one would put `\\u003c` through most of
    the comparison operators and C++ templates in a kernel corpus: a large, permanent
    distortion of the training text to defend against a narrow case.
    """
    try:
        text = json.dumps(obj, separators=(", ", ": "), allow_nan=False)
    except ValueError as exc:
        # NaN and Infinity are accepted by Python's json and rejected by every other
        # runtime that will read this corpus, so they must not reach the wire.
        raise ProtocolError(f"{what}: not serializable as JSON: {exc}") from exc
    return _TAG_START_RE.sub(lambda _: "\\u003c", text)


def render_tool_call(name: str, arguments: dict[str, Any]) -> str:
    """One `<tool_call>` block, `name` first as the chat templates emit it."""
    if not name.strip():
        raise ProtocolError("a tool call needs a name")
    if not isinstance(arguments, dict):
        # The renderer must not be able to emit something the parser calls malformed.
        raise ProtocolError(f"{name}: arguments must be an object, got {type(arguments).__name__}")
    return f"{CALL_OPEN}\n{_payload({'name': name, 'arguments': arguments}, name)}\n{CALL_CLOSE}"


def render_tool_calls(calls: list[tuple[str, dict[str, Any]]]) -> str:
    """Several calls in one assistant turn.

    Separate blocks, not a JSON array in one block: the protocol has no array form, and a
    model taught one would emit something no Hermes parser reads.
    """
    if not calls:
        raise ProtocolError("no calls to render; an abstention is empty assistant text, not an empty block")
    return "\n".join(render_tool_call(name, args) for name, args in calls)


def render_tool_response(name: str, content: Any) -> str:
    """One observation coming back.

    The `{"name", "content"}` envelope is convention rather than template-enforced -- the
    template emits message content raw -- but it is what the reference implementation and
    every model card show, and it is the only thing carrying the tool's identity back to a
    model that has no call ids to match on.
    """
    if not name.strip():
        raise ProtocolError("a tool response needs the name of the tool that produced it")
    return f"{RESPONSE_OPEN}\n{_payload({'name': name, 'content': content}, name)}\n{RESPONSE_CLOSE}"


def render_scratch_pad(goal: str, actions: list[str], observation: str = "None", reflection: str = "") -> str:
    """The GOAP block: Goal, Actions, Observation, Reflection.

    `observation` defaults to the literal `"None"` because that is what the template's own
    example uses on a turn that is issuing calls rather than reading them -- the agent has
    not observed anything yet, and writing an observation there would be a prediction
    dressed as evidence.
    """
    if not goal.strip():
        raise ProtocolError("a scratchpad needs a goal")
    if isinstance(actions, str):
        # A bare string is iterable, so this would otherwise render one bullet per
        # character and look like a very confused plan.
        raise ProtocolError("actions must be a list of strings, not a single string")

    for label, text in (
        ("goal", goal),
        ("observation", observation),
        ("reflection", reflection),
        *(("action", a) for a in actions),
    ):
        # The pad is prose, not JSON, so there is nothing here to escape into -- a value
        # quoting `</scratch_pad>` closes the block from inside and everything after it
        # becomes assistant text. An argument carrying a `<tool_call>` then reparses as a
        # call the trajectory never made, for a tool that may not even be in `<tools>`:
        # the invented-tool failure, manufactured by the renderer. The assistant side is
        # already protected by `_payload`; this is the same invariant, same threat.
        if "</scratch_pad>" in text or CALL_OPEN in text:
            raise ProtocolError(
                f"{label} quotes a protocol tag; the GOAP block is plain text and cannot "
                f"contain </scratch_pad> or {CALL_OPEN}"
            )
        # Line-oriented, so an embedded newline silently produces an unprefixed line that
        # reads as a different GOAP field.
        if "\n" in text:
            raise ProtocolError(f"{label} {text[:30]!r} contains a newline; the scratchpad is line-oriented")

    body = [f"Goal: {goal.strip()}", "Actions:"]
    body.extend(f"- {a.strip()}" for a in actions) if actions else body.append("None")
    body.append(f"Observation: {observation.strip() or 'None'}")
    # Always emitted. The template's example lists all four fields, and a pad that silently
    # omits Reflection teaches the model that the field is optional -- which is how the
    # "did that work?" step stops appearing.
    body.append(f"Reflection: {reflection.strip() or 'None'}")
    return "<scratch_pad>\n" + "\n".join(body) + "\n</scratch_pad>"


@dataclass(frozen=True)
class ParsedCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ParsedTurn:
    """What an assistant turn actually contained.

    `abstained` and `malformed` are separate on purpose, and the separation is the whole
    point of the type. A turn whose `<tool_call>` block holds unparseable JSON, or is cut
    off mid-call by a token limit, or spells the tag some other way, contains no usable
    calls -- and every one of those resolves to "no calls found" unless something looks for
    it. Folding them into `abstained` would score the protocol's hardest failures as its
    most disciplined behaviour, and a worker whose every generation truncates would post a
    perfect abstention rate.

    An abstention also requires *something said*. Hermes's rule is to answer in natural
    language when no tool fits; an empty completion is a crashed worker, not a decision.
    """

    calls: tuple[ParsedCall, ...]
    text: str
    scratch_pad: str = ""
    malformed: tuple[str, ...] = ()

    @property
    def abstained(self) -> bool:
        return not self.calls and not self.malformed and bool(self.text.strip())

    @property
    def well_formed(self) -> bool:
        return not self.malformed

    @property
    def safe_calls(self) -> tuple[ParsedCall, ...]:
        """Calls only when nothing in the turn was malformed.

        A truncated call can leave a *syntactically clean* call behind it -- the tail of a
        broken block, or a second block smuggled inside a string that the scanner resumed
        past. Anything that executes what it parses should read this rather than `calls`,
        so a turn that confused the parser executes nothing at all.
        """
        return () if self.malformed else self.calls


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            # `json.loads` takes the last one silently, so `{"name": "a", "name": "b"}`
            # would call `b` with no indication that `a` was ever requested.
            raise ProtocolError(f"duplicate key {key!r}")
        seen[key] = value
    return seen


_DECODER = json.JSONDecoder(object_pairs_hook=_no_duplicate_keys)


def _check_call(payload: Any) -> tuple[ParsedCall | None, str]:
    """Validate one decoded call. Returns (call, reason); exactly one is set."""
    if not isinstance(payload, dict):
        return None, f"call is {type(payload).__name__}, expected an object"
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        return None, "call has no name"
    # `parameters` is the schema's word for a tool's signature, never a call's arguments.
    # Accepting it would let a model drift onto a key no Hermes runtime reads and never be
    # told -- including when it supplies both and the real arguments are the discarded ones.
    if "parameters" in payload:
        return None, f"{name}: used 'parameters'; a call's arguments key is 'arguments'"
    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        return None, f"{name}: arguments is {type(arguments).__name__}, expected an object"
    return ParsedCall(name=name, arguments=arguments), ""


def parse_turn(content: str) -> ParsedTurn:
    """Read tool calls out of assistant text.

    Calls are found by decoding JSON from just after each `<tool_call>`, not by matching
    to the next `</tool_call>`. The difference matters because an argument may legitimately
    *contain* the closing tag -- an agent writing code about the protocol, on this repo,
    routinely would -- and a regex stopping at the first inner match truncates a correct
    call into a malformed one, or worse, resumes inside a string and finds a call the model
    never made.

    Calls are scanned *before* reasoning blocks are located, and the decoded JSON spans are
    masked out before the reasoning tags are matched. Doing it the other way round lets an
    argument containing `</scratch_pad>` close a block it was never inside, which silently
    moves the boundary between what the agent planned and what it did.

    A call that lands inside a reasoning block is then dropped. Both tags are recognised,
    not just Hermes 3's: `<scratch_pad>` and `<think>` are the same field in different
    generations, and excising only the older one meant a Hermes 4 worker that deliberated
    about a call and decided against it was graded as having made it.

    Either key order is accepted -- the templates emit `name` first and the model cards
    show `arguments` first, so both are in the training distribution and a parser insisting
    on one would reject half of Nous's own examples.
    """
    calls: list[ParsedCall] = []
    malformed: list[str] = []
    spans: list[tuple[int, int]] = []
    json_spans: list[tuple[int, int]] = []
    pos = 0

    while (start := content.find(CALL_OPEN, pos)) != -1:
        cursor = start + len(CALL_OPEN)
        while cursor < len(content) and content[cursor].isspace():
            cursor += 1
        try:
            payload, decoded_to = _DECODER.raw_decode(content, cursor)
        except (json.JSONDecodeError, ProtocolError) as exc:
            reason = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
            malformed.append(f"invalid JSON: {reason}")
            close = content.find(CALL_CLOSE, cursor)
            stop = close + len(CALL_CLOSE) if close != -1 else len(content)
            spans.append((start, stop))
            pos = stop
            continue

        json_spans.append((cursor, decoded_to))
        close = content.find(CALL_CLOSE, decoded_to)
        if close == -1:
            # A generation that hit its token limit mid-call. Without this the block leaves
            # no trace at all and the turn reads as a clean decline.
            malformed.append("unclosed <tool_call>: no </tool_call>")
            spans.append((start, len(content)))
            break
        if content[decoded_to:close].strip():
            malformed.append(f"trailing content before </tool_call>: {content[decoded_to:close].strip()[:40]!r}")
            spans.append((start, close + len(CALL_CLOSE)))
        else:
            call, reason = _check_call(payload)
            spans.append((start, close + len(CALL_CLOSE)))
            if call is not None:
                calls.append((call, len(spans) - 1))  # type: ignore[arg-type]
            else:
                malformed.append(reason)
        pos = close + len(CALL_CLOSE)

    # Reasoning tags are matched against a copy with the decoded JSON blanked, so a pad tag
    # quoted inside an argument can neither open nor close a block.
    masked = list(content)
    for lo, hi in json_spans:
        masked[lo:hi] = " " * (hi - lo)
    pads: list[str] = []
    pad_spans: list[tuple[int, int]] = []
    for match in _REASONING_RE.finditer("".join(masked)):
        pads.append(content[match.start(2) : match.end(2)].strip())
        pad_spans.append((match.start(), match.end()))

    def inside_pad(span: tuple[int, int]) -> bool:
        return any(lo <= span[0] and span[1] <= hi for lo, hi in pad_spans)

    # A call written in a reasoning block is an intention, not an action.
    kept = tuple(call for call, index in calls if not inside_pad(spans[index]))  # type: ignore[misc]

    remainder = _strip_spans(content, sorted(spans + pad_spans))
    # Anything still tag-shaped after the real blocks are gone -- `<tool_call >`,
    # `<TOOL_CALL>`, `<tool_call id="1">`. The parser cannot execute it, and calling it an
    # abstention would resolve yet another ambiguity in the worker's favour.
    for stray in _LOOSE_CALL_RE.findall(remainder):
        malformed.append(f"unrecognised tag {stray.strip()!r}; the tag is exactly {CALL_OPEN}")

    return ParsedTurn(
        calls=kept,
        text=_LOOSE_CALL_RE.sub("", remainder).strip(),
        scratch_pad="\n".join(pads),
        malformed=tuple(malformed),
    )


def _strip_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Remove the given spans. Overlaps are tolerated: a call span sits inside a pad span
    whenever the model planned in the pad, and dropping the text twice would corrupt it."""
    kept: list[str] = []
    cursor = 0
    for start, stop in spans:
        if start >= cursor:
            kept.append(text[cursor:start])
        cursor = max(cursor, stop)
    kept.append(text[cursor:])
    return "".join(kept)


def pair_responses(calls: tuple[ParsedCall, ...], responses: list[Any]) -> list[tuple[ParsedCall, Any]]:
    """Match observations to the calls that produced them, positionally.

    The protocol carries no ids, so order is the only correlation available. A mismatch in
    count is refused rather than zipped short: silently dropping the tail would attribute
    one tool's output to another call, which is worse than having no observation at all.
    """
    if len(calls) != len(responses):
        raise ProtocolError(
            f"{len(calls)} tool calls but {len(responses)} responses; the protocol has no "
            "call ids, so correlation is positional and a mismatch cannot be resolved"
        )
    return list(zip(calls, responses, strict=True))


@dataclass
class Conversation:
    """Message list in the shape a Hermes chat template expects.

    Tool results are appended with role `tool`; the dialect decides what the template
    renders that as. Hermes 4 rewrites it to `user`, which is recorded here rather than
    left for a reader to discover from rendered output.
    """

    dialect: Dialect
    messages: list[dict[str, str]] = field(default_factory=list)
    # Hermes 4 puts results in the `user` role, so "previous message is a user turn" is not
    # enough to tell a run of observations from the user's own turn before it.
    _last_was_result: bool = False

    def system(self, content: str) -> Conversation:
        self.messages.append({"role": "system", "content": content})
        self._last_was_result = False
        return self

    def user(self, content: str) -> Conversation:
        self.messages.append({"role": "user", "content": content})
        self._last_was_result = False
        return self

    def assistant(self, content: str) -> Conversation:
        self.messages.append({"role": "assistant", "content": content})
        self._last_was_result = False
        return self

    def tool_result(self, name: str, content: Any) -> Conversation:
        """Append an observation, joining a run of them into one turn.

        Both Hermes templates open a single turn for a whole run of consecutive results and
        put each `<tool_response>` block inside it. One message per observation inserts a
        turn boundary between calls that were issued together, contradicting the commitment
        the assistant side already makes -- one turn per *batch* of calls -- and teaching
        the model that parallel calls come back one conversational turn at a time.
        """
        rendered = render_tool_response(name, content)
        role = self.dialect.tool_result_role
        if self._last_was_result and self.messages and self.messages[-1]["role"] == role:
            self.messages[-1]["content"] += "\n" + rendered
        else:
            self.messages.append({"role": role, "content": rendered})
        self._last_was_result = True
        return self

    def to_record(self) -> dict[str, Any]:
        return {"dialect": self.dialect.name, "messages": list(self.messages)}
