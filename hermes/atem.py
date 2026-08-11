"""The ATEM wire format, as Muse-Glimmer-30B natively speaks it.

`hermes.protocol` covers what changed *between Hermes generations* -- a result role, a reasoning
tag, whether the pydantic line is restated -- and holds it as data because those differences are
enumerable. ATEM is not a Hermes generation. It is a different markup family, and the parts that
differ are the parts `Dialect` cannot express as fields:

    Hermes 4        <tool_call>{"name": "terminal", "arguments": {"command": "ls"}}</tool_call>
    ATEM            <atem:function_calls>
                      <atem:invoke name="terminal">
                        <atem:parameter name="command">ls</atem:parameter>
                      </atem:invoke>
                    </atem:function_calls>

One JSON object per call becomes one element per *parameter*, and a parameter's value is text
rather than JSON. So there is no arguments object to decode, no place for a JSON syntax error to
occur, and correspondingly a whole class of malformed turn that cannot happen here -- and a new
one that can, which the rest of this module is mostly about.

Everything returns `hermes.protocol.ParsedTurn` and `ParsedCall`. Same types, so
`hermesbench.policy` and `hermesbench.toolchoice` need a dispatch and not a rewrite, and
`malformed_turns` keeps meaning the same thing across both formats. A second `ParsedTurn`-alike
would drift from the first, and every metric computed from the two would silently stop comparing.

## Values are text, so types have to be recovered

The template this is read from writes booleans as bare `true`/`false`, `None` as `null`, and
anything list- or dict-shaped through `tojson`. Nothing marks which of those happened, so a
parameter reading `true` is indistinguishable from the *string* "true" on the wire. That is a
property of the format, not a defect here, and it cannot be fixed by parsing harder.

What it can be handled by is the tool schema, which the caller has: `coerce` takes the declared
parameter types and applies them. Without a schema the value stays a string, because guessing is
how `{"path": "123"}` becomes `{"path": 123}` and a tool receives an integer where it declared a
filename. Recovering less is the safe direction.

## The reasoning channel is not an inline tag

Hermes reasons inside `<think>` in the same completion. ATEM emits reasoning as a separate turn
addressed to `self` -- `<|start|>assistant to=self<|message|>...<|eom|>` -- which the serving
layer surfaces as `reasoning_content` rather than in the text. So a turn's reasoning may arrive
beside the content instead of inside it, and `parse_turn` takes it as an argument. Reading it out
of the text would find nothing and report every turn as having skipped deliberation.
"""

from __future__ import annotations

import json
import re
from typing import Any

from hermes.protocol import ParsedCall, ParsedTurn, ProtocolError

DIALECT_NAME = "atem"

CALLS_OPEN = "<atem:function_calls>"
CALLS_CLOSE = "</atem:function_calls>"
OUTPUT_OPEN = "<tool_output"
OUTPUT_CLOSE = "</tool_output>"

# Reasoning arrives on its own channel, so unlike Hermes there is no tag to look for.
REASONING_CHANNEL = "self"

_BLOCK_RE = re.compile(re.escape(CALLS_OPEN) + r"(.*?)" + re.escape(CALLS_CLOSE), re.DOTALL)
_INVOKE_RE = re.compile(r'<atem:invoke\s+name="([^"]*)"\s*>(.*?)</atem:invoke>', re.DOTALL)
_PARAM_RE = re.compile(r'<atem:parameter\s+name="([^"]*)"\s*>(.*?)</atem:parameter>', re.DOTALL)

# Tag-shaped and not the tag: a namespace typo, a missing close, an unquoted name. Looked for only
# after the real blocks are removed, so a near-miss is reported rather than read as silence -- the
# same reason `hermes.protocol` scans for loose `<tool_call>` tags.
_LOOSE_RE = re.compile(
    r"<\s*/?\s*atem\s*:\s*(?:function_calls|invoke|parameter)\b[^>]*>?|<\s*/?\s*function_calls\b[^>]*>?",
    re.IGNORECASE,
)

# An opened block with no close. Truncation by token limit lands here, and it is the failure this
# format makes most likely: an ATEM call is several times longer than the equivalent JSON one, so
# there is more of it to cut off.
_UNCLOSED_RE = re.compile(re.escape(CALLS_OPEN) + r"(?!.*" + re.escape(CALLS_CLOSE) + r")", re.DOTALL)


def render_tool_call(name: str, arguments: dict[str, Any]) -> str:
    """One call, in the format the model's own template writes.

    Parameter order follows the mapping's order rather than being sorted. The template does the
    same, and a renderer that sorted would produce training rows the model never saw itself
    generate -- teaching it a normalisation nothing else in the stack applies.
    """
    if not name:
        raise ProtocolError("a tool call needs a name")
    parts = [CALLS_OPEN, f'\n<atem:invoke name="{_attr(name)}">\n']
    for key, value in arguments.items():
        parts.append(f'<atem:parameter name="{_attr(key)}">{_value(value)}</atem:parameter>\n')
    parts.append("</atem:invoke>\n")
    parts.append(CALLS_CLOSE)
    return "".join(parts)


def render_tool_calls(calls: list[tuple[str, dict[str, Any]]]) -> str:
    """Several calls. One block per call, matching the template.

    The template emits a separate `<|start|>assistant to=<name>` turn per call and therefore a
    separate block. Nesting several `<atem:invoke>` in one block would parse here and is not what
    the model produces, so training on it would teach a shape the serving layer never sends.
    """
    return "\n".join(render_tool_call(name, arguments) for name, arguments in calls)


def render_tool_response(name: str, content: Any) -> str:
    if not name:
        raise ProtocolError("a tool response needs the name of the tool that produced it")
    body = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return f'{OUTPUT_OPEN} name="{_attr(name)}">\n{body}\n{OUTPUT_CLOSE}'


def _attr(text: str) -> str:
    """Escape a value going into a `name="..."` attribute.

    A tool or parameter named with a quote would otherwise close the attribute early and the
    remainder would parse as more attributes -- so a name is a place where a crafted string
    changes the *structure* of the call, not just its content.
    """
    return str(text).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


def _value(value: Any) -> str:
    """A parameter value, the way the template writes it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        # Not escaped beyond the closing tag. The format is explicitly "not expected to be valid
        # XML and is parsed with regular expressions", so escaping everything would produce text
        # the model does not write and a tool would receive entities instead of characters.
        return value.replace("</atem:parameter>", "&lt;/atem:parameter>")
    if isinstance(value, (int, float)):
        return repr(value) if isinstance(value, float) else str(value)
    return json.dumps(value, ensure_ascii=False)


def coerce(arguments: dict[str, str], schema: dict[str, Any] | None) -> dict[str, Any]:
    """Recover declared types from text values.

    Everything arrives as a string because the format has no types. The tool's own JSON Schema is
    the only thing that says what a parameter should be, so it is the only thing consulted.

    Anything not covered by the schema stays a string. Sniffing -- "it looks like a number, make
    it one" -- turns `{"path": "123"}` into `{"path": 123}` and hands a tool an integer where it
    declared a filename. Under-recovering is the direction that fails loudly at the tool boundary
    rather than quietly inside it.
    """
    properties = ((schema or {}).get("properties") or {}) if isinstance(schema, dict) else {}
    out: dict[str, Any] = {}
    for key, raw in arguments.items():
        declared = properties.get(key) if isinstance(properties, dict) else None
        kind = (declared or {}).get("type") if isinstance(declared, dict) else None
        out[key] = _cast(raw, kind)
    return out


def _cast(raw: str, kind: str | None) -> Any:
    if kind in (None, "string"):
        return raw
    text = raw.strip()
    try:
        if kind == "boolean":
            if text in ("true", "false"):
                return text == "true"
            return raw
        if kind == "integer":
            return int(text)
        if kind == "number":
            return float(text)
        if kind in ("object", "array"):
            loaded = json.loads(text)
            # The declared type still has to hold: a schema saying `array` and a payload holding
            # an object is a disagreement, and passing it through would move the failure into the
            # tool where the reason is no longer visible.
            if kind == "array" and not isinstance(loaded, list):
                return raw
            if kind == "object" and not isinstance(loaded, dict):
                return raw
            return loaded
        if kind == "null":
            return None if text == "null" else raw
    except (TypeError, ValueError):
        # Unparseable against its declared type. Returned as the text it was, so the tool refuses
        # it with the real value in the message instead of receiving a silently coerced one.
        return raw
    return raw


def parse_turn(content: str, *, reasoning: str = "", schemas: dict[str, dict[str, Any]] | None = None) -> ParsedTurn:
    """Read one assistant turn.

    `reasoning` is passed in rather than found: ATEM puts deliberation on a separate channel, so
    searching the text for it would find nothing and report every turn as having skipped it.

    `schemas` maps tool name to its JSON Schema and is what `coerce` needs. Without it every
    argument stays a string, which is a usable degradation -- a tool that declares
    `{"command": "string"}` needs no recovery at all, and most do.
    """
    malformed: list[str] = []
    calls: list[ParsedCall] = []
    spans: list[tuple[int, int]] = []

    for block in _BLOCK_RE.finditer(content):
        spans.append(block.span())
        body = block.group(1)
        invokes = list(_INVOKE_RE.finditer(body))
        if not invokes:
            # A block with no invoke is not an abstention and not a call. Left unreported it would
            # resolve to "no calls found", which is the reading `ParsedTurn` exists to prevent.
            malformed.append(
                f"{CALLS_OPEN} block contains no <atem:invoke>; the model opened a call block and "
                "wrote nothing callable in it"
            )
            continue
        for invoke in invokes:
            name = invoke.group(1).strip()
            if not name:
                malformed.append("<atem:invoke> has an empty name attribute")
                continue
            raw = {p.group(1).strip(): p.group(2) for p in _PARAM_RE.finditer(invoke.group(2))}
            leftover = _PARAM_RE.sub("", invoke.group(2)).strip()
            if leftover and "<atem:parameter" in leftover:
                # An unterminated parameter: the value swallowed the rest of the invoke. Reported,
                # because the parameters that DID parse look like a complete call.
                malformed.append(f"{name}: an <atem:parameter> is not closed, so its value ran to the end of the call")
                continue
            calls.append(ParsedCall(name=name, arguments=coerce(raw, (schemas or {}).get(name))))

    remainder = _strip(content, spans)

    if _UNCLOSED_RE.search(remainder):
        malformed.append(
            f"{CALLS_OPEN} was opened and never closed, so the call is truncated. An ATEM call is "
            "several times longer than the JSON equivalent, which makes this the likeliest way a "
            "turn breaks under a token limit."
        )
    for loose in _LOOSE_RE.finditer(remainder):
        malformed.append(
            f"{loose.group(0)[:40]!r} is call-shaped but not a call. The namespace prefix is part of "
            "the tag: `<function_calls>` and `<atem :invoke>` are both unparseable."
        )

    text = _LOOSE_RE.sub("", _UNCLOSED_RE.sub("", remainder)).strip()
    return ParsedTurn(calls=tuple(calls), text=text, scratch_pad=reasoning.strip(), malformed=tuple(malformed))


def _strip(text: str, spans: list[tuple[int, int]]) -> str:
    out, last = [], 0
    for start, end in spans:
        out.append(text[last:start])
        last = end
    out.append(text[last:])
    return "".join(out)


__all__ = [
    "CALLS_CLOSE",
    "CALLS_OPEN",
    "DIALECT_NAME",
    "OUTPUT_CLOSE",
    "OUTPUT_OPEN",
    "REASONING_CHANNEL",
    "coerce",
    "parse_turn",
    "render_tool_call",
    "render_tool_calls",
    "render_tool_response",
]
