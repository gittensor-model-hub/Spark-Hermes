"""A served model as an `AgentPolicy`: the one interface everything was waiting on.

`hermesbench.runner` has always been able to run an episode. It has never had anything to
run: the only `AgentPolicy` was `ReplayPolicy`, which replays a trajectory somebody
already recorded, so `main()` refuses to run a suite and says so rather than reporting
`success_rate 0.0` as a measurement. Thirteen modules in this repo are reachable only from
tests for that one reason -- not because they are badly wired, but because the thing that
would call them had no model to call.

This is that thing. It speaks to any OpenAI-compatible chat endpoint, which covers SGLang -- what
this project serves on, and the engine every measurement here was taken through -- along with vLLM
and every hosted gateway the teachers already use.

Three properties it does not compromise on:

**It never authors an observation.** The policy returns tool *calls*; the runner executes
them and appends what actually happened. A model that emitted a `tool_result` alongside its
call would be describing a tool run rather than running one, so those steps are dropped
before they reach the runner -- which drops them again, because the invariant is worth
enforcing at both ends.

**Malformed output is a visible failure, not silence.** A truncated `<tool_call>`, a
misspelled tag or unparseable JSON all produce *no calls*, and "no calls" is
indistinguishable from "the model decided to stop" unless something looks. The turn is
recorded as a thinking step carrying the parse errors, so the episode ends on the record
rather than on a quiet `final`, and `hermesbench.metrics` counts it as the failure it is.

**Tokens are counted from the provider, not estimated.** `mean_tokens` has been a column of
zeros because `ReplayPolicy.tokens` defaults to 0 and nothing else implements the property.
Usage comes back from the endpoint on every call and is normalised through `hermes.cost`,
so the same numbers that price a run are the ones the metrics report.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hermes.cost import OPENAI, Usage, usage_from_provider
from hermes.protocol import (
    Dialect,
    ParsedCall,
    ParsedTurn,
    parse_turn,
    render_system,
    render_tool_response,
    tool_schema,
)
from hermes.trajectory import FINAL, THINKING, TOOL_CALL, TOOL_RESULT, Step
from hermesbench.tasks import Task

# A chat completion: messages in, (text, raw usage dict) out. Deliberately this small --
# anything that can be adapted to it can drive the benchmark, including a local process,
# a hosted gateway, or a stub in a test.
# `tools` is optional and only supplied for dialects whose definitions belong to the serving
# template rather than the system prompt -- see `Dialect.tools_in_prompt`. Kept as a keyword with
# a default so every existing caller and test stub is unaffected: a stub written for Hermes is
# never handed the argument.
Completion = Callable[..., tuple[str, dict[str, Any]]]


class PolicyError(RuntimeError):
    """The policy cannot produce a usable turn."""


@dataclass
class ServedModelPolicy:
    """Drives an episode from a served chat endpoint, in the Hermes wire format.

    `tool_schemas` is required rather than derived from `task.tools`. Hermes advertises
    full signatures inside `<tools>`, and synthesising them from bare names would show the
    model parameters that do not exist -- the same refusal `hermes.format.to_hermes_record`
    makes on the training side, for the same reason. A benchmark that advertises invented
    signatures is measuring how well a model copes with a broken prompt.
    """

    complete: Completion
    dialect: Dialect
    tool_schemas: dict[str, dict[str, Any]]
    system: str = ""
    scratch_pad: bool = False
    _tokens: int = field(default=0, init=False)
    usage: Usage = field(default_factory=Usage, init=False)
    parse_failures: int = field(default=0, init=False)

    @property
    def tokens_used(self) -> int:
        return self._tokens

    def _system_turn(self, task: Task) -> str:
        missing = [t for t in task.tools if t not in self.tool_schemas]
        if missing:
            raise PolicyError(
                f"{task.task_id}: no schema for advertised tools {sorted(missing)}; showing the model "
                "a bare name would either hide the signature or invent one, and both measure the prompt"
            )
        tools = [
            tool_schema(
                name, self.tool_schemas[name].get("description", ""), self.tool_schemas[name].get("parameters", {})
            )
            for name in task.tools
        ]
        if not self.dialect.tools_in_prompt:
            # The definitions go with the request instead, so the serving template renders them,
            # the valid-recipient list and the reasoning line as one consistent block. What is
            # left for the system turn is the operator's own framing, which is what `system` is.
            return self.system.strip()
        return render_system(tools, dialect=self.dialect, scratch_pad=self.scratch_pad, extra=self.system)

    def _tool_payload(self, task: Task) -> list[dict[str, Any]]:
        """The tool definitions, in the shape an OpenAI-compatible request takes."""
        return [
            tool_schema(
                name, self.tool_schemas[name].get("description", ""), self.tool_schemas[name].get("parameters", {})
            )
            for name in task.tools
        ]

    def _messages(self, task: Task, history: list[Step]) -> list[dict[str, str]]:
        """Rebuild the conversation from the trajectory so far.

        Rebuilt each turn rather than accumulated, so the messages sent always match the
        trajectory that will be scored. An independently-maintained message list can drift
        from the recorded history, and then the episode that was graded is not the one the
        model saw.
        """
        system = self._system_turn(task)
        # Omitted when empty rather than sent blank. The ATEM template injects its own default
        # system message when none is present -- knowledge cutoff, reasoning strength, tool
        # definitions, valid recipients -- and an empty system message would suppress that and
        # leave the model with no tool definitions at all.
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": task.prompt})
        call_names: dict[str, str] = {}
        pending: list[str] = []
        for step in history:
            if step.kind == THINKING:
                pending.append(step.content)
            elif step.kind == TOOL_CALL:
                if step.call_id:
                    call_names[step.call_id] = step.tool or ""
                pending.append(_render_call(step, self.dialect))
            elif step.kind == TOOL_RESULT:
                if pending:
                    messages.append({"role": "assistant", "content": "\n".join(p for p in pending if p)})
                    pending = []
                name = call_names.get(step.call_id or "", "")
                body = step.content if step.ok else f"ERROR: {step.content}"
                content = _render_response(name, body, self.dialect)
                role = self.dialect.tool_result_role
                merge_prefix = "<tool_output" if self.dialect.family == "atem" else "<tool_response>"
                if messages and messages[-1]["role"] == role and messages[-1]["content"].startswith(merge_prefix):
                    messages[-1]["content"] += "\n" + content
                else:
                    messages.append({"role": role, "content": content})
            elif step.kind == FINAL:
                pending.append(step.content)
        if pending:
            messages.append({"role": "assistant", "content": "\n".join(p for p in pending if p)})
        return messages

    def next_steps(self, task: Task, history: list[Step]) -> list[Step]:
        messages = self._messages(task, history)
        if self.dialect.tools_in_prompt:
            text, raw = self.complete(messages)
        else:
            text, raw = self.complete(messages, tools=self._tool_payload(task))
        if raw:
            turn_usage = usage_from_provider(raw, shape=OPENAI)
            self.usage = self.usage + turn_usage
            self._tokens += turn_usage.total
        # Reasoning arrives beside the content rather than inside it for a dialect with no inline
        # tag, so it is read off the response and handed to the parser. Looking for a tag in the
        # text would find nothing and report every turn as having skipped deliberation.
        reasoning = str(raw.get("reasoning_content") or "") if isinstance(raw, dict) else ""
        structured = raw.get("tool_calls") if isinstance(raw, dict) else None
        if structured:
            # The server parsed the wire format, and it does not always parse ALL of it -- and the
            # channel it leaves a call in is usually `reasoning_content`, not `content`. An ATEM turn
            # is `assistant to=self` deliberation followed by `assistant to=<tool>` carrying the call,
            # so a reasoning parser that does not stop cleanly at the end of the first swallows the
            # second. Measured over one 19-task run: 15 turns arrived with complete call markup inside
            # the reasoning, in 11 of the 19 episodes, and 8 of those calls matched nothing the server
            # returned structurally. Those were never executed, never counted in `tool_calls`, and not
            # malformed either -- markup inside a THINKING step, which no metric distinguishes from
            # real reasoning.
            #
            # So both channels are parsed and merged, deduplicated by (name, arguments): the other 7
            # were echoes of a call the server did return, and executing those twice is worse than
            # losing them.
            turn = _turn_from_tool_calls(structured, text=text, reasoning=reasoning)
            turn = _merge_leftover_calls(
                turn, text=text, reasoning=reasoning, dialect=self.dialect, schemas=self.tool_schemas
            )
        else:
            turn = parse_turn(text, dialect=self.dialect, reasoning=reasoning, schemas=self.tool_schemas)
        # `parse_turn` already separates malformed from abstained, and `steps_from_turn`
        # already says these are "a failure the metrics should see" -- but the failure left
        # here as prose inside a THINKING step, which no metric can distinguish from real
        # reasoning. Counting it is what makes Hermes conformance measurable, and it has to
        # be measurable before it can be a gate: emitting a well-formed <tool_call> costs
        # tokens, so anything scoring efficiency is scoring against the protocol.
        self.parse_failures += len(turn.malformed)
        return steps_from_turn(turn)


def _fingerprint(call: ParsedCall) -> tuple[str, str]:
    return call.name, json.dumps(call.arguments, sort_keys=True)


def _merge_leftover_calls(
    turn: ParsedTurn, *, text: str, reasoning: str, dialect: Dialect, schemas: dict[str, dict[str, Any]]
) -> ParsedTurn:
    """Recover calls the server left in the text or the reasoning, and clean both channels.

    A serving layer's tool parser is not obliged to return every call in a turn, and the one measured
    here does not. Over a 19-task run, 15 turns in 11 episodes arrived with complete ATEM call markup
    inside `reasoning_content`; 8 of those calls matched nothing the server returned structurally, so
    reading only the structured list dropped them. That is the worst shape a loss can take: the
    episode still succeeds when the model reissues the call, and the only symptom is an agent that
    appears to need more turns than it does.

    Both channels are read because both carry it -- and `reasoning_content` is the likelier one. An
    ATEM turn is `assistant to=self` deliberation followed by `assistant to=<tool>` carrying the call,
    so a reasoning parser that does not stop cleanly at the end of the first swallows the second. The
    markup is a parser artifact, not the model deliberating about a call it chose not to make, which
    is why these are executed rather than dropped the way Hermes drops a call inside `<think>`.

    Deduplicated on (name, arguments): the other 7 were echoes of a call the server did return, and
    executing those twice is worse than losing them.

    The cleaned prose from each channel replaces it, so the recorded trajectory -- and therefore the
    SFT corpus built from it -- does not carry raw wire markup inside a reasoning block and teach the
    model to emit a call where its own template renders private deliberation.
    """
    from_text = parse_turn(text, dialect=dialect, reasoning="", schemas=schemas) if text.strip() else None
    from_reasoning = (
        parse_turn(reasoning, dialect=dialect, reasoning="", schemas=schemas) if reasoning.strip() else None
    )

    seen = {_fingerprint(c) for c in turn.calls}
    extra: list[ParsedCall] = []
    malformed: tuple[str, ...] = turn.malformed
    for parsed in (from_text, from_reasoning):
        if parsed is None:
            continue
        malformed = malformed + parsed.malformed
        for call in parsed.calls:
            if _fingerprint(call) not in seen:
                seen.add(_fingerprint(call))
                extra.append(call)

    return ParsedTurn(
        calls=turn.calls + tuple(extra),
        # The parser's prose for each channel, so a turn that was half answer and half call keeps the
        # answer while the markup stops being recorded as something the model said.
        text=from_text.text if from_text is not None else turn.text,
        scratch_pad=from_reasoning.text.strip() if from_reasoning is not None else turn.scratch_pad,
        malformed=malformed,
    )


def _turn_from_tool_calls(calls: list[dict[str, Any]], *, text: str, reasoning: str) -> ParsedTurn:
    """Build a `ParsedTurn` from calls the serving layer parsed.

    Arguments arrive as a JSON string, which is the OpenAI shape whatever the model's own wire
    format was. A string that will not decode is recorded as malformed rather than dropped: it is
    the same failure `parse_turn` reports for an unreadable call, and folding it into "no calls
    found" would score the protocol's hardest failure as its most disciplined behaviour.
    """
    import json

    parsed: list[ParsedCall] = []
    malformed: list[str] = []
    for call in calls:
        name = str(call.get("name") or "")
        if not name:
            malformed.append("the server returned a tool call with no function name")
            continue
        raw_args = call.get("arguments")
        if isinstance(raw_args, dict):
            parsed.append(ParsedCall(name=name, arguments=raw_args))
            continue
        try:
            decoded = json.loads(raw_args or "{}")
        except (TypeError, ValueError) as exc:
            malformed.append(f"{name}: the server's tool-call arguments are not readable JSON ({exc})")
            continue
        if not isinstance(decoded, dict):
            malformed.append(f"{name}: tool-call arguments decoded to {type(decoded).__name__}, not an object")
            continue
        parsed.append(ParsedCall(name=name, arguments=decoded))
    return ParsedTurn(calls=tuple(parsed), text=text.strip(), scratch_pad=reasoning.strip(), malformed=tuple(malformed))


def _render_call(step: Step, dialect: Dialect) -> str:
    """Rebuild an assistant turn's call in the dialect the model speaks.

    The history sent back has to be in the same format the model emits, or it is being shown a
    conversation it did not have -- and a model reading its own prior turns in a foreign format
    is being taught, mid-episode, that the format is negotiable.
    """
    if dialect.family == "atem":
        from hermes.atem import render_tool_call as render_atem

        return render_atem(step.tool or "", step.args)
    from hermes.protocol import render_tool_call

    return render_tool_call(step.tool or "", step.args)


def _render_response(name: str, content: str, dialect: Dialect) -> str:
    if dialect.family == "atem":
        from hermes.atem import render_tool_response as render_atem_response

        return render_atem_response(name, content)
    return render_tool_response(name, content)


def steps_from_turn(turn: ParsedTurn) -> list[Step]:
    """Convert one parsed assistant turn into trajectory steps.

    A malformed turn becomes a *thinking* step naming the parse errors, never a `final`.
    Returning nothing would stall the episode and returning a final would record the
    model as having answered, and neither says what happened: the model emitted something
    the protocol could not read, which is a failure the metrics should see.
    """
    steps: list[Step] = []
    if turn.scratch_pad:
        steps.append(Step(kind=THINKING, content=turn.scratch_pad))
    if turn.malformed:
        steps.append(Step(kind=THINKING, content="unparseable tool call: " + "; ".join(turn.malformed)))
        return steps
    if turn.text:
        steps.append(Step(kind=THINKING, content=turn.text) if turn.calls else Step(kind=FINAL, content=turn.text))
    for index, call in enumerate(turn.calls):
        steps.append(Step(kind=TOOL_CALL, tool=call.name, args=call.arguments, call_id=f"c{index}"))
    return steps


def openai_completion(
    *, base_url: str, model: str, api_key: str = "", timeout_s: int = 300, **params: Any
) -> Completion:
    """A `Completion` over any OpenAI-compatible chat endpoint.

    Covers SGLang, vLLM and every hosted gateway the teachers already use, which is why the
    adapter is this thin: the benchmark should not care which of them is serving, only that
    the same messages go in and the usage comes back.

    It does care about one thing, and not by choice: a server that parses the wire format itself
    returns structured `tool_calls` and an EMPTY `content`, so both are carried back. See
    `next_steps`, and docs/serving-muse-glimmer.md for what reading only `content` would score.

    Sampling parameters are passed through and belong in the run manifest, not here. They
    change the result as surely as the prompt does -- two runs at different temperatures are
    not the same measurement -- and burying a default inside the adapter would hide that.
    """
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key or "not-needed", timeout=timeout_s)

    def complete(
        messages: list[dict[str, str]], *, tools: list[dict[str, Any]] | None = None
    ) -> tuple[str, dict[str, Any]]:
        extra = {"tools": tools} if tools else {}
        response = client.chat.completions.create(model=model, messages=messages, **extra, **params)  # type: ignore[arg-type]
        choice = response.choices[0]
        usage = response.usage.model_dump() if response.usage else {}
        # Carried beside the usage because a dialect that reasons on its own channel returns it
        # here rather than in the content. `usage_from_provider` reads the keys it knows and
        # ignores this one; dropping it would blank the reasoning for every ATEM turn.
        reasoning = getattr(choice.message, "reasoning_content", None)
        if reasoning:
            usage["reasoning_content"] = reasoning
        # Structured calls, when the server parsed them itself. SGLang with --tool-call-parser muse
        # returns OpenAI tool_calls and leaves `content` EMPTY -- so a policy that only reads content
        # would see nothing said, score an abstention, and report a model that never calls a tool as
        # a model that chose not to. Carried through so the caller can prefer them.
        calls = getattr(choice.message, "tool_calls", None)
        if calls:
            usage["tool_calls"] = [
                {"name": c.function.name, "arguments": c.function.arguments}
                for c in calls
                if getattr(c, "function", None) is not None
            ]
        return choice.message.content or "", usage

    return complete
