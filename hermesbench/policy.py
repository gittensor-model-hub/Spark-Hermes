"""A served model as an `AgentPolicy`: the one interface everything was waiting on.

`hermesbench.runner` has always been able to run an episode. It has never had anything to
run: the only `AgentPolicy` was `ReplayPolicy`, which replays a trajectory somebody
already recorded, so `main()` refuses to run a suite and says so rather than reporting
`success_rate 0.0` as a measurement. Thirteen modules in this repo are reachable only from
tests for that one reason -- not because they are badly wired, but because the thing that
would call them had no model to call.

This is that thing. It speaks to any OpenAI-compatible chat endpoint, which covers vLLM,
SGLang, and every hosted gateway the teachers already use.

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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hermes.cost import OPENAI, Usage, usage_from_provider
from hermes.protocol import Dialect, ParsedTurn, parse_turn, render_system, render_tool_response, tool_schema
from hermes.trajectory import FINAL, THINKING, TOOL_CALL, TOOL_RESULT, Step
from hermesbench.tasks import Task

# A chat completion: messages in, (text, raw usage dict) out. Deliberately this small --
# anything that can be adapted to it can drive the benchmark, including a local process,
# a hosted gateway, or a stub in a test.
Completion = Callable[[list[dict[str, str]]], tuple[str, dict[str, Any]]]


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
        return render_system(tools, dialect=self.dialect, scratch_pad=self.scratch_pad, extra=self.system)

    def _messages(self, task: Task, history: list[Step]) -> list[dict[str, str]]:
        """Rebuild the conversation from the trajectory so far.

        Rebuilt each turn rather than accumulated, so the messages sent always match the
        trajectory that will be scored. An independently-maintained message list can drift
        from the recorded history, and then the episode that was graded is not the one the
        model saw.
        """
        messages = [{"role": "system", "content": self._system_turn(task)}, {"role": "user", "content": task.prompt}]
        call_names: dict[str, str] = {}
        pending: list[str] = []
        for step in history:
            if step.kind == THINKING:
                pending.append(step.content)
            elif step.kind == TOOL_CALL:
                if step.call_id:
                    call_names[step.call_id] = step.tool or ""
                pending.append(_render_call(step))
            elif step.kind == TOOL_RESULT:
                if pending:
                    messages.append({"role": "assistant", "content": "\n".join(p for p in pending if p)})
                    pending = []
                name = call_names.get(step.call_id or "", "")
                content = render_tool_response(name, step.content if step.ok else f"ERROR: {step.content}")
                role = self.dialect.tool_result_role
                if messages and messages[-1]["role"] == role and messages[-1]["content"].startswith("<tool_response>"):
                    messages[-1]["content"] += "\n" + content
                else:
                    messages.append({"role": role, "content": content})
            elif step.kind == FINAL:
                pending.append(step.content)
        if pending:
            messages.append({"role": "assistant", "content": "\n".join(p for p in pending if p)})
        return messages

    def next_steps(self, task: Task, history: list[Step]) -> list[Step]:
        text, raw = self.complete(self._messages(task, history))
        if raw:
            turn_usage = usage_from_provider(raw, shape=OPENAI)
            self.usage = self.usage + turn_usage
            self._tokens += turn_usage.total
        turn = parse_turn(text)
        # `parse_turn` already separates malformed from abstained, and `steps_from_turn`
        # already says these are "a failure the metrics should see" -- but the failure left
        # here as prose inside a THINKING step, which no metric can distinguish from real
        # reasoning. Counting it is what makes Hermes conformance measurable, and it has to
        # be measurable before it can be a gate: emitting a well-formed <tool_call> costs
        # tokens, so anything scoring efficiency is scoring against the protocol.
        self.parse_failures += len(turn.malformed)
        return steps_from_turn(turn)


def _render_call(step: Step) -> str:
    from hermes.protocol import render_tool_call

    return render_tool_call(step.tool or "", step.args)


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

    Covers vLLM, SGLang and every hosted gateway the teachers already use, which is why the
    adapter is this thin: the benchmark should not care which of them is serving, only that
    the same messages go in and the usage comes back.

    Sampling parameters are passed through and belong in the run manifest, not here. They
    change the result as surely as the prompt does -- two runs at different temperatures are
    not the same measurement -- and burying a default inside the adapter would hide that.
    """
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key or "not-needed", timeout=timeout_s)

    def complete(messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        response = client.chat.completions.create(model=model, messages=messages, **params)  # type: ignore[arg-type]
        choice = response.choices[0]
        usage = response.usage.model_dump() if response.usage else {}
        return choice.message.content or "", usage

    return complete
