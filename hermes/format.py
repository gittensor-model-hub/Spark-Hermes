"""CLI: agent trajectories -> SFT-ready messages records.

    python -m hermes.format \
        --in data/processed/hermes_trajectories.jsonl \
        --out data/processed/hermes_sft.jsonl

Renders each trajectory into the OpenAI-style `messages` shape Axolotl consumes with
`type: chat_template`, using the tool-call/tool-result roles the Qwen3 chat template
already understands. Reasoning is folded into the assistant turn as a leading
`<think>...</think>` block -- the same convention `teacher/format.py` uses for
non-agentic data, so a worker's output shape stays consistent whether the row came from
a chat corpus or an execution trace.

One assistant turn is emitted per *batch* of consecutive tool calls, so a model that
issues two independent calls before observing either learns that it may do so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from hermes.protocol import (
    DIALECTS,
    HERMES_4,
    Conversation,
    Dialect,
    ProtocolError,
    render_scratch_pad,
    render_system,
    render_tool_calls,
    tool_schema,
)
from hermes.trajectory import (
    FINAL,
    THINKING,
    TOOL_CALL,
    TOOL_RESULT,
    AgentTrajectory,
    Step,
    TrajectoryError,
    load_jsonl,
    validate,
)

DEFAULT_SYSTEM = (
    "You are a Hermes agent. Solve the task by using the tools available to you. "
    "Verify your results before reporting them, recover from failures instead of "
    "giving up, and explain the final state when you are done."
)


# What to do with the prompt that produced a row, when the model will be served without it.
#
# The project's bet is that guidance is scaffolding: a miner supplies it, the model
# internalises it, users run stock Hermes. `hermes.leakage` measures what that costs and the
# answer on the canonical export is 62.4% -- 83 of 133 rows carry assistant text whose only
# source is the prompt, 78 of them a validator sentinel it asked for and the user did not. So this is a decision with a measured price, not a
# default, and there is no automatic repair: whether a behaviour should survive without its
# prompt is a judgement about what the model ought to do.
#
# Three honest options, and every row records which one it got. A corpus that cannot say how
# it was built cannot be reweighted or discarded later by anyone who did not build it.
KEEP = "keep"  # train on the prompt that ran. No shift, and no internalisation either.
REPLACE = "replace"  # substitute a canonical prompt that ships with the model.
STRIP = "strip"  # train toward the empty prompt. The largest shift; measure before choosing.

SYSTEM_POLICIES = (KEEP, REPLACE, STRIP)


def _system_content(trajectory: AgentTrajectory, default_system: str) -> str:
    """System turn, with the episode's tool names spelled out.

    The names have to appear in text the student is actually trained on. The record's
    top-level `tools` key is metadata that no recipe here maps into the prompt, so
    without this a row teaches the model to emit `tool_calls` for names it was never
    shown -- training exactly the invented-tool behavior stage C is meant to remove.

    The default applies only to hand-authored trajectories, which never had a system
    prompt to record. An EXECUTED trajectory did: some specific text caused those exact
    tokens, and substituting a different one pairs the assistant turns with a prompt that
    did not produce them. The harness prompt tells the agent to "inspect before you change
    anything" and to "recover from a failed tool call rather than giving up"; DEFAULT_SYSTEM
    says neither. A row built that way teaches inspect-first behaviour as though it were
    unprompted, and the mismatch is invisible in the output because both are plausible
    system prompts. Refusing is the same call `to_hermes_record` already makes on missing
    tool schemas -- a row that misdescribes its own context is worse than no row.
    """
    if trajectory.system is None and trajectory.metadata.get("executed"):
        raise ValueError(
            f"executed trajectory {trajectory.task_id or '<no id>'!r} recorded no system prompt, so the "
            "prompt that produced these tokens is unknown; exporting it would pair the assistant turns "
            "with a system prompt that did not cause them. Record `system=` at execution time."
        )
    system = trajectory.system or default_system
    if not trajectory.tools_available:
        return system
    return f"{system}\n\nAvailable tools: {', '.join(trajectory.tools_available)}"


def _think_block(reasoning: str) -> str:
    return f"<think>\n{reasoning.strip()}\n</think>"


def _reasoning_text(step: Step) -> str:
    """What a thinking step contributes to the trained `<think>` block.

    Structured state wins and the prose is dropped entirely -- keeping both would train
    the student to emit the labelled form *and* the accent, which is worse than either.
    """
    if step.state is not None:
        return step.state.render()
    return step.content


def _tool_field(names: tuple[str, ...], *, dialect: Dialect, schemas: dict[str, dict[str, Any]] | None) -> list[Any]:
    """The row's `tools` field: bare names, or full definitions when the template renders them.

    Hermes puts the definitions in the system prompt, which is already in `messages`, so the names
    are a record of what was available and nothing reads them as a signature. The ATEM template
    renders definitions from this field and calls `.name` on each entry -- given a string it raises
    `'str object' has no attribute 'name'`, which is how an ATEM corpus failed on its first row.

    Refused rather than filled in with stubs. A synthesised signature shows the model parameters
    that do not exist, which is the same refusal `to_hermes_record` already makes.
    """
    if dialect.tools_in_prompt:
        return list(names)
    if not schemas:
        raise ProtocolError(
            f"dialect {dialect.name!r} renders tool definitions from the row, so {list(names)} cannot be "
            "written as bare names; pass tool_schemas (hermes.pin.load_tool_schemas reads the committed set)"
        )
    missing = [n for n in names if n not in schemas]
    if missing:
        raise ProtocolError(
            f"no schema for {missing}; a corpus that advertises an invented signature teaches the model "
            "parameters the tool does not have"
        )
    return [tool_schema(n, schemas[n].get("description", ""), schemas[n].get("parameters", {})) for n in names]


def _assistant_turn(
    reasoning: list[str], calls: list[Step], answer: str | None, *, dialect: Dialect = HERMES_4
) -> dict[str, Any]:
    """One assistant message, shaped for the chat template that will render it.

    Both branches below exist because a real trajectory was rendered through the pinned ATEM
    template and it raised. The row looked correct in every test in this repository, because every
    test in this repository checked the row rather than what a template does with it.
    """
    parts: list[str] = []
    joined = "\n\n".join(r.strip() for r in reasoning if r.strip())
    message: dict[str, Any] = {"role": "assistant"}
    if joined and dialect.reasoning_in_content:
        parts.append(_think_block(joined))
    if answer:
        parts.append(answer.strip())
    message["content"] = "\n\n".join(parts)
    if joined and not dialect.reasoning_in_content:
        # A separate field, not a tag inside the content, because that is where this dialect's
        # template looks. Putting it in `content` loses it entirely on a turn that also carries
        # tool_calls -- the template renders the calls and never reads `content` at all -- so the
        # reasoning would vanish from precisely the turns whose reasoning is the lesson.
        message["reasoning_content"] = joined
    if calls:
        message["tool_calls"] = [
            {
                "id": call.call_id,
                "type": "function",
                # A JSON string for Hermes, whose templates parse it; a mapping for a template that
                # cannot. The ATEM template raises `a JSON string cannot be parsed in the HF jinja
                # sandbox` rather than rendering something wrong, so every tool-calling row in an
                # ATEM corpus failed at train time until this branch existed.
                "function": {
                    "name": call.tool,
                    "arguments": json.dumps(call.args, ensure_ascii=False)
                    if dialect.tool_arguments_json
                    else call.args,
                },
            }
            for call in calls
        ]
    return message


def apply_system_policy(
    messages: list[dict[str, Any]],
    *,
    policy: str,
    replacement: str = "",
) -> list[dict[str, Any]]:
    """Keep, replace, or drop the system turn, per an explicit choice.

    `strip` removes the turn entirely rather than emptying it. An empty system message is
    not the serving condition it is meant to imitate -- a model served without a system
    prompt sees no system turn at all -- and training on `""` teaches the shape of a blank
    instruction rather than the absence of one.

    `replace` is the option the design discussion did not have and probably wants. The rule
    that Hermes is upstream forbids forking the *agent*; it does not forbid a model shipping
    its own recommended prompt in its model card and chat template. Training toward a small
    canonical prompt that travels with the weights is a far smaller distribution shift than
    training toward nothing, and it is still not a Hermes fork.
    """
    if policy not in SYSTEM_POLICIES:
        raise ValueError(f"unknown system policy {policy!r}; expected one of {list(SYSTEM_POLICIES)}")
    if policy == KEEP:
        return messages
    if policy == STRIP:
        return [m for m in messages if m.get("role") != "system"]
    if not replacement.strip():
        raise ValueError("system policy 'replace' needs a replacement prompt; an empty one is 'strip' by another name")
    return [{**m, "content": replacement} if m.get("role") == "system" else m for m in messages]


def to_messages_record(
    trajectory: AgentTrajectory,
    *,
    default_system: str = DEFAULT_SYSTEM,
    system_policy: str = KEEP,
    system_replacement: str = "",
    dialect: Dialect = HERMES_4,
    tool_schemas: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Render one trajectory as an Axolotl `chat_template` messages record.

    `dialect` decides the row's *shape*, not just its markup, and the default keeps every existing
    caller on the Hermes shape they were written against. Passing the dialect the episode was
    actually run in is what makes the corpus trainable: see `_assistant_turn`.

    `tool_schemas` resolves `tools_available` -- which is a list of bare names -- into the function
    definitions a chat template needs. Required for a dialect that renders tool definitions from the
    row, because the alternative is emitting names into a field the template will call `.name` on.
    """
    messages: list[dict[str, Any]] = [{"role": "system", "content": _system_content(trajectory, default_system)}]
    messages.append({"role": "user", "content": trajectory.task})

    reasoning: list[str] = []
    calls: list[Step] = []

    for step in trajectory.steps:
        if step.kind == THINKING:
            reasoning.append(_reasoning_text(step))
        elif step.kind == TOOL_CALL:
            calls.append(step)
        elif step.kind == TOOL_RESULT:
            if calls:
                messages.append(_assistant_turn(reasoning, calls, None, dialect=dialect))
                reasoning, calls = [], []
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": step.call_id,
                    "content": step.content if step.ok else f"ERROR: {step.content}",
                }
            )
        elif step.kind == FINAL:
            messages.append(_assistant_turn(reasoning, calls, step.content, dialect=dialect))
            reasoning, calls = [], []

    record: dict[str, Any] = {
        "messages": apply_system_policy(messages, policy=system_policy, replacement=system_replacement)
    }
    if trajectory.tools_available:
        record["tools"] = _tool_field(trajectory.tools_available, dialect=dialect, schemas=tool_schemas)
    if system_policy != KEEP:
        # Recorded on the row, not only in the command that produced it. A corpus is read
        # months later by someone who did not build it, and "were these rows trained toward
        # a prompt that will be there at inference?" is not answerable from the messages --
        # a stripped row and a row that never had a prompt look identical.
        record["system_policy"] = system_policy
    return record


def _scratch_pad_from(reasoning: list[Step], calls: list[Step], *, observed_results: bool = False) -> str:
    """A GOAP block from the episode's structured state.

    `ReasoningState` already carries the fields the scratchpad wants under different
    names: goal is Goal, the pending calls are Actions, expected-vs-observed is what an
    Observation records, and decision is the Reflection. Rendering from the state rather
    than the prose keeps the property the state exists for -- three teachers with the same
    reasoning produce one identical trained form.
    """
    states = [s.state for s in reasoning if s.state is not None]
    if not states:
        return ""
    first = states[0]
    actions = [f"{c.tool}({', '.join(f'{k}={v!r}' for k, v in c.args.items())})" for c in calls]
    reflection = " ".join(s.decision.strip() for s in states if s.decision.strip())
    # An Observation may only appear once results have actually come back. `observed_signal`
    # is recorded after the fact, so a turn that is issuing calls -- or an episode where no
    # tool ever ran, such as an abstention -- must write `None`. Otherwise the model learns
    # to fill in a result before the tool has run: a prediction dressed as evidence, in the
    # one field whose whole job is to hold evidence.
    observed = (
        " ".join(s.observed_signal.strip() for s in states if s.observed_signal.strip())
        if observed_results and not calls
        else ""
    )
    return render_scratch_pad(
        first.goal,
        actions or [first.action],
        observation=observed or "None",
        reflection=reflection,
    )


def to_hermes_record(
    trajectory: AgentTrajectory,
    *,
    dialect: Dialect,
    default_system: str = DEFAULT_SYSTEM,
) -> dict[str, Any]:
    """Render one trajectory in the Hermes wire format rather than the OpenAI one.

    The difference is not cosmetic. `to_messages_record` emits a structured `tool_calls`
    array, which the Qwen3 template understands and a Hermes runtime does not; this emits
    `<tool_call>` blocks inside assistant text, which is what Hermes was trained on. A
    worker distilled from the first shape has learned tool use without learning the
    protocol, and the gap shows up as a model emitting nothing its runtime recognises.

    **Requires recorded tool schemas.** Hermes advertises tools as full signatures inside
    `<tools>`, and a trajectory carrying only names cannot produce that turn. Inventing
    plausible parameter schemas would train the model on signatures that do not exist --
    the invented-tool failure moved one level down, from names to arguments -- so this
    refuses instead. Producers must record `metadata.tool_schemas` alongside the names
    they advertised.
    """
    schemas = trajectory.metadata.get("tool_schemas")
    if not schemas:
        raise ProtocolError(
            f"{trajectory.task_id or trajectory.task[:40]!r}: no metadata.tool_schemas; "
            "Hermes advertises full signatures in <tools>, and synthesizing them from bare "
            "names would train the model on parameters that do not exist"
        )
    declared = set(trajectory.tools_available)
    named = {s.get("name") for s in schemas}
    if declared and named != declared:
        # The allowlist and the advertised signatures must be the same set, or the model is
        # shown one thing and graded against another.
        raise ProtocolError(
            f"tool_schemas {sorted(n for n in named if n)} do not match tools_available {sorted(declared)}"
        )

    tools = [tool_schema(s["name"], s.get("description", ""), s.get("parameters", {})) for s in schemas]
    use_pad = dialect.supports_scratch_pad
    convo = Conversation(dialect)
    convo.system(render_system(tools, dialect=dialect, scratch_pad=use_pad, extra=trajectory.system or ""))
    convo.user(trajectory.task)

    reasoning: list[Step] = []
    calls: list[Step] = []
    call_names: dict[str, str] = {}
    saw_result = False

    def flush(answer: str | None) -> None:
        nonlocal saw_result
        parts: list[str] = []
        if use_pad:
            pad = _scratch_pad_from(reasoning, calls, observed_results=saw_result)
            if pad:
                parts.append(pad)
            elif reasoning:
                # Unnormalized reasoning: no state to build a GOAP block from. Falling
                # through to a bare tool call would silently drop the thinking, training
                # the model to act without visible deliberation on exactly the rows where
                # the teacher deliberated most.
                joined = "\n\n".join(_reasoning_text(x).strip() for x in reasoning if _reasoning_text(x).strip())
                if joined:
                    parts.append(f"<scratch_pad>\n{joined}\n</scratch_pad>")
        else:
            joined = "\n\n".join(_reasoning_text(s).strip() for s in reasoning if _reasoning_text(s).strip())
            if joined:
                parts.append(f"<{dialect.reasoning_tag}>\n{joined}\n</{dialect.reasoning_tag}>")
        if calls:
            parts.append(render_tool_calls([(c.tool or "", c.args) for c in calls]))
        if answer:
            parts.append(answer.strip())
        if parts:
            convo.assistant("\n".join(parts))

    for step in trajectory.steps:
        if step.kind == THINKING:
            reasoning.append(step)
        elif step.kind == TOOL_CALL:
            calls.append(step)
            if step.call_id:
                call_names[step.call_id] = step.tool or ""
        elif step.kind == TOOL_RESULT:
            if calls:
                flush(None)
                reasoning, calls = [], []
            name = call_names.get(step.call_id or "", "")
            convo.tool_result(name, step.content if step.ok else f"ERROR: {step.content}")
            saw_result = True
        elif step.kind == FINAL:
            flush(step.content)
            reasoning, calls = [], []

    record = convo.to_record()
    record["tools"] = tools
    return record


def convert(
    trajectories: list[AgentTrajectory],
    *,
    require_success: bool = False,
    keep_invalid: bool = False,
    executed_only: bool = False,
    keep_harness_finals: bool = False,
    structured_only: bool = False,
    dialect: Dialect | None = None,
    system_policy: str = KEEP,
    system_replacement: str = "",
) -> tuple[list[dict[str, Any]], list[str]]:
    """Convert trajectories to SFT records, returning (records, skip reasons).

    With `dialect` set, rows are rendered in the Hermes wire format instead of the
    OpenAI-style one. A row that cannot be rendered faithfully is skipped with its reason
    rather than downgraded to the other shape: a corpus half in each format teaches the
    model that both are acceptable, and it would then emit either at random.
    """
    records: list[dict[str, Any]] = []
    skipped: list[str] = []
    for index, trajectory in enumerate(trajectories):
        try:
            validate(trajectory)
        except TrajectoryError as exc:
            if not keep_invalid:
                skipped.append(f"row {index}: {exc}")
                continue
        if require_success and not trajectory.success:
            skipped.append(f"row {index}: unverified outcome (success=false)")
            continue
        if executed_only and not trajectory.metadata.get("executed"):
            skipped.append(f"row {index}: simulated tool results (metadata.executed is not true)")
            continue
        if structured_only and not trajectory.fully_structured:
            # Multi-teacher corpora are where this matters: a partially-normalized row
            # still trains raw prose on the steps that were missed, reintroducing the
            # style mixing normalization was meant to remove.
            skipped.append(f"row {index}: unnormalized reasoning (not fully structured)")
            continue
        if not keep_harness_finals and trajectory.metadata.get("harness_final"):
            # The closing turn was written by the bench runner ("step budget
            # exhausted"), not the agent. Rendered as-is it becomes the assistant's
            # final answer and teaches the student to say it.
            skipped.append(f"row {index}: harness-authored final step (metadata.harness_final)")
            continue
        if dialect is None:
            records.append(
                to_messages_record(trajectory, system_policy=system_policy, system_replacement=system_replacement)
            )
            continue
        try:
            records.append(to_hermes_record(trajectory, dialect=dialect))
        except ProtocolError as exc:
            skipped.append(f"row {index}: {exc}")
    return records, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, required=True, help="jsonl of agent trajectories")
    parser.add_argument("--out", dest="out_path", type=Path, required=True, help="jsonl of SFT messages records")
    parser.add_argument(
        "--require-success",
        action="store_true",
        help="keep only verified-successful trajectories (drops the failure/recovery signal)",
    )
    parser.add_argument(
        "--keep-invalid",
        action="store_true",
        help="do not drop schema-invalid trajectories (debugging only; never for training data)",
    )
    parser.add_argument(
        "--executed-only",
        action="store_true",
        help="keep only rows whose tool results came from real execution (metadata.executed); "
        "required for the stage-c tool-reliability mix",
    )
    parser.add_argument(
        "--structured-only",
        action="store_true",
        help="keep only trajectories whose every thinking step carries a normalized reasoning "
        "state; use for multi-teacher mixes, where raw prose reintroduces style mixing",
    )
    parser.add_argument(
        "--keep-harness-finals",
        action="store_true",
        help="keep episodes whose closing turn was written by the bench runner rather than the agent",
    )
    parser.add_argument(
        "--target",
        choices=("openai", *sorted(DIALECTS)),
        default="openai",
        help=(
            "wire format. 'openai' emits a tool_calls array for the Qwen3 template; a hermes "
            "dialect emits <tool_call> blocks in assistant text, which is what a Hermes "
            "runtime reads. The dialects differ in the role tool results come back under."
        ),
    )
    parser.add_argument(
        "--system-policy",
        choices=SYSTEM_POLICIES,
        default=KEEP,
        help=(
            "what to do with the prompt that produced each row. 'keep' trains on it, which "
            "means no distribution shift and no internalisation either. 'strip' trains toward "
            "the empty prompt -- run `python -m hermes.leakage` first, because on the canonical "
            "export 62%% of rows carry assistant text whose only source is that prompt. "
            "'replace' substitutes a canonical prompt shipped with the model, which is a much "
            "smaller shift than stripping and is still not a Hermes fork."
        ),
    )
    parser.add_argument(
        "--system-replacement",
        default="",
        help="the canonical prompt to substitute; required by --system-policy replace",
    )
    args = parser.parse_args(argv)

    trajectories = list(load_jsonl(args.in_path))
    records, skipped = convert(
        trajectories,
        require_success=args.require_success,
        keep_invalid=args.keep_invalid,
        executed_only=args.executed_only,
        keep_harness_finals=args.keep_harness_finals,
        structured_only=args.structured_only,
        dialect=DIALECTS.get(args.target),
        system_policy=args.system_policy,
        system_replacement=args.system_replacement,
    )

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} records to {args.out_path}", file=sys.stderr)
    if args.system_policy != KEEP:
        # Said out loud, because this is the choice with a measured price and the person
        # running the command months from now is not the person who read hermes/leakage.py.
        print(
            f"system policy: {args.system_policy}. These rows train toward a context the "
            "prompt that produced them will not be in; `python -m hermes.leakage` measures "
            "how much of their assistant text traced to it.",
            file=sys.stderr,
        )
    if skipped:
        print(f"skipped {len(skipped)} trajectories:", file=sys.stderr)
        for reason in skipped[:20]:
            print(f"  - {reason}", file=sys.stderr)
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
