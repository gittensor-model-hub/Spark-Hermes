"""Agent execution-trajectory schema.

The training unit for a Hermes worker is an ordered sequence of steps the agent
actually took -- thinking, a tool call, the observed result of that call, and a final
answer -- not a prompt/response pair. `teacher.providers.Trajectory` is the older,
non-agentic record (one prompt, one response); this module is deliberately separate
because the two are not interchangeable and silently mixing them would train chat
behavior into a worker.

The invariant that makes this a trajectory rather than a transcript of guessing:
**every tool call has an observed result**. A model that emits tool calls whose outputs
it never sees has not used a tool, it has described using one, and training on that
teaches confident fabrication. `validate` enforces the pairing.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hermes.state import ReasoningState, StateError

SCHEMA_VERSION = 1

THINKING = "thinking"
TOOL_CALL = "tool_call"
TOOL_RESULT = "tool_result"
FINAL = "final"

STEP_KINDS = frozenset({THINKING, TOOL_CALL, TOOL_RESULT, FINAL})


class TrajectoryError(ValueError):
    """A trajectory violates the schema and must not enter training data."""


@dataclass(frozen=True)
class Step:
    """One step of an agent trajectory.

    Field meaning by `kind`:

    - `thinking`    -> `content` is the reasoning trace.
    - `tool_call`   -> `tool` is the tool name, `args` its arguments, `call_id` links to
                       the matching result.
    - `tool_result` -> `call_id` matches the call, `content` is the observed output,
                       `ok` records whether the tool succeeded.
    - `final`       -> `content` is the answer delivered to the user.
    """

    kind: str
    content: str = ""
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None
    ok: bool = True
    # Structured reasoning, on `thinking` steps. When present this is what gets trained
    # and `content` is dropped: the labelled form transfers across teachers, the prose
    # is one provider's accent. See hermes/state.py.
    state: ReasoningState | None = None

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {"kind": self.kind}
        if self.content:
            record["content"] = self.content
        if self.tool is not None:
            record["tool"] = self.tool
        if self.args:
            record["args"] = self.args
        if self.call_id is not None:
            record["call_id"] = self.call_id
        if self.kind == TOOL_RESULT:
            record["ok"] = self.ok
        if self.state is not None:
            record["state"] = self.state.to_record()
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Step:
        kind = record.get("kind")
        if kind not in STEP_KINDS:
            raise TrajectoryError(f"unknown step kind {kind!r}; expected one of {sorted(STEP_KINDS)}")
        args = record.get("args") or {}
        if not isinstance(args, dict):
            raise TrajectoryError(f"step args must be an object, got {type(args).__name__}")
        raw_state = record.get("state")
        try:
            state = ReasoningState.from_record(raw_state) if raw_state else None
        except StateError as exc:
            # Surface as TrajectoryError so load_jsonl can annotate it with file:line.
            raise TrajectoryError(f"invalid reasoning state: {exc}") from exc
        return cls(
            kind=kind,
            content=str(record.get("content") or ""),
            tool=record.get("tool"),
            args=args,
            call_id=record.get("call_id"),
            ok=bool(record.get("ok", True)),
            state=state,
        )


@dataclass(frozen=True)
class AgentTrajectory:
    """A complete agent episode: one task, the steps taken, and how it ended.

    `success` is the *verified* outcome -- whether the task's own check passed -- not
    whether the agent claimed to be done. A trajectory may end with a confident `final`
    step and still carry `success=False`; those rows are what teach a model that
    declaring victory is not the same as winning.
    """

    task: str
    steps: tuple[Step, ...]
    success: bool
    system: str | None = None
    source: str | None = None
    task_id: str | None = None
    tools_available: tuple[str, ...] = ()
    abstention: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    @property
    def tool_calls(self) -> tuple[Step, ...]:
        return tuple(s for s in self.steps if s.kind == TOOL_CALL)

    @property
    def tool_results(self) -> tuple[Step, ...]:
        return tuple(s for s in self.steps if s.kind == TOOL_RESULT)

    @property
    def failed_steps(self) -> tuple[Step, ...]:
        return tuple(s for s in self.steps if s.kind == TOOL_RESULT and not s.ok)

    @property
    def recovery_steps(self) -> tuple[Step, ...]:
        """Tool calls issued after an observed failure.

        A corpus of only-successful trajectories teaches a model that tools never fail.
        These are the steps that teach it what to do when they do -- so they are counted
        explicitly rather than left implicit in the step list.
        """
        recoveries: list[Step] = []
        seen_failure = False
        for step in self.steps:
            if step.kind == TOOL_RESULT and not step.ok:
                seen_failure = True
            elif step.kind == TOOL_CALL and seen_failure:
                recoveries.append(step)
        return tuple(recoveries)

    @property
    def structured_thinking(self) -> tuple[Step, ...]:
        """Thinking steps carrying a normalized state rather than raw prose."""
        return tuple(s for s in self.steps if s.kind == THINKING and s.state is not None)

    @property
    def fully_structured(self) -> bool:
        """Whether every thinking step was normalized.

        A partially-normalized trajectory still trains prose on the steps that were
        missed, so this is the flag `hermes.format --structured-only` filters on.
        """
        thinking = [s for s in self.steps if s.kind == THINKING]
        return bool(thinking) and len(thinking) == len(self.structured_thinking)

    @property
    def final_answer(self) -> str:
        for step in reversed(self.steps):
            if step.kind == FINAL:
                return step.content
        return ""

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema_version": self.schema_version,
            "task": self.task,
            "steps": [s.to_record() for s in self.steps],
            "success": self.success,
        }
        if self.system:
            record["system"] = self.system
        if self.source:
            record["source"] = self.source
        if self.task_id:
            record["task_id"] = self.task_id
        if self.tools_available:
            record["tools_available"] = list(self.tools_available)
        if self.abstention:
            record["abstention"] = True
        if self.metadata:
            record["metadata"] = self.metadata
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> AgentTrajectory:
        task = record.get("task")
        if not task or not str(task).strip():
            raise TrajectoryError("trajectory has no task")
        raw_steps = record.get("steps")
        if not isinstance(raw_steps, list):
            raise TrajectoryError("trajectory steps must be a list")
        raw_version = record.get("schema_version", SCHEMA_VERSION)
        try:
            version = int(raw_version)
        except (TypeError, ValueError) as exc:
            # Must surface as TrajectoryError: load_jsonl only annotates that type with
            # the file and line number, and a bare ValueError would abort a whole corpus
            # with no way to find the offending row.
            raise TrajectoryError(f"schema_version must be an integer, got {raw_version!r}") from exc
        if version > SCHEMA_VERSION:
            raise TrajectoryError(f"trajectory schema_version {version} is newer than supported {SCHEMA_VERSION}")
        tools = record.get("tools_available") or ()
        return cls(
            task=str(task),
            steps=tuple(Step.from_record(s) for s in raw_steps),
            success=bool(record.get("success", False)),
            system=record.get("system"),
            source=record.get("source"),
            task_id=record.get("task_id"),
            tools_available=tuple(str(t) for t in tools),
            abstention=bool(record.get("abstention", False)),
            metadata=record.get("metadata") or {},
            schema_version=version,
        )


def validate(trajectory: AgentTrajectory) -> None:
    """Raise `TrajectoryError` if the trajectory is not trainable.

    Rejects the shapes that quietly poison an agent corpus:
    a call whose result is never observed, a result with no call, a tool the agent was
    never offered, and an episode with no tool use at all (that is a chat record --
    train it here and the worker learns to answer instead of act).

    Note that the tool allowlist can only be checked against a declared `tools_available`;
    a trajectory that declares none permits every tool name, including invented ones. Any
    producer that advertises a tool set to a model must record that same set on the
    trajectory, or this check silently stops enforcing anything.
    """
    if not trajectory.steps:
        raise TrajectoryError("trajectory has no steps")

    pending: dict[str, Step] = {}
    seen_call_ids: set[str] = set()
    allowed = set(trajectory.tools_available)

    for index, step in enumerate(trajectory.steps):
        if step.kind == TOOL_CALL:
            if not step.tool:
                raise TrajectoryError(f"step {index}: tool_call has no tool name")
            if allowed and step.tool not in allowed:
                raise TrajectoryError(f"step {index}: tool {step.tool!r} is not in tools_available")
            if not step.call_id:
                raise TrajectoryError(f"step {index}: tool_call {step.tool!r} has no call_id")
            if step.call_id in seen_call_ids:
                raise TrajectoryError(f"step {index}: duplicate call_id {step.call_id!r}")
            seen_call_ids.add(step.call_id)
            pending[step.call_id] = step
        elif step.kind == TOOL_RESULT:
            if not step.call_id:
                raise TrajectoryError(f"step {index}: tool_result has no call_id")
            if step.call_id not in pending:
                raise TrajectoryError(
                    f"step {index}: tool_result for unknown or already-resolved call {step.call_id!r}"
                )
            del pending[step.call_id]

    if pending:
        unresolved = ", ".join(sorted(pending))
        raise TrajectoryError(f"tool calls with no observed result: {unresolved}")

    if trajectory.abstention:
        # Declining to act is part of the protocol, not a failure to participate in it:
        # Hermes's own system prompt tells the model to answer in words when the offered
        # tools are not relevant. Without this branch the schema could not express a
        # correct abstention at all, so every trajectory in the corpus would end in a tool
        # call and the worker would learn that reaching for a tool is always the move.
        if trajectory.tool_calls:
            raise TrajectoryError(
                "trajectory is marked as an abstention but calls tools; if a tool was the "
                "right move this is not an abstention"
            )
        if not trajectory.tools_available:
            # An abstention with nothing on offer is a chat record wearing a flag: there
            # was no decision to get right.
            raise TrajectoryError(
                "an abstention must advertise the tools it declined; with none offered "
                "there was no choice to make and the record teaches nothing"
            )
        if not trajectory.final_answer.strip():
            # Hermes's rule is to *answer in natural language* when no tool fits. An empty
            # final is a crashed worker, and training it teaches the model that silence is
            # an acceptable response to a question it could have answered.
            raise TrajectoryError(
                "an abstention must say something; declining to act is answering in words, "
                "and an empty final is a crashed episode rather than a decision"
            )
    elif not trajectory.tool_calls:
        raise TrajectoryError("trajectory has no tool calls; this is a chat record, not an agent trajectory")

    if trajectory.steps[-1].kind != FINAL:
        raise TrajectoryError("trajectory does not end with a final step")


def is_valid(trajectory: AgentTrajectory) -> bool:
    try:
        validate(trajectory)
    except TrajectoryError:
        return False
    return True


def load_jsonl(path: Path) -> Iterator[AgentTrajectory]:
    """Yield trajectories from a JSONL file. Malformed rows raise with their line number."""
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield AgentTrajectory.from_record(json.loads(line))
            except (json.JSONDecodeError, TrajectoryError) as exc:
                raise TrajectoryError(f"{path}:{lineno}: {exc}") from exc


def write_jsonl(path: Path, trajectories: Sequence[AgentTrajectory]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            handle.write(json.dumps(trajectory.to_record(), ensure_ascii=False) + "\n")
    return len(trajectories)
