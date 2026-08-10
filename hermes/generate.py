"""CLI: generate synthetic agent trajectories from frontier teachers.

    python -m hermes.generate \
        --tasks hermes/tasks/phase0.jsonl \
        --out data/processed/hermes_trajectories.jsonl \
        --provider anthropic

Reuses the pinned teacher clients from `teacher.providers`, so trajectory generation
inherits the same provider/model policy the dataset track already enforces.

**Simulated, not executed.** A teacher asked to solve a task in one pass *writes down*
what it thinks the tools would return; nothing is actually run. Those rows are useful for
teaching trajectory shape -- when to call a tool, how to phrase a call, how to react to
an error -- but their tool results are fiction, and a corpus of nothing else teaches a
model that its predictions about the world are the world. Every row produced here is
stamped `metadata.executed = false`. Rows whose results came from real execution (the
HermesBench runner) carry `executed = true`; keep the ratio honest and prefer executed
data for anything that trains verification behavior.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from hermes.trajectory import AgentTrajectory, TrajectoryError, validate, write_jsonl
from teacher.providers import get_teacher

# Used when a task row does not name its own tools. The same tuple is both advertised to
# the teacher and recorded as the trajectory's `tools_available`: if the two drift apart,
# the schema's tool allowlist silently stops enforcing anything (an empty recorded set
# permits every tool), and invented tools reach training data.
DEFAULT_TOOLS = ("terminal", "file_read", "edit", "python")

AGENT_BRIEF = """You are an expert Hermes agent. Solve the task below.

You must:
- use tools rather than answering from memory
- verify your results before reporting them
- recover from failures instead of giving up
- explain the final state when you are done

Available tools: {tools}

Respond with a single JSON object in a ```json fenced block, and nothing else. Shape:

```json
{{
  "steps": [
    {{"kind": "thinking", "content": "what you need to find out and why"}},
    {{"kind": "tool_call", "call_id": "c1", "tool": "terminal", "args": {{"command": "pytest"}}}},
    {{"kind": "tool_result", "call_id": "c1", "ok": false, "content": "3 failed"}},
    {{"kind": "thinking", "content": "reacting to that result"}},
    {{"kind": "tool_call", "call_id": "c2", "tool": "file_read", "args": {{"path": "src/x.py"}}}},
    {{"kind": "tool_result", "call_id": "c2", "ok": true, "content": "<file contents>"}},
    {{"kind": "final", "content": "what you did and the verified end state"}}
  ],
  "success": true
}}
```

Rules: every tool_call needs a matching tool_result with the same call_id; call_ids are
unique; the last step is "final"; only use the tools listed above. Include at least one
failed tool_result and the recovery from it where the task realistically produces one --
trajectories where everything works first try are the least useful kind.

Task: {task}"""

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class GenerationError(RuntimeError):
    """A teacher response could not be parsed into a trajectory."""


def parse_response(
    text: str,
    *,
    task: str,
    tools: tuple[str, ...],
    task_id: str | None = None,
    source: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentTrajectory:
    """Parse a teacher's fenced-JSON response into a validated trajectory."""
    match = _JSON_BLOCK.search(text)
    payload = match.group(1) if match else text.strip()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise GenerationError(f"teacher response is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise GenerationError(f"teacher response is {type(parsed).__name__}, expected an object")

    record = {
        "task": task,
        "steps": parsed.get("steps", []),
        "success": parsed.get("success", False),
        "tools_available": list(tools),
        "task_id": task_id,
        "source": source,
        "metadata": {**(metadata or {}), "executed": False},
    }
    trajectory = AgentTrajectory.from_record(record)
    validate(trajectory)
    return trajectory


def _iter_tasks(path: Path, limit: int | None) -> Iterator[dict[str, Any]]:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if limit is not None and count >= limit:
                return
            yield json.loads(line)
            count += 1


def generate_trajectories(
    tasks: list[dict[str, Any]],
    *,
    provider: str,
    model: str | None = None,
    max_tokens: int = 8192,
    temperature: float = 0.7,
) -> tuple[list[AgentTrajectory], list[str]]:
    """Generate one trajectory per task. Returns (trajectories, failure reasons)."""
    teacher = get_teacher(provider, model)
    trajectories: list[AgentTrajectory] = []
    failures: list[str] = []

    for index, task in enumerate(tasks):
        prompt_text = task.get("task") or task.get("prompt")
        if not prompt_text:
            failures.append(f"task {index}: no 'task' field")
            continue
        tools = tuple(task.get("tools") or DEFAULT_TOOLS)
        brief = AGENT_BRIEF.format(tools=", ".join(tools), task=prompt_text)

        try:
            response = teacher.generate(brief, max_tokens=max_tokens, temperature=temperature)
            trajectory = parse_response(
                response.response,
                task=prompt_text,
                tools=tools,
                task_id=task.get("task_id"),
                source=f"synthetic:{teacher.name}:{teacher.model}",
                metadata={"teacher_model": teacher.model, "teacher_provider": teacher.name},
            )
        except (GenerationError, TrajectoryError) as exc:
            failures.append(f"task {index} ({task.get('task_id') or 'unnamed'}): {exc}")
            continue
        trajectories.append(trajectory)

    return trajectories, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", type=Path, required=True, help="jsonl of {task, tools, task_id}")
    parser.add_argument("--out", type=Path, required=True, help="jsonl of generated trajectories")
    parser.add_argument("--provider", default="anthropic", help="teacher provider (anthropic, openai)")
    parser.add_argument("--model", default=None, help="teacher model override")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N tasks")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args(argv)

    tasks = list(_iter_tasks(args.tasks, args.limit))
    trajectories, failures = generate_trajectories(
        tasks,
        provider=args.provider,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    written = write_jsonl(args.out, trajectories)
    print(f"wrote {written} trajectories to {args.out}", file=sys.stderr)
    if failures:
        print(f"{len(failures)} tasks failed:", file=sys.stderr)
        for reason in failures[:20]:
            print(f"  - {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
