"""Consume committed measured curriculum through the existing task synthesis gate."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable

from admin.artifacts import StageError, canonical, content_digest
from admin.candidates import checked_file, file_identity
from admin.curriculum import build_curriculum
from admin.pipeline import Workspace
from admin.replay import ReplayStore
from hermes.taskgen.cli import _write_accepted
from hermes.taskgen.dna import TaskDNA
from hermes.taskgen.gate import gate
from hermes.taskgen.synth import synthesise
from hermesbench.tasks import load_suite


def generate_from_feedback(
    replay: ReplayStore,
    *,
    curriculum_id: str,
    request_id: str,
    workspace: Workspace,
    complete: Callable[..., Any],
    salt: str,
    allow_unsandboxed: bool = False,
) -> dict[str, Any]:
    """Generate one new task; source requests confer no correctness or release authority.

    The completion is an external adapter. Its output must pass the actual executable
    gate before publication, including distinct private checks and shortcut rejection.
    """
    from admin.artifacts import same_domain

    same_domain(replay.identity, workspace.identity)
    if not allow_unsandboxed or len(salt) < 16:
        raise StageError("task feedback generation requires a disposable execution host and withheld salt")
    original = replay.authority.get(curriculum_id, kind="curriculum")["payload"]
    if original["diagnostic_sources"]:
        raise StageError("generation requires a curriculum bound only to retained settled experience")
    current = build_curriculum(replay, config=original["config"], identifiers=original["settled_inputs"])
    if canonical({k: v for k, v in current.items() if k != "authority_id"}) != canonical(original):
        raise StageError("curriculum differs from original measured feedback")
    requests = [r for r in original["requests"] if r["request_id"] == request_id]
    if len(requests) != 1:
        raise StageError("request is absent from committed curriculum")
    request = requests[0]
    if request["category"] not in {"task_correctness", "withheld_generalization", "tool_recovery", "breadth"}:
        raise StageError("unsupported measured learning category")
    task_id = "gen-feedback-" + request_id.removeprefix("sha256:")[:16]
    if (workspace.tasks / "generated" / (task_id + ".yaml")).exists():
        raise StageError("feedback task already published; retain and reuse its original generation record")
    dna = TaskDNA(
        domain="repository_engineering",
        skills=("error_recovery",) if request["category"] == "tool_recovery" else ("self_verification",),
        environment="shell",
        difficulty=2,
        horizon=(2, 4),
        failure_mode=request["category"],
        required_tools=("terminal",),
        verification="public and independent withheld execution",
        sub_domain=request["family_id"],
        source={"curriculum": curriculum_id, "request": request_id, "source_ids": request["source_ids"]},
    )
    result = synthesise(dna, task_id=task_id, complete=complete)
    verdict = gate(result.candidate, timeout_s=30)
    if not verdict.accepted:
        raise StageError("feedback task failed synthesis gate: " + verdict.failed_check)
    published = _write_accepted(
        workspace.tasks / "generated", workspace.tasks / "withheld", result, salt=salt, max_steps=12
    )
    task = next(t for t in load_suite("generated", root=workspace.tasks) if t.task_id == task_id)
    payload = {
        "schema": "spark-feedback-task-v1",
        "origin": replay.identity,
        "curriculum": curriculum_id,
        "request": request,
        "dna": dna.to_record(),
        "task": published,
        "task_version": content_digest(asdict(task)),
        "checks_run": list(verdict.checks_run),
    }
    payload["files"] = [file_identity(workspace.tasks / "generated" / (task_id + ".yaml"))] + [
        file_identity(workspace.tasks / "withheld" / (task_id + suffix)) for suffix in (".sh", ".solution.sh")
    ]
    record = replay.authority.put("feedback-task", payload)
    workspace.record("generate", {"accepted": [task_id], "origin": workspace.identity, "feedback": record})
    return record


def verify_feedback_task(replay: ReplayStore, identifier: str) -> dict[str, Any]:
    record = replay.authority.get(identifier, kind="feedback-task")
    payload = record["payload"]
    source = replay.authority.get(payload["curriculum"], kind="curriculum")["payload"]
    replay.experiences(source["settled_inputs"])
    if payload["request"] not in source["requests"] or payload["origin"] != replay.identity:
        raise StageError("task generation is not bound to original measured feedback")
    for artifact in payload["files"]:
        checked_file(artifact)
    return record
