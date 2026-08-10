"""Objective verification of a finished HermesBench episode.

This is the layer the roadmap calls *verified execution*: a claim is worth nothing until
something other than the claimant runs the check. The agent's final message is not
consulted here at all -- only the exit status of the task's own `verify` command, run in
the workspace the agent actually modified.

Same posture SparkProof takes toward datasets, applied to behavior: nothing counts
because the model said so.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from hermesbench.tasks import Task

# How much of a command's output the agent is allowed to see. This is a harness
# parameter, not an implementation detail: change it and the agent observes a different
# world, takes different actions and scores differently, with no diff visible in any task
# file. Defined once here and imported by the runner, so the two cannot drift apart, and
# folded into the harness digest so a change to it makes runs incomparable rather than
# silently comparable.
OBSERVATION_LIMIT = 8000


def sanitize_path(path_value: str, workspace: Path | None) -> str:
    """Drop PATH entries the agent can write to.

    A task sets `PATH: "./bin:$PATH"` so that *the agent* meets a broken tool -- that is
    the failure mode the task exists to create. The same env was also handed to the
    verifier, which put an agent-writable directory at the front of the grader's own
    PATH: an agent could write `bin/tr`, never do the work, and have the grader's own
    `tr` report whatever it liked. Confirmed against `recover-from-bad-command`, where a
    report file containing the word "cheated" passed.

    So the grader's PATH keeps only absolute entries that lie outside the workspace.
    Relative entries resolve against cwd -- which *is* the workspace during verification
    -- and absolute entries under the workspace are writable too.
    """
    root = workspace.resolve() if workspace else None
    keep: list[str] = []
    for entry in path_value.split(os.pathsep):
        if not entry or not entry.startswith("/"):
            continue
        if root is not None:
            resolved = Path(entry).resolve()
            if resolved == root or root in resolved.parents:
                continue
        keep.append(entry)
    return os.pathsep.join(keep)


def resolve_env(
    overrides: dict[str, str] | None,
    *,
    workspace: Path | None = None,
    for_verification: bool = False,
) -> dict[str, str] | None:
    """Overlay task env onto the current environment, expanding `$VAR` references.

    Returns None when there is nothing to override, so subprocesses inherit normally.
    Expansion happens against the live environment, which is what makes
    `PATH: "./bin:$PATH"` prepend instead of clobbering the caller's PATH.

    `for_verification=True` additionally strips agent-writable PATH entries -- see
    `sanitize_path`. The agent still meets the shadowed tool; the grader does not.
    """
    if not overrides:
        return None
    env = dict(os.environ)
    for key, value in overrides.items():
        env[key] = os.path.expandvars(value)
    if for_verification and "PATH" in env:
        env["PATH"] = sanitize_path(env["PATH"], workspace)
    return env


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False

    def to_record(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
        }


def _truncate(text: str) -> str:
    if len(text) <= OBSERVATION_LIMIT:
        return text
    return text[:OBSERVATION_LIMIT] + f"\n... [truncated {len(text) - OBSERVATION_LIMIT} chars]"


def run_command(
    command: str,
    *,
    cwd: Path,
    timeout_s: int,
    env: dict[str, str] | None = None,
) -> VerificationResult:
    """Run a shell command in `cwd` and report how it exited.

    The command comes from the task file (maintainer-authored), not from the model --
    model-issued commands go through `hermesbench.runner`, which is where the sandbox
    requirement lives.
    """
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return VerificationResult(
            passed=False,
            exit_code=None,
            stdout=_truncate(exc.stdout or "" if isinstance(exc.stdout, str) else ""),
            stderr=f"verification timed out after {timeout_s}s",
            timed_out=True,
        )
    return VerificationResult(
        passed=completed.returncode == 0,
        exit_code=completed.returncode,
        stdout=_truncate(completed.stdout),
        stderr=_truncate(completed.stderr),
    )


def verify_task(task: Task, workspace: Path) -> VerificationResult:
    """Run a task's verification command against the agent's finished workspace."""
    return run_command(
        task.verify,
        cwd=workspace,
        timeout_s=task.timeout_s,
        env=resolve_env(task.env, workspace=workspace, for_verification=True),
    )


def run_checkpoints(task: Task, workspace: Path) -> dict[str, bool]:
    """Evaluate every sub-objective against the workspace as it stands right now.

    Called repeatedly during a long-horizon episode to build a pass/fail timeline. Each
    checkpoint gets a shorter timeout than the task's own: sampling happens many times
    per episode, so a slow checkpoint would otherwise dominate the run.
    """
    if not task.checkpoints:
        return {}
    env = resolve_env(task.env, workspace=workspace, for_verification=True)
    timeout = max(10, task.timeout_s // 4)
    return {
        checkpoint.checkpoint_id: run_command(checkpoint.verify, cwd=workspace, timeout_s=timeout, env=env).passed
        for checkpoint in task.checkpoints
    }


def verify_hidden(task: Task, workspace: Path) -> VerificationResult | None:
    """Run the withheld checks. Returns None when the task declares none.

    Kept separate from `verify_task` so the two verdicts can be reported apart: an
    episode that passes the published checks and fails these has optimised to the
    benchmark, and collapsing them into one boolean would erase exactly that signal.
    """
    if not task.has_hidden_tests:
        return None
    return run_command(
        task.hidden_verify,
        cwd=workspace,
        timeout_s=task.timeout_s,
        env=resolve_env(task.env, workspace=workspace, for_verification=True),
    )


def redact_for_release(record: dict, *, salt: str = "") -> dict:
    """Strip withheld checks from a task record before publishing it.

    Returns a copy. Publishing a task whose `hidden_verify` travels with it defeats the
    entire point, and doing this by hand is precisely the step someone forgets.

    With a `salt`, a commitment to the withheld check is published in its place. Deleting
    the key alone leaves a released task carrying no evidence that its withheld check ever
    had a particular value, so a maintainer could substitute a different one between two
    runs and nothing would show it. The commitment closes that without revealing anything.

    **The salt is not optional decoration.** A withheld check is a short shell command
    drawn from a small space -- a near neighbour of the `verify` published alongside it --
    so a bare digest is a verification oracle rather than a commitment: it turns "guess the
    withheld test" from unfalsifiable into a check-your-guess loop. Salted, the digest
    proves two runs used the same check; only revealing the salt later proves *which*.
    """
    public = {k: v for k, v in record.items() if k != "hidden_verify"}
    hidden = record.get("hidden_verify") or ""
    metadata = public.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    metadata["has_hidden_tests"] = bool(hidden)
    if hidden and salt:
        from hermes.harness import salted_digest

        metadata["hidden_verify_commitment"] = salted_digest(hidden, salt)
    public["metadata"] = metadata
    return public


def setup_task(task: Task, workspace: Path) -> VerificationResult | None:
    """Prepare a task workspace. Returns None when the task declares no setup."""
    if not task.setup:
        return None
    workspace.mkdir(parents=True, exist_ok=True)
    return run_command(task.setup, cwd=workspace, timeout_s=task.timeout_s, env=resolve_env(task.env))
