"""Executing the lazy strategy, to find out whether the trap is real.

The gate everyone reaches for first is: build the workspace, confirm both checks fail, apply
the known-good solution, confirm both checks pass, repeat a few times to confirm determinism,
seal. That gate proves a task is **well-formed**. It says nothing at all about whether the
task is **trivial**, and the difference has cost this corpus twice:

  * `tc-log-rotation-order` shipped a NOTES.md hint -- "read the directory top to bottom,
    with app.log last" -- and because the peak event fell inside the last segment, that
    instruction alone produced the graded-correct answer. 86% of plausible orderings passed.
  * `lh-i18n-catalog-parity` listed its five objectives so that the prompt's own stated order
    was one of the 15 orderings (out of 120) that never drift. A single forward pass, each
    objective done once and never revisited, satisfied all five checkpoints and the published
    check.

Both had a fresh workspace that failed and a correct solution that passed. Both were
perfectly deterministic. Neither defect is reachable from those facts. The only thing that
finds them is running the lazy strategy and seeing that it wins -- which is what this module
does, and what turns a trap described in a comment into an assertion that executes.

This was checked against the real defect rather than argued. Replaying the pre-fix objective
ordering of `lh-i18n-catalog-parity` through `run_shortcut` -- logout appended, dead keys
pruned, sort applied afterwards (silently repairing the unsorted append), checksums refreshed
last -- returns:

    verdict=ESCAPED  public_passed=True  is_failure=True
    "the shortcut passes the published check, so this trap is decorative"

So the gate would have blocked that task from shipping. Note what the reorder did and did not
do: 15 of the 120 orderings still finish clean, so a careful agent can still pass in one
forward pass. What changed is that the prompt no longer hands one of those 15 to a reader who
follows it literally.

## Why the declaration lives in the task file

A generic sweep cannot guess the lazy strategy for an arbitrary task; that is task-specific
knowledge held by whoever wrote the trap. So the author writes it down as a script, and this
module runs it. The useful consequence is that a task claiming a trap in a comment while
declaring no shortcut is visibly an untested claim, which is precisely the state both defective
tasks were in.

## What a missing withheld check means here

`passes_public_fails_hidden` is the more valuable expectation -- an overfit path, caught only
by the withheld check, which is what `overfit_rate` measures. It is also undecidable in a
checkout that does not hold the withheld bodies, which is every public clone. Those come back
`UNRESOLVED` rather than passing, because a sweep that reported success for a check it could
not run would be worse than no sweep: it would certify the exact property it failed to test.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermesbench.tasks import FAILS_PUBLIC, PASSES_PUBLIC_FAILS_HIDDEN, Shortcut, Task
from hermesbench.verify import resolve_env, run_command

# A shortcut behaved as the task claimed: the trap caught it.
CAUGHT = "caught"
# The shortcut won. The trap is decorative and the task is easier than it claims.
ESCAPED = "escaped"
# Could not be decided here -- almost always a withheld check this checkout does not hold.
UNRESOLVED = "unresolved"
# The shortcut script itself failed to run. Not a verdict about the task.
BROKEN = "broken"


@dataclass(frozen=True)
class ShortcutOutcome:
    task_id: str
    shortcut_id: str
    expectation: str
    verdict: str
    detail: str = ""
    public_passed: bool | None = None
    hidden_passed: bool | None = None

    @property
    def is_failure(self) -> bool:
        """Whether this outcome should fail a build.

        `UNRESOLVED` deliberately does not: a public clone cannot run withheld checks, and
        making that a build failure would mean the public suite could never be green. It is
        reported instead, and the private verifier is where it becomes decidable.
        """
        return self.verdict in (ESCAPED, BROKEN)

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "shortcut_id": self.shortcut_id,
            "expectation": self.expectation,
            "verdict": self.verdict,
            "detail": self.detail,
            "public_passed": self.public_passed,
            "hidden_passed": self.hidden_passed,
        }


def run_shortcut(task: Task, shortcut: Shortcut, workspace: Path) -> ShortcutOutcome:
    """Build a fresh workspace, apply one lazy strategy, and report whether it was caught.

    The workspace is rebuilt from `task.setup` rather than reused, because a shortcut that
    mutates the tree would otherwise contaminate the next one -- and the contamination would
    read as a stronger trap, which is the wrong direction to be wrong in.
    """
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    def outcome(verdict: str, detail: str, **kw: Any) -> ShortcutOutcome:
        return ShortcutOutcome(
            task_id=task.task_id,
            shortcut_id=shortcut.shortcut_id,
            expectation=shortcut.expectation,
            verdict=verdict,
            detail=detail,
            **kw,
        )

    if task.setup:
        prepared = run_command(task.setup, cwd=workspace, timeout_s=task.timeout_s, env=resolve_env(task.env))
        if not prepared.passed:
            return outcome(BROKEN, f"task setup failed: {(prepared.stderr or prepared.stdout)[:400]}")

    applied = run_command(
        shortcut.apply,
        cwd=workspace,
        timeout_s=task.timeout_s,
        # The lazy strategy stands in for an AGENT, so it gets the agent's environment --
        # not the grader's. Handing it `for_verification=True` would give it a sanitized PATH
        # and PYTHONSAFEPATH, i.e. protections the real agent never has, and a shortcut that
        # only fails under those is not actually caught.
        env=resolve_env(task.env, workspace=workspace),
    )
    if not applied.passed:
        return outcome(
            BROKEN,
            f"the shortcut script itself failed, so it tested nothing: {(applied.stderr or applied.stdout)[:400]}",
        )

    grader_env = resolve_env(task.env, workspace=workspace, for_verification=True)
    public = run_command(task.verify, cwd=workspace, timeout_s=task.timeout_s, env=grader_env)

    if shortcut.expectation == FAILS_PUBLIC:
        if public.passed:
            return outcome(
                ESCAPED,
                "the shortcut passes the published check, so this trap is decorative: the task "
                "is solvable without doing the work it claims to require",
                public_passed=True,
            )
        return outcome(CAUGHT, "the published check rejects it", public_passed=False)

    if shortcut.expectation == PASSES_PUBLIC_FAILS_HIDDEN:
        if not public.passed:
            return outcome(
                ESCAPED,
                "declared as an overfit path but the PUBLISHED check already rejects it, so it "
                "measures nothing about overfitting; either the expectation or the script is wrong",
                public_passed=False,
            )
        if not task.has_hidden_tests:
            return outcome(
                UNRESOLVED,
                "passes the published check as declared, but this checkout holds no withheld "
                "body, so whether the withheld check catches it cannot be decided here",
                public_passed=True,
            )
        hidden = run_command(task.hidden_verify, cwd=workspace, timeout_s=task.timeout_s, env=grader_env)
        if hidden.passed:
            return outcome(
                ESCAPED,
                "passes BOTH the published and the withheld check, so nothing distinguishes it "
                "from a correct solution and overfit_rate cannot fire on it",
                public_passed=True,
                hidden_passed=True,
            )
        return outcome(
            CAUGHT,
            "passes the published check and the withheld check rejects it, which is the overfit "
            "signal this expectation exists to prove",
            public_passed=True,
            hidden_passed=False,
        )

    return outcome(BROKEN, f"unknown expectation {shortcut.expectation!r}")


def sweep(tasks: list[Task], root: Path) -> list[ShortcutOutcome]:
    """Run every declared shortcut for every task. Workspaces are named per shortcut."""
    outcomes: list[ShortcutOutcome] = []
    for task in tasks:
        for shortcut in task.shortcuts:
            outcomes.append(run_shortcut(task, shortcut, root / f"{task.task_id}#{shortcut.shortcut_id}"))
    return outcomes


def undeclared(tasks: list[Task]) -> list[str]:
    """Task ids that claim a trap in prose but declare no shortcut to prove it.

    Heuristic and deliberately so: it reads the words the authors of this corpus actually use
    when describing a trap. Its job is to stop a NEW task shipping an untested claim the quiet
    way, which is how both known defects arrived -- not to be a proof of anything.
    """
    words = ("trap", "shortcut", "cheapest wrong", "lazy", "decorative", "giveaway")
    flagged = []
    for task in tasks:
        if task.shortcuts:
            continue
        haystack = " ".join([task.prompt, *(c.description for c in task.checkpoints)]).lower()
        if any(word in haystack for word in words):
            flagged.append(task.task_id)
    return flagged


__all__ = [
    "BROKEN",
    "CAUGHT",
    "ESCAPED",
    "UNRESOLVED",
    "ShortcutOutcome",
    "run_shortcut",
    "sweep",
    "undeclared",
]
