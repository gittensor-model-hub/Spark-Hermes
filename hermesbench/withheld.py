"""Withheld checks, kept out of the public suite and merged back at run time.

The tasks have to be public. Miners compete on them, so a competition whose tasks nobody
can read is not a competition. What must not be public is the *withheld check* -- the one
that decides the real outcome, and the only reason `overfit_rate` measures anything. A
model can be optimised against a verifier it can read; that is what "learned the benchmark
rather than the job" means, and a published withheld check is just a second published one.

So the split runs through the task file, not around it:

    public repo    the task -- prompt, setup, tools, budgets, `verify`, and a salted
                   commitment to the withheld check
    private tree   `<root>/<task_id>.sh`, the withheld check itself

`overlay` merges the second onto the first and **verifies every body against its published
commitment**. Without that check the private tree could drift from what was published and
nothing would show it -- which is the failure the commitment exists to prevent, arriving
from the maintainer's side instead of the miner's.

## Why a commitment and not just a missing key

Deleting `hidden_verify` before publishing leaves a task carrying no evidence that its
withheld check ever had a particular value. A maintainer could then substitute a different
one between two runs and no reader could tell. The commitment closes that while revealing
nothing.

It is salted, and the salt is not decoration. A withheld check is a short shell command
drawn from a small space -- a near neighbour of the `verify` published beside it -- so a
bare digest is a check-your-guess oracle rather than a commitment. `hermes.harness.
salted_digest` refuses a salt under 16 characters for that reason.

## What a checkout without the private tree must do

Report that it cannot score, never that there is nothing to score. `Task.declares_hidden_
tests` stays true from the commitment alone; `has_hidden_tests` goes false because the body
is absent. Anything reading only the second would turn `overfit_rate` from *unavailable*
into a confident zero, and a confident zero on the metric that detects benchmark gaming is
worse than no number at all.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from hermesbench.tasks import Task, TaskError

WITHHELD_ROOT_ENV = "SPARKDISTILL_WITHHELD_ROOT"
WITHHELD_SALT_ENV = "HERMESBENCH_WITHHELD_SALT"


class WithheldError(TaskError):
    """A withheld check is missing, unreadable, or does not match its published commitment."""


def withheld_root(root: Path | None = None) -> Path | None:
    """The private tree of withheld checks, or None when this checkout has none."""
    if root is not None:
        return root
    configured = os.environ.get(WITHHELD_ROOT_ENV, "").strip()
    return Path(configured).expanduser() if configured else None


def withheld_path(task_id: str, root: Path) -> Path:
    return root / f"{task_id}.sh"


def overlay(tasks: Iterable[Task], *, root: Path | None = None, salt: str = "") -> list[Task]:
    """Attach each task's withheld check from the private tree.

    Returns tasks unchanged when no private tree is configured -- a public checkout is a
    legitimate state, not an error, and it is the state most contributors will be in.

    Every attached body is checked against the task's published commitment. A mismatch is
    fatal rather than a warning: a withheld check that is not the one the task committed to
    is scoring a different benchmark than the one that was published, and the whole point
    of the commitment is that this cannot happen quietly.
    """
    resolved = withheld_root(root)
    if resolved is None:
        return list(tasks)
    if not resolved.is_dir():
        raise WithheldError(f"{WITHHELD_ROOT_ENV} points at {resolved}, which is not a directory")

    salt = salt or os.environ.get(WITHHELD_SALT_ENV, "")
    out: list[Task] = []
    for task in tasks:
        path = withheld_path(task.task_id, resolved)
        if not path.is_file():
            if task.hidden_verify_commitment:
                raise WithheldError(
                    f"{task.task_id} commits to a withheld check but {path} does not exist; a suite "
                    "scored without it reports no overfit signal, which reads as a clean result"
                )
            out.append(task)
            continue

        body = path.read_text(encoding="utf-8")
        commitment = task.hidden_verify_commitment
        if commitment:
            if not salt:
                raise WithheldError(
                    f"{task.task_id} publishes a commitment but {WITHHELD_SALT_ENV} is unset, so the "
                    "withheld check cannot be verified against it. Attaching it unchecked would let "
                    "the private tree drift from what was published."
                )
            from hermes.harness import derive_task_salt, salted_digest

            # The per-task salt derived from the master, never the master itself. Opening one
            # spent task must not unseal every task still sealed -- see `derive_task_salt`.
            if salted_digest(body, derive_task_salt(salt, task.task_id)) != commitment:
                raise WithheldError(
                    f"{task.task_id}: the withheld check at {path} does not match the commitment "
                    "published with the task. Either the private tree changed or the task did; "
                    "either way this would score a different benchmark than the published one."
                )
        out.append(_with_hidden(task, body))
    return out


def _with_hidden(task: Task, body: str) -> Task:
    import dataclasses

    return dataclasses.replace(task, hidden_verify=body)


def unscorable(tasks: Iterable[Task]) -> list[str]:
    """Task ids that declare a withheld check this checkout cannot run.

    Reported rather than raised so a public checkout can still execute the suite and say
    honestly what its numbers do and do not cover.
    """
    return [t.task_id for t in tasks if t.withheld_check_missing]


__all__ = [
    "WITHHELD_ROOT_ENV",
    "WITHHELD_SALT_ENV",
    "WithheldError",
    "overlay",
    "unscorable",
    "withheld_path",
    "withheld_root",
]
