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

## Where the private tree is

`gittensor-model-hub/Spark-Hermes-Withheld`, private: one `<task_id>.sh` per committed task, the
`.notes.md` that explains each, and the master salt in `SALT`.

    git clone git@github.com:gittensor-model-hub/Spark-Hermes-Withheld.git ../spark-hermes-withheld
    export SPARKDISTILL_WITHHELD_ROOT=../spark-hermes-withheld
    export HERMESBENCH_WITHHELD_SALT="$(cat "$SPARKDISTILL_WITHHELD_ROOT/SALT")"
    python -m hermesbench.withheld      # attached 19, unscorable 0

Written down because its absence was expensive. Nothing in this repository named the tree, so
establishing that the withheld half existed at all took a filesystem search of two hosts, a scan of
every tree in eighty commits of history, and finally a look at the organisation's other
repositories -- which is where it was the whole time. The conclusion reached just before that last
step was that the bodies were unrecoverable and nineteen checks would have to be rewritten.

Naming the repository costs nothing. The withheld half is protected by access control, not by the
name being unguessable, and the paragraphs above already say a private tree exists.

The salt lives in that repository and nowhere else, which is one access boundary and one failure
domain shared with the bodies it opens. It wants a copy outside GitHub; losing it makes every
commitment permanently unopenable, and `reveal` has no other input.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def solution_path(task_id: str, root: Path) -> Path:
    """Where a task's reference solution lives, beside its withheld check."""
    return root / f"{task_id}.solution.sh"


def solution_for(task_id: str, *, root: Path | None = None) -> str | None:
    """A known-good solution for one task, from the private tree. `None` when there is none.

    In the private tree and not in the task file, because a reference solution is the answer:
    publishing it beside a public prompt ends the task. It is not part of the graded contract
    either -- nothing scores against it and no commitment covers it -- so unlike `hidden_verify`
    it is read straight off disk rather than checked against a digest.

    What it is for: `suitecheck` asserts that a verifier FAILS an unsolved workspace, which is the
    damaging direction, and its own docstring notes it cannot assert the other one without a
    solution. That gap has a cost on the record. `verify-speedup-claim` ended a `&&` chain with an
    interpreter lookup, so on a clean workspace the earlier `test` failed first and the verifier
    exited 1 -- passing the fresh-workspace assertion while being incapable of ever passing. It
    scored 0/10 on the first real baseline and read as a capability gap in the model.
    """
    resolved = withheld_root(root)
    if resolved is None:
        return None
    path = solution_path(task_id, resolved)
    return path.read_text(encoding="utf-8") if path.is_file() else None


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
            # Left unattached, so `withheld_check_missing` stays true and `unscorable` names it.
            #
            # This used to raise, on the reasoning that a suite scored without a check it is
            # supposed to have reports no overfit signal and reads as a clean result. That
            # reasoning was right about the danger and wrong about where to put the guard: it
            # made a *partial* private tree refuse the entire suite, so nineteen committed tasks
            # could not be re-authored one at a time -- the first one written would abort every
            # run until the last one was. Discovered by trying to do exactly that.
            #
            # The danger is handled where the reading happens instead. `status` reports the
            # unscorable set and exits non-zero, the runner names every unscorable task on
            # stderr before it runs, and `shortcut_sweep` returns UNRESOLVED rather than a pass
            # for an expectation it could not decide. Absence is loud in all three; what it no
            # longer is, is fatal.
            #
            # A body that is present and does NOT match its commitment still raises below. That
            # is the case worth refusing: it means the private tree drifted from what was
            # published, and scoring against it measures a different benchmark than the one named.
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


@dataclass(frozen=True)
class WithheldStatus:
    """What this checkout can and cannot score.

    A record rather than a dict: every other report in this repo is one, and the first version of
    `main` read `len()` off a value typed `object` -- which pyright caught and a reader would not.
    """

    tasks: int
    committed: tuple[str, ...]
    attached: tuple[str, ...]
    unscorable: tuple[str, ...]
    root: str
    salt_length: int
    problem: str = ""

    @property
    def complete(self) -> bool:
        return not self.unscorable and not self.problem

    @property
    def salt_long_enough(self) -> bool:
        # The bound `salted_digest` enforces: a withheld check is a short command from a small
        # space, so a short salt publishes a guessable digest rather than a commitment.
        return self.salt_length >= 16

    def to_record(self) -> dict[str, Any]:
        return {
            "tasks": self.tasks,
            "committed": list(self.committed),
            "attached": list(self.attached),
            "unscorable": list(self.unscorable),
            "root": self.root,
            "root_env": WITHHELD_ROOT_ENV,
            "salt_env": WITHHELD_SALT_ENV,
            # A length is a fact about a secret; the secret is not. This output is meant to be
            # pasteable into an issue.
            "salt_length": self.salt_length,
            "salt_long_enough": self.salt_long_enough,
            "complete": self.complete,
            "problem": self.problem,
        }


def status(tasks: Iterable[Task], *, root: Path | None = None, salt: str = "") -> WithheldStatus:
    """What this checkout can and cannot score, as facts rather than as an absence.

    Answering "is the withheld half present" took a filesystem search, three env vars and a read
    of two modules. It is one call now, because the state it reports is the difference between a
    suite that measures overfitting and one that silently does not -- and the second looks like a
    clean result.
    """
    listed = list(tasks)
    resolved = withheld_root(root)
    salt = salt or os.environ.get(WITHHELD_SALT_ENV, "")
    problem = ""
    # Measured after attaching, not before. Computing `unscorable` on the input would report a
    # complete private tree as scoring nothing -- the answer would be identical whether the tree
    # was there or not, which is the one distinction this function exists to draw.
    resulting = listed
    if resolved is not None:
        try:
            resulting = overlay(listed, root=resolved, salt=salt)
        except WithheldError as exc:
            problem = str(exc)
    return WithheldStatus(
        tasks=len(listed),
        committed=tuple(t.task_id for t in listed if t.hidden_verify_commitment),
        attached=tuple(t.task_id for t in resulting if t.hidden_verify),
        unscorable=tuple(unscorable(resulting)),
        root=str(resolved) if resolved is not None else "",
        salt_length=len(salt),
        problem=problem,
    )


def main(argv: list[str] | None = None) -> int:
    """Exit 0 when every committed task can be scored, 1 otherwise, so a script can gate on it."""
    import argparse
    import json

    from hermesbench.tasks import load_suite

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default="all")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = status(load_suite(args.suite))
    if args.json:
        print(json.dumps(report.to_record(), indent=2, sort_keys=True))
    else:
        print(f"tasks              {report.tasks}")
        print(f"commit to withheld {len(report.committed)}")
        print(f"withheld attached  {len(report.attached)}")
        print(f"{WITHHELD_ROOT_ENV:<18} {report.root or '(unset)'}")
        print(f"{WITHHELD_SALT_ENV:<18} {report.salt_length} chars" if report.salt_length else "salt (unset)")
        if report.problem:
            print(f"problem            {report.problem}")
        if report.unscorable:
            listed = ", ".join(report.unscorable[:8])
            print(
                f"\nUNSCORABLE {len(report.unscorable)}: {listed}{' ...' if len(report.unscorable) > 8 else ''}\n"
                "These publish a commitment whose check is not here, so a run scores only the published\n"
                "verifier -- the half a strategy can fit -- and reports no overfit signal."
            )
        else:
            print("\nevery committed task can be scored")
    return 0 if report.complete else 1


__all__ = [
    "WITHHELD_ROOT_ENV",
    "WITHHELD_SALT_ENV",
    "WithheldError",
    "WithheldStatus",
    "main",
    "solution_for",
    "solution_path",
    "status",
    "overlay",
    "unscorable",
    "withheld_path",
    "withheld_root",
]


if __name__ == "__main__":
    raise SystemExit(main())
