"""Where a validator keeps its rounds between restarts.

    from validator.store import RoundStore
    store = RoundStore()
    store.save(window)
    windows = store.load_all()

`hermes.round` is pure logic with no storage, and `validator.api` held its rounds in a module
dict populated by nothing -- so a live server answered 404 to every round, and a restart during
grading lost the grading. This is the missing half.

## Why the store is not under `datasets/`

`datasets/rounds/` holds round *announcements*, which are meant to be public: `load_round` reads
them out of the base ref precisely so a submitter cannot have written them. A round *window* is
the opposite kind of record. Its snapshot carries every verdict, including verdicts recorded
while the round was still FROZEN, and publishing one of those turns the withheld verifier into a
check-your-guess oracle.

So windows go under `var/rounds/`, which is gitignored, and `store_is_private()` checks that by
asking git rather than by asserting it in a comment. A directory that is only *described* as
private is one commit away from being published, and the commit that does it will look like
housekeeping.

## Why it persists the snapshot rather than the ledger

`RoundWindow.to_record()` withholds `verdicts` until GRADED, by design. Saving through it would
drop every verdict recorded before grading finished, the reload would succeed, and the round
would come back up publishing `verdicts_recorded: 0` as though none had been made -- a silent
loss that looks exactly like an honest empty round. `RoundWindow.snapshot()` exists for this,
and `screen_public_payload` refuses it on sight because it carries a `verdicts` key.

## One file per round, written whole

A round is saved by writing a temporary file and renaming it over the target. A crash midway
through a `write_text` leaves a half-written JSON file that no reload can parse, and the round it
described is then unrecoverable -- whereas `rename` within a filesystem is atomic, so a reader
sees either the previous snapshot or the new one.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from hermes.evidence_json import evidence_object
from hermes.round import RoundError, RoundWindow
from validator.persistence import atomic_write, locked, state_identity

# Not `datasets/`: a window snapshot carries verdicts. See the module docstring.
WINDOW_DIR = Path("var/rounds")


class StoreError(RoundError):
    """A round cannot be saved or loaded."""


def store_is_private(root: Path | None = None) -> bool:
    """Whether git ignores the store directory. Asked, not assumed.

    The snapshot this store writes carries verdicts recorded before grading, so committing one
    would publish exactly what the withheld half exists to withhold. Checked by running
    `git check-ignore` rather than by trusting the `.gitignore` entry to still be there --
    someone reorganising ignore rules has no way to know this one is load-bearing.

    Returns True outside a git repository: there is nothing to publish to.
    """
    target = (root or WINDOW_DIR) / "probe.json"
    try:
        done = subprocess.run(
            ["git", "check-ignore", "-q", str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return True
    if done.returncode == 0:
        return True
    # 1 means "not ignored"; 128 means git could not answer (not a repo, no work tree).
    return done.returncode not in (0, 1)


class RoundStore:
    """File-backed rounds, one JSON snapshot per round id."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        require_private: bool = True,
        mode: str | None = None,
        namespace: str | None = None,
    ) -> None:
        self.root = root or WINDOW_DIR
        # Refused at construction rather than at the first save. A validator that has already
        # graded a round and then cannot store it has done the expensive part twice.
        if require_private and not store_is_private(self.root):
            raise StoreError(
                f"{self.root} is not gitignored, and a round snapshot carries verdicts recorded "
                "before grading finished. Committing one publishes the withheld half of the "
                "benchmark. Add it to .gitignore, or pass require_private=False if this really is "
                "a scratch directory outside any repository."
            )

        try:
            self.identity = state_identity(self.root, mode=mode, namespace=namespace)
        except ValueError as exc:
            raise StoreError(str(exc)) from exc

    def lock(self, round_id: str) -> Any:
        return locked(self.path_for(round_id).with_suffix(".lock"))

    def path_for(self, round_id: str) -> Path:
        if not round_id or "/" in round_id or round_id in (".", ".."):
            # A round id reaches this from a request in some deployments, and `Path / ".."`
            # escapes the store without complaining.
            raise StoreError(f"unusable round id {round_id!r}: it must be a single path segment")
        return self.root / f"{round_id}.json"

    def save(self, window: RoundWindow) -> Path:
        """Write one round's private snapshot. Atomic: temp file, then rename."""
        path = self.path_for(window.round_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        origin = window.store_identity or self.identity
        if origin != self.identity:
            raise StoreError("round belongs to another state root or trust domain")
        record = window.snapshot()
        record["store_identity"] = self.identity
        record["admissions"] = getattr(window, "admissions", {})
        if window.assignment is not None:
            record["assignment"] = window.assignment.to_record(reveal_seed=True)
        atomic_write(path, (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
        return path

    def load(self, round_id: str, *, assignment: Any = None, contract: Any = None) -> RoundWindow:
        path = self.path_for(round_id)
        if not path.is_file():
            raise StoreError(f"no stored round at {path}")
        try:
            record = evidence_object(path.read_bytes())
        except ValueError as exc:
            raise StoreError(f"invalid round snapshot: {exc}") from exc
        origin = record.get("store_identity")
        if origin is not None and origin != self.identity:
            raise StoreError("round belongs to another state root or trust domain")
        if "assignment" in record:
            from hermes.seed import Round

            try:
                stored_assignment = Round.from_record(record["assignment"])
            except ValueError as exc:
                raise StoreError(f"invalid stored assignment: {exc}") from exc
            if assignment is not None and (
                type(assignment) is not Round
                or assignment.commitment != stored_assignment.commitment
                or assignment.state != stored_assignment.state
            ):
                raise StoreError("supplied assignment differs from stored assignment")
            assignment = stored_assignment
        window = RoundWindow.from_snapshot(record, assignment=assignment, contract=contract)
        window.store_identity = self.identity
        window.admissions = record.get("admissions", {})
        return window

    def round_ids(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.stem for p in self.root.glob("*.json"))

    def load_all(self) -> tuple[dict[str, RoundWindow], list[tuple[str, str]]]:
        """Every stored round, plus the ones that would not load and why.

        Returns rather than raises on a bad file. One unreadable round must not stop a validator
        from serving the others: the alternative is a single corrupt snapshot taking the whole
        server down, which converts a recoverable problem into an outage.
        """
        loaded: dict[str, RoundWindow] = {}
        failed: list[tuple[str, str]] = []
        for round_id in self.round_ids():
            try:
                loaded[round_id] = self.load(round_id)
            except (StoreError, RoundError, ValueError, KeyError, TypeError) as exc:
                failed.append((round_id, f"{type(exc).__name__}: {exc}"))
        return loaded, failed


__all__ = ["WINDOW_DIR", "RoundStore", "StoreError", "store_is_private"]
