"""Split the suite into a public half and a private half.

    python -m hermesbench.split_suite --withheld-out ../spark-hermes-withheld

Rewrites every task in place with `hidden_verify` replaced by a salted commitment, and
writes the check bodies to `<withheld-out>/<task_id>.sh`. After this the tasks are safe to
publish and the private tree is what a validator points `SPARKDISTILL_WITHHELD_ROOT` at.

Run once, deliberately, and commit the two results to their two repositories. It is not
idempotent in the useful direction: a second run over already-redacted tasks finds nothing
to move and says so rather than overwriting the private tree with empty files.

**This is a one-way door for the checks it moves.** Once a task is published with its
withheld check redacted, the check is only as secret as the private tree -- and a check
that was ever in a public git history is spent, because history is not rewritten by a
later redaction. `--check` reports what would move without moving it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

from hermesbench.tasks import TASKS_ROOT
from hermesbench.withheld import WITHHELD_SALT_ENV, withheld_path


def redact_text(text: str) -> tuple[str, str]:
    """Cut the withheld check out of a task file. Returns (public, notes).

    A text operation, not a YAML round-trip. `yaml.safe_dump` would reformat the whole
    file: on the real suite it dropped 39 comment lines from a single task and folded the
    prompt into one escaped string. Those comments are where each trap is explained, and a
    redaction that guts the task in order to hide one field has traded the wrong thing.

    **The comment block above `hidden_verify` travels with it.** It describes what the
    withheld check tests -- one of them names "the last diverging row and the count of
    stable half-cent rows", which is the answer. Publishing the prose while hiding the
    command withholds almost nothing.
    """
    lines = text.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if line.startswith("hidden_verify:")), None)
    if start is None:
        return text, ""

    # Walk back over the contiguous comment block that introduces it, stopping at the first
    # blank or non-comment line so an unrelated comment further up is left alone.
    head = start
    while head > 0 and lines[head - 1].lstrip().startswith("#"):
        head -= 1

    # The body runs until the next top-level key. `checkpoints:` follows it in two tasks,
    # so "to end of file" would swallow a field the public task needs.
    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.strip() and not line[0].isspace() and not line.startswith("#"):
            end = index
            break

    public = "".join(lines[:head] + lines[end:]).rstrip() + "\n"
    notes = "".join(lines[head:start])
    return public, notes


def drop_commitment(text: str) -> str:
    """Remove an existing `metadata.hidden_verify_commitment`, and the block if that empties it.

    Needed because splitting is no longer a once-ever operation. Every task in the suite is
    already redacted and carries a commitment, so re-authoring a withheld check and sealing it
    again appends a *second* `metadata:` block -- and the first one holds the stale commitment.

    PyYAML takes the last duplicate key, so the value happens to come out right here and the
    file is wrong: another parser, or a stricter mode, takes the first, which commits the task
    to the body that was lost. A file that parses correctly under exactly one reader is not a
    published artifact. The blocks also accumulate, one per re-split.

    A text operation for the reason `redact_text` is: `yaml.safe_dump` reformats the whole file
    and drops the comments that explain each trap.
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("metadata:"):
            block = index + 1
            while block < len(lines) and (not lines[block].strip() or lines[block][0].isspace()):
                block += 1
            kept = [
                inner for inner in lines[index + 1 : block] if inner.strip() and "hidden_verify_commitment" not in inner
            ]
            # Only re-emit `metadata:` if something other than the commitment lived under it.
            # Today nothing does, and a bare `metadata:` with no children parses as None, which
            # `Task.from_record` would read as a task declaring no withheld check at all.
            if kept:
                out.append(line)
                out.extend(kept)
            index = block
            continue
        out.append(line)
        index += 1
    return "".join(out).rstrip() + "\n"


def split(
    *,
    withheld_out: Path,
    tasks_root: Path | None = None,
    salt: str,
    dry_run: bool = False,
) -> tuple[list[str], list[str]]:
    """Redact every task and write its check out. Returns (moved, already_redacted)."""
    from hermes.harness import derive_task_salt, salted_digest

    root = tasks_root or TASKS_ROOT
    moved: list[str] = []
    already: list[str] = []

    # The task files directly, not `load_suite`. A rewriting tool needs the path it is
    # rewriting, and `Task` does not carry one -- `origin` is a `from_record` parameter
    # used for error messages and is not kept on the instance.
    for source in sorted(
        p for d in sorted(root.iterdir()) if d.is_dir() for p in [*d.glob("*.yaml"), *d.glob("*.yml")]
    ):
        record = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            continue
        task_id = str(record.get("task_id") or source.stem)
        body = str(record.get("hidden_verify") or "")
        if not body.strip():
            already.append(task_id)
            continue

        if not dry_run:
            withheld_out.mkdir(parents=True, exist_ok=True)
            public_text, notes = redact_text(source.read_text(encoding="utf-8"))
            # The .sh holds the check EXACTLY as `hidden_verify` parsed it, so the digest
            # the overlay recomputes is the digest the split committed to. The prose that
            # explained it goes beside it rather than inside it -- folding it in would make
            # the commitment change every time a comment was reworded.
            withheld_path(task_id, withheld_out).write_text(body, encoding="utf-8")
            if notes.strip():
                withheld_path(task_id, withheld_out).with_suffix(".notes.md").write_text(notes, encoding="utf-8")
            # The commitment goes in metadata, where `Task.hidden_verify_commitment` reads
            # it, so a redacted task still *declares* a withheld check. Dropping the key
            # without leaving one would make the task look like it never had a check, and
            # a suite of those reports a clean overfit rate rather than an unavailable one.
            #
            # Committed over the BODY only, so the digest is stable against edits to the
            # explanatory comment that travels beside it.
            #
            # Under the PER-TASK salt derived from the master, never the master itself. This
            # line is why the split path had to change too: it writes commitments without
            # going through `redact_for_release`, so fixing that function alone left every
            # freshly split task committing under one shared salt -- and one shared salt means
            # the first time a spent task's salt is published, every commitment still sealed
            # becomes brute-forceable. See `hermes.harness.derive_task_salt`.
            commitment = salted_digest(body, derive_task_salt(salt, task_id))
            source.write_text(
                drop_commitment(public_text) + f"\nmetadata:\n  hidden_verify_commitment: {commitment}\n",
                encoding="utf-8",
            )
        moved.append(task_id)

    return moved, already


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--withheld-out", type=Path, required=True, help="private tree to write check bodies into")
    parser.add_argument("--tasks-root", type=Path, default=None)
    parser.add_argument("--check", action="store_true", help="report what would move, change nothing")
    args = parser.parse_args(argv)

    salt = os.environ.get(WITHHELD_SALT_ENV, "")
    if not args.check and len(salt) < 16:
        # Checked before anything is written. Discovering it afterwards means the private
        # tree exists and the tasks are half-redacted, which is the worst of both states.
        print(
            f"{WITHHELD_SALT_ENV} must be set to at least 16 characters before splitting.\n"
            "An unsalted commitment to a withheld check is a check-your-guess oracle, not a "
            "commitment: these are short shell commands drawn from a small space.",
            file=sys.stderr,
        )
        return 2

    moved, already = split(withheld_out=args.withheld_out, tasks_root=args.tasks_root, salt=salt, dry_run=args.check)
    verb = "would move" if args.check else "moved"
    print(f"{verb} {len(moved)} withheld check(s); {len(already)} task(s) already redacted or declaring none")
    for task_id in moved:
        print(f"  {task_id}")
    if moved and not args.check:
        print(
            f"\nCommit the tasks to the public repository and {args.withheld_out} to the private one.\n"
            f"Validators set {WITHHELD_SALT_ENV} and SPARKDISTILL_WITHHELD_ROOT to score the full suite.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
