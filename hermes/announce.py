"""Write a round announcement to `datasets/rounds/`, so the gate has something to read.

    python -m hermes.announce commit --round r-042 --tasks a,b,c --miners alice,bob
    python -m hermes.announce open   --round r-042
    python -m hermes.announce close  --round r-042
    python -m hermes.announce show   --round r-042

`hermes.seed` computes an assignment and `eval.rollout_track.check_scope` recomputes it to
decide whether a miner was owed the work. Between them there was nothing that *published* one:
`ROUNDS_DIR = Path("datasets/rounds")` was read by `eval.rollout_track_cli.load_round` and
written by nothing, and the directory did not exist. So every rollout submission failed with
"round ... has no announcement in the base ref; it was never opened" -- the gate failing closed,
correctly, on a record no code path could produce.

## Commit then reveal, and why it is two commands rather than one

`commit` writes the announcement WITHOUT the seed. `open` adds the seed to the same file.

The seed decides who draws which task, so a seed a miner can predict is a seed a miner can
grind against: register identities until the assignment hands you the tasks you have already
solved. Publishing the digest first fixes the assignment before anyone knows what it is, and
publishing the value afterwards makes it computable by everyone at once. Collapsing the two
into a single command would leave nothing between "the round exists" and "the assignment is
known", which is the whole property.

`show` prints the assignment and the coverage. Rendezvous hashing balances in expectation
rather than exactly, and a round where one miner drew most of the pool is worth seeing before
the work starts rather than after.

## Read from the base ref, written to a tracked file

`load_round` reads the announcement with `git show <base>:datasets/rounds/<id>.json`,
deliberately from the base rather than the head -- a record read from the submitter's branch is
a record the submitter could have written, and the assignment it encodes is the thing being
checked. That only works if the file is committed, so this writes into the working tree and
leaves the commit to the operator. Nothing here touches git: a tool that committed on your
behalf would make "the announcement came from the base ref" a claim about its own behaviour.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

# The one the gate reads. Imported rather than re-declared: two spellings of this path is how
# the writer and the reader end up disagreeing about where a round lives.
from eval.rollout_track import ROUNDS_DIR
from hermes.evidence_json import evidence_object
from hermes.seed import CLOSED, COMMITTED, OPEN, Round, SeedError, coverage, duplicated
from hermes.seed import open_round as seed_round


def announcement_path(round_id: str, root: Path | None = None) -> Path:
    return (root or ROUNDS_DIR) / f"{round_id}.json"


def _load(round_id: str, root: Path | None = None) -> dict[str, Any]:
    path = announcement_path(round_id, root)
    if not path.is_file():
        raise SeedError(f"no announcement at {path}; commit the round first")
    try:
        record = evidence_object(path.read_bytes())
    except ValueError as exc:
        raise SeedError(f"invalid round announcement: {exc}") from exc
    if record.get("round_id") != round_id:
        raise SeedError("announcement round_id differs from requested round")
    return record


def _write(record: dict[str, Any], round_id: str, root: Path | None = None) -> Path:
    path = announcement_path(round_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def commit_round(
    *,
    round_id: str,
    task_ids: list[str],
    miner_ids: list[str],
    replicas: int = 1,
    seed: str = "",
    root: Path | None = None,
) -> tuple[Path, Round]:
    """Publish the commitment: the round exists, the assignment is fixed, the seed is not out.

    The seed is generated here and held back rather than generated at `open`. Generating it
    later would mean the commitment published now could not be a commitment to anything -- there
    would be no value yet for the digest to bind.
    """
    path = announcement_path(round_id, root)
    if path.exists():
        raise SeedError(
            f"{path} already exists. Rewriting an announcement changes an assignment miners may "
            "already be working against, so a re-run is refused rather than allowed to overwrite."
        )
    round_ = replace(
        seed_round(round_id, task_ids, miner_ids, replicas=replicas, seed=seed),
        state=COMMITTED,
    )
    # Seed withheld. `Round.from_record` refuses to rebuild from this file for exactly that
    # reason, so a validator cannot accidentally compute an assignment that is not public yet.
    return _write(round_.to_record(reveal_seed=False), round_id, root), round_


def open_seed(*, round_id: str, seed: str, root: Path | None = None) -> tuple[Path, Round]:
    """Reveal the seed the commitment already bound, moving the round to OPEN.

    The seed is checked against the published commitment before anything is written. Without
    that, `open` would accept any seed at all and the commit-reveal would prove nothing: a
    validator could publish one digest and later reveal whatever value produced a convenient
    assignment.
    """
    record = _load(round_id, root)
    if record.get("state") != COMMITTED:
        raise SeedError(
            f"round {round_id} is {record.get('state')!r}; only a COMMITTED round can have its seed "
            "revealed. Re-revealing would let an assignment change after miners saw it."
        )
    if "seed" in record and record["seed"] != seed:
        raise SeedError("supplied seed differs from the seed already recorded")
    candidate = replace(Round.from_record({**record, "seed": seed}), state=OPEN)
    return _write(candidate.to_record(reveal_seed=True), round_id, root), candidate


def close_round(*, round_id: str, root: Path | None = None) -> tuple[Path, dict[str, Any]]:
    """Mark the round closed. It no longer owes anyone work.

    `check_scope` already refuses a submission naming a CLOSED round, so this is the switch
    that turns that refusal on rather than a bookkeeping note.
    """
    record = _load(round_id, root)
    if record.get("state") == CLOSED:
        Round.from_record(record)
        return announcement_path(round_id, root), record
    if record.get("state") != OPEN:
        raise SeedError(
            f"round {round_id} is {record.get('state')!r}; closing a round that never opened would "
            "record a window nobody could submit to as though it had run"
        )
    Round.from_record(record)
    record["state"] = CLOSED
    return _write(record, round_id, root), record


def describe(round_id: str, root: Path | None = None) -> dict[str, Any]:
    """The assignment and its balance, or just the envelope while the seed is withheld."""
    record = _load(round_id, root)
    if record.get("state") == COMMITTED:
        return {
            "round_id": record.get("round_id"),
            "state": record.get("state"),
            "commitment": record.get("commitment"),
            "tasks": len(record.get("task_ids") or ()),
            "miners": len(record.get("miner_ids") or ()),
            "assignment": "not computable yet: the seed is committed and not revealed",
        }
    round_ = Round.from_record(record)
    return {
        "round_id": round_.round_id,
        "state": round_.state,
        "commitment": round_.commitment,
        "assignments": [
            {"task_id": a.task_id, "miner_id": a.miner_id, "replica": a.replica} for a in round_.assignments()
        ],
        "coverage": coverage(round_),
        "cross_checked": [d.to_record() for d in duplicated(round_)],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["commit", "open", "close", "show"])
    parser.add_argument("--round", required=True, dest="round_id")
    parser.add_argument("--tasks", default="", help="comma-separated task ids (commit only)")
    parser.add_argument("--miners", default="", help="comma-separated miner ids (commit only)")
    parser.add_argument("--replicas", type=int, default=1, help="how many miners each task goes to")
    parser.add_argument("--seed", default="", help="the seed to reveal (open only)")
    parser.add_argument("--rounds-dir", type=Path, default=None, help="override datasets/rounds")
    args = parser.parse_args(argv)

    try:
        if args.action == "commit":
            tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
            miners = [m.strip() for m in args.miners.split(",") if m.strip()]
            path, round_ = commit_round(
                round_id=args.round_id,
                task_ids=tasks,
                miner_ids=miners,
                replicas=args.replicas,
                root=args.rounds_dir,
            )
            print(f"wrote {path}")
            print(f"commitment: {round_.commitment}")
            # The one thing this tool cannot recover. The announcement deliberately omits it, so
            # a seed lost between commit and open is a round that can never be opened.
            print(f"SEED (keep this; `open` needs it and the file does not hold it): {round_.seed}")
        elif args.action == "open":
            if not args.seed:
                print("hermes.announce: --seed is required to open a round", file=sys.stderr)
                return 2
            path, round_ = open_seed(round_id=args.round_id, seed=args.seed, root=args.rounds_dir)
            print(f"wrote {path}")
            print(json.dumps(describe(args.round_id, args.rounds_dir), indent=2, sort_keys=True))
        elif args.action == "close":
            path, _ = close_round(round_id=args.round_id, root=args.rounds_dir)
            print(f"wrote {path}")
        else:
            print(json.dumps(describe(args.round_id, args.rounds_dir), indent=2, sort_keys=True))
    except SeedError as exc:
        print(f"hermes.announce: {exc}", file=sys.stderr)
        return 2

    if args.action in ("commit", "open", "close"):
        print(
            "commit the file: the gate reads the announcement from the BASE ref, because a record "
            "read from a submitter's branch is one the submitter could have written.",
            file=sys.stderr,
        )
    return 0


__all__ = ["announcement_path", "close_round", "commit_round", "describe", "main", "open_seed"]


if __name__ == "__main__":
    raise SystemExit(main())
