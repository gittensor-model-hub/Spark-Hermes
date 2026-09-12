"""A snapshot of the board, for somewhere that cannot call the validator.

    python -m validator.board --out docs/board/state.json

The board at `GET /` reads the validator's own endpoints, which is right for whoever runs the
validator and useless anywhere else. GitHub Pages serves static files from a different origin: the
page's relative fetches would hit nothing, and it would honestly report a validator that answered
404 while showing an empty round.

Three ways to fix that, and this is the third.

**Live cross-origin** needs CORS on the validator, a publicly reachable HTTPS validator -- Pages is
HTTPS, so an http:// one is blocked as mixed content -- and a URL for the page to point at. That
last part undoes what serving from the validator's own origin bought: no second place to configure
a URL, and no way to aim the board at a validator whose receipts nobody published. A `?validator=`
parameter means anyone can render another validator's numbers under this project's name.

**A proxy** works and inserts a trusted intermediary between a reader and the thing being audited,
which is the wrong direction for a project whose whole argument is that the validator's word should
be checkable without trusting it.

**A published file** is what this is, and it is what the rest of the design already does: audit
bundles are files, receipts are a committed JSONL, challenge packets are committed JSON. A reader
sees what the validator published rather than what a page was told to fetch.

## Why this is not a workflow

`var/` is gitignored, so the round store does not exist in CI. A workflow that generated this would
be inventing state -- so it is produced where the store lives, by whoever runs the validator, and
the file carries its own timestamp so nothing downstream has to guess how old it is.

## Staleness is the cost, so it is recorded rather than hidden

`generated_at` goes in the file and the board renders its age. A snapshot that looks live is worse
than no snapshot: it turns "the round moved on and nobody republished" into a reader's wrong belief
about the current state. The board says which mode it is in, always.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from hermes.round import RECEIPT_FIELDS, ScoreLeakError, refuse_withheld_body
from validator.intake import Intake
from validator.store import RoundStore

SCHEMA = "spark-board-v1"

# The page itself. Published beside the snapshot rather than maintained as a second copy: one board
# reads the live API when there is one and this file when there is not, so a static host and a
# validator show the same thing. Two hand-kept copies would agree today and diverge on the first
# change, and the copy downstream of the divergence is the one a reader sees.
PAGE = Path(__file__).with_name("dashboard.html")


class BoardError(ValueError):
    """A snapshot cannot be built from what was supplied."""


def snapshot(*, store: RoundStore, intake: Intake | None = None, now: float | None = None) -> dict[str, Any]:
    """The newest round's public view, plus the public receipts.

    Screened through `hermes.round.refuse_withheld_body`, the same call the API's read paths make.
    That is not belt-and-braces: this file is written to a *public* directory by a process that also
    holds the private store, so it is the one place in the system where a withheld body could be
    published by a path that never went through the API at all.
    """
    rounds = store.round_ids()
    if not rounds:
        raise BoardError(f"no rounds in {store.root}; there is nothing to publish")

    newest = max((store.load(r) for r in rounds), key=lambda w: w.opened_at)
    view = refuse_withheld_body(newest.public_view(), where=f"board snapshot of {newest.round_id}")

    receipts = [r for r in (intake or Intake()).read_receipts() if r.round_id == newest.round_id]
    allowed = RECEIPT_FIELDS | {
        "submission_id",
        "round_id",
        "miner_id",
        "bundle_sha256",
        "received_at",
        "files",
        "bytes",
        "status",
        "origin",
    }
    published = []
    for receipt in receipts:
        record = receipt.to_record()
        unexpected = sorted(set(record) - allowed)
        if unexpected:
            # Refused rather than filtered. A receipt that grew a field is a receipt whose shape
            # nobody here has considered, and dropping it silently is how the next one gets
            # published by a version of this function that forgot to.
            raise BoardError(f"receipt {receipt.submission_id} carries unexpected field(s) {unexpected}")
        published.append(record)

    return {
        "schema_version": SCHEMA,
        # Seconds, matching every other timestamp the API serves, so the board formats one way.
        "generated_at": time.time() if now is None else now,
        "round": view,
        "submissions": published,
    }


def publish_page(directory: Path) -> Path:
    """Copy the board beside its snapshot, so the static host serves the same page as the validator."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "index.html"
    target.write_text(PAGE.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def write(path: Path, payload: dict[str, Any]) -> None:
    """Write the snapshot, atomically.

    Temp-then-rename because this file is read by a web server: a reader that arrives mid-write
    would otherwise get a truncated JSON document and the board would report the validator as
    unreachable, which is a different and wrong thing to tell them.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("docs/board/state.json"))
    parser.add_argument("--store", type=Path, default=None, help="round store; defaults to var/rounds")
    parser.add_argument(
        "--allow-public-store",
        action="store_true",
        help="write even when the round store is not gitignored. Off by default: a store inside the "
        "repository means the private half of a round is one `git add` from being published.",
    )
    args = parser.parse_args(argv)

    store = RoundStore(args.store, require_private=not args.allow_public_store) if args.store else RoundStore()
    try:
        payload = snapshot(store=store)
    except (BoardError, ScoreLeakError) as exc:
        print(f"validator.board: {exc}")
        return 1

    write(args.out, payload)
    page = publish_page(args.out.parent)
    view = payload["round"]
    print(f"wrote {args.out}")
    print(f"wrote {page}")
    print(f"  round        {view.get('round_id')} on {view.get('task_id')} ({view.get('state')})")
    print(f"  submissions  {len(payload['submissions'])}")
    print(f"  generated_at {payload['generated_at']:.0f}")
    print("\nCommit it to publish. The board prefers the live API and falls back to this file,")
    print("and renders its age -- a snapshot that looks live is worse than no snapshot.")
    return 0


__all__ = ["PAGE", "SCHEMA", "BoardError", "main", "publish_page", "snapshot", "write"]


if __name__ == "__main__":
    raise SystemExit(main())
