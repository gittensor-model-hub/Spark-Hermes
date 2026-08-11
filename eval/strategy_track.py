"""Accepting a miner's commitment: a digest in a pull request, a bundle already held privately.

    strategy:ACCEPT   the commitment stands; the validator will run that bundle
    strategy:REJECT   with every reason, not the first

A miner uploads their surface to the validator API and opens a pull request appending one line to
`datasets/strategies.jsonl` naming the digest of what they uploaded. This decides whether that line
may merge.

## This replaced a gate that expected the files themselves

An earlier version of this module took the surface *in the pull request* and checked the committed
files against the contract. That made the surface public, which destroys the thing a miner is
competing with. The bundle is private now and the pull request carries only its digest, so every
file-shaped check here has moved to `validator.intake`, which validates a bundle before storing it.

What is left is the part that only a public, timestamped, attributable record can do: bind a miner
to a specific bundle before anything runs.

## The attack this exists to stop

Receipts are public. Every digest the validator has accepted is visible, including other people's.
So the obvious move is to open a pull request naming someone else's digest and have their work
evaluated under your name.

Three identities therefore have to agree: the pull request's author, the `miner_id` on the line,
and the `miner_id` on the receipt that digest belongs to. Any disagreement is a rejection.
`check_author` is the only check here that cannot be inferred from the repository alone -- it needs
the author GitHub reports -- which is why it takes it as an argument rather than reading it from
the record, where a submitter could write anything.

## A commitment names a bundle that already exists

The digest must match a receipt the validator published. A pull request naming a digest nothing was
ever uploaded for is not a commitment; it is a claim on a bundle that may be produced later to fit
whatever result would have been convenient.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval.rollout_track import check_scope

STRATEGY_REGISTRY = Path("datasets/strategies.jsonl")

SCHEMA_VERSION = 2

REQUIRED_FIELDS = (
    "schema_version",
    "round_id",
    "miner_id",
    "bundle_sha256",
)

REJECT = "strategy:REJECT"
ACCEPT = "strategy:ACCEPT"


class StrategyError(ValueError):
    """A commitment is malformed."""


@dataclass(frozen=True)
class Commitment:
    """One line: this miner, this round, this bundle."""

    round_id: str
    miner_id: str
    bundle_sha256: str
    task_ids: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION
    notes: str = field(default="")

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "round_id": self.round_id,
            "miner_id": self.miner_id,
            "bundle_sha256": self.bundle_sha256,
            "task_ids": sorted(self.task_ids),
            "notes": self.notes,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Commitment:
        missing = [f for f in REQUIRED_FIELDS if not record.get(f)]
        if missing:
            raise StrategyError(f"commitment is missing {', '.join(missing)}")
        return cls(
            round_id=str(record["round_id"]),
            miner_id=str(record["miner_id"]),
            bundle_sha256=str(record["bundle_sha256"]),
            task_ids=tuple(str(t) for t in record.get("task_ids") or ()),
            schema_version=int(record.get("schema_version") or SCHEMA_VERSION),
            notes=str(record.get("notes") or ""),
        )


def check_shape(record: dict[str, Any]) -> list[str]:
    try:
        commitment = Commitment.from_record(record)
    except StrategyError as exc:
        return [str(exc)]
    issues: list[str] = []
    if commitment.schema_version != SCHEMA_VERSION:
        issues.append(
            f"schema_version {commitment.schema_version} is not {SCHEMA_VERSION}. Version 1 carried the "
            "surface files themselves; the surface is private now and the line carries a digest."
        )
    if not commitment.bundle_sha256.startswith("sha256:") or len(commitment.bundle_sha256) != 71:
        issues.append(f"bundle_sha256 {commitment.bundle_sha256!r} is not a sha256 digest")
    return issues


def check_author(commitment: Commitment, pull_request_author: str | None) -> list[str]:
    """Whether the person opening the pull request is the miner claiming the bundle.

    Taken as an argument rather than read from the record: the record is written by the submitter,
    so a `miner_id` field alone proves nothing about who opened the pull request. GitHub is the
    only party that can say, and this is the one check that depends on it.

    `None` means the caller could not determine an author, which is reported rather than passed.
    An unauthenticated commitment is exactly the case this check exists for.
    """
    if pull_request_author is None:
        return [
            "no pull request author was supplied, so this commitment cannot be attributed. The "
            "receipt is public, so an unattributed line could name any bundle the validator holds."
        ]
    if pull_request_author != commitment.miner_id:
        return [
            f"the pull request is opened by {pull_request_author!r} and the line claims "
            f"{commitment.miner_id!r}. A miner commits to their own bundle."
        ]
    return []


def check_commitment(commitment: Commitment, receipts: list[Any]) -> list[str]:
    """Whether the digest names a bundle this validator actually holds, uploaded by this miner.

    Two failures with one shape and different meanings. A digest nothing was uploaded for is a
    claim on a bundle that could be produced afterwards to fit whatever result would have been
    convenient. A digest belonging to *someone else* is an attempt to have their work scored under
    your name, and receipts being public is what makes that worth defending against.
    """
    from validator.intake import receipt_for_digest

    receipt = receipt_for_digest(receipts, round_id=commitment.round_id, digest=commitment.bundle_sha256)
    if receipt is None:
        return [
            f"no bundle with digest {commitment.bundle_sha256[:23]}... was uploaded for round "
            f"{commitment.round_id}. A commitment names something the validator already holds; a "
            "digest with nothing behind it is a claim on a bundle that could be written later."
        ]
    if receipt.miner_id != commitment.miner_id:
        return [
            f"digest {commitment.bundle_sha256[:23]}... was uploaded by {receipt.miner_id!r}, and this "
            f"line claims it for {commitment.miner_id!r}. Receipts are public; a bundle is not."
        ]
    return []


def check_paths(changed_paths: list[str] | None) -> list[str]:
    """A commitment PR touches one file. Data-only, so auto-merge can never carry code."""
    if changed_paths is None:
        return []
    registry = STRATEGY_REGISTRY.as_posix()
    unexpected = sorted({p for p in changed_paths if p != registry})
    if unexpected:
        return [f"a strategy PR may only change {registry}; unexpected paths: {unexpected!r}"]
    return []


def check_append_only(base_text: str, head_text: str) -> list[str]:
    base_lines = [line.strip() for line in base_text.splitlines() if line.strip()]
    head_lines = [line.strip() for line in head_text.splitlines() if line.strip()]
    if head_lines[: len(base_lines)] != base_lines:
        return [
            f"{STRATEGY_REGISTRY.as_posix()} is append-only; rebase onto the latest base and preserve "
            "every existing line in order"
        ]
    return []


def added_lines(base_text: str, head_text: str) -> list[dict[str, Any]]:
    """The JSON objects this PR appends, positionally."""
    base_lines = [line.strip() for line in base_text.splitlines() if line.strip()]
    head_lines = [line.strip() for line in head_text.splitlines() if line.strip()]
    out: list[dict[str, Any]] = []
    for line in head_lines[len(base_lines) :]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise StrategyError(f"appended line is not JSON: {exc}") from exc
    return out


def check_one_commitment_per_round(commitment: Commitment, base_text: str) -> list[str]:
    """One standing commitment per miner per round.

    A second line is a miner changing which bundle they are judged on after the first was public.
    Uploading twice is allowed and cheap -- the digest decides which is evaluated -- so the moment
    that choice becomes binding has to be a single, dated, public one.
    """
    for line in base_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            existing = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            str(existing.get("round_id") or "") == commitment.round_id
            and str(existing.get("miner_id") or "") == commitment.miner_id
        ):
            return [
                f"{commitment.miner_id!r} already has a commitment standing in round "
                f"{commitment.round_id}. Which bundle you are judged on is fixed once it is public."
            ]
    return []


def gate(
    *,
    record: dict[str, Any],
    round_record: dict[str, Any],
    receipts: list[Any],
    base_text: str,
    head_text: str,
    changed_paths: list[str] | None = None,
    pull_request_author: str | None = None,
) -> list[str]:
    """Every reason this commitment may not merge. Empty means ACCEPT.

    All of them, not the first: a miner who learns one problem per pull request stops opening them.

    Shape is checked before anything else because every later check reads fields off the record, and
    a check running on a malformed record reports the malformation as its own kind of failure.
    """
    shape = check_shape(record)
    if shape:
        return shape

    commitment = Commitment.from_record(record)
    return [
        *check_paths(changed_paths),
        *check_append_only(base_text, head_text),
        *check_one_commitment_per_round(commitment, base_text),
        *check_author(commitment, pull_request_author),
        *check_scope(record, round_record),
        *check_commitment(commitment, receipts),
    ]


__all__ = [
    "ACCEPT",
    "REJECT",
    "REQUIRED_FIELDS",
    "SCHEMA_VERSION",
    "STRATEGY_REGISTRY",
    "Commitment",
    "StrategyError",
    "added_lines",
    "check_append_only",
    "check_author",
    "check_commitment",
    "check_one_commitment_per_round",
    "check_paths",
    "check_shape",
    "gate",
]
