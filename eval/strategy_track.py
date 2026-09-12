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

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval.rollout_track import check_scope
from hermes.evidence_json import evidence_records

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
        if not isinstance(record, dict):
            raise StrategyError("commitment must be an object")
        missing = [f for f in REQUIRED_FIELDS if not record.get(f)]
        if missing:
            raise StrategyError(f"commitment is missing {', '.join(missing)}")
        if type(record["schema_version"]) is not int:
            raise StrategyError("schema_version must be an integer")
        for key in ("round_id", "miner_id", "bundle_sha256"):
            if not isinstance(record[key], str) or not record[key].strip():
                raise StrategyError(f"{key} must be a nonempty string")
        for key in ("round_id", "miner_id"):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", record[key]):
                raise StrategyError(f"invalid {key}")
        tasks = record.get("task_ids", [])
        if (
            not isinstance(tasks, list)
            or any(not isinstance(t, str) or not t for t in tasks)
            or len(set(tasks)) != len(tasks)
        ):
            raise StrategyError("task_ids must be a list of unique nonempty strings")
        if not isinstance(record.get("notes", ""), str):
            raise StrategyError("notes must be a string")
        if set(record) - {*REQUIRED_FIELDS, "task_ids", "notes"}:
            raise StrategyError("unknown commitment fields")
        return cls(
            round_id=record["round_id"],
            miner_id=record["miner_id"],
            bundle_sha256=record["bundle_sha256"],
            task_ids=tuple(tasks),
            schema_version=record["schema_version"],
            notes=record.get("notes", ""),
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
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", commitment.bundle_sha256):
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

    receipt = receipt_for_digest(
        receipts, round_id=commitment.round_id, digest=commitment.bundle_sha256, miner_id=commitment.miner_id
    )
    if receipt is None:
        other = receipt_for_digest(receipts, round_id=commitment.round_id, digest=commitment.bundle_sha256)
        if other is not None:
            return [f"digest was uploaded by {other.miner_id!r}, not {commitment.miner_id!r}"]
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
    if not changed_paths:
        return ["verified changed paths are required"]
    registry = STRATEGY_REGISTRY.as_posix()
    unexpected = sorted({p for p in changed_paths if p != registry})
    if unexpected:
        return [f"a strategy PR may only change {registry}; unexpected paths: {unexpected!r}"]
    return []


def check_append_only(base_text: str, head_text: str) -> list[str]:
    if not head_text.startswith(base_text) or (
        base_text
        and not base_text.endswith("\n")
        and head_text[len(base_text) :]
        and not head_text[len(base_text) :].startswith("\n")
    ):
        return [f"{STRATEGY_REGISTRY.as_posix()} is append-only; preserve every existing byte in order"]
    return []


def added_lines(base_text: str, head_text: str) -> list[dict[str, Any]]:
    """The positional delta, after decoding complete base and head snapshots."""
    try:
        base = evidence_records(base_text)
        head = evidence_records(head_text)
    except ValueError as exc:
        raise StrategyError(f"registry is not JSON object metadata: {exc}") from exc
    return head[len(base) :]


def check_one_commitment_per_round(commitment: Commitment, base_text: str) -> list[str]:
    """One standing commitment per miner per round.

    A second line is a miner changing which bundle they are judged on after the first was public.
    Uploading twice is allowed and cheap -- the digest decides which is evaluated -- so the moment
    that choice becomes binding has to be a single, dated, public one.
    """
    try:
        existing_records = evidence_records(base_text)
    except ValueError as exc:
        return [f"base registry contains malformed metadata: {exc}"]
    for existing in existing_records:
        # History may contain older schemas or identity-only records. They still
        # exclude this miner/round; never require a current payload to recognize it.
        if any(
            not isinstance(existing.get(k), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", existing[k])
            for k in ("round_id", "miner_id")
        ):
            return ["base registry contains a malformed commitment identity"]
        if "schema_version" in existing and (
            type(existing["schema_version"]) is not int or existing["schema_version"] < 1
        ):
            return ["base registry contains a malformed schema_version"]
        if "bundle_sha256" in existing and (
            not isinstance(existing["bundle_sha256"], str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", existing["bundle_sha256"])
        ):
            return ["base registry contains a malformed bundle_sha256"]
        if "task_ids" in existing:
            tasks = existing["task_ids"]
            if (
                not isinstance(tasks, list)
                or any(not isinstance(t, str) or not t for t in tasks)
                or len(set(tasks)) != len(tasks)
            ):
                return ["base registry contains malformed task_ids"]
        if "notes" in existing and not isinstance(existing["notes"], str):
            return ["base registry contains malformed notes"]
        if existing["round_id"] == commitment.round_id and existing["miner_id"] == commitment.miner_id:
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
    try:
        delta = added_lines(base_text, head_text)
    except StrategyError as exc:
        return [str(exc)]
    if delta != [record]:
        return ["PR must append exactly the one commitment being gated"]
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
