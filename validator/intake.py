"""Accepting a private bundle over HTTP, and refusing everything that is not one.

A miner uploads their surface here and opens a pull request carrying only its digest. The bundle
stays private, so the surface remains the miner's edge; the digest is public, timestamped and
attributable, so the validator cannot evaluate a different bundle than the one committed and the
miner cannot revise after the fact.

This module is the validation. `validator.api` is the transport, and keeping them apart is what
makes the rules testable without a running server -- which matters more here than anywhere else in
the repository, because this is the only code that takes input from someone who wants to beat it.

## A JSON map of path to text, not an archive

The obvious upload format is a tarball or a zip, and both bring a class of bug this does not have:
path traversal from an entry named `../../etc/x`, symlinks that resolve outside the extraction
root, and decompression bombs that are small on the wire and enormous on disk.

A JSON object of `{path: content}` has no compression to expand, no symlink to follow, and paths
that are ordinary strings the validator checks before anything reaches a filesystem. The cost is
that the upload is bigger. A surface is a few kilobytes of prose, so that cost is nothing and the
class of bug is gone.

## Everything is checked before anything is stored

Size, path shape, and the contract, in that order, and nothing is written until all three pass. A
validator that stored first and validated second would need a delete path that itself has to be
right, and the first bug in it is a file on disk that nobody meant to accept.

## What the digest covers

`bundle_digest` is over the canonical JSON of the file map: sorted keys, no incidental whitespace.
So the same surface digests the same regardless of upload order or formatting, and any change to
any file or filename moves it. This is the value that goes in the pull request, and it is what the
gate later checks the receipt against.

## Receipts carry no correctness information

A receipt says a bundle arrived, from whom, when, and where it is in the queue. That is an
envelope fact, and the same reasoning that lets `/receipts` be served during an open round applies:
a miner has to be able to tell a rejected upload from a lost one, and none of it says anything
about whether the surface is any good.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

# A surface is prose. These are generous for that and small enough that a thousand of them is not
# a storage problem, which is the point: the limit exists so that refusing is cheap.
MAX_FILES = 64
MAX_FILE_BYTES = 256 * 1024
MAX_TOTAL_BYTES = 1024 * 1024

PENDING = "pending"
EVALUATING = "evaluating"
DONE = "result"
STATUSES = (PENDING, EVALUATING, DONE)

SUBMISSION_DIR = Path("var/submissions")
RECEIPTS = Path("datasets/receipts.jsonl")


class IntakeError(ValueError):
    """A bundle is not acceptable. The message is what the miner is told."""


@dataclass(frozen=True)
class Receipt:
    """The public record of one upload. No correctness information, ever."""

    submission_id: str
    round_id: str
    miner_id: str
    bundle_sha256: str
    received_at: float
    files: int
    bytes: int
    status: str = PENDING

    def to_record(self) -> dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "round_id": self.round_id,
            "miner_id": self.miner_id,
            "bundle_sha256": self.bundle_sha256,
            "received_at": self.received_at,
            "files": self.files,
            "bytes": self.bytes,
            "status": self.status,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Receipt:
        return cls(
            submission_id=str(record["submission_id"]),
            round_id=str(record["round_id"]),
            miner_id=str(record["miner_id"]),
            bundle_sha256=str(record["bundle_sha256"]),
            received_at=float(record.get("received_at") or 0.0),
            files=int(record.get("files") or 0),
            bytes=int(record.get("bytes") or 0),
            status=str(record.get("status") or PENDING),
        )


def canonical(files: dict[str, str]) -> bytes:
    """The bytes the digest is taken over. Sorted keys, no incidental whitespace."""
    return json.dumps(files, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def bundle_digest(files: dict[str, str]) -> str:
    return "sha256:" + hashlib.sha256(canonical(files)).hexdigest()


def check_paths(files: dict[str, str]) -> list[str]:
    """Path shape, before the contract and before any filesystem call.

    The contract matches names against patterns; it is not a path-safety check. `../../etc/x`
    matches no allowed pattern and would be refused, but relying on that is relying on a rule
    written for a different purpose, and the day someone adds a permissive pattern the traversal
    arrives with it.
    """
    problems: list[str] = []
    for raw in sorted(files):
        if not raw or raw != raw.strip():
            problems.append(f"{raw!r}: a path may not be empty or carry surrounding whitespace")
            continue
        if raw.startswith("/") or (len(raw) > 1 and raw[1] == ":"):
            problems.append(f"{raw!r}: absolute paths are not accepted")
            continue
        if "\\" in raw:
            problems.append(f"{raw!r}: use forward slashes; a backslash is a filename here, not a separator")
            continue
        if "\x00" in raw:
            problems.append(f"{raw!r}: contains a null byte")
            continue
        parts = PurePosixPath(raw).parts
        if any(part in ("..", ".") for part in parts):
            problems.append(f"{raw!r}: path escapes the bundle root")
    return problems


def check_size(files: dict[str, str]) -> list[str]:
    """Counts and bytes. Reported per file so a miner knows which one to cut."""
    problems: list[str] = []
    if len(files) > MAX_FILES:
        problems.append(f"{len(files)} files; at most {MAX_FILES} are accepted")
    total = 0
    for path in sorted(files):
        size = len(files[path].encode("utf-8"))
        total += size
        if size > MAX_FILE_BYTES:
            problems.append(f"{path}: {size:,} bytes; at most {MAX_FILE_BYTES:,} per file")
    if total > MAX_TOTAL_BYTES:
        problems.append(f"{total:,} bytes in total; at most {MAX_TOTAL_BYTES:,}")
    return problems


def check_contract(files: dict[str, str]) -> list[str]:
    """The same contract the runner enforces before it contacts the model.

    Called rather than reimplemented: a second copy of the rules would agree today and diverge on
    the first change, and the copy downstream of the divergence is the one that decides what runs.
    """
    from hermes.miner_contract import load as load_contract

    return [str(v) for v in load_contract().check(sorted(files))]


def validate(files: Any) -> list[str]:
    """Every reason this bundle is not acceptable. Empty means it is."""
    if not isinstance(files, dict):
        return ["the bundle must be an object mapping path to file content"]
    if not files:
        return ["the bundle is empty; an empty surface runs as the unmodified baseline"]

    bad_types = [k for k, v in files.items() if not isinstance(k, str) or not isinstance(v, str)]
    if bad_types:
        return [f"{k!r}: both the path and the content must be strings" for k in sorted(map(str, bad_types))]

    # Order matters: a path that escapes the root must be refused before anything measures or
    # matches it, and size before the contract so an enormous upload is rejected cheaply.
    problems = check_paths(files)
    if problems:
        return problems
    problems = check_size(files)
    if problems:
        return problems
    return check_contract(files)


def submission_id(*, round_id: str, miner_id: str, digest: str) -> str:
    """A short, stable id derived from what it identifies.

    Derived rather than random so the same bundle re-uploaded by the same miner in the same round
    yields the same id -- which makes a retry after a dropped connection idempotent instead of
    creating a second pending submission nobody meant.
    """
    material = f"{round_id}\x00{miner_id}\x00{digest}".encode()
    return hashlib.sha256(material).hexdigest()[:16]


def canonical_path(bundle_dir: Path) -> Path:
    """The canonical serialisation that sits beside a bundle directory, never within it."""
    return bundle_dir.with_name(bundle_dir.name + ".bundle.json")


@dataclass
class Intake:
    """Stores accepted bundles privately and appends public receipts.

    The default paths are resolved through `default_factory`, at instantiation, rather than being
    written as plain defaults. A plain default is captured into the generated `__init__` when the
    class is created, so reassigning `SUBMISSION_DIR` afterwards changes nothing for new instances.

    That is not hypothetical tidiness. `validator.api`'s upload endpoint constructs `Intake()`
    itself, and its tests isolate by monkeypatching these module globals -- a fixture whose
    docstring says "so uploads do not touch the repository" and which, with plain defaults, did
    not. Running the API tests wrote real bundles into `var/submissions` and appended to the real
    `datasets/receipts.jsonl`, and every test still passed, because they assert on responses and
    nothing asserted on where the files landed.
    """

    root: Path = field(default_factory=lambda: SUBMISSION_DIR)
    receipts: Path = field(default_factory=lambda: RECEIPTS)

    def canonical_path_for(self, receipt: Receipt) -> Path:
        """Where the canonical serialisation of a stored bundle lives."""
        return canonical_path(self.bundle_dir(receipt))

    def accept(self, *, round_id: str, miner_id: str, files: dict[str, str], now: float | None = None) -> Receipt:
        """Validate, store, and record. Raises `IntakeError` with every reason on refusal."""
        for name, value in (("round_id", round_id), ("miner_id", miner_id)):
            if not value or "/" in value or value in (".", ".."):
                # Both become directory components below.
                raise IntakeError(f"{name} {value!r} must be a single path segment")

        problems = validate(files)
        if problems:
            raise IntakeError("; ".join(problems))

        digest = bundle_digest(files)
        ident = submission_id(round_id=round_id, miner_id=miner_id, digest=digest)
        target = self.root / round_id / miner_id / ident
        target.mkdir(parents=True, exist_ok=True)
        for path, content in sorted(files.items()):
            destination = target / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
        # BESIDE the bundle directory, not inside it. This is the canonical serialisation the
        # digest is computed over -- validator metadata, not part of the miner's surface -- and
        # `validator.judge` hands that directory to the runner as `--miner-dir`. The miner contract
        # admits SOUL.md, skills/*/SKILL.md and skills/*/references/*.md and refuses everything
        # else rather than assuming it harmless, so a bundle.json inside made the runner exit 2 on
        # EVERY judged submission. The competition could not run, and nothing caught it: the judge
        # filters this name out of its own path accounting, so the file was invisible from both
        # sides until an end-to-end round was actually attempted.
        canonical_path(target).write_text(canonical(files).decode("utf-8") + "\n", encoding="utf-8")

        receipt = Receipt(
            submission_id=ident,
            round_id=round_id,
            miner_id=miner_id,
            bundle_sha256=digest,
            received_at=time.time() if now is None else now,
            files=len(files),
            bytes=sum(len(v.encode("utf-8")) for v in files.values()),
        )
        self.append_receipt(receipt)
        return receipt

    def append_receipt(self, receipt: Receipt) -> None:
        """Append one receipt, replacing any earlier line for the same submission id.

        Rewritten in place rather than appended blindly because a status moves -- pending, then
        evaluating, then result -- and a reader taking the first match would report a finished
        submission as pending forever.
        """
        self.receipts.parent.mkdir(parents=True, exist_ok=True)
        existing = [r for r in self.read_receipts() if r.submission_id != receipt.submission_id]
        existing.append(receipt)
        existing.sort(key=lambda r: (r.received_at, r.submission_id))
        with self.receipts.open("w", encoding="utf-8") as handle:
            for entry in existing:
                handle.write(json.dumps(entry.to_record(), sort_keys=True) + "\n")

    def read_receipts(self) -> list[Receipt]:
        if not self.receipts.is_file():
            return []
        out: list[Receipt] = []
        for line in self.receipts.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(Receipt.from_record(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                # A malformed line is skipped rather than fatal: one bad receipt must not make the
                # dashboard unreadable or stop a later status from being recorded.
                continue
        return out

    def set_status(self, submission_id_: str, status: str) -> Receipt:
        if status not in STATUSES:
            raise IntakeError(f"{status!r} is not one of {list(STATUSES)}")
        for receipt in self.read_receipts():
            if receipt.submission_id == submission_id_:
                updated = Receipt(**{**receipt.to_record(), "status": status})
                self.append_receipt(updated)
                return updated
        raise IntakeError(f"no submission {submission_id_!r}")

    def bundle_dir(self, receipt: Receipt) -> Path:
        return self.root / receipt.round_id / receipt.miner_id / receipt.submission_id


def receipt_for_digest(receipts: list[Receipt], *, round_id: str, digest: str) -> Receipt | None:
    """The receipt a pull request's digest refers to, or None.

    This is what makes the public commitment authoritative: the pull request names a digest, and
    the validator evaluates the bundle that digest identifies rather than whatever the miner
    uploaded most recently. A miner who uploads twice and commits to the first is evaluated on the
    first.
    """
    for receipt in receipts:
        if receipt.round_id == round_id and receipt.bundle_sha256 == digest:
            return receipt
    return None


__all__ = [
    "DONE",
    "EVALUATING",
    "MAX_FILES",
    "MAX_FILE_BYTES",
    "MAX_TOTAL_BYTES",
    "PENDING",
    "RECEIPTS",
    "STATUSES",
    "SUBMISSION_DIR",
    "Intake",
    "IntakeError",
    "Receipt",
    "bundle_digest",
    "canonical_path",
    "canonical",
    "check_contract",
    "check_paths",
    "check_size",
    "receipt_for_digest",
    "submission_id",
    "validate",
]
