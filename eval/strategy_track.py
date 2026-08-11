"""Accepting a miner's surface by pull request, so the validator can run it.

    strategy_track:ACCEPT   the surface may merge; the validator will run it
    strategy_track:REJECT   with every reason, not the first

At the first stage the miner submits their surface -- `SOUL.md`, `skills/*/SKILL.md`,
`skills/*/references/*.md` -- and the **validator runs it**: pinned model, pinned environment,
pinned runtime, its own hardware, its own withheld verifiers. Nothing about the execution is taken
on the miner's word, because the miner does not perform it.

## Why this is a different record from the rollout track

`eval.rollout_track.Submission` requires `hf_url`, `hf_revision`, `export_digests` and `rows`: it
accepts *rollouts a miner generated and published*, and its whole apparatus exists to prove which
hardware produced files the validator never watched being made. A surface submission has none of
those and needs none of them. Reusing that record would have meant inventing an `hf_url` for a
submission that has no export -- a field that validates and means nothing.

The two tracks answer different questions. There, "did this data come from the hardware you
claim". Here, "may the validator run these files", which it then does.

## What this makes verifiable that miner-side generation cannot

Because the validator executes the surface, three things stop being promises:

**The files are inspectable.** They are in the pull request. `hermes.miner_contract` is enforced
against what was actually committed, and it is prose-only -- so the same property that lets the
rollout track auto-merge data holds here: a `.md` surface cannot carry code.

**The model, environment and runtime are the pinned ones** because the validator supplies them.
There is no guest image to measure and no attestation gap: `check_tdx_measurement` returns `None`
for want of a pinned MRTD, and at this stage nothing depends on it.

**The verifiers stay withheld.** The miner never holds the withheld half, so `overfit_rate` means
what it says.

## The security spine, unchanged

The workflow checks out the trusted base and fetches the pull request head as a git object it
never executes. Reading a `.md` file is not executing it. The gate below reads the committed
surface, digests it, and checks the digests against what the registry line claims -- so a
submission cannot cite one surface and ship another.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval.rollout_track import check_scope

STRATEGY_REGISTRY = Path("datasets/strategies.jsonl")
SUBMISSIONS_ROOT = Path("submissions")

SCHEMA_VERSION = 1

REQUIRED_FIELDS = (
    "schema_version",
    "round_id",
    "miner_id",
    "task_ids",
    "surface_digests",
)

REJECT = "strategy:REJECT"
ACCEPT = "strategy:ACCEPT"


class StrategyError(ValueError):
    """A surface submission is malformed."""


@dataclass(frozen=True)
class StrategySubmission:
    """One miner's surface, offered for one round."""

    round_id: str
    miner_id: str
    task_ids: tuple[str, ...]
    surface_digests: dict[str, str]
    schema_version: int = SCHEMA_VERSION
    notes: str = field(default="")

    @property
    def surface_dir(self) -> Path:
        """Where the files live. Derived, never taken from the record.

        A path read out of the submission is a path the submitter chose, and the one thing this
        gate must not let a miner choose is which files the validator is about to load.
        """
        return SUBMISSIONS_ROOT / self.round_id / self.miner_id

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "round_id": self.round_id,
            "miner_id": self.miner_id,
            "task_ids": sorted(self.task_ids),
            "surface_digests": dict(sorted(self.surface_digests.items())),
            "notes": self.notes,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> StrategySubmission:
        missing = [f for f in REQUIRED_FIELDS if not record.get(f)]
        if missing:
            raise StrategyError(f"submission is missing {', '.join(missing)}")
        digests = record["surface_digests"]
        if not isinstance(digests, dict):
            raise StrategyError("surface_digests must be a mapping of path to sha256")
        return cls(
            round_id=str(record["round_id"]),
            miner_id=str(record["miner_id"]),
            task_ids=tuple(str(t) for t in record["task_ids"]),
            surface_digests={str(k): str(v) for k, v in digests.items()},
            schema_version=int(record.get("schema_version") or SCHEMA_VERSION),
            notes=str(record.get("notes") or ""),
        )


def digest_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def digest_surface(root: Path) -> dict[str, str]:
    """Digest every file in a surface, as posix paths relative to its root.

    Symlinks are digested as links rather than followed. Following one would digest whatever it
    points at, so the recorded digest would describe a file that is not in the submission -- and
    the contract check refuses symlinks anyway, so this only has to avoid reading them.
    """
    digests: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            digests[rel] = "symlink:" + hashlib.sha256(str(path.readlink()).encode("utf-8")).hexdigest()
        elif path.is_file():
            digests[rel] = digest_file(path)
    return digests


def check_shape(record: dict[str, Any]) -> list[str]:
    try:
        submission = StrategySubmission.from_record(record)
    except StrategyError as exc:
        return [str(exc)]
    issues: list[str] = []
    if submission.schema_version != SCHEMA_VERSION:
        issues.append(f"schema_version {submission.schema_version} is not {SCHEMA_VERSION}")
    if "/" in submission.miner_id or submission.miner_id in (".", ".."):
        # The miner id becomes a directory component. `..` would place the surface outside the
        # submissions root, and the validator loads whatever is at that path.
        issues.append(f"miner_id {submission.miner_id!r} must be a single path segment")
    if "/" in submission.round_id or submission.round_id in (".", ".."):
        issues.append(f"round_id {submission.round_id!r} must be a single path segment")
    for path, digest in submission.surface_digests.items():
        if not digest.startswith(("sha256:", "symlink:")):
            issues.append(f"{path}: digest {digest!r} is not a sha256")
    return issues


def check_surface_present(submission: StrategySubmission, head_root: Path) -> list[str]:
    """Whether the committed files are exactly the ones the record claims, byte for byte.

    Both directions matter and they catch different things. A claimed file that is absent is a
    submission citing work it did not ship. A present file the record does not claim is worse: it
    would be loaded by the validator without ever having been digested, so the registry line
    would be an incomplete description of what actually ran.
    """
    root = head_root / submission.surface_dir
    if not root.is_dir():
        return [f"{submission.surface_dir.as_posix()} is not in this pull request; there is nothing to run"]

    found = digest_surface(root)
    issues: list[str] = []
    for path, claimed in sorted(submission.surface_digests.items()):
        actual = found.get(path)
        if actual is None:
            issues.append(f"{path}: claimed in the registry line and not present in the submission")
        elif actual != claimed:
            issues.append(f"{path}: committed file digests {actual}, the registry line claims {claimed}")
    for path in sorted(set(found) - set(submission.surface_digests)):
        issues.append(
            f"{path}: present in the submission and not claimed in the registry line. The validator "
            "loads the directory, so an unclaimed file would run undeclared."
        )
    return issues


def check_contract(submission: StrategySubmission, head_root: Path) -> list[str]:
    """Whether the committed surface is one the pinned runtime may load.

    Load-bearing in a way it is not on the rollout track: the validator is about to run these
    files. `hermes.miner_contract` is prose-only, which is the same property that lets a data-only
    registry line auto-merge -- a `.md` surface cannot carry code.
    """
    from hermes.miner_contract import load as load_contract

    root = head_root / submission.surface_dir
    if not root.is_dir():
        return []

    paths, links = [], []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            links.append(rel)
        elif path.is_file():
            paths.append(rel)

    issues = [
        f"{link}: is a symlink. The contract matches names, so a link called `notes.md` satisfies "
        "every rule while resolving to anything the validator's filesystem holds."
        for link in links
    ]
    if not paths and not links:
        issues.append(
            f"{submission.surface_dir.as_posix()} contains no files. An empty surface runs as the "
            "unmodified baseline while occupying a submission slot."
        )
    issues.extend(str(v) for v in load_contract().check(paths + links))
    return issues


def allowed_paths(submission: StrategySubmission) -> tuple[str, ...]:
    """The only paths this submission may change: the registry, and its own surface directory."""
    return (STRATEGY_REGISTRY.as_posix(), submission.surface_dir.as_posix() + "/")


def check_paths(submission: StrategySubmission, changed_paths: list[str] | None) -> list[str]:
    """A surface PR touches the registry and one directory, and nothing else.

    Scoped to *this* submission's directory rather than to `submissions/` as a whole, because a PR
    that edits another miner's surface is not a submission, it is an attack on theirs.
    """
    if changed_paths is None:
        return []
    registry = STRATEGY_REGISTRY.as_posix()
    own = submission.surface_dir.as_posix() + "/"
    unexpected = sorted({p for p in changed_paths if p != registry and not p.startswith(own)})
    if unexpected:
        return [
            f"a strategy PR may only change {registry} and {own}; unexpected paths: {unexpected!r}. "
            "Editing another miner's surface is not a submission."
        ]
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


def check_one_submission_per_round(submission: StrategySubmission, base_text: str) -> list[str]:
    """One standing surface per miner per round.

    A second line for the same pair is a resubmission, and the round window is what decides
    whether a replacement is allowed -- not this gate. Refusing here keeps the registry a record
    of what stood rather than a log of what was tried.
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
            str(existing.get("round_id") or "") == submission.round_id
            and str(existing.get("miner_id") or "") == submission.miner_id
        ):
            return [
                f"{submission.miner_id!r} already has a surface standing in round {submission.round_id}. "
                "Replacements go through the round window, which records what replaced what."
            ]
    return []


def gate(
    *,
    record: dict[str, Any],
    round_record: dict[str, Any],
    head_root: Path,
    base_text: str,
    head_text: str,
    changed_paths: list[str] | None = None,
) -> list[str]:
    """Every reason this surface may not merge. Empty means ACCEPT.

    All of them, not the first: a miner who learns one problem per pull request stops opening
    them, and the reasons cost nothing to collect.

    Shape is checked before anything else because every later check reads fields off the record,
    and a check that runs on a malformed record reports the malformation as its own kind of
    failure.
    """
    shape = check_shape(record)
    if shape:
        return shape

    submission = StrategySubmission.from_record(record)
    return [
        *check_paths(submission, changed_paths),
        *check_append_only(base_text, head_text),
        *check_one_submission_per_round(submission, base_text),
        *check_scope(record, round_record),
        *check_surface_present(submission, head_root),
        *check_contract(submission, head_root),
    ]


__all__ = [
    "ACCEPT",
    "REJECT",
    "REQUIRED_FIELDS",
    "SCHEMA_VERSION",
    "STRATEGY_REGISTRY",
    "SUBMISSIONS_ROOT",
    "StrategyError",
    "StrategySubmission",
    "added_lines",
    "allowed_paths",
    "check_append_only",
    "check_contract",
    "check_one_submission_per_round",
    "check_paths",
    "check_shape",
    "check_surface_present",
    "digest_file",
    "digest_surface",
    "gate",
]
