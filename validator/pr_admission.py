"""Authenticated, read-only PR admission: python -m validator.pr_admission --help.

Only GitHub responses fetched here confer authority. Uploads and caller JSON do not.
The configured credential is passed in the subprocess environment, never its argv.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from eval.strategy_track import Commitment, StrategyError, added_lines, gate
from hermes.evidence_json import evidence_object, evidence_value
from hermes.seed import COMMITTED, OPEN, Round, seed_commitment
from validator.intake import Intake, IntakeError, receipt_for_digest
from validator.store import RoundStore, StoreError

_ISSUER = object()
REGISTRY = "datasets/strategies.jsonl"


class AdmissionError(ValueError):
    """The trusted source or commitment could not be verified."""


def digest(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    )


@dataclass(frozen=True)
class VerifiedPR:
    repository: str
    number: int
    author: str
    head_sha: str
    base_sha: str
    base_text: str
    head_text: str
    changed_paths: tuple[str, ...]
    round_record_json: str
    authenticated_as: str
    _issuer: object


class GitHubSource:
    """GitHub REST over gh; controlled transports exercise this same production adapter."""

    def __init__(
        self, repository: str, *, credential_env: str = "GH_TOKEN", transport: Callable[..., Any] | None = None
    ):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise AdmissionError("configure an owner/repository")
        credential = os.environ.get(credential_env, "")
        if not credential.strip():
            raise AdmissionError(f"missing GitHub credential in {credential_env}")
        self.repository = repository
        self._env = {**os.environ, "GH_TOKEN": credential, "GH_HOST": "github.com", "GH_PROMPT_DISABLED": "1"}
        self._transport = transport or subprocess.run

    def get(self, endpoint: str) -> Any:
        try:
            result = self._transport(
                ["gh", "api", "--hostname", "github.com", "--method", "GET", endpoint],
                env=self._env,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            if result.returncode != 0:
                raise AdmissionError("authenticated GitHub read failed")
            return evidence_value(result.stdout)
        except (OSError, subprocess.SubprocessError, ValueError, TypeError) as exc:
            raise AdmissionError("GitHub transport failed or returned malformed JSON") from exc

    def content(self, repository: str, path: str, sha: str) -> str:
        record = self.get(f"repos/{repository}/contents/{path}?ref={sha}")
        try:
            if record["type"] != "file" or record["path"] != path or record["encoding"] != "base64":
                raise ValueError("unexpected content metadata")
            raw = base64.b64decode("".join(record["content"].splitlines()), validate=True)
            blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            if record["sha"] != blob:
                raise ValueError("blob identity mismatch")
            return raw.decode("utf-8")
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise AdmissionError("malformed or mismatched GitHub content") from exc

    def collect(self, number: int, *, round_id: str, expected_head: str | None = None) -> VerifiedPR:
        from validator.intake import check_segment

        check_segment(round_id, "round_id")
        if type(number) is not int or number <= 0:
            raise AdmissionError("PR number must be positive")
        who = self.get("user")
        if not isinstance(who, dict) or not isinstance(who.get("login"), str) or not who["login"]:
            raise AdmissionError("GitHub credential identity is unknown")
        endpoint = f"repos/{self.repository}/pulls/{number}"

        def identity(pr: Any) -> tuple[str, str, str, str, int]:
            try:
                author, head, base = pr["user"]["login"], pr["head"]["sha"], pr["base"]["sha"]
                head_repo = pr["head"]["repo"]["full_name"]
                count = pr["changed_files"]
                if (
                    type(pr["number"]) is not int
                    or pr["number"] != number
                    or pr["base"]["repo"]["full_name"] != self.repository
                ):
                    raise ValueError("repository or PR mismatch")
                if pr["state"] != "open" or pr["merged"] is not False or pr["draft"] is not False:
                    raise ValueError("PR must be open and ready")
                if not isinstance(author, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}", author):
                    raise ValueError("unknown author")
                if not all(isinstance(s, str) and re.fullmatch(r"[0-9a-f]{40}", s) for s in (head, base)):
                    raise ValueError("malformed commit SHA")
                if not isinstance(head_repo, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", head_repo):
                    raise ValueError("unknown head repository")
                if type(count) is not int or not 1 <= count <= 3000 or expected_head and head != expected_head:
                    raise ValueError("wrong head or file count")
                return author, head, base, head_repo, count
            except (KeyError, TypeError, ValueError) as exc:
                raise AdmissionError("malformed/mismatched PR identity or state") from exc

        first = identity(self.get(endpoint))
        author, head, base, head_repo, count = first
        files: list[str] = []
        for page in range(1, (count + 99) // 100 + 1):
            batch = self.get(f"{endpoint}/files?per_page=100&page={page}")
            if not isinstance(batch, list):
                raise AdmissionError("malformed GitHub diff")
            for entry in batch:
                if (
                    not isinstance(entry, dict)
                    or entry.get("status") not in ("added", "modified")
                    or not isinstance(entry.get("filename"), str)
                    or "previous_filename" in entry
                ):
                    raise AdmissionError("malformed or unsupported diff metadata")
                files.append(entry["filename"])
        if len(files) != count or len(set(files)) != count or set(files) != {REGISTRY}:
            raise AdmissionError("strategy PR must change only its registry; diff incomplete or out of track")
        base_text = self.content(self.repository, REGISTRY, base)
        head_text = self.content(head_repo, REGISTRY, head)
        round_json = self.content(self.repository, f"datasets/rounds/{round_id}.json", base)
        if identity(self.get(endpoint)) != first:
            raise AdmissionError("PR changed head/state during collection")
        return VerifiedPR(
            self.repository,
            number,
            author,
            head,
            base,
            base_text,
            head_text,
            tuple(files),
            round_json,
            who["login"],
            _ISSUER,
        )


def assignment_identity(assignment: Round) -> str:
    """Canonical immutable assignment; pool order and lifecycle state do not change it."""
    if type(assignment) is not Round:
        raise AdmissionError("trusted admission requires a typed round assignment")
    return digest(
        {
            "round_id": assignment.round_id,
            "seed_commitment": seed_commitment(assignment.seed),
            "task_ids": sorted(assignment.task_ids),
            "miner_ids": sorted(assignment.miner_ids),
            "replicas": assignment.replicas,
        }
    )


def round_identity(window: Any) -> str:
    """Immutable evaluation identity; lifecycle state and admissions are deliberately excluded."""
    return digest(
        {
            "round_id": window.round_id,
            "assignment_identity": assignment_identity(window.assignment),
            "challenge": window.challenge.snapshot(),
            "origin": window.store_identity,
        }
    )


def admission_for(window: Any, miner_id: str) -> dict[str, Any]:
    record = window.admissions.get(miner_id)
    standing = window.submissions.get(miner_id)
    if not isinstance(record, dict) or standing is None:
        raise AdmissionError("no trusted admission for this miner")
    verify_admission(record)
    expected = {
        "author": miner_id,
        "round_id": window.round_id,
        "task_id": window.task_id,
        "bundle_sha256": standing.payload_digest,
        "epoch": window.challenge.epoch,
        "origin": window.store_identity,
        "assignment_identity": assignment_identity(window.assignment),
        "round_identity": round_identity(window),
    }
    if any(record.get(k) != value for k, value in expected.items()):
        raise AdmissionError("admission does not match standing bundle/active epoch/round identity")
    if window.assignment.state == COMMITTED:
        raise AdmissionError("admitted assignment cannot return to committed state")
    return record


def verify_admission(record: dict[str, Any]) -> None:
    if not isinstance(record.get("assignment_identity"), str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", record["assignment_identity"]
    ):
        raise AdmissionError("admission lacks assignment binding; trusted recovery and fresh admission required")
    if type(record.get("pr_number")) is not int or record["pr_number"] <= 0:
        raise AdmissionError("admission PR number must be a positive integer")
    body = {k: v for k, v in record.items() if k != "admission_id"}
    if record.get("admission_id") != digest(body):
        raise AdmissionError("admission record identity is corrupt")


def admit(
    *, metadata: VerifiedPR, round_id: str, store: RoundStore, intake: Intake, now: float | None = None
) -> dict[str, Any]:
    if type(metadata) is not VerifiedPR or metadata._issuer is not _ISSUER:
        raise AdmissionError("authenticated GitHub metadata is required; caller JSON has no authority")
    try:
        records = added_lines(metadata.base_text, metadata.head_text)
        round_record = evidence_object(metadata.round_record_json)
        source_assignment = Round.from_record(round_record)
    except ValueError as exc:
        raise AdmissionError(f"malformed registry or round metadata: {exc}") from exc
    if len(records) != 1:
        raise AdmissionError("PR must append exactly one commitment")
    try:
        commitment = Commitment.from_record(records[0])
    except StrategyError as exc:
        raise AdmissionError(f"malformed commitment: {exc}") from exc
    if commitment.round_id != round_id:
        raise AdmissionError("commitment round mismatch")
    with store.lock(round_id):
        window = store.load(round_id)
        issues = gate(
            record=records[0],
            round_record=round_record,
            receipts=intake.read_receipts(),
            base_text=metadata.base_text,
            head_text=metadata.head_text,
            changed_paths=list(metadata.changed_paths),
            pull_request_author=metadata.author,
        )
        if issues:
            raise AdmissionError("; ".join(issues))
        source_identity = assignment_identity(source_assignment)
        if source_identity != assignment_identity(window.assignment):
            raise AdmissionError("source round assignment differs from stored round assignment")
        if window.assignment.state != OPEN:
            raise AdmissionError("stored round assignment is not open for admission")
        if commitment.task_ids != (window.task_id,):
            raise AdmissionError("commitment must name exactly this round's task")
        receipt = receipt_for_digest(
            intake.read_receipts(), round_id=round_id, digest=commitment.bundle_sha256, miner_id=metadata.author
        )
        if receipt is None:
            raise AdmissionError("no matching receipt")
        if any(intake.identity[k] != store.identity[k] for k in ("mode", "namespace")):
            raise AdmissionError("intake and round trust domains differ")
        root = intake.verify(receipt)
        record = {
            "repository": metadata.repository,
            "pr_number": metadata.number,
            "author": metadata.author,
            "head_sha": metadata.head_sha,
            "base_sha": metadata.base_sha,
            "round_id": round_id,
            "registry_delta": records[0],
            "registry_base_digest": digest(metadata.base_text),
            "registry_head_digest": digest(metadata.head_text),
            "receipt_origin": receipt.origin,
            "submission_id": receipt.submission_id,
            "bundle_sha256": receipt.bundle_sha256,
            "assignment_identity": source_identity,
            "round_identity": round_identity(window),
            "epoch": window.challenge.epoch,
            "task_id": window.task_id,
            "origin": store.identity,
            "authenticated_as": metadata.authenticated_as,
        }
        record["admission_id"] = digest(record)
        admissions = getattr(window, "admissions", {})
        prior = admissions.get(metadata.author)
        if prior is not None:
            verify_admission(prior)
            if prior != record:
                raise AdmissionError("miner already admitted a different commitment")
            return prior
        if metadata.author in window.submissions:
            raise AdmissionError("standing submission has no matching trusted admission")
        stamp = time.time() if now is None else now
        result = window.submit(
            metadata.author,
            paths=sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()),
            payload_digest=receipt.bundle_sha256,
            received_at=stamp,
        )
        if result.outcome != "accepted":
            raise AdmissionError(f"round refused admission: {result.outcome}")
        window.admissions = {**admissions, metadata.author: record}
        store.save(window)
        return record


def main(argv: list[str] | None = None, *, transport: Callable[..., Any] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--round", required=True, dest="round_id")
    parser.add_argument("--head", default=None)
    parser.add_argument("--credential-env", default="GH_TOKEN")
    parser.add_argument("--store", type=Path, default=None)
    parser.add_argument("--intake-root", type=Path, default=None)
    parser.add_argument("--receipts", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        source = GitHubSource(args.repository, credential_env=args.credential_env, transport=transport)
        metadata = source.collect(args.pr, round_id=args.round_id, expected_head=args.head)
        intake = Intake()
        if args.intake_root is not None:
            intake.root = args.intake_root
        if args.receipts is not None:
            intake.receipts = args.receipts
        result = admit(metadata=metadata, round_id=args.round_id, store=RoundStore(args.store), intake=intake)
    except (AdmissionError, IntakeError, StoreError, ValueError, KeyError, TypeError) as exc:
        print(f"admission refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
