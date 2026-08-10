"""Assemble a rollout submission from a pull request and run the gate on it.

`eval.rollout_track.gate` decides one submission and takes structured inputs -- two texts,
a round record, a manifest, the miner's attestation, the changed paths, a directory of
fetched exports. Something has to turn a pull request into those, and until now nothing did,
which is why a complete and adversarially-tested gate had no production caller.

The shape mirrors `eval.training_track_gate` because that gate already learned this the hard
way and its workflow encodes the lessons: read the submission, never execute it; resolve
everything from the base ref; fail closed on anything that cannot be checked.

**Nothing here executes the submission.** The head ref is read with `git show`, the exports
are downloaded as data, and the only code that runs is this file and what it imports -- all
of it from the base ref. A rollout PR may only touch `datasets/rollouts.jsonl`, and
`check_paths` refuses anything else, but that refusal is a check inside the gate rather than
a property of how the gate is invoked. Both are needed: one stops a submission that adds
code, the other stops that code from running while the gate decides whether to reject it.

**The exports are fetched at the pinned revision or not at all.** `check_exports` refuses
when `export_dir` is None -- a submission whose published rows nobody read cannot be
accepted, and that refusal is load-bearing: a miner who published nothing at all was
accepted before it existed. So a download failure here produces no export directory and the
gate rejects, rather than being waved through as "could not check".
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from eval.hf_pin import require_revision

# SubmissionError comes from the gate rather than being redeclared here. Defining a second
# class of the same name meant `except SubmissionError` caught only the local one, so a
# malformed row -- which `added_lines` already rejects with exactly this error -- escaped as
# a traceback instead of the rejection it is.
from eval.rollout_track import (
    ACCEPT,
    REJECT,
    ROLLOUT_REGISTRY,
    ROUNDS_DIR,
    SubmissionError,
    added_lines,
    gate,
)


def git_show(ref: str, path: str) -> str:
    """One file at one ref, or empty. Reading, never checking out."""
    result = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True, check=False)
    return result.stdout if result.returncode == 0 else ""


def submitted_row(base_text: str, head_text: str) -> dict[str, Any]:
    """The single row this PR adds.

    Exactly one: a PR adding two rows is either two submissions sharing one review or a
    mistake, and both are better refused than half-processed.
    """
    added = added_lines(base_text, head_text)
    if not added:
        raise SubmissionError("the PR adds no row to the rollout registry")
    if len(added) > 1:
        raise SubmissionError(f"the PR adds {len(added)} rows; one submission per pull request")
    record = added[0]
    if not isinstance(record, dict):
        raise SubmissionError("the added row must be a JSON object")
    return record


def load_round(round_id: str, base_ref: str) -> dict[str, Any]:
    """The round announcement, read from the BASE ref.

    From the base deliberately. A round record read from the head is a record the submitter
    could have written, and the assignment it encodes is the thing being checked.
    """
    text = git_show(base_ref, (ROUNDS_DIR / f"{round_id}.json").as_posix())
    if not text.strip():
        raise SubmissionError(f"round {round_id!r} has no announcement in the base ref; it was never opened")
    return json.loads(text)


def fetch_exports(record: dict[str, Any], into: Path) -> Path | None:
    """Download the published exports at the pinned revision, or None.

    Returning None on failure is not a shrug -- `check_exports` treats it as a refusal. The
    alternative, skipping the check when the download fails, is how a submission that
    published nothing at all was once accepted.
    """
    from huggingface_hub import snapshot_download

    repo = str(record.get("hf_url", "")).rstrip("/").split("huggingface.co/")[-1]
    repo = repo.removeprefix("datasets/")
    try:
        revision = require_revision(record.get("hf_revision"))
        return Path(
            snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                revision=revision,
                local_dir=str(into),
                # attestation.json is downloaded because the gate verifies it from here. It is
                # safe to read from the miner's own snapshot: its tokens are signed by NVIDIA
                # and Intel and commit to this bundle's claim digest, so editing it breaks a
                # signature and substituting another miner's breaks the binding.
                allow_patterns=["*.jsonl", "manifest.json", "attestation.json"],
            )
        )
    except Exception as exc:  # noqa: BLE001 -- any fetch failure is a rejection, not a crash
        print(f"could not fetch exports from {repo}: {exc}", file=sys.stderr)
        return None


def load_attestation(export_dir: Path | None) -> dict[str, Any] | None:
    """The miner's published attestation, read from the exports it covers.

    Read from the export snapshot rather than fetched from a service, and that is safe here
    in a way it would not have been for a Cathedral receipt. A receipt was an assertion
    signed by a third party about a run, so a copy supplied by the submitter was a document
    the submitter chose to hand over; it had to come from the issuing API. This is a bundle
    of tokens signed by NVIDIA and Intel that *commit to the exports it sits beside*. A
    miner who edits it breaks a signature, and a miner who substitutes someone else's
    breaks the binding to their own claim digest.

    Which also means no API key, no fetch step, and no shared secret in the workflow.
    """
    if export_dir is None:
        return None
    path = export_dir / "attestation.json"
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return record if isinstance(record, dict) else None


def decide(
    *,
    base_ref: str,
    head_ref: str,
    changed_paths: list[str],
    workdir: Path,
) -> tuple[Any, dict[str, Any], list[str]]:
    """Assemble everything the gate needs and run it. Returns (GateResult, row, extra issues)."""

    base_text = git_show(base_ref, ROLLOUT_REGISTRY.as_posix())
    head_text = git_show(head_ref, ROLLOUT_REGISTRY.as_posix())
    record = submitted_row(base_text, head_text)

    round_record = load_round(str(record.get("round_id", "")), base_ref)
    export_dir = fetch_exports(record, workdir / "exports")
    manifest_path = (export_dir / "manifest.json") if export_dir else None
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path and manifest_path.is_file() else {}
    )
    attestation = load_attestation(export_dir)

    result = gate(
        base_text=base_text,
        head_text=head_text,
        round_record=round_record,
        manifest=manifest,
        attestation=attestation,
        changed_paths=changed_paths,
        export_dir=export_dir,
    )
    return result, record, []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-ref", required=True, help="the trusted base ref; everything is resolved from it")
    parser.add_argument("--head-ref", required=True, help="the PR head, READ ONLY -- never checked out or executed")
    parser.add_argument("--changed-paths-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="write the gate report here")
    args = parser.parse_args(argv)

    changed = [p.strip() for p in args.changed_paths_file.read_text(encoding="utf-8").splitlines() if p.strip()]

    with tempfile.TemporaryDirectory(prefix="rollout-gate-") as tmp:
        try:
            result, record, extra = decide(
                base_ref=args.base_ref,
                head_ref=args.head_ref,
                changed_paths=changed,
                workdir=Path(tmp),
            )
        except (SubmissionError, json.JSONDecodeError) as exc:
            # A submission that cannot be assembled is rejected with the reason, not crashed
            # on: the miner needs to know what to fix, and a traceback in a CI log is not that.
            print(f"{REJECT} {exc}", file=sys.stderr)
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps({"verdict": REJECT, "issues": [str(exc)]}, indent=2), encoding="utf-8")
            return 1

        issues = [*extra, *result.issues]
        verdict = REJECT if extra else result.verdict
        report = {"verdict": verdict, "issues": issues, "row": record}
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)
        print(verdict, file=sys.stderr)
        return 0 if verdict == ACCEPT else 1


if __name__ == "__main__":
    raise SystemExit(main())
