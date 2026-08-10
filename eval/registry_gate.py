"""Automated gate for datasets/registry.jsonl pull requests.

Miners append one JSON line per HF dataset submission. This module parses the
added lines, rejects malformed or duplicate registry entries, and runs full
SparkProof production verification against each HF repo's `proof/` artifacts
before merge.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from eval.dataset_verify import MERGE_THRESHOLD_ROWS, REWARDED_DATASET_LABELS, verify_dataset_submission
from eval.fair_dataset_label import apply_fair_label, compute_rows_selected_for_entry
from eval.gpu_architecture import dataset_architecture_allowed, normalize_gpu_architecture

REGISTRY_PATH = Path("datasets/registry.jsonl")
REQUIRED_FIELDS = ("miner", "hf_url", "trajectories_sha256", "rows_total", "dataset_version", "gpu_architecture")
_HF_REPO_RE = re.compile(r"^https://huggingface\.co/datasets/([^/]+/[^/#?]+)")
_DATASET_TRACK_CHECKBOX_RE = re.compile(
    r"^\s*-\s*\[[xX]\]\s+\*{0,2}Dataset track submission\*{0,2}\s*$",
    re.MULTILINE,
)
_AUTO_CLOSE_LABELS = frozenset({"dataset:REJECT", "dataset:none"})
DATASET_LABELS = REWARDED_DATASET_LABELS | _AUTO_CLOSE_LABELS
_LABEL_COLORS = {
    "dataset:xl": "1d76db",
    "dataset:l": "0e8a16",
    "dataset:m": "2cbe4e",
    "dataset:s": "7bd88f",
    "dataset:xs": "c5def5",
    "dataset:none": "d4c5f9",
    "dataset:REJECT": "b60205",
}


def _load_registry_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def parse_added_registry_lines(base_text: str, head_text: str) -> list[dict[str, Any]]:
    """Return JSON objects newly appended in `head_text` relative to `base_text`.

    Raises ValueError for invalid JSON or a non-object JSON value. Callers that
    drive the CI gate (``gate_registry_pr``) must catch that and return a clean
    ``dataset:REJECT`` instead of letting the traceback fail the job.
    """
    base_lines = {line.strip() for line in base_text.splitlines() if line.strip()}
    added: list[dict[str, Any]] = []
    for line_no, line in enumerate(head_text.splitlines(), start=1):
        line = line.strip()
        if not line or line in base_lines:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"added registry line {line_no}: invalid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"added registry line {line_no}: must be a JSON object, got {type(obj).__name__}")
        added.append(obj)
    return added


def validate_append_only_registry(base_text: str, head_text: str) -> list[str]:
    """Require the PR registry to preserve every base line, in order, then append."""
    base_lines = [line.strip() for line in base_text.splitlines() if line.strip()]
    head_lines = [line.strip() for line in head_text.splitlines() if line.strip()]
    if head_lines[: len(base_lines)] != base_lines:
        return [
            "datasets/registry.jsonl is append-only; rebase onto the latest base "
            "and preserve every existing line in order"
        ]
    return []


def is_dataset_track_pr(pr_body: str | None) -> bool:
    """Return whether the machine-readable dataset-track checkbox is selected."""
    return bool(_DATASET_TRACK_CHECKBOX_RE.search(pr_body or ""))


def validate_changed_paths(changed_paths: list[str] | None) -> list[str]:
    """Dataset PRs are data-only: auto-merge must never carry executable changes."""
    if changed_paths is None:
        return []
    unexpected = sorted({path for path in changed_paths if path != REGISTRY_PATH.as_posix()})
    if unexpected:
        return [f"dataset-track PRs may only change datasets/registry.jsonl; unexpected paths: {unexpected!r}"]
    return []


def reward_eligible(report: dict[str, Any]) -> bool:
    """A verified dataset earns a reward at the `dataset:xs` floor (25 rows) or above."""
    return bool(report.get("verified")) and report.get("label") in REWARDED_DATASET_LABELS


def merge_eligible(report: dict[str, Any]) -> bool:
    """Only rewarded, fully verified dataset submissions may be auto-merged."""
    return reward_eligible(report)


def update_pr_dataset_label(pr_number: int, label: str) -> list[str]:
    """Replace any existing dataset:* PR label with the gate's computed label."""
    if label not in DATASET_LABELS:
        return [f"refusing to apply unknown dataset label {label!r}"]

    create = subprocess.run(
        [
            "gh",
            "label",
            "create",
            label,
            "--force",
            "--color",
            _LABEL_COLORS[label],
            "--description",
            "SparkDistill dataset registry gate result",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if create.returncode != 0:
        return [f"could not create/update GitHub label {label!r}: {(create.stderr or create.stdout).strip()}"]

    current = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{{owner}}/{{repo}}/issues/{pr_number}/labels",
            "--jq",
            ".[].name",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if current.returncode != 0:
        return [f"could not read PR labels: {(current.stderr or current.stdout).strip()}"]

    issues: list[str] = []
    current_labels = {line.strip() for line in current.stdout.splitlines() if line.strip()}
    for stale in sorted((current_labels & DATASET_LABELS) - {label}):
        remove = subprocess.run(
            [
                "gh",
                "api",
                "--method",
                "DELETE",
                f"repos/{{owner}}/{{repo}}/issues/{pr_number}/labels/{stale}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if remove.returncode != 0:
            issues.append(f"could not remove stale label {stale!r}: {(remove.stderr or remove.stdout).strip()}")

    if label not in current_labels:
        add = subprocess.run(
            [
                "gh",
                "api",
                "--method",
                "POST",
                f"repos/{{owner}}/{{repo}}/issues/{pr_number}/labels",
                "-f",
                f"labels[]={label}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if add.returncode != 0:
            issues.append(f"could not apply label {label!r}: {(add.stderr or add.stdout).strip()}")
    return issues


def _pr_state(pr_number: int) -> str | None:
    """Return GitHub PR state (OPEN, MERGED, CLOSED) or None on lookup failure."""
    result = subprocess.run(
        ["gh", "pr", "view", str(pr_number), "--json", "state"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout or "{}").get("state")
    except json.JSONDecodeError:
        return None


def close_dataset_pr(
    pr_number: int,
    *,
    label: str = "dataset:REJECT",
    issues: list[str] | None = None,
) -> list[str]:
    """Close a dataset-track PR that is not merge-eligible."""
    if label == "dataset:none":
        summary = "Closed automatically: verified proof is below the 25-row merge threshold (dataset:none)."
    else:
        summary = "Closed automatically: dataset registry gate rejected this submission."
    if issues:
        bullets = "\n".join(f"- {issue}" for issue in issues[:8])
        summary = f"{summary}\n\n{bullets}"
    result = subprocess.run(
        ["gh", "pr", "close", str(pr_number), "--comment", summary],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return [f"could not close PR #{pr_number}: {(result.stderr or result.stdout).strip()}"]
    return []


def hf_repo_from_url(url: str) -> str:
    match = _HF_REPO_RE.match(url.strip())
    if not match:
        raise ValueError(f"hf_url must be https://huggingface.co/datasets/<org>/<repo>, got {url!r}")
    return match.group(1)


def validate_registry_entry(entry: dict[str, Any]) -> list[str]:
    if not isinstance(entry, dict):
        return ["registry entry must be a JSON object"]
    issues: list[str] = []
    for field in REQUIRED_FIELDS:
        if not entry.get(field):
            issues.append(f"missing required field: {field}")
    if entry.get("trajectories_sha256") and len(str(entry["trajectories_sha256"])) != 64:
        issues.append("trajectories_sha256 must be a 64-char sha256 hex digest")
    try:
        rows_total = int(entry.get("rows_total", 0))
        if rows_total <= 0:
            issues.append("rows_total must be positive")
    except (TypeError, ValueError):
        issues.append("rows_total must be an integer")
    try:
        hf_repo_from_url(str(entry.get("hf_url", "")))
    except ValueError as exc:
        issues.append(str(exc))
    if entry.get("gpu_architecture"):
        gpu_architecture = normalize_gpu_architecture(str(entry["gpu_architecture"]))
        if gpu_architecture is None:
            issues.append(f"gpu_architecture {entry['gpu_architecture']!r} is not recognized")
        elif not dataset_architecture_allowed(gpu_architecture):
            issues.append(f"gpu_architecture {gpu_architecture!r} is not an accepted dataset-generation architecture")
    return issues


def check_registry_duplicates(
    existing: list[dict[str, Any]],
    new_entries: list[dict[str, Any]],
) -> list[str]:
    issues: list[str] = []
    seen_hf: set[str] = set()
    seen_sha: set[str] = set()
    for row in existing:
        # Malformed hf_url / trajectories_sha256 on prior registry lines must not
        # crash duplicate detection for a new submission (same class as #173).
        raw_url = row.get("hf_url")
        try:
            repo = hf_repo_from_url(raw_url) if raw_url else None
        except ValueError:
            repo = None
        if repo is not None:
            seen_hf.add(repo)
        sha = row.get("trajectories_sha256")
        if sha:
            seen_sha.add(sha)

    for entry in new_entries:
        # A missing or malformed hf_url / trajectories_sha256 is already reported by
        # validate_registry_entry; skip it here so a bad line yields a clean
        # dataset:REJECT instead of crashing the gate on the duplicate check.
        raw_url = entry.get("hf_url")
        try:
            repo = hf_repo_from_url(raw_url) if raw_url else None
        except ValueError:
            repo = None
        if repo is not None:
            if repo in seen_hf:
                issues.append(f"duplicate hf_url repo already in registry: {repo}")
            seen_hf.add(repo)

        sha = entry.get("trajectories_sha256")
        if sha:
            if sha in seen_sha:
                issues.append(f"duplicate trajectories_sha256 already in registry: {sha}")
            seen_sha.add(sha)
    return issues


def gate_registry_submission(
    entry: dict[str, Any],
    *,
    sparkproof_root: Path,
    existing_registry: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate one registry entry end-to-end."""
    issues = validate_registry_entry(entry)
    if existing_registry is not None:
        issues.extend(check_registry_duplicates(existing_registry, [entry]))

    # A malformed hf_url is already reported by validate_registry_entry (so `issues`
    # is non-empty and this function returns a clean dataset:REJECT below); don't let
    # parsing it for the report field crash the gate first.
    try:
        hf_repo = hf_repo_from_url(entry["hf_url"]) if entry.get("hf_url") else ""
    except ValueError:
        hf_repo = ""
    report: dict[str, Any] = {
        "miner": entry.get("miner"),
        "hf_repo": hf_repo,
        "trajectories_sha256": entry.get("trajectories_sha256"),
        "issues": issues,
        "verified": False,
        "label": "dataset:REJECT",
    }
    if issues:
        return report

    verification = verify_dataset_submission(
        claimed_sha256=entry["trajectories_sha256"],
        sparkproof_root=sparkproof_root,
        hf_repo=hf_repo,
        production=True,
    )
    report.update(verification)
    if int(entry.get("rows_total", 0)) != verification.get("rows_total", -1):
        report["issues"] = list(report.get("issues") or []) + [
            f"rows_total mismatch: PR claims {entry['rows_total']} but verified bundle has {verification.get('rows_total')}"
        ]
        report["verified"] = False
        report["label"] = "dataset:REJECT"
    claimed_arch = normalize_gpu_architecture(str(entry.get("gpu_architecture") or ""))
    if claimed_arch != verification.get("gpu_architecture"):
        report["issues"] = list(report.get("issues") or []) + [
            f"gpu_architecture mismatch: PR claims {entry.get('gpu_architecture')!r} "
            f"but verified bundle has {verification.get('gpu_architecture')!r}"
        ]
        report["verified"] = False
        report["label"] = "dataset:REJECT"
    return report


def gate_registry_pr(
    *,
    base_registry_text: str,
    head_registry_text: str,
    sparkproof_root: Path,
    pr_body: str | None = None,
    changed_paths: list[str] | None = None,
    mining_dataset_repo_id: str | None = None,
    publish_mining_dataset: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Gate every newly appended registry line in a PR."""
    preflight_issues = validate_changed_paths(changed_paths)
    preflight_issues.extend(validate_append_only_registry(base_registry_text, head_registry_text))
    if pr_body is not None and not is_dataset_track_pr(pr_body):
        preflight_issues.append("check 'Dataset track submission' in the pull request template")
    if preflight_issues:
        return {
            "verified": False,
            "reward_eligible": False,
            "merge_eligible": False,
            "label": "dataset:REJECT",
            "issues": preflight_issues,
            "submissions": [],
        }

    existing = []
    for line in base_registry_text.splitlines():
        line = line.strip()
        if line:
            existing.append(json.loads(line))

    # Invalid / non-object JSON in the appended line used to raise ValueError /
    # AttributeError out of the gate; catch it so CI gets dataset:REJECT + a
    # helpful close comment instead of a traceback (same class as #173).
    try:
        added = parse_added_registry_lines(base_registry_text, head_registry_text)
    except ValueError as exc:
        return {
            "verified": False,
            "reward_eligible": False,
            "merge_eligible": False,
            "label": "dataset:REJECT",
            "issues": [str(exc)],
            "submissions": [],
        }
    if not added:
        return {
            "verified": False,
            "reward_eligible": False,
            "merge_eligible": False,
            "label": "dataset:REJECT",
            "issues": ["PR must append exactly one new line to datasets/registry.jsonl"],
            "submissions": [],
        }
    if len(added) != 1:
        return {
            "verified": False,
            "reward_eligible": False,
            "merge_eligible": False,
            "label": "dataset:REJECT",
            "issues": [f"PR must append exactly one registry line per submission (got {len(added)})"],
            "submissions": [],
        }

    report = gate_registry_submission(added[0], sparkproof_root=sparkproof_root, existing_registry=existing)
    issues = list(report.get("issues", []))

    if report.get("verified"):
        from eval.mining_dataset import aggregate_registry_text, mining_dedupe_mode

        proposed_registry = aggregate_registry_text(base_registry_text, head_registry_text)
        fair_mix = compute_rows_selected_for_entry(
            proposed_registry,
            str(added[0]["trajectories_sha256"]),
            sparkproof_root=sparkproof_root,
            dedupe=mining_dedupe_mode(),
        )
        if not fair_mix.get("verified"):
            report["verified"] = False
            report["label"] = "dataset:REJECT"
            issues.extend(list(fair_mix.get("issues") or ["fair-label mix failed"]))
        else:
            report = apply_fair_label(report, rows_selected=int(fair_mix["rows_selected"]))
            if report.get("fair_label_note"):
                issues.append(str(report["fair_label_note"]))

    eligible = reward_eligible(report)
    if report.get("verified") and not eligible:
        issues.append(
            f"dataset proof is valid but fewer than {MERGE_THRESHOLD_ROWS} canonical-mix rows does not meet "
            "the dataset:xs merge/reward threshold"
        )

    mining_report: dict[str, Any] | None = None
    merge_eligible_flag = eligible
    if eligible and mining_dataset_repo_id:
        from eval.mining_dataset import aggregate_and_publish_mining_dataset, aggregate_registry_text

        proposed_registry = aggregate_registry_text(base_registry_text, head_registry_text)
        try:
            publish = publish_mining_dataset or aggregate_and_publish_mining_dataset
            mining_report = publish(
                proposed_registry,
                repo_id=mining_dataset_repo_id,
                sparkproof_root=sparkproof_root,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            merge_eligible_flag = False
            issues.append(f"mining dataset aggregation failed: {exc}")
        else:
            if not mining_report.get("published"):
                merge_eligible_flag = False
                issues.extend(list(mining_report.get("issues") or ["mining dataset publish failed"]))
            else:
                from eval.export_registry_snapshot import verify_remote_registry_snapshot

                snapshot_issues = verify_remote_registry_snapshot(
                    proposed_registry,
                    repo_id=mining_dataset_repo_id,
                    hf_token=os.environ.get("HF_TOKEN"),
                    sparkproof_root=sparkproof_root,
                )
                if snapshot_issues:
                    merge_eligible_flag = False
                    issues.extend(snapshot_issues)

    return {
        "verified": report.get("verified", False),
        "reward_eligible": eligible,
        "merge_eligible": merge_eligible_flag,
        "label": report.get("label"),
        "issues": issues,
        "submissions": [report],
        "mining_dataset": mining_report,
    }


def _git_show(ref: str, path: Path) -> str:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path.as_posix()}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout


def main(argv: list[str] | None = None) -> int:
    from eval.mining_dataset import DEFAULT_MINING_DATASET_REPO, mining_dataset_repo

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-ref", default="origin/main", help="git ref for the merge base registry")
    parser.add_argument("--head-ref", default="HEAD", help="git ref for the PR head registry")
    parser.add_argument("--sparkproof-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--pr-body-file",
        type=Path,
        default=None,
        help="PR body file; when supplied, the dataset-track checkbox must be selected",
    )
    parser.add_argument(
        "--changed-paths-file",
        type=Path,
        default=None,
        help="newline-delimited changed paths; dataset auto-merge permits only the registry",
    )
    parser.add_argument(
        "--apply-label",
        action="store_true",
        help="replace the PR's dataset:* label with the computed gate label (CI only)",
    )
    parser.add_argument(
        "--merge-on-pass",
        action="store_true",
        help="merge the current GitHub PR when verification passes (CI only)",
    )
    parser.add_argument(
        "--close-on-reject",
        action="store_true",
        help="close the GitHub PR when the gate label is dataset:REJECT or dataset:none (CI only)",
    )
    parser.add_argument("--pr-number", type=int, default=None)
    parser.add_argument(
        "--mining-dataset-repo",
        default=None,
        help=f"canonical HF datasets repo to update before merge (default: env or {DEFAULT_MINING_DATASET_REPO})",
    )
    parser.add_argument(
        "--skip-mining-publish",
        action="store_true",
        help="skip aggregating into the canonical mining dataset (local dev only)",
    )
    args = parser.parse_args(argv)

    base_text = _git_show(args.base_ref, REGISTRY_PATH)
    head_text = _git_show(args.head_ref, REGISTRY_PATH)
    if not head_text.strip():
        head_path = REGISTRY_PATH
        if head_path.exists():
            head_text = head_path.read_text(encoding="utf-8")

    pr_body = args.pr_body_file.read_text(encoding="utf-8") if args.pr_body_file else None
    changed_paths = None
    if args.changed_paths_file:
        changed_paths = [
            line.strip() for line in args.changed_paths_file.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    report = gate_registry_pr(
        base_registry_text=base_text,
        head_registry_text=head_text,
        sparkproof_root=args.sparkproof_root,
        pr_body=pr_body,
        changed_paths=changed_paths,
        mining_dataset_repo_id=None
        if args.skip_mining_publish
        else (args.mining_dataset_repo or mining_dataset_repo()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        f"{report['label']} verified={report['verified']} issues={len(report.get('issues') or [])}",
        file=sys.stderr,
    )
    if report.get("mining_dataset"):
        mining = report["mining_dataset"]
        print(
            f"mining dataset published={mining.get('published')} "
            f"rows={mining.get('rows_total')} url={mining.get('hf_url')}",
            file=sys.stderr,
        )
    if report.get("issues"):
        for issue in report["issues"]:
            print(f"  - {issue}", file=sys.stderr)

    if args.pr_number is not None and _pr_state(args.pr_number) == "MERGED":
        print(f"PR #{args.pr_number} already merged; skipping label/merge/close", file=sys.stderr)
        return 0

    if args.apply_label:
        if args.pr_number is None:
            print("--apply-label requires --pr-number", file=sys.stderr)
            return 1
        label_issues = update_pr_dataset_label(args.pr_number, report["label"])
        if label_issues:
            for issue in label_issues:
                print(f"  - {issue}", file=sys.stderr)
            return 1

    if args.close_on_reject and report.get("label") in _AUTO_CLOSE_LABELS and args.pr_number is not None:
        close_issues = close_dataset_pr(
            args.pr_number,
            label=str(report["label"]),
            issues=list(report.get("issues") or []),
        )
        if close_issues:
            for issue in close_issues:
                print(f"  - {issue}", file=sys.stderr)
            return 1
        print(f"closed PR #{args.pr_number}", file=sys.stderr)

    if args.merge_on_pass and report.get("merge_eligible") and args.pr_number is not None:
        merge = subprocess.run(
            ["gh", "pr", "merge", str(args.pr_number), "--merge"],
            capture_output=True,
            text=True,
            check=False,
        )
        if merge.returncode != 0:
            print(merge.stderr or merge.stdout, file=sys.stderr)
            return 1
        print(f"merged PR #{args.pr_number}", file=sys.stderr)

        from eval.update_canonical_pin import commit_canonical_pin_to_main

        fetch = subprocess.run(
            ["git", "fetch", "origin", "main"],
            capture_output=True,
            text=True,
            check=False,
        )
        if fetch.returncode != 0:
            print(fetch.stderr or fetch.stdout, file=sys.stderr)
            return 1
        checkout = subprocess.run(
            ["git", "checkout", "-B", "main", "origin/main"],
            capture_output=True,
            text=True,
            check=False,
        )
        if checkout.returncode != 0:
            print(checkout.stderr or checkout.stdout, file=sys.stderr)
            return 1
        pin_issues = commit_canonical_pin_to_main(hf_token=os.environ.get("HF_TOKEN"))
        if pin_issues:
            for issue in pin_issues:
                print(f"  - {issue}", file=sys.stderr)
            return 1

    # Proof-verified submissions below the merge threshold (`dataset:none`) still
    # succeed CI; the workflow closes them when --close-on-reject is set.
    proof_verified = bool(report.get("submissions") and report["submissions"][0].get("verified"))
    return 0 if proof_verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
