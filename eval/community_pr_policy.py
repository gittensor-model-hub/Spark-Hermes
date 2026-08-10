"""Auto-close community PRs that are not optimization submissions.

Policy (see `.gittensor/weights.json`): the eval/scoring harness, tooling, and docs
are maintainer-owned. Community contributors compete on the two rewarded tracks —
training (recipe/hyperparameter improvement) and dataset (verified rows). A PR stays
open only when it is either:

  * from a trusted author — OWNER / MEMBER / COLLABORATOR, or a bot (e.g. dependabot); or
  * an optimization submission — a training-track or dataset-track PR.

Everything else from the community is commented on and closed. This never executes
untrusted PR code: it reads the PR body, the changed path names, and author metadata
only, so it is safe to run from `pull_request_target`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from eval.training_track_gate import is_dataset_track_pr, is_training_track_pr

TRUSTED_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})
_DATASET_REGISTRY_PATH = "datasets/registry.jsonl"

_CLOSE_COMMENT = (
    "## Closed automatically\n\n"
    "Community pull requests here are limited to **training-track or dataset-track "
    "optimization submissions**. The eval/scoring harness, tooling, and docs are "
    "maintainer-owned (see `.gittensor/weights.json` and `CONTRIBUTING.md`), so a PR that "
    "changes them is closed.\n\n"
    "**Found a bug or have an improvement idea? Please open a detailed issue instead of a "
    "pull request** — describe the problem, its impact, and your suggested fix. Maintainers "
    "triage issues and make the harness/tooling/docs changes themselves.\n\n"
    "If you intended a training or dataset submission, re-open with the *Training/evaluation "
    "improvement* or *Dataset track submission* checkbox plus the required proof-bundle / "
    "registry artifact."
)


def _is_bot(author_login: str, author_type: str) -> bool:
    return (author_type or "").strip().lower() == "bot" or (author_login or "").endswith("[bot]")


def is_trusted_author(author_association: str, author_login: str, author_type: str) -> bool:
    """Maintainers (OWNER/MEMBER/COLLABORATOR) and bots are exempt from auto-close."""
    if (author_association or "").strip().upper() in TRUSTED_ASSOCIATIONS:
        return True
    return _is_bot(author_login, author_type)


def is_optimization_pr(pr_body: str | None, changed_paths: list[str] | None) -> bool:
    """Whether a PR is a training-track or dataset-track optimization submission.

    Detected by the track checkbox in the body, or by touching a track's submission
    artifact — a recipe file (training) or the dataset registry (dataset).
    """
    body = pr_body or ""
    if is_training_track_pr(body) or is_dataset_track_pr(body):
        return True
    for path in changed_paths or []:
        if path.startswith("recipes/") and path.endswith((".yaml", ".yml")):
            return True
        if path == _DATASET_REGISTRY_PATH:
            return True
    return False


def should_close_community_pr(
    *,
    author_association: str,
    author_login: str,
    author_type: str,
    pr_body: str | None,
    changed_paths: list[str] | None,
) -> bool:
    """Close only when the author is not trusted AND the PR is not an optimization."""
    if is_trusted_author(author_association, author_login, author_type):
        return False
    return not is_optimization_pr(pr_body, changed_paths)


def close_community_pr(pr_number: int) -> list[str]:
    """Comment the policy, then close the PR."""
    comment = subprocess.run(
        ["gh", "pr", "comment", str(pr_number), "--body", _CLOSE_COMMENT],
        capture_output=True,
        text=True,
        check=False,
    )
    if comment.returncode != 0:
        return [comment.stderr.strip() or comment.stdout.strip() or "gh pr comment failed"]
    close = subprocess.run(
        ["gh", "pr", "close", str(pr_number)],
        capture_output=True,
        text=True,
        check=False,
    )
    if close.returncode != 0:
        return [close.stderr.strip() or close.stdout.strip() or "gh pr close failed"]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--author-association", default="")
    parser.add_argument("--author-login", default="")
    parser.add_argument("--author-type", default="")
    parser.add_argument("--pr-body-file", type=Path, default=None)
    parser.add_argument("--changed-paths-file", type=Path, default=None)
    parser.add_argument("--pr-number", type=int, default=None)
    parser.add_argument("--apply", action="store_true", help="actually comment + close (default: dry-run)")
    args = parser.parse_args(argv)

    pr_body = args.pr_body_file.read_text(encoding="utf-8") if args.pr_body_file else None
    changed_paths = None
    if args.changed_paths_file:
        changed_paths = [
            line.strip() for line in args.changed_paths_file.read_text(encoding="utf-8").splitlines() if line.strip()
        ]

    close = should_close_community_pr(
        author_association=args.author_association,
        author_login=args.author_login,
        author_type=args.author_type,
        pr_body=pr_body,
        changed_paths=changed_paths,
    )
    if not close:
        if is_trusted_author(args.author_association, args.author_login, args.author_type):
            reason = "trusted author (maintainer/bot)"
        else:
            reason = "optimization submission (training/dataset track)"
        print(f"keep-open: {reason}", file=sys.stderr)
        return 0

    print(f"close: community non-optimization PR by {args.author_login or 'unknown'}", file=sys.stderr)
    if args.apply and args.pr_number is not None:
        issues = close_community_pr(args.pr_number)
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)
        if issues:
            return 1
        print(f"closed PR #{args.pr_number}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
