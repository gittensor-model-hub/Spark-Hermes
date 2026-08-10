"""Publish committed changes to ``main``, with a squash-merge-PR fallback.

A branch-protection ruleset ("changes must be made through a pull request", GH013)
blocks the automation bot's *direct* push to ``main``. This lands the same change
through a short-lived PR instead — the mechanism the canonical-pin refresh already
uses (``eval.update_canonical_pin``), generalized so the training-track
ledger/frontier writer can reuse it.
"""

from __future__ import annotations

import subprocess
import sys


def _git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def _is_branch_protection_error(stderr: str) -> bool:
    s = (stderr or "").lower()
    return "gh013" in s or "protected branch" in s or "must be made through a pull request" in s


def publish_paths_to_main(
    pathspec: str,
    *,
    commit_message: str,
    pr_branch: str,
    pr_title: str,
    pr_body: str,
) -> list[str]:
    """Commit changes under ``pathspec`` and land them on ``main``.

    Tries a direct push; on a branch-protection rejection, re-lands the same commit
    through a squash-merge PR (``gh pr create`` + ``gh pr merge``). Returns ``[]`` on
    success or when there is nothing to commit, else a list of error strings.
    """
    status = _git(["git", "status", "--porcelain", "--", pathspec])
    if status.returncode != 0:
        return [status.stderr.strip() or "git status failed"]
    if not status.stdout.strip():
        print(f"no changes under {pathspec} to publish", file=sys.stderr)
        return []

    for step in (["git", "add", "--", pathspec], ["git", "commit", "-m", commit_message]):
        result = _git(step)
        if result.returncode != 0:
            return [result.stderr.strip() or result.stdout.strip() or f"git {step[1]} failed"]

    push = _git(["git", "push", "origin", "HEAD:main"])
    if push.returncode == 0:
        print("published on main", file=sys.stderr)
        return []
    if not _is_branch_protection_error(push.stderr):
        return [push.stderr.strip() or push.stdout.strip() or "git push failed"]

    # Direct push blocked by branch protection — re-land the commit through a PR.
    print("direct push blocked by branch protection; opening a PR", file=sys.stderr)
    for step in (
        ["git", "checkout", "-B", pr_branch],
        ["git", "push", "-u", "origin", f"HEAD:{pr_branch}"],
    ):
        result = _git(step)
        if result.returncode != 0:
            return [result.stderr.strip() or result.stdout.strip() or f"git {step[1]} failed"]

    create = _git(["gh", "pr", "create", "--title", pr_title, "--body", pr_body, "--base", "main", "--head", pr_branch])
    if create.returncode != 0:
        return [create.stderr.strip() or create.stdout.strip() or "gh pr create failed"]

    pr_number = create.stdout.strip().split("/")[-1]
    merge = _git(["gh", "pr", "merge", pr_number, "--squash", "--delete-branch"])
    if merge.returncode != 0:
        return [merge.stderr.strip() or merge.stdout.strip() or "gh pr merge failed"]
    print(f"published on main via PR #{pr_number}", file=sys.stderr)
    return []
