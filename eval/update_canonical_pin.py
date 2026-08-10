"""CLI to refresh datasets/canonical.json from Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from eval.canonical_dataset import CANONICAL_PATH, write_pin_from_remote


def refresh_canonical_pin_file(
    *,
    repo_id: str | None = None,
    hf_token: str | None = None,
    out_path: Path = CANONICAL_PATH,
) -> dict:
    """Download HF mix_manifest and rewrite datasets/canonical.json."""
    return write_pin_from_remote(repo_id=repo_id, hf_token=hf_token, out_path=out_path)


def _git_run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=check)


def _pin_changed(
    out_path: Path,
    *,
    repo_id: str | None = None,
    hf_token: str | None = None,
) -> tuple[dict, bool]:
    before = out_path.read_text(encoding="utf-8") if out_path.exists() else None
    payload = refresh_canonical_pin_file(repo_id=repo_id, hf_token=hf_token, out_path=out_path)
    after = out_path.read_text(encoding="utf-8")
    return payload, before != after


def _commit_pin_change(out_path: Path) -> list[str]:
    add = _git_run(["git", "add", out_path.as_posix()], check=False)
    if add.returncode != 0:
        return [add.stderr.strip() or add.stdout.strip() or "git add failed"]

    commit = _git_run(
        [
            "git",
            "commit",
            "-m",
            "Refresh canonical mining dataset pin after registry update.\n",
        ],
        check=False,
    )
    if commit.returncode != 0:
        return [commit.stderr.strip() or commit.stdout.strip() or "git commit failed"]
    return []


def publish_canonical_pin_via_pr(
    *,
    repo_id: str | None = None,
    hf_token: str | None = None,
    out_path: Path = CANONICAL_PATH,
) -> list[str]:
    """Refresh the pin and land it on main through a squash-merge PR."""
    payload, changed = _pin_changed(out_path, repo_id=repo_id, hf_token=hf_token)
    if not changed:
        return []

    rows = (payload.get("mix_manifest") or {}).get("rows_total")
    sft_sha = (payload.get("mix_manifest") or {}).get("sft_sha256") or ""
    branch = f"chore/refresh-canonical-pin-{rows or 'unknown'}"
    if len(sft_sha) >= 8:
        branch = f"{branch}-{sft_sha[:8]}"

    checkout = _git_run(["git", "checkout", "-B", branch], check=False)
    if checkout.returncode != 0:
        return [checkout.stderr.strip() or checkout.stdout.strip() or "git checkout failed"]

    issues = _commit_pin_change(out_path)
    if issues:
        return issues

    push = _git_run(["git", "push", "-u", "origin", f"HEAD:{branch}"], check=False)
    if push.returncode != 0:
        return [push.stderr.strip() or push.stdout.strip() or "git push failed"]

    title = f"Refresh canonical pin after mining republish ({rows} rows)"
    body = (
        "Automated pin refresh from the live HF `mix_manifest.json`.\n\n"
        f"- rows_total: {rows}\n"
        f"- sft_sha256: `{sft_sha}`\n"
    )
    create = _git_run(
        [
            "gh",
            "pr",
            "create",
            "--title",
            title,
            "--body",
            body,
            "--base",
            "main",
            "--head",
            branch,
        ],
        check=False,
    )
    if create.returncode != 0:
        return [create.stderr.strip() or create.stdout.strip() or "gh pr create failed"]

    pr_number = create.stdout.strip().split("/")[-1]
    merge = _git_run(
        ["gh", "pr", "merge", pr_number, "--squash", "--delete-branch"],
        check=False,
    )
    if merge.returncode != 0:
        return [merge.stderr.strip() or merge.stdout.strip() or "gh pr merge failed"]
    return []


def commit_canonical_pin_to_main(
    *,
    repo_id: str | None = None,
    hf_token: str | None = None,
    out_path: Path = CANONICAL_PATH,
) -> list[str]:
    """Refresh the pin and publish to main (PR when branch protection blocks direct push)."""
    payload, changed = _pin_changed(out_path, repo_id=repo_id, hf_token=hf_token)
    if not changed:
        return []

    checkout = _git_run(["git", "checkout", "-B", "main"], check=False)
    if checkout.returncode != 0:
        return [checkout.stderr.strip() or checkout.stdout.strip() or "git checkout failed"]

    issues = _commit_pin_change(out_path)
    if issues:
        return issues

    push = _git_run(["git", "push", "origin", "HEAD:main"], check=False)
    if push.returncode != 0:
        stderr = push.stderr or ""
        if "GH013" in stderr or "pull request" in stderr.lower():
            reset = _git_run(["git", "reset", "--hard", "origin/main"], check=False)
            if reset.returncode != 0:
                return [reset.stderr.strip() or reset.stdout.strip() or "git reset failed"]
            return publish_canonical_pin_via_pr(
                repo_id=repo_id,
                hf_token=hf_token,
                out_path=out_path,
            )
        return [stderr.strip() or push.stdout.strip() or "git push failed"]

    rows = (payload.get("mix_manifest") or {}).get("rows_total")
    print(f"canonical pin published on main ({rows} rows)", file=sys.stderr)
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=CANONICAL_PATH,
        help="path to write datasets/canonical.json",
    )
    parser.add_argument("--repo-id", default=None, help="HF datasets repo (default: canonical default)")
    parser.add_argument(
        "--publish",
        action="store_true",
        help="commit and publish the refreshed pin to main (PR fallback on branch protection)",
    )
    parser.add_argument(
        "--publish-via-pr",
        action="store_true",
        help="always publish through a squash-merge PR",
    )
    args = parser.parse_args(argv)

    try:
        if args.publish_via_pr:
            issues = publish_canonical_pin_via_pr(
                repo_id=args.repo_id,
                hf_token=os.environ.get("HF_TOKEN"),
                out_path=args.out,
            )
            if issues:
                for issue in issues:
                    print(issue, file=sys.stderr)
                return 1
            payload = json.loads(args.out.read_text(encoding="utf-8"))
        elif args.publish:
            issues = commit_canonical_pin_to_main(
                repo_id=args.repo_id,
                hf_token=os.environ.get("HF_TOKEN"),
                out_path=args.out,
            )
            if issues:
                for issue in issues:
                    print(issue, file=sys.stderr)
                return 1
            payload = json.loads(args.out.read_text(encoding="utf-8"))
        else:
            payload = refresh_canonical_pin_file(
                repo_id=args.repo_id,
                hf_token=os.environ.get("HF_TOKEN"),
                out_path=args.out,
            )
    except Exception as exc:
        print(f"update canonical pin failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
