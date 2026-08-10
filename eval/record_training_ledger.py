"""CLI: append a merged training-track PR's result to runs/ledger.jsonl.

Run once, after merge (not at PR-gate time, unlike eval.training_track_gate) —
reads the PR body for its cited HF proof-bundle repo, re-runs eval.verify's
no-GPU attested checks against the merged commit, and if a report was
produced, appends the run to runs/ledger.jsonl, writes
runs/<run-id>/result.json (+ attestation.json when present), and seeds / raises
runs/frontiers.json for the run's GPU architecture. Idempotent: skips ledger
double-append when run_id is already present (frontier highs may still heal).
Non-training-track PRs (no cited HF proof-bundle repo) are a silent no-op.

    python -m eval.record_training_ledger --pr-url <merged PR URL> \\
        --pr-body-file /tmp/pr_body.md --head-ref HEAD \\
        --changed-paths-file /tmp/changed_paths.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from eval.git_publish import publish_paths_to_main
from eval.training_track_gate import record_merged_ledger_entry


def _pr_number_from_url(pr_url: str) -> str:
    tail = pr_url.rstrip("/").rsplit("/", 1)[-1]
    return tail if tail.isdigit() else "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pr-url", required=True)
    parser.add_argument("--pr-body-file", type=Path, default=None)
    parser.add_argument("--head-ref", default="HEAD")
    parser.add_argument(
        "--merge-base-ref",
        default=None,
        help="git ref for merge-base with the PR base; enables canonical-pin grace window",
    )
    parser.add_argument("--changed-paths-file", type=Path, default=None)
    parser.add_argument("--ledger-path", type=Path, default=Path("runs/ledger.jsonl"))
    parser.add_argument(
        "--publish",
        action="store_true",
        help="commit the ledger/frontier changes to main (PR fallback on branch protection)",
    )
    args = parser.parse_args(argv)

    pr_body = args.pr_body_file.read_text(encoding="utf-8") if args.pr_body_file else None
    changed_paths = None
    if args.changed_paths_file:
        changed_paths = [
            line.strip() for line in args.changed_paths_file.read_text(encoding="utf-8").splitlines() if line.strip()
        ]

    issues = record_merged_ledger_entry(
        pr_url=args.pr_url,
        pr_body=pr_body,
        head_ref=args.head_ref,
        changed_paths=changed_paths,
        merge_base_ref=args.merge_base_ref,
        hf_token=os.environ.get("HF_TOKEN"),
        ledger_path=args.ledger_path,
    )
    for issue in issues:
        print(f"  - {issue}", file=sys.stderr)
    if issues:
        return 1

    if args.publish:
        pr_number = _pr_number_from_url(args.pr_url)
        runs_dir = args.ledger_path.parent.as_posix()
        publish_issues = publish_paths_to_main(
            runs_dir,
            commit_message=f"Record ledger/frontier for PR #{pr_number}",
            pr_branch=f"chore/ledger-pr-{pr_number}",
            pr_title=f"Record ledger/frontier for PR #{pr_number}",
            pr_body=(
                f"Automated training-track ledger + frontier update for #{pr_number}.\n\n"
                "Direct push to `main` was blocked by branch protection, so the merged run's "
                "`runs/ledger.jsonl`, `runs/frontiers.json`, and `runs/<run-id>/result.json` land here."
            ),
        )
        for issue in publish_issues:
            print(f"  - {issue}", file=sys.stderr)
        if publish_issues:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
