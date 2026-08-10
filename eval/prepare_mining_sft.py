"""Export the canonical Hugging Face mining dataset to local Axolotl jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eval.canonical_dataset import (
    CANONICAL_TRAINING_DATASET_PATH,
    canonical_hf_url,
    canonical_repo_id,
    fetch_remote_mix_manifest,
    sha256_matches_canonical_export,
    verify_manifest_matches_pin,
)
from eval.mining_dataset import DEFAULT_MINING_DATASET_REPO, mining_dataset_repo


def export_mining_sft(
    *,
    out_path: Path,
    repo_id: str | None = None,
    hf_token: str | None = None,
    verify_pin: bool = True,
    mix_manifest_out: Path | None = None,
) -> dict[str, Any]:
    """Download HF mining split and write messages-only jsonl for Axolotl."""
    from datasets import load_dataset

    repo = (repo_id or mining_dataset_repo()).strip()
    if not repo:
        raise ValueError("mining dataset repo id is empty")
    if repo != canonical_repo_id():
        raise ValueError(f"training exports must use the canonical mining repo {canonical_repo_id()!r}, got {repo!r}")

    remote_manifest = fetch_remote_mix_manifest(repo_id=repo, hf_token=hf_token)
    if verify_pin:
        pin_issues = verify_manifest_matches_pin(remote_manifest, repo_id=repo)
        if pin_issues:
            raise ValueError("; ".join(pin_issues))

    mix_manifest_path = (mix_manifest_out or out_path.parent / "mix_manifest.json").resolve()
    mix_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    mix_manifest_path.write_text(json.dumps(remote_manifest, indent=2) + "\n", encoding="utf-8")

    ds = load_dataset(repo, split="train", token=hf_token)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in ds:
            if not isinstance(row, dict):
                raise ValueError(f"{repo} produced a non-object row")
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{repo} row missing non-empty messages list")
            record: dict[str, Any] = {"messages": messages}
            metadata = row.get("metadata")
            if metadata:
                record["metadata"] = metadata
            # Must match mix_registry mining_sft.jsonl serialization for sft_sha256 pin checks.
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            rows_written += 1

    if rows_written == 0:
        raise ValueError(f"{repo} train split is empty")

    resolved_out = out_path.resolve()
    if verify_pin and resolved_out.as_posix().endswith(CANONICAL_TRAINING_DATASET_PATH):
        sha_issues = sha256_matches_canonical_export(resolved_out)
        if sha_issues:
            raise ValueError("; ".join(sha_issues))

    return {
        "repo_id": repo,
        "dataset_url": canonical_hf_url(),
        "rows_written": rows_written,
        "out_path": str(resolved_out),
        "mix_manifest_path": str(mix_manifest_path),
        "mix_manifest_sft_sha256": remote_manifest.get("sft_sha256"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/sparkproof-mining_sft.jsonl"),
        help="output messages jsonl",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help=f"HF datasets repo (default: {DEFAULT_MINING_DATASET_REPO})",
    )
    parser.add_argument(
        "--mix-manifest-out",
        type=Path,
        default=None,
        help="write HF mix_manifest.json (default: beside --out, e.g. data/processed/mix_manifest.json)",
    )
    parser.add_argument(
        "--skip-pin-check",
        action="store_true",
        help="skip verification against datasets/canonical.json (local dev only)",
    )
    args = parser.parse_args(argv)

    import os

    try:
        result = export_mining_sft(
            out_path=args.out,
            repo_id=args.repo_id,
            hf_token=os.environ.get("HF_TOKEN"),
            verify_pin=not args.skip_pin_check,
            mix_manifest_out=args.mix_manifest_out,
        )
    except Exception as exc:
        print(f"prepare mining sft failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
