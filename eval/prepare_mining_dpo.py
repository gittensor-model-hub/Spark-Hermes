"""Export the canonical Hugging Face preference dataset to local Axolotl DPO jsonl.

The DPO track's counterpart to ``eval.prepare_mining_sft``: it downloads the pinned
canonical **preference** dataset (chosen/rejected pairs, produced by SparkProof's
``--pair-type correctness`` export) and writes a local jsonl that a ``rl: dpo`` recipe
trains on, verifying the bytes against the pinned ``pref_sha256``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eval.canonical_dataset import (
    CANONICAL_PREFERENCE_DATASET_PATH,
    canonical_pref_hf_url,
    canonical_pref_repo_id,
    pref_sha256_matches_canonical_export,
)

_REQUIRED_FIELDS = ("prompt", "chosen", "rejected")


def export_mining_dpo(
    *,
    out_path: Path,
    repo_id: str | None = None,
    hf_token: str | None = None,
    verify_pin: bool = True,
) -> dict[str, Any]:
    """Download the canonical preference split and write chosen/rejected jsonl for Axolotl."""
    from datasets import load_dataset

    repo = (repo_id or canonical_pref_repo_id()).strip()
    if not repo:
        raise ValueError("canonical preference dataset repo id is empty")
    if repo != canonical_pref_repo_id():
        raise ValueError(
            f"DPO exports must use the canonical preference repo {canonical_pref_repo_id()!r}, got {repo!r}"
        )

    ds = load_dataset(repo, split="train", token=hf_token)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in ds:
            if not isinstance(row, dict):
                raise ValueError(f"{repo} produced a non-object row")
            missing = [field for field in _REQUIRED_FIELDS if not str(row.get(field) or "").strip()]
            if missing:
                raise ValueError(f"{repo} preference row missing {missing}")
            record: dict[str, Any] = {field: row[field] for field in _REQUIRED_FIELDS}
            metadata = row.get("metadata")
            if metadata:
                record["metadata"] = metadata
            # Byte-canonical serialization so the local export matches the pinned pref_sha256.
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            rows_written += 1

    if rows_written == 0:
        raise ValueError(f"{repo} preference train split is empty")

    resolved_out = out_path.resolve()
    if verify_pin and resolved_out.as_posix().endswith(CANONICAL_PREFERENCE_DATASET_PATH):
        sha_issues = pref_sha256_matches_canonical_export(resolved_out)
        if sha_issues:
            raise ValueError("; ".join(sha_issues))

    return {
        "repo_id": repo,
        "dataset_url": canonical_pref_hf_url(),
        "rows_written": rows_written,
        "out_path": str(resolved_out),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(CANONICAL_PREFERENCE_DATASET_PATH),
        help="output chosen/rejected preference jsonl",
    )
    parser.add_argument("--repo-id", default=None, help="HF preference datasets repo (default: canonical pin)")
    parser.add_argument(
        "--skip-pin-check",
        action="store_true",
        help="skip verification against datasets/canonical.json pref_sha256 (local dev only)",
    )
    args = parser.parse_args(argv)

    import os

    try:
        result = export_mining_dpo(
            out_path=args.out,
            repo_id=args.repo_id,
            hf_token=os.environ.get("HF_TOKEN"),
            verify_pin=not args.skip_pin_check,
        )
    except Exception as exc:
        print(f"prepare mining dpo failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
