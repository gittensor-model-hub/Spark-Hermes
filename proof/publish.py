"""Publish a proof bundle (checkpoint + eval scores) to Hugging Face Hub.

The resulting repo URL is the "proof of verification" link a miner puts in their PR.

    python -m proof.publish --bundle proof/_bundles/2026-07-09-qwen3.5-4b-001 --repo-id <hf-username>/sparkdistill-2026-07-09-qwen3.5-4b-001

Requires `huggingface_hub` (`uv sync --extra proof`) and `HF_TOKEN` set in the
environment (see `.env.example`) with write access to `--repo-id`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def publish_bundle(bundle_dir: Path, repo_id: str, private: bool = False) -> str:
    """Create (if needed) `repo_id` on the Hub and upload `bundle_dir`'s contents.

    Returns the resulting repo URL.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(bundle_dir), repo_id=repo_id)
    return f"https://huggingface.co/{repo_id}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, required=True, help="bundle dir from proof.bundle")
    parser.add_argument("--repo-id", required=True, help="e.g. <hf-username>/sparkdistill-<run-id>")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args(argv)

    url = publish_bundle(args.bundle, args.repo_id, private=args.private)
    print(url)
    print(f"published {args.bundle} to {url}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
