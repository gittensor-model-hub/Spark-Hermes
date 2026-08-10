"""Build a datasets/registry.jsonl line from a SparkProof bundle manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eval.gpu_architecture import dataset_architecture_allowed, normalize_gpu_architecture


def normalize_hf_url(hf_url_or_repo: str) -> str:
    value = hf_url_or_repo.strip().rstrip("/")
    if value.startswith("https://"):
        return value
    return f"https://huggingface.co/datasets/{value.lstrip('/')}"


def load_dataset_manifest(bundle_dir: Path) -> dict[str, Any]:
    path = bundle_dir / "dataset_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path} — run sparkproof-publish-dataset with release gate first")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return manifest


def build_registry_entry(*, miner: str, hf_url: str, manifest: dict[str, Any]) -> dict[str, Any]:
    sha = manifest.get("trajectories_sha256")
    if not isinstance(sha, str) or len(sha) != 64:
        raise ValueError("dataset_manifest.json missing a 64-character trajectories_sha256")

    rows_total = manifest.get("rows_total")
    if not isinstance(rows_total, int) or rows_total <= 0:
        raise ValueError("dataset_manifest.json missing a positive integer rows_total")

    dataset_version = manifest.get("dataset_version")
    if not isinstance(dataset_version, str) or not dataset_version.strip():
        raise ValueError("dataset_manifest.json missing dataset_version")

    gpu_architecture = normalize_gpu_architecture(manifest.get("gpu_architecture"))
    if gpu_architecture is None:
        raise ValueError("dataset_manifest.json missing or unrecognized gpu_architecture")
    if not dataset_architecture_allowed(gpu_architecture):
        raise ValueError(f"gpu_architecture {gpu_architecture!r} is not an accepted dataset-generation architecture")

    return {
        "miner": miner.strip(),
        "hf_url": normalize_hf_url(hf_url),
        "trajectories_sha256": sha,
        "rows_total": rows_total,
        "dataset_version": dataset_version,
        "gpu_architecture": gpu_architecture,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="SparkProof bundle directory")
    parser.add_argument("--miner", required=True, help="GitHub handle for the registry line")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--hf-url", help="Published Hugging Face datasets URL")
    group.add_argument("--repo-id", help="HF datasets repo id (org/name)")
    parser.add_argument(
        "--append",
        type=Path,
        default=None,
        help="append the JSON line to datasets/registry.jsonl (creates parent dirs if needed)",
    )
    args = parser.parse_args(argv)

    try:
        manifest = load_dataset_manifest(args.bundle)
        entry = build_registry_entry(
            miner=args.miner,
            hf_url=args.hf_url or args.repo_id,
            manifest=manifest,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"registry line failed: {exc}", file=sys.stderr)
        return 1

    line = json.dumps(entry, separators=(",", ":"))
    print(line)
    if args.append is not None:
        args.append.parent.mkdir(parents=True, exist_ok=True)
        with args.append.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(f"appended to {args.append}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
