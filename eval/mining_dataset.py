"""Canonical mining dataset on Hugging Face.

Before a dataset registry PR merges, CI aggregates every registry line (existing +
the proposed submission) into the default mining dataset repo. Training miners can
point recipes at that single HF URL instead of hand-mixing components.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from eval.dataset_verify import _sha256_file
from eval.mix_registry import mix_registry_datasets, verify_mix_manifest

DEFAULT_MINING_DATASET_REPO = "gittensor-model-hub/sparkproof-mining"
MINING_MANIFEST_PATH = "mix_manifest.json"
DEFAULT_MINING_DEDUPE = "exact"


def mining_dataset_repo() -> str:
    return os.environ.get("SPARKDISTILL_MINING_DATASET_REPO", DEFAULT_MINING_DATASET_REPO).strip()


def mining_dedupe_mode() -> str:
    """Cross-miner dedupe when building sparkproof-mining.

    Row **quality** is enforced upstream by SparkProof (release gate, decontamination,
    per-row GPU validation). Mix-time dedupe only removes redundant copies.

    Default ``exact``: drop identical prompts / assistant-code only — keeps
    structurally similar but distinct verified rows. ``near`` also drops similar
    tasks and can over-shrink large submissions. ``none`` is for local debugging.
    Override with ``SPARKDISTILL_MINING_DEDUPE``.
    """
    value = os.environ.get("SPARKDISTILL_MINING_DEDUPE", DEFAULT_MINING_DEDUPE).strip().lower()
    if value not in {"exact", "near", "none"}:
        return DEFAULT_MINING_DEDUPE
    return value


def publish_sft_dataset(
    *,
    sft_path: Path,
    manifest_path: Path,
    repo_id: str,
    private: bool = False,
) -> dict[str, Any]:
    """Upload mixed SFT rows + mix_manifest.json to a Hugging Face datasets repo."""
    from huggingface_hub import HfApi

    from datasets import Dataset

    rows: list[dict[str, Any]] = []
    with sft_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("no SFT rows to publish")

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private)
    api.upload_file(
        path_or_fileobj=str(manifest_path),
        path_in_repo=MINING_MANIFEST_PATH,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Update mining dataset mix manifest",
    )

    ds = Dataset.from_list(rows)
    ds.push_to_hub(repo_id, split="train", commit_message="Update canonical SparkDistill mining dataset")

    manifest_sha = _sha256_file(manifest_path)
    sft_sha = _sha256_file(sft_path)
    return {
        "published": True,
        "hf_url": f"https://huggingface.co/datasets/{repo_id}",
        "repo_id": repo_id,
        "rows_total": len(rows),
        "mix_manifest_sha256": manifest_sha,
        "sft_sha256": sft_sha,
        "issues": [],
    }


def aggregate_mining_mix(
    registry_entries: list[dict[str, Any]],
    *,
    repo_id: str = DEFAULT_MINING_DATASET_REPO,
    sparkproof_root: Path | None = None,
    work_dir: Path | None = None,
    dedupe: str | None = None,
    download_proof: Callable[[str, Path | None], Path] | None = None,
) -> dict[str, Any]:
    """Mix registry entries locally without publishing to Hugging Face."""
    if not registry_entries:
        return {"verified": False, "issues": ["registry is empty — nothing to aggregate"]}

    cleanup = work_dir is None
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="sparkdistill-mining-"))
    work_dir.mkdir(parents=True, exist_ok=True)

    sft_path = work_dir / "mining_sft.jsonl"
    manifest_path = work_dir / "mix_manifest.json"

    try:
        registry_path = work_dir / "registry.jsonl"
        registry_path.write_text("".join(json.dumps(row) + "\n" for row in registry_entries), encoding="utf-8")

        mix_result = mix_registry_datasets(
            registry_entries,
            out_path=sft_path,
            manifest_path=manifest_path,
            mix_id=f"mining-{repo_id.replace('/', '-')}",
            sparkproof_root=sparkproof_root,
            dedupe=(dedupe or mining_dedupe_mode()),  # type: ignore[arg-type]
            download_proof=download_proof,
        )
        verify_report = verify_mix_manifest(manifest_path, sft_path=sft_path, registry_path=registry_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not verify_report.get("verified"):
            return {
                "verified": False,
                "issues": list(verify_report.get("issues") or ["mix verification failed"]),
                "rows_total": mix_result.rows_total,
                "manifest": manifest,
                "components": manifest.get("components") or [],
                "dedupe": mix_result.dedupe,
            }

        return {
            "verified": True,
            "issues": [],
            "rows_total": mix_result.rows_total,
            "manifest": manifest,
            "components": manifest.get("components") or [],
            "dedupe": mix_result.dedupe,
            "sft_path": sft_path,
            "manifest_path": manifest_path,
            "registry_entries": registry_entries,
        }
    finally:
        if cleanup:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)


def aggregate_and_publish_mining_dataset(
    registry_entries: list[dict[str, Any]],
    *,
    repo_id: str,
    sparkproof_root: Path | None = None,
    work_dir: Path | None = None,
    dedupe: str | None = None,
    download_proof: Callable[[str, Path | None], Path] | None = None,
    publish_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Mix all registry entries and publish to the canonical mining dataset repo."""
    if not registry_entries:
        return {"published": False, "issues": ["registry is empty — nothing to aggregate"]}

    cleanup = work_dir is None
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="sparkdistill-mining-"))
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        mix_report = aggregate_mining_mix(
            registry_entries,
            repo_id=repo_id,
            sparkproof_root=sparkproof_root,
            work_dir=work_dir,
            dedupe=dedupe,
            download_proof=download_proof,
        )
        if not mix_report.get("verified"):
            return {
                "published": False,
                "issues": list(mix_report.get("issues") or ["mix verification failed"]),
                "rows_total": mix_report.get("rows_total"),
                "components": mix_report.get("components") or [],
            }

        publish = publish_fn or publish_sft_dataset
        from eval.export_registry_snapshot import (
            attach_snapshot_pins_to_manifest,
            publish_registry_snapshot,
            verify_registry_snapshot_pins,
            write_registry_snapshot,
        )

        snapshot_path = work_dir / "accepted_registry_snapshot.jsonl"
        task_ids_path = work_dir / "accepted_task_ids.json"
        snapshot_report = write_registry_snapshot(
            mix_report["registry_entries"],
            out_path=snapshot_path,
            task_ids_path=task_ids_path,
            sparkproof_root=sparkproof_root,
            dedupe=dedupe,
            download_proof=download_proof,
        )
        manifest = attach_snapshot_pins_to_manifest(mix_report["manifest_path"], snapshot_report)
        pin_issues = verify_registry_snapshot_pins(
            mix_report["registry_entries"],
            manifest=manifest,
            snapshot_path=snapshot_path,
            task_ids_path=task_ids_path,
            sparkproof_root=sparkproof_root,
            dedupe=dedupe,
            download_proof=download_proof,
        )
        if pin_issues:
            return {
                "published": False,
                "issues": pin_issues,
                "rows_total": mix_report.get("rows_total"),
                "components": mix_report.get("components") or [],
            }

        pub_report = publish(
            sft_path=mix_report["sft_path"],
            manifest_path=mix_report["manifest_path"],
            repo_id=repo_id,
        )
        pub_report["registry_snapshot"] = publish_registry_snapshot(
            snapshot_path,
            repo_id=repo_id,
            task_ids_path=task_ids_path,
        )
        pub_report["registry_snapshot"]["rows_total"] = snapshot_report["rows_total"]
        pub_report["registry_snapshot"]["sha256"] = snapshot_report["sha256"]
        pub_report["component_count"] = len(registry_entries)
        pub_report["dedupe"] = mix_report.get("dedupe")
        pub_report["components"] = mix_report.get("components") or []
        pub_report["manifest"] = mix_report.get("manifest")
        return pub_report
    finally:
        if cleanup:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)


def aggregate_registry_text(
    base_registry_text: str,
    head_registry_text: str,
) -> list[dict[str, Any]]:
    """Return the full registry as it would exist immediately after a successful merge."""
    rows: list[dict[str, Any]] = []
    for text in (base_registry_text, head_registry_text):
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # Deduplicate by trajectories_sha256 while preserving order.
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        sha = row["trajectories_sha256"]
        if sha in seen:
            continue
        seen.add(sha)
        unique.append(row)
    return unique
