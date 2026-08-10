"""Export the accepted-registry fingerprint snapshot miners pass to SparkProof.

SparkProof's release gate already accepts ``--registry-snapshot`` (JSONL of prior
trajectory rows). SparkDistill builds that file by simulating the same cross-registry
mix dedupe used for ``sparkproof-mining``, so miners can see registry duplicates
*before* opening a dataset PR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from eval.mining_dataset import DEFAULT_MINING_DATASET_REPO, mining_dedupe_mode
from eval.mix_registry import (
    DedupeMode,
    _add_row_to_registry,
    _classify_row,
    _import_trajectory_exporter,
    _make_dedupe_registry,
    _should_skip,
    load_trajectories_jsonl,
    resolve_proof_dir,
)

ACCEPTED_REGISTRY_SNAPSHOT_PATH = Path("datasets/accepted_registry_snapshot.jsonl")
ACCEPTED_TASK_IDS_PATH = Path("datasets/accepted_task_ids.json")
HF_SNAPSHOT_FILENAME = "accepted_registry_snapshot.jsonl"
HF_TASK_IDS_FILENAME = "accepted_task_ids.json"
MANIFEST_SNAPSHOT_SHA256_KEY = "accepted_registry_snapshot_sha256"
MANIFEST_SNAPSHOT_ROWS_KEY = "accepted_registry_snapshot_rows_total"
MANIFEST_TASK_IDS_SHA256_KEY = "accepted_task_ids_sha256"
MANIFEST_TASK_IDS_ROWS_KEY = "accepted_task_ids_total"
MANIFEST_SNAPSHOT_FILENAME_KEY = "accepted_registry_snapshot_filename"
MANIFEST_TASK_IDS_FILENAME_KEY = "accepted_task_ids_filename"


def _task_id_from_trajectory(trajectory: dict[str, Any]) -> str | None:
    meta = trajectory.get("metadata") or {}
    prompt_meta = meta.get("prompt_meta") or {}
    value = prompt_meta.get("task_id") or prompt_meta.get("problem_id") or meta.get("task_id")
    return str(value) if value else None


def collect_accepted_trajectories(
    registry_entries: list[dict[str, Any]],
    *,
    sparkproof_root: Path | None = None,
    dedupe: DedupeMode | str | None = None,
    download_proof: Callable[[str, Path | None], Path] | None = None,
    proof_cache: Path | None = None,
    export_fn: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
) -> list[dict[str, Any]]:
    """Return trajectory rows whose fingerprints occupy the accepted registry mix state.

    Applies the SAME exportability filter as
    ``eval.mix_registry.mix_registry_datasets``: rows SparkProof's publish exporter
    drops (empty/failed episodes) never enter the canonical mix, so they must not
    occupy the accepted snapshot / task-id index either. Otherwise a miner would
    dedup against — and skip regenerating — a task whose only prior submission was a
    dropped row the mix never actually filled, and the snapshot's dedup state would
    diverge from the mix's (an unexportable row would wrongly block a later good
    duplicate). ``export_fn`` defaults to SparkProof's exporter; tests inject a stub.
    """
    mode: DedupeMode = dedupe or mining_dedupe_mode()  # type: ignore[assignment]
    working_registry, fingerprint_row = _make_dedupe_registry(sparkproof_root)
    working = working_registry.copy() if hasattr(working_registry, "copy") else working_registry
    if export_fn is None:
        export_fn = _import_trajectory_exporter(sparkproof_root)

    accepted: list[dict[str, Any]] = []
    for entry in registry_entries:
        proof_dir = resolve_proof_dir(entry, proof_cache=proof_cache, download_proof=download_proof)
        for trajectory in load_trajectories_jsonl(proof_dir / "trajectories.jsonl"):
            verdict = _classify_row(trajectory, working, dedupe=mode, fingerprint_row=fingerprint_row)
            if _should_skip(verdict, dedupe=mode):
                continue
            if export_fn(trajectory) is None:
                # Unexportable row: the mix drops it and does not record its
                # fingerprint, so neither does the snapshot.
                continue
            accepted.append(trajectory)
            _add_row_to_registry(working, trajectory, fingerprint_row)
    return accepted


def write_registry_snapshot(
    registry_entries: list[dict[str, Any]],
    *,
    out_path: Path = ACCEPTED_REGISTRY_SNAPSHOT_PATH,
    task_ids_path: Path = ACCEPTED_TASK_IDS_PATH,
    sparkproof_root: Path | None = None,
    dedupe: DedupeMode | str | None = None,
    download_proof: Callable[[str, Path | None], Path] | None = None,
    export_fn: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Write SparkProof-compatible snapshot JSONL plus a lightweight task-id index."""
    accepted = collect_accepted_trajectories(
        registry_entries,
        sparkproof_root=sparkproof_root,
        dedupe=dedupe,
        download_proof=download_proof,
        export_fn=export_fn,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in accepted:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    task_ids = sorted({task_id for row in accepted if (task_id := _task_id_from_trajectory(row))})
    payload = {
        "rows_total": len(accepted),
        "task_ids_total": len(task_ids),
        "dedupe_mode": dedupe or mining_dedupe_mode(),
        "task_ids": task_ids,
    }
    task_ids_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    task_ids_sha256 = hashlib.sha256(task_ids_path.read_bytes()).hexdigest()
    return {
        "snapshot_path": out_path,
        "task_ids_path": task_ids_path,
        "rows_total": len(accepted),
        "task_ids_total": len(task_ids),
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "task_ids_sha256": task_ids_sha256,
    }


def snapshot_manifest_pins(snapshot_report: dict[str, Any]) -> dict[str, Any]:
    """Fields to embed in mix_manifest.json so miners can pin the published snapshot."""
    pins = {
        MANIFEST_SNAPSHOT_FILENAME_KEY: HF_SNAPSHOT_FILENAME,
        MANIFEST_SNAPSHOT_SHA256_KEY: snapshot_report["sha256"],
        MANIFEST_SNAPSHOT_ROWS_KEY: int(snapshot_report["rows_total"]),
        MANIFEST_TASK_IDS_FILENAME_KEY: HF_TASK_IDS_FILENAME,
        MANIFEST_TASK_IDS_SHA256_KEY: snapshot_report.get("task_ids_sha256"),
        MANIFEST_TASK_IDS_ROWS_KEY: int(snapshot_report.get("task_ids_total") or 0),
    }
    return {key: value for key, value in pins.items() if value is not None}


def attach_snapshot_pins_to_manifest(manifest_path: Path, snapshot_report: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(snapshot_manifest_pins(snapshot_report))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_registry_snapshot_pins(
    registry_entries: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    snapshot_path: Path | None = None,
    task_ids_path: Path | None = None,
    sparkproof_root: Path | None = None,
    dedupe: DedupeMode | str | None = None,
    download_proof: Callable[[str, Path | None], Path] | None = None,
    export_fn: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
) -> list[str]:
    """Recompute the snapshot and confirm it matches mix_manifest pins."""
    issues: list[str] = []
    expected_sha = manifest.get(MANIFEST_SNAPSHOT_SHA256_KEY)
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        issues.append(f"mix_manifest missing {MANIFEST_SNAPSHOT_SHA256_KEY}")
        return issues

    if snapshot_path is not None and snapshot_path.exists():
        actual_sha = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            issues.append(f"{MANIFEST_SNAPSHOT_SHA256_KEY} mismatch: manifest={expected_sha} file={actual_sha}")
        expected_rows = manifest.get(MANIFEST_SNAPSHOT_ROWS_KEY)
        if expected_rows is not None:
            actual_rows = sum(1 for line in snapshot_path.read_text().splitlines() if line.strip())
            if int(expected_rows) != actual_rows:
                issues.append(f"{MANIFEST_SNAPSHOT_ROWS_KEY} mismatch: manifest={expected_rows} file={actual_rows}")

    recomputed_dir = Path(tempfile.mkdtemp(prefix="sparkdistill-snapshot-verify-"))
    try:
        recomputed = write_registry_snapshot(
            registry_entries,
            out_path=recomputed_dir / HF_SNAPSHOT_FILENAME,
            task_ids_path=recomputed_dir / HF_TASK_IDS_FILENAME,
            sparkproof_root=sparkproof_root,
            dedupe=dedupe,
            download_proof=download_proof,
            export_fn=export_fn,
        )
        if recomputed["sha256"] != expected_sha:
            issues.append("recomputed accepted_registry_snapshot_sha256 does not match mix_manifest pin")

        expected_task_ids_sha = manifest.get(MANIFEST_TASK_IDS_SHA256_KEY)
        if isinstance(expected_task_ids_sha, str) and len(expected_task_ids_sha) == 64:
            if recomputed.get("task_ids_sha256") != expected_task_ids_sha:
                issues.append("recomputed accepted_task_ids_sha256 does not match mix_manifest pin")
    finally:
        import shutil

        shutil.rmtree(recomputed_dir, ignore_errors=True)

    return issues


def verify_remote_registry_snapshot(
    registry_entries: list[dict[str, Any]],
    *,
    repo_id: str = DEFAULT_MINING_DATASET_REPO,
    hf_token: str | None = None,
    sparkproof_root: Path | None = None,
    dedupe: DedupeMode | str | None = None,
    download_proof: Callable[[str, Path | None], Path] | None = None,
) -> list[str]:
    """Download HF mix_manifest + snapshot and confirm pins match a local recompute."""
    from huggingface_hub import hf_hub_download

    from eval.mining_dataset import MINING_MANIFEST_PATH

    try:
        manifest_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=MINING_MANIFEST_PATH,
            token=hf_token,
        )
        snapshot_hf_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=HF_SNAPSHOT_FILENAME,
            token=hf_token,
        )
    except Exception as exc:
        return [f"failed to download registry snapshot artifacts from {repo_id}: {exc}"]

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if MANIFEST_SNAPSHOT_SHA256_KEY not in manifest:
        return [f"{repo_id}/{MINING_MANIFEST_PATH} missing {MANIFEST_SNAPSHOT_SHA256_KEY}"]

    task_ids_path: Path | None = None
    if manifest.get(MANIFEST_TASK_IDS_SHA256_KEY):
        try:
            task_ids_path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=HF_TASK_IDS_FILENAME,
                    token=hf_token,
                )
            )
        except Exception as exc:
            return [f"failed to download {HF_TASK_IDS_FILENAME} from {repo_id}: {exc}"]

    return verify_registry_snapshot_pins(
        registry_entries,
        manifest=manifest,
        snapshot_path=Path(snapshot_hf_path),
        task_ids_path=task_ids_path,
        sparkproof_root=sparkproof_root,
        dedupe=dedupe,
        download_proof=download_proof,
    )


def publish_registry_snapshot(
    snapshot_path: Path,
    *,
    repo_id: str = DEFAULT_MINING_DATASET_REPO,
    task_ids_path: Path | None = None,
) -> dict[str, Any]:
    """Upload snapshot artifacts beside mix_manifest.json on the mining HF repo."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(snapshot_path),
        path_in_repo=HF_SNAPSHOT_FILENAME,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Update accepted registry snapshot for miner novelty checks",
    )
    if task_ids_path is not None and task_ids_path.exists():
        api.upload_file(
            path_or_fileobj=str(task_ids_path),
            path_in_repo=HF_TASK_IDS_FILENAME,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Update accepted registry task-id index",
        )
    return {
        "published": True,
        "hf_url": f"https://huggingface.co/datasets/{repo_id}/blob/main/{HF_SNAPSHOT_FILENAME}",
        "repo_id": repo_id,
        "rows_total": sum(1 for line in snapshot_path.read_text().splitlines() if line.strip()),
    }


def main(argv: list[str] | None = None) -> int:
    from eval.mix_registry import load_registry

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("datasets/registry.jsonl"),
        help="registry file to export (default: datasets/registry.jsonl)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ACCEPTED_REGISTRY_SNAPSHOT_PATH,
        help="local snapshot JSONL for SparkProof --registry-snapshot",
    )
    parser.add_argument(
        "--task-ids-out",
        type=Path,
        default=ACCEPTED_TASK_IDS_PATH,
        help="local JSON index of accepted task_ids",
    )
    parser.add_argument("--sparkproof-root", type=Path, default=None)
    parser.add_argument("--publish", action="store_true", help="upload snapshot to sparkproof-mining HF repo")
    parser.add_argument("--repo-id", default=DEFAULT_MINING_DATASET_REPO)
    args = parser.parse_args(argv)

    try:
        report = write_registry_snapshot(
            load_registry(args.registry),
            out_path=args.out,
            task_ids_path=args.task_ids_out,
            sparkproof_root=args.sparkproof_root,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"export registry snapshot failed: {exc}", file=sys.stderr)
        return 1

    if args.publish:
        try:
            report.update(
                publish_registry_snapshot(
                    args.out,
                    repo_id=args.repo_id,
                    task_ids_path=args.task_ids_out,
                )
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"publish registry snapshot failed: {exc}", file=sys.stderr)
            return 1

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
