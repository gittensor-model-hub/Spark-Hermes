"""Compose multiple registry-indexed SparkProof datasets into one SFT mix.

Each component must be a merged line in ``datasets/registry.jsonl`` with a pinned
``trajectories_sha256``. The mix downloads ``proof/trajectories.jsonl`` from each
Hugging Face repo, deduplicates across sources, converts to Axolotl ``messages``
records, and writes a small ``mix_manifest.json`` that auditors can re-check.

    # Build a mix from two merged registry entries (SparkProof checkout required for export)
    python -m eval.mix_registry mix \\
        --registry datasets/registry.jsonl \\
        --sha256 <sha-a> --sha256 <sha-b> \\
        --out data/processed/mix_sft.jsonl \\
        --manifest-out data/processed/mix_manifest.json \\
        --sparkproof-root ../SparkProof

    # Verify a committed mix manifest + SFT file
    python -m eval.mix_registry verify \\
        --manifest data/processed/mix_manifest.json \\
        --sft data/processed/mix_sft.jsonl \\
        --registry datasets/registry.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal

from eval.dataset_verify import _sha256_file, check_proof_dir
from eval.hf_pin import download_with_optional_pin
from eval.registry_gate import hf_repo_from_url, validate_registry_entry

MIX_VERSION = "sparkdistill-mix-v0"
REGISTRY_PATH = Path("datasets/registry.jsonl")
DedupeMode = Literal["near", "exact", "none"]

_HF_REPO_RE = re.compile(r"^https://huggingface\.co/datasets/([^/]+/[^/#?]+)")


@dataclass
class MixComponent:
    registry_entry: dict[str, Any]
    rows_in_source: int = 0
    rows_selected: int = 0
    rows_skipped_dedupe: int = 0
    rows_skipped_unexportable: int = 0


@dataclass
class MixResult:
    mix_id: str
    sft_path: Path
    manifest_path: Path
    rows_total: int
    components: list[MixComponent]
    dedupe: dict[str, int] = field(default_factory=dict)
    dedupe_mode: DedupeMode = "exact"

    def to_manifest(self) -> dict[str, Any]:
        return {
            "mix_version": MIX_VERSION,
            "mix_id": self.mix_id,
            "created_at": datetime.now(UTC).isoformat(),
            "sft_format": "messages",
            "sft_sha256": _sha256_file(self.sft_path),
            "rows_total": self.rows_total,
            "dedupe_mode": self.dedupe_mode,
            "dedupe": self.dedupe,
            "components": [
                {
                    "miner": component.registry_entry["miner"],
                    "hf_url": component.registry_entry["hf_url"],
                    "trajectories_sha256": component.registry_entry["trajectories_sha256"],
                    "rows_in_source": component.rows_in_source,
                    "rows_selected": component.rows_selected,
                    "rows_skipped_dedupe": component.rows_skipped_dedupe,
                    "rows_skipped_unexportable": component.rows_skipped_unexportable,
                    "dataset_version": component.registry_entry.get("dataset_version"),
                    "gpu_architecture": component.registry_entry.get("gpu_architecture"),
                }
                for component in self.components
            ],
        }


def load_registry(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        issues = validate_registry_entry(row)
        if issues:
            raise ValueError(f"{path}:{line_no}: invalid registry entry: {'; '.join(issues)}")
        rows.append(row)
    return rows


def select_registry_entries(
    registry: list[dict[str, Any]],
    *,
    sha256s: list[str] | None = None,
    miners: list[str] | None = None,
    all_merged: bool = False,
) -> list[dict[str, Any]]:
    if all_merged:
        if not registry:
            raise ValueError("registry is empty")
        return list(registry)

    if not sha256s and not miners:
        raise ValueError("specify at least one of --sha256, --miner, or --all")

    by_sha = {row["trajectories_sha256"]: row for row in registry}
    selected: list[dict[str, Any]] = []
    seen_sha: set[str] = set()

    for sha in sha256s or []:
        entry = by_sha.get(sha)
        if entry is None:
            raise ValueError(f"trajectories_sha256 not found in registry: {sha}")
        if sha not in seen_sha:
            selected.append(entry)
            seen_sha.add(sha)

    if miners:
        by_miner: dict[str, list[dict[str, Any]]] = {}
        for row in registry:
            by_miner.setdefault(row["miner"], []).append(row)
        for miner in miners:
            entries = by_miner.get(miner)
            if not entries:
                raise ValueError(f"miner not found in registry: {miner}")
            for entry in entries:
                sha = entry["trajectories_sha256"]
                if sha not in seen_sha:
                    selected.append(entry)
                    seen_sha.add(sha)

    if not selected:
        raise ValueError("no registry entries matched the selection")
    return selected


def resolve_sparkproof_root(sparkproof_root: Path | None) -> Path:
    """Return a SparkProof checkout (explicit arg, ``SPARKPROOF_ROOT``, or sibling clone)."""
    if sparkproof_root is not None:
        root = sparkproof_root.resolve()
    else:
        env = os.environ.get("SPARKPROOF_ROOT")
        if env:
            root = Path(env).resolve()
        else:
            root = (Path(__file__).resolve().parents[1] / ".." / "SparkProof").resolve()
    marker = root / "sparkproof" / "publish" / "hf_dataset.py"
    if not marker.is_file():
        raise ValueError(
            f"SparkProof checkout required for registry mix export (missing {marker}). "
            "Pass --sparkproof-root or set SPARKPROOF_ROOT."
        )
    return root


def _ensure_sparkproof_on_path(root: Path) -> None:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _import_trajectory_exporter(sparkproof_root: Path | None) -> Callable[[dict[str, Any]], dict[str, Any] | None]:
    """SparkProof HF publish export: skips empty/failed rows, prefers multi-turn episodes."""
    root = resolve_sparkproof_root(sparkproof_root)
    _ensure_sparkproof_on_path(root)
    from sparkproof.publish.hf_dataset import trajectory_to_messages_record  # pyright: ignore[reportMissingImports]

    return trajectory_to_messages_record


def _import_novelty(sparkproof_root: Path | None):
    try:
        root = resolve_sparkproof_root(sparkproof_root)
    except ValueError:
        return None
    _ensure_sparkproof_on_path(root)
    try:
        from sparkproof.triton_dataset.novelty import (  # pyright: ignore[reportMissingImports]
            NoveltyRegistry,
            fingerprint_row,
        )
    except ImportError:
        return None

    return NoveltyRegistry, fingerprint_row


class _PromptDedupeRegistry:
    """Fallback exact dedupe when SparkProof is not on the path."""

    def __init__(self) -> None:
        self._prompt_keys: set[str] = set()

    @staticmethod
    def _task_prompt(row: dict[str, Any]) -> str:
        meta = (row.get("metadata") or {}).get("prompt_meta") or {}
        return str(meta.get("prompt") or row.get("prompt") or "").strip().lower()

    @staticmethod
    def _row_gpu_architecture(row: dict[str, Any]) -> str:
        meta = (row.get("metadata") or {}).get("prompt_meta") or {}
        return str(row.get("gpu_architecture") or meta.get("gpu_architecture") or "blackwell")

    def classify(self, row: dict[str, Any]) -> Literal["exact", "near", "novel"]:
        prompt = self._task_prompt(row)
        if not prompt:
            return "novel"
        arch = self._row_gpu_architecture(row)
        key = f"{arch}:{prompt}"
        if key in self._prompt_keys:
            return "exact"
        return "novel"

    def add(self, row: dict[str, Any]) -> None:
        prompt = self._task_prompt(row)
        if prompt:
            arch = self._row_gpu_architecture(row)
            self._prompt_keys.add(f"{arch}:{prompt}")

    def copy(self) -> "_PromptDedupeRegistry":
        clone = _PromptDedupeRegistry()
        clone._prompt_keys = set(self._prompt_keys)
        return clone


def _make_dedupe_registry(sparkproof_root: Path | None):
    novelty = _import_novelty(sparkproof_root)
    if novelty is None:
        return _PromptDedupeRegistry(), None
    registry_cls, fingerprint_row = novelty
    return registry_cls(), fingerprint_row


def _should_skip(verdict: str, *, dedupe: DedupeMode) -> bool:
    if dedupe == "none":
        return False
    if dedupe == "exact":
        return verdict == "exact"
    return verdict in {"exact", "near"}


def _classify_row(
    row: dict[str, Any],
    working: Any,
    *,
    dedupe: DedupeMode,
    fingerprint_row: Callable[[dict[str, Any]], Any] | None,
) -> str:
    if dedupe == "none":
        return "novel"
    if fingerprint_row is not None:
        fp = fingerprint_row(row)
        return working.classify(fp)
    return working.classify(row)


def _add_row_to_registry(
    working: Any, row: dict[str, Any], fingerprint_row: Callable[[dict[str, Any]], Any] | None
) -> None:
    if fingerprint_row is not None:
        working.add(fingerprint_row(row))
    else:
        working.add(row)


def load_trajectories_jsonl(path: Path) -> list[dict[str, Any]]:
    # Miner-controlled input: report a clean per-line ValueError (the gate's caught
    # set) instead of an uncaught AttributeError downstream (issue #214), mirroring
    # load_registry's convention.
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: trajectory row must be a JSON object")
            rows.append(row)
    return rows


def resolve_proof_dir(
    entry: dict[str, Any],
    *,
    proof_cache: Path | None = None,
    download_proof: Callable[[str, Path | None], Path] | None = None,
) -> Path:
    repo = hf_repo_from_url(entry["hf_url"])
    # From the entry, exactly as the repo is. A row that named a repo but not a revision
    # could be re-fetched into different bytes; `check_revision` refuses the ones that can.
    revision = entry.get("hf_revision")
    if download_proof is not None:
        proof_dir = download_proof(repo, proof_cache)
    else:
        from huggingface_hub import snapshot_download

        cache_dir = str(proof_cache) if proof_cache else None
        snapshot = Path(
            # Used when the entry carries one, left to the Hub's default otherwise. The
            # dataset registry is live and its existing rows predate the field; requiring a
            # pin here would invalidate every one of them. Only the rollout track -- which
            # has no rows yet -- requires it outright.
            download_with_optional_pin(
                snapshot_download,
                revision=revision,
                repo_id=repo,
                repo_type="dataset",
                allow_patterns=["proof/*"],
                cache_dir=cache_dir,
            )
        )
        proof_dir = snapshot / "proof"
    issues, _rows, _gpu_architecture = check_proof_dir(proof_dir, claimed_sha256=entry["trajectories_sha256"])
    if issues:
        raise ValueError(f"{repo}: proof check failed: {'; '.join(issues)}")
    return proof_dir


def trajectory_to_sft_record(
    trajectory: dict[str, Any],
    *,
    component: dict[str, Any],
    row_index: int,
    export_fn: Callable[[dict[str, Any]], dict[str, Any] | None],
) -> dict[str, Any] | None:
    """Convert one proof trajectory to an SFT row using SparkProof's publish exporter."""
    record = export_fn(trajectory)
    if record is None:
        return None
    mix_meta = {
        "mix_source_miner": component["miner"],
        "mix_source_hf_url": component["hf_url"],
        "mix_source_sha256": component["trajectories_sha256"],
        "mix_row_index": row_index,
        "mix_task_id": ((trajectory.get("metadata") or {}).get("prompt_meta") or {}).get("task_id"),
    }
    record_meta = dict(record.get("metadata") or {})
    record_meta.update(mix_meta)
    record["metadata"] = record_meta
    return record


def mix_registry_datasets(
    entries: list[dict[str, Any]],
    *,
    out_path: Path,
    manifest_path: Path,
    mix_id: str,
    sparkproof_root: Path | None = None,
    proof_cache: Path | None = None,
    dedupe: DedupeMode = "near",
    max_rows_per_source: int | None = None,
    download_proof: Callable[[str, Path | None], Path] | None = None,
) -> MixResult:
    working_registry, fingerprint_row = _make_dedupe_registry(sparkproof_root)
    working = working_registry.copy() if hasattr(working_registry, "copy") else working_registry

    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    export_fn = _import_trajectory_exporter(sparkproof_root)

    components: list[MixComponent] = []
    dedupe_counts = {
        "exact_skipped": 0,
        "near_skipped": 0,
        "intra_mix_skipped": 0,
        "unexportable_skipped": 0,
    }
    rows_written = 0

    with out_path.open("w", encoding="utf-8", newline="\n") as out_f:
        for entry in entries:
            proof_dir = resolve_proof_dir(entry, proof_cache=proof_cache, download_proof=download_proof)
            trajectories = load_trajectories_jsonl(proof_dir / "trajectories.jsonl")
            component = MixComponent(registry_entry=entry, rows_in_source=len(trajectories))

            selected_for_source = 0
            for trajectory in trajectories:
                if max_rows_per_source is not None and selected_for_source >= max_rows_per_source:
                    break
                verdict = _classify_row(trajectory, working, dedupe=dedupe, fingerprint_row=fingerprint_row)
                if _should_skip(verdict, dedupe=dedupe):
                    component.rows_skipped_dedupe += 1
                    if verdict == "exact":
                        dedupe_counts["exact_skipped"] += 1
                    elif verdict == "near":
                        dedupe_counts["near_skipped"] += 1
                    else:
                        dedupe_counts["intra_mix_skipped"] += 1
                    continue

                record = trajectory_to_sft_record(
                    trajectory,
                    component=entry,
                    row_index=rows_written,
                    export_fn=export_fn,
                )
                if record is None:
                    component.rows_skipped_unexportable += 1
                    dedupe_counts["unexportable_skipped"] += 1
                    continue
                out_f.write(json.dumps(record, separators=(",", ":")) + "\n")
                rows_written += 1
                selected_for_source += 1
                component.rows_selected += 1
                _add_row_to_registry(working, trajectory, fingerprint_row)

            components.append(component)

    if rows_written == 0:
        raise ValueError("mix produced zero rows after deduplication")

    result = MixResult(
        mix_id=mix_id,
        sft_path=out_path,
        manifest_path=manifest_path,
        rows_total=rows_written,
        components=components,
        dedupe=dedupe_counts,
        dedupe_mode=dedupe,
    )
    manifest_path.write_text(json.dumps(result.to_manifest(), indent=2) + "\n", encoding="utf-8")
    return result


def verify_mix_manifest(
    manifest_path: Path,
    *,
    sft_path: Path | None = None,
    registry_path: Path = REGISTRY_PATH,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    issues: list[str] = []

    if manifest.get("mix_version") != MIX_VERSION:
        issues.append(f"unsupported mix_version: {manifest.get('mix_version')!r}")

    registry = load_registry(registry_path)
    by_sha = {row["trajectories_sha256"]: row for row in registry}

    for component in manifest.get("components") or []:
        sha = component.get("trajectories_sha256")
        if not sha:
            issues.append("component missing trajectories_sha256")
            continue
        registry_row = by_sha.get(sha)
        if registry_row is None:
            issues.append(f"component sha256 not in registry: {sha}")
            continue
        if component.get("hf_url") and component["hf_url"] != registry_row["hf_url"]:
            issues.append(f"hf_url mismatch for {sha}")
        if component.get("miner") and component["miner"] != registry_row["miner"]:
            issues.append(f"miner mismatch for {sha}")

    if sft_path is not None:
        if not sft_path.exists():
            issues.append(f"missing sft file: {sft_path}")
        else:
            actual_sha = _sha256_file(sft_path)
            expected_sha = manifest.get("sft_sha256")
            if expected_sha and actual_sha != expected_sha:
                issues.append("sft_sha256 does not match mix_manifest")
            actual_rows = sum(1 for line in sft_path.read_text(encoding="utf-8").splitlines() if line.strip())
            expected_rows = int(manifest.get("rows_total") or 0)
            if expected_rows and actual_rows != expected_rows:
                issues.append(f"rows_total mismatch: manifest={expected_rows} sft={actual_rows}")

            selected_total = sum(
                int(component.get("rows_selected") or 0) for component in manifest.get("components") or []
            )
            if selected_total and actual_rows != selected_total:
                issues.append(f"component rows_selected sum ({selected_total}) != sft rows ({actual_rows})")

    return {
        "verified": not issues,
        "issues": issues,
        "rows_total": manifest.get("rows_total"),
        "mix_id": manifest.get("mix_id"),
        "component_count": len(manifest.get("components") or []),
    }


def _cmd_mix(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    entries = select_registry_entries(
        registry,
        sha256s=args.sha256 or None,
        miners=args.miner or None,
        all_merged=args.all,
    )
    try:
        result = mix_registry_datasets(
            entries,
            out_path=args.out,
            manifest_path=args.manifest_out,
            mix_id=args.mix_id,
            sparkproof_root=args.sparkproof_root,
            proof_cache=args.proof_cache,
            dedupe=args.dedupe,
            max_rows_per_source=args.max_rows_per_source,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"mix failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"wrote {result.rows_total} rows to {result.sft_path} "
        f"(manifest {result.manifest_path}, dedupe={result.dedupe})",
        file=sys.stderr,
    )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    report = verify_mix_manifest(args.manifest, sft_path=args.sft, registry_path=args.registry)
    print(
        f"mix verified={report['verified']} rows={report.get('rows_total')} components={report.get('component_count')}",
        file=sys.stderr,
    )
    if report["issues"]:
        for issue in report["issues"]:
            print(f"  - {issue}", file=sys.stderr)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["verified"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    mix_parser = subparsers.add_parser("mix", help="compose registry datasets into one SFT file")
    mix_parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    mix_parser.add_argument("--sha256", action="append", default=[], help="registry trajectories_sha256 (repeatable)")
    mix_parser.add_argument("--miner", action="append", default=[], help="registry miner handle (repeatable)")
    mix_parser.add_argument("--all", action="store_true", help="include every line in the registry")
    mix_parser.add_argument("--out", type=Path, required=True, help="output SFT jsonl (messages format)")
    mix_parser.add_argument("--manifest-out", type=Path, required=True, help="provenance manifest json")
    mix_parser.add_argument("--mix-id", default=None, help="identifier recorded in mix_manifest.json")
    mix_parser.add_argument(
        "--sparkproof-root",
        type=Path,
        default=None,
        help="SparkProof checkout for novelty dedupe and SFT export (default: SPARKPROOF_ROOT or ../SparkProof)",
    )
    mix_parser.add_argument("--proof-cache", type=Path, default=None, help="HF snapshot cache directory")
    mix_parser.add_argument("--dedupe", choices=["near", "exact", "none"], default="near")
    mix_parser.add_argument("--max-rows-per-source", type=int, default=None)
    mix_parser.set_defaults(func=_cmd_mix)

    verify_parser = subparsers.add_parser("verify", help="verify a mix manifest against the registry")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    verify_parser.add_argument("--sft", type=Path, default=None)
    verify_parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    verify_parser.add_argument("--out", type=Path, default=None)
    verify_parser.set_defaults(func=_cmd_verify)

    args = parser.parse_args(argv)
    if args.command == "mix" and args.mix_id is None:
        args.mix_id = args.manifest_out.stem
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
