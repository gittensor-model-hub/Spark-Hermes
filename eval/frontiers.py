"""Per-GPU-architecture frontier records (Blackwell vs Hopper)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.benchmarks import BENCHMARKS
from eval.frontier import merge_frontier_scores
from eval.gpu_architecture import (
    DEFAULT_GPU_ARCHITECTURE,
    GPU_ARCHITECTURES,
    GpuArchitecture,
    normalize_gpu_architecture,
)

FRONTIERS_PATH = Path("runs/frontiers.json")
LEGACY_FRONTIER_PATH = Path("runs/frontier.json")

# Training tracks get independent frontier buckets so each is seeded/tiered on its own.
# SFT keeps the bare architecture key (byte-identical with existing frontiers.json); a
# DPO run uses a `<arch>::dpo` bucket. Consequences that fall out for free: the first
# verified run on an empty bucket scores eval:BASELINE (verify_submission labels BASELINE
# when the frontier is unset), and later runs tier over that bucket — so the first DPO run
# is a legit phase baseline and the rest compete as usual, without touching SFT.
TRAINING_TRACK_SFT = "sft"
TRAINING_TRACK_DPO = "dpo"


def training_track_of(mapping: dict[str, Any]) -> str:
    """Track declared by a bundle manifest or verify report (`train_objective`); default sft."""
    return (
        TRAINING_TRACK_DPO
        if str(mapping.get("train_objective") or "").strip().lower() == TRAINING_TRACK_DPO
        else TRAINING_TRACK_SFT
    )


def frontier_bucket(gpu_architecture: str, track: str = TRAINING_TRACK_SFT) -> str:
    """Frontier bucket key: the bare arch for SFT (backward-compatible), else `<arch>::<track>`."""
    track = (track or TRAINING_TRACK_SFT).strip().lower()
    return gpu_architecture if track == TRAINING_TRACK_SFT else f"{gpu_architecture}::{track}"


def _bucket_arch(bucket: str) -> str:
    return bucket.split("::", 1)[0]


def _empty_record(arch: str) -> dict[str, Any]:
    return {
        "gpu_architecture": arch,
        "run_id": None,
        "proof_bundle": None,
        "scores": {},
    }


def _numeric_scores(raw: Any) -> dict[str, float]:
    """Coerce a flat `{key: score}` map, or a `{key: {"candidate": score}}` one."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            if "candidate" not in value:
                continue
            value = value["candidate"]
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def candidate_scores_from_report(report: dict[str, Any]) -> dict[str, float]:
    """Extract candidate benchmark scores from an ``eval.verify`` report.

    Prefers the report's full ``scores`` claim. ``per_benchmark`` is only a
    fallback (for reports written before ``scores`` was recorded): `eval.score`
    restricts it to keys present in both the candidate and the frontier, so
    relying on it silently drops any benchmark the frontier does not carry yet.
    """
    scores = _numeric_scores(report.get("scores"))
    if scores:
        return scores
    return _numeric_scores(report.get("per_benchmark"))


def write_frontiers(frontiers: dict[str, dict[str, Any]], path: Path = FRONTIERS_PATH) -> None:
    """Persist ``runs/frontiers.json`` (and sibling legacy ``frontier.json``)."""
    payload: dict[str, Any] = {}
    # SFT arch buckets are always written (stable order, empty if unseeded); any extra
    # track buckets (e.g. `blackwell::dpo`) follow, only when they actually exist.
    buckets = list(GPU_ARCHITECTURES) + [key for key in frontiers if key not in GPU_ARCHITECTURES]
    for bucket in buckets:
        arch = _bucket_arch(bucket)
        record = frontiers.get(bucket) or _empty_record(arch)
        payload[bucket] = {
            "gpu_architecture": record.get("gpu_architecture") or arch,
            "run_id": record.get("run_id"),
            "proof_bundle": record.get("proof_bundle"),
            "scores": record.get("scores") if isinstance(record.get("scores"), dict) else {},
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Keep legacy single-file Blackwell frontier in sync for older tooling.
    if path.name == "frontiers.json":
        blackwell = payload.get("blackwell") or _empty_record("blackwell")
        path.with_name("frontier.json").write_text(
            json.dumps(
                {
                    "run_id": blackwell.get("run_id"),
                    "proof_bundle": blackwell.get("proof_bundle"),
                    "scores": blackwell.get("scores") or {},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def apply_verified_report_to_frontiers(
    report: dict[str, Any],
    *,
    proof_bundle: str | None,
    path: Path = FRONTIERS_PATH,
) -> list[str]:
    """Raise per-benchmark highs for a verified non-REJECT training run.

    Called at merge time (ledger workflow). Safe to re-run: only updates when a
    candidate score beats the current bucket high (or seeds an empty bucket).
    """
    if not report.get("verified"):
        return []
    label = str(report.get("label") or "")
    if label == "eval:REJECT" or label.endswith(":REJECT"):
        return []

    candidate = candidate_scores_from_report(report)
    if not candidate:
        return []

    arch = resolve_gpu_architecture(report.get("gpu_architecture"))
    bucket = frontier_bucket(arch, training_track_of(report))
    frontiers = load_frontiers(path)
    current = dict(frontiers.get(bucket) or _empty_record(arch))
    current_scores_obj = current.get("scores")
    current_scores = current_scores_obj if isinstance(current_scores_obj, dict) else {}
    # Official basket keys via merge_frontier_scores; also raise diagnostic TritonBench
    # breakdown keys (triton_syntax_pass_rate, …) that seed alongside the basket.
    basket = {key: value for key, value in candidate.items() if key in BENCHMARKS}
    extras = {key: value for key, value in candidate.items() if key not in BENCHMARKS}
    merged_scores, updates = merge_frontier_scores(current_scores, basket)
    for key, value in extras.items():
        if key not in merged_scores or value > float(merged_scores[key]):
            merged_scores[key] = value
            updates.append(key)

    seeding = not current.get("run_id") and not current_scores
    if not updates and not seeding:
        return []

    current["gpu_architecture"] = arch
    current["scores"] = merged_scores
    if report.get("run_id") is not None and (updates or seeding):
        current["run_id"] = report.get("run_id")
    if proof_bundle is not None and (updates or seeding or not current.get("proof_bundle")):
        current["proof_bundle"] = proof_bundle
    frontiers = dict(frontiers)
    frontiers[bucket] = current
    write_frontiers(frontiers, path=path)
    return updates


def load_frontiers(path: Path = FRONTIERS_PATH) -> dict[str, dict[str, Any]]:
    """Load all architecture frontiers from `runs/frontiers.json`."""
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a JSON object")
        # Preserve every bucket present (SFT arch buckets + track buckets like
        # `blackwell::dpo`); a track frontier must not be dropped on load.
        out: dict[str, dict[str, Any]] = {key: record for key, record in data.items() if isinstance(record, dict)}
        for arch in GPU_ARCHITECTURES:
            out.setdefault(arch, _empty_record(arch))
        return out

    # Legacy single-file frontier seeds Blackwell only.
    if LEGACY_FRONTIER_PATH.exists():
        legacy = json.loads(LEGACY_FRONTIER_PATH.read_text(encoding="utf-8"))
        scores = legacy.get("scores") if isinstance(legacy.get("scores"), dict) else {}
        return {
            "blackwell": {
                "gpu_architecture": "blackwell",
                "run_id": legacy.get("run_id"),
                "proof_bundle": legacy.get("proof_bundle"),
                "scores": scores,
            },
            "hopper": _empty_record("hopper"),
        }

    return {arch: _empty_record(arch) for arch in GPU_ARCHITECTURES}


def load_frontier_scores(
    gpu_architecture: GpuArchitecture,
    *,
    track: str = TRAINING_TRACK_SFT,
    path: Path = FRONTIERS_PATH,
) -> dict[str, float] | None:
    """Return frontier scores for an (architecture, track) bucket, or None when unset.

    None is the BASELINE signal — so a bundle on a track whose bucket has never been
    seeded (e.g. the first DPO run) scores eval:BASELINE and seeds it.
    """
    bucket = frontier_bucket(gpu_architecture, track)
    record = load_frontiers(path).get(bucket) or _empty_record(gpu_architecture)
    scores = record.get("scores")
    if not isinstance(scores, dict) or not scores:
        return None
    return {key: float(value) for key, value in scores.items()}


def load_frontier_record(
    gpu_architecture: GpuArchitecture,
    *,
    track: str = TRAINING_TRACK_SFT,
    path: Path = FRONTIERS_PATH,
) -> dict[str, Any]:
    bucket = frontier_bucket(gpu_architecture, track)
    return load_frontiers(path).get(bucket) or _empty_record(gpu_architecture)


def merge_frontier_record(
    frontiers: dict[str, dict[str, Any]],
    gpu_architecture: GpuArchitecture,
    candidate_scores: dict[str, float],
    *,
    run_id: str | None = None,
    proof_bundle: str | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Merge per-benchmark highs into one architecture bucket."""
    record = dict(frontiers.get(gpu_architecture) or _empty_record(gpu_architecture))
    record_scores_obj = record.get("scores")
    current_scores = record_scores_obj if isinstance(record_scores_obj, dict) else {}
    merged_scores, updates = merge_frontier_scores(current_scores, candidate_scores)
    record["gpu_architecture"] = gpu_architecture
    record["scores"] = merged_scores
    if run_id is not None:
        record["run_id"] = run_id
    if proof_bundle is not None:
        record["proof_bundle"] = proof_bundle
    frontiers = dict(frontiers)
    frontiers[gpu_architecture] = record
    return frontiers, updates


def resolve_gpu_architecture(
    value: str | None, *, default: GpuArchitecture = DEFAULT_GPU_ARCHITECTURE
) -> GpuArchitecture:
    arch = normalize_gpu_architecture(value)
    if arch is None:
        return default
    return arch
