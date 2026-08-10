"""Fair dataset reward labels from canonical-mix contribution, not bundle size.

SparkProof sizes novelty within a single bundle. SparkDistill sizes rewards from
``mix_manifest.components[].rows_selected`` after cross-registry deduplication.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eval.dataset_verify import size_label


def rows_selected_for_sha(manifest: dict[str, Any], trajectories_sha256: str) -> int | None:
    """Return mix-selected row count for one registry submission."""
    for component in manifest.get("components") or []:
        if component.get("trajectories_sha256") == trajectories_sha256:
            return int(component.get("rows_selected") or 0)
    return None


def fair_label_from_rows_selected(rows_selected: int) -> str:
    return size_label(rows_selected)


def apply_fair_label(
    report: dict[str, Any],
    *,
    rows_selected: int,
) -> dict[str, Any]:
    """Downgrade (or upgrade) the gate label to match mix-selected rows."""
    bundle_label = str(report.get("label") or "dataset:REJECT")
    fair_label = fair_label_from_rows_selected(rows_selected)
    report = dict(report)
    report["bundle_label"] = bundle_label
    report["rows_selected"] = rows_selected
    report["label"] = fair_label
    if bundle_label != fair_label:
        report["fair_label_note"] = (
            f"fair label {fair_label} from {rows_selected} canonical-mix rows "
            f"(bundle has {report.get('rows_total')} verified rows, sized as {bundle_label})"
        )
    return report


def mix_manifest_from_result(mix_result: Any) -> dict[str, Any]:
    import json

    return json.loads(mix_result.manifest_path.read_text(encoding="utf-8"))


def compute_rows_selected_for_entry(
    registry_entries: list[dict[str, Any]],
    trajectories_sha256: str,
    *,
    sparkproof_root: Path | None = None,
    dedupe: str | None = None,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """Dry-run the mining mix and return rows_selected for one submission."""
    from eval.mining_dataset import aggregate_mining_mix

    mix_report = aggregate_mining_mix(
        registry_entries,
        sparkproof_root=sparkproof_root,
        work_dir=work_dir,
        dedupe=dedupe,
    )
    if not mix_report.get("verified"):
        return {
            "verified": False,
            "issues": list(mix_report.get("issues") or ["mining mix failed"]),
            "rows_selected": 0,
            "manifest": mix_report.get("manifest"),
        }

    manifest = mix_report["manifest"]
    rows_selected = rows_selected_for_sha(manifest, trajectories_sha256)
    if rows_selected is None:
        return {
            "verified": False,
            "issues": [f"mix manifest missing component for {trajectories_sha256}"],
            "rows_selected": 0,
            "manifest": manifest,
        }
    return {
        "verified": True,
        "issues": [],
        "rows_selected": rows_selected,
        "manifest": manifest,
        "mix_rows_total": manifest.get("rows_total"),
    }
