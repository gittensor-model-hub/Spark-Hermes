"""Pinned canonical mining dataset for fair training-track competition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.dataset_verify import _sha256_file
from eval.mining_dataset import DEFAULT_MINING_DATASET_REPO, MINING_MANIFEST_PATH

CANONICAL_PATH = Path("datasets/canonical.json")
CANONICAL_TRAINING_DATASET_PATH = "data/processed/sparkproof-mining_sft.jsonl"
# DPO training track: a canonical preference dataset (chosen/rejected pairs) published
# by SparkProof, pinned separately from the SFT mix so the two accepted-sha sets stay
# disjoint (an SFT submission can never satisfy the DPO pin, or vice-versa).
CANONICAL_PREFERENCE_DATASET_PATH = "data/processed/sparkproof-mining_dpo.jsonl"

TRAINING_TRACK_SFT = "sft"
TRAINING_TRACK_DPO = "dpo"


def recipe_training_track(recipe: dict[str, Any]) -> str:
    """Which training track a recipe declares: ``dpo`` iff Axolotl ``rl: dpo``, else ``sft``.

    SFT recipes carry no ``rl`` key, so existing recipes keep the exact SFT behavior.
    """
    return TRAINING_TRACK_DPO if str(recipe.get("rl") or "").strip().lower() == "dpo" else TRAINING_TRACK_SFT


def canonical_dataset_path_for_track(track: str) -> str:
    """The one dataset path a recipe on ``track`` is allowed to point at."""
    return CANONICAL_PREFERENCE_DATASET_PATH if track == TRAINING_TRACK_DPO else CANONICAL_TRAINING_DATASET_PATH


def load_canonical(path: Path = CANONICAL_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing canonical dataset pin: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def canonical_hf_url(path: Path = CANONICAL_PATH) -> str:
    return str(load_canonical(path)["hf_url"])


def canonical_repo_id(path: Path = CANONICAL_PATH) -> str:
    return str(load_canonical(path).get("repo_id") or DEFAULT_MINING_DATASET_REPO)


def canonical_sft_sha256(path: Path = CANONICAL_PATH) -> str:
    manifest = load_canonical(path).get("mix_manifest") or {}
    value = manifest.get("sft_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{path} mix_manifest.sft_sha256 must be a 64-char hex digest")
    return value


def sft_sha256_from_canonical_text(text: str) -> str | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    value = (payload.get("mix_manifest") or {}).get("sft_sha256")
    if isinstance(value, str) and len(value) == 64:
        return value
    return None


def canonical_pref_sha256(path: Path = CANONICAL_PATH) -> str:
    manifest = load_canonical(path).get("pref_manifest") or {}
    value = manifest.get("pref_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{path} pref_manifest.pref_sha256 must be a 64-char hex digest")
    return value


def pref_sha256_from_canonical_text(text: str) -> str | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    value = (payload.get("pref_manifest") or {}).get("pref_sha256")
    if isinstance(value, str) and len(value) == 64:
        return value
    return None


def canonical_sha256_for_track(track: str, path: Path = CANONICAL_PATH) -> str:
    """Select the pinned digest for a track (DPO → pref_sha256, else sft_sha256)."""
    return canonical_pref_sha256(path) if track == TRAINING_TRACK_DPO else canonical_sft_sha256(path)


def canonical_pref_hf_url(path: Path = CANONICAL_PATH) -> str:
    """Hugging Face URL of the canonical preference dataset (DPO track)."""
    manifest = load_canonical(path).get("pref_manifest") or {}
    url = manifest.get("hf_url")
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"{path} pref_manifest.hf_url is required for the DPO track")
    return url


def fetch_remote_mix_manifest(
    *,
    repo_id: str | None = None,
    hf_token: str | None = None,
) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    repo = (repo_id or canonical_repo_id()).strip()
    manifest_path = hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        filename=MINING_MANIFEST_PATH,
        token=hf_token,
    )
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{repo}/{MINING_MANIFEST_PATH} must be a JSON object")
    return manifest


def verify_manifest_matches_pin(
    remote: dict[str, Any],
    *,
    repo_id: str | None = None,
    pin_path: Path = CANONICAL_PATH,
) -> list[str]:
    """Return issues when a fetched HF mix_manifest does not match datasets/canonical.json."""
    issues: list[str] = []
    pin = load_canonical(pin_path)
    expected_repo = str(pin.get("repo_id") or DEFAULT_MINING_DATASET_REPO)
    repo = (repo_id or expected_repo).strip()
    if repo != expected_repo:
        issues.append(f"repo_id {repo!r} does not match pinned {expected_repo!r}")

    pinned_manifest = pin.get("mix_manifest") or {}
    remote_sft_sha = remote.get("sft_sha256")
    expected_sft_sha = pinned_manifest.get("sft_sha256")
    if expected_sft_sha and remote_sft_sha != expected_sft_sha:
        issues.append(
            "canonical pin is stale: HF mix_manifest.sft_sha256 "
            f"{remote_sft_sha!r} != pinned {expected_sft_sha!r}; "
            "run scripts/update_canonical_pin.sh after the mining dataset is republished"
        )

    expected_rows = pinned_manifest.get("rows_total")
    remote_rows = remote.get("rows_total")
    if expected_rows is not None and remote_rows != expected_rows:
        issues.append(f"canonical rows_total mismatch: HF has {remote_rows!r}, pin has {expected_rows!r}")
    return issues


def verify_remote_matches_pin(
    *,
    repo_id: str | None = None,
    hf_token: str | None = None,
    pin_path: Path = CANONICAL_PATH,
) -> list[str]:
    """Return issues when the live HF canonical dataset does not match datasets/canonical.json."""
    repo = (repo_id or canonical_repo_id(pin_path)).strip()
    try:
        remote = fetch_remote_mix_manifest(repo_id=repo, hf_token=hf_token)
    except Exception as exc:
        return [f"failed to download {repo}/{MINING_MANIFEST_PATH}: {exc}"]
    return verify_manifest_matches_pin(remote, repo_id=repo, pin_path=pin_path)


def write_pin_from_remote(
    *,
    repo_id: str | None = None,
    hf_token: str | None = None,
    out_path: Path = CANONICAL_PATH,
) -> dict[str, Any]:
    """Refresh datasets/canonical.json from the live HF mining dataset manifest."""
    repo = (repo_id or DEFAULT_MINING_DATASET_REPO).strip()
    remote = fetch_remote_mix_manifest(repo_id=repo, hf_token=hf_token)
    sft_sha = remote.get("sft_sha256")
    if not isinstance(sft_sha, str) or len(sft_sha) != 64:
        raise ValueError(f"{repo} mix_manifest missing sft_sha256")

    payload = {
        "repo_id": repo,
        "hf_url": f"https://huggingface.co/datasets/{repo}",
        "training_dataset_path": CANONICAL_TRAINING_DATASET_PATH,
        "mix_manifest": {
            "mix_id": remote.get("mix_id"),
            "rows_total": remote.get("rows_total"),
            "sft_sha256": sft_sha,
        },
    }
    mix_manifest = payload["mix_manifest"]
    if isinstance(remote.get("accepted_registry_snapshot_sha256"), str):
        mix_manifest["accepted_registry_snapshot_sha256"] = remote["accepted_registry_snapshot_sha256"]
    if remote.get("accepted_registry_snapshot_rows_total") is not None:
        mix_manifest["accepted_registry_snapshot_rows_total"] = remote["accepted_registry_snapshot_rows_total"]
    if isinstance(remote.get("accepted_task_ids_sha256"), str):
        mix_manifest["accepted_task_ids_sha256"] = remote["accepted_task_ids_sha256"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def assert_recipe_uses_canonical_dataset(recipe: dict[str, Any]) -> list[str]:
    """Training recipes may only point at the canonical export path for their track.

    SFT recipes (no ``rl`` key) must use the canonical mining SFT mix — unchanged. A DPO
    recipe (``rl: dpo``) must instead use the canonical preference dataset; pointing a DPO
    recipe at the SFT path (or an SFT recipe at the preference path) is rejected, so the two
    tracks' accepted datasets never cross.
    """
    issues: list[str] = []
    datasets = recipe.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        return issues

    track = recipe_training_track(recipe)
    allowed_path = canonical_dataset_path_for_track(track)
    for index, entry in enumerate(datasets):
        if not isinstance(entry, dict):
            continue
        data_path = entry.get("path")
        if not isinstance(data_path, str) or not data_path.strip():
            continue
        normalized = data_path.strip()
        if normalized != allowed_path:
            issues.append(
                f"datasets[{index}].path must be {allowed_path!r} (canonical {track} dataset only), got {normalized!r}"
            )
    return issues


def sha256_matches_canonical_export(export_path: Path, pin_path: Path = CANONICAL_PATH) -> list[str]:
    if not export_path.exists():
        return [f"missing canonical export {export_path}"]
    actual = _sha256_file(export_path)
    expected = canonical_sft_sha256(pin_path)
    if actual != expected:
        return [f"{export_path} sha256 {actual} does not match pinned canonical sft_sha256 {expected}"]
    return []


def canonical_pref_repo_id(path: Path = CANONICAL_PATH) -> str:
    """HF repo id of the canonical preference dataset (from ``pref_manifest``)."""
    manifest = load_canonical(path).get("pref_manifest") or {}
    repo = manifest.get("repo_id")
    if isinstance(repo, str) and repo.strip():
        return repo.strip()
    # Fall back to deriving it from the pinned preference dataset URL.
    return canonical_pref_hf_url(path).rstrip("/").split("huggingface.co/datasets/", 1)[-1]


def pref_sha256_matches_canonical_export(export_path: Path, pin_path: Path = CANONICAL_PATH) -> list[str]:
    """Check a prepared DPO preference export against the pinned ``pref_sha256``."""
    if not export_path.exists():
        return [f"missing canonical preference export {export_path}"]
    actual = _sha256_file(export_path)
    expected = canonical_pref_sha256(pin_path)
    if actual != expected:
        return [f"{export_path} sha256 {actual} does not match pinned canonical pref_sha256 {expected}"]
    return []
