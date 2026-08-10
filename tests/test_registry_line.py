import json
from pathlib import Path

import pytest

from eval.registry_line import build_registry_entry, load_dataset_manifest, normalize_hf_url


def test_normalize_hf_url_accepts_repo_id():
    assert normalize_hf_url("org/sparkproof-triton-v0") == ("https://huggingface.co/datasets/org/sparkproof-triton-v0")


def test_build_registry_entry_from_manifest(tmp_path: Path):
    manifest = {
        "dataset_version": "triton-distill-v0.2",
        "rows_total": 25,
        "trajectories_sha256": "a" * 64,
        "gpu_architecture": "blackwell",
    }
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_dataset_manifest(tmp_path)
    entry = build_registry_entry(
        miner="alice",
        hf_url="https://huggingface.co/datasets/org/sparkproof-triton-v0",
        manifest=loaded,
    )
    assert entry == {
        "miner": "alice",
        "hf_url": "https://huggingface.co/datasets/org/sparkproof-triton-v0",
        "trajectories_sha256": "a" * 64,
        "rows_total": 25,
        "dataset_version": "triton-distill-v0.2",
        "gpu_architecture": "blackwell",
    }


def test_build_registry_entry_from_hopper_manifest(tmp_path: Path):
    manifest = {
        "dataset_version": "triton-distill-v0.2",
        "rows_total": 25,
        "trajectories_sha256": "a" * 64,
        "gpu_architecture": "hopper-h100",
    }
    entry = build_registry_entry(miner="alice", hf_url="org/repo", manifest=manifest)
    assert entry["gpu_architecture"] == "hopper"


def test_build_registry_entry_requires_sha256(tmp_path: Path):
    with pytest.raises(ValueError, match="trajectories_sha256"):
        build_registry_entry(
            miner="alice",
            hf_url="org/repo",
            manifest={"rows_total": 1, "dataset_version": "v0", "gpu_architecture": "blackwell"},
        )


def test_build_registry_entry_requires_gpu_architecture(tmp_path: Path):
    with pytest.raises(ValueError, match="gpu_architecture"):
        build_registry_entry(
            miner="alice",
            hf_url="org/repo",
            manifest={"rows_total": 1, "dataset_version": "v0", "trajectories_sha256": "a" * 64},
        )
