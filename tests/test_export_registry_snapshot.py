import json

import pytest

from eval.export_registry_snapshot import (
    _task_id_from_trajectory,
    attach_snapshot_pins_to_manifest,
    collect_accepted_trajectories,
    verify_registry_snapshot_pins,
    write_registry_snapshot,
)
from eval.mix_registry import load_trajectories_jsonl


@pytest.fixture(autouse=True)
def _force_fallback_dedupe(monkeypatch):
    # These tests assert the built-in prompt-based fallback dedupe. Force it so results are
    # deterministic even when a sibling SparkProof checkout would otherwise make
    # resolve_sparkproof_root(None) import the real (more aggressive) NoveltyRegistry.
    monkeypatch.setattr("eval.mix_registry._import_novelty", lambda *args, **kwargs: None)


# Stub SparkProof publish exporter: keeps every row (returns it), so tests exercise the
# dedup/order logic without a SparkProof checkout. Tests that need the unexportable path
# pass their own stub returning None for the dropped rows.
def _keep_all(trajectory):
    return trajectory


def _traj(task_id: str, prompt: str, *, gpu_architecture: str = "blackwell") -> dict:
    return {
        "prompt": prompt,
        "response": f"```python\n# {task_id}\n```",
        "metadata": {
            "prompt_meta": {
                "task_id": task_id,
                "origin": "torch_op",
                "split": "train",
                "gpu_architecture": gpu_architecture,
            }
        },
        "gpu_architecture": gpu_architecture,
        "sparkproof_validation": {"passed": True},
    }


def test_collect_accepted_trajectories_respects_registry_order(tmp_path, monkeypatch):
    bundle_a = tmp_path / "a" / "proof"
    bundle_b = tmp_path / "b" / "proof"
    bundle_a.mkdir(parents=True)
    bundle_b.mkdir(parents=True)
    (bundle_a / "trajectories.jsonl").write_text(
        json.dumps(_traj("task_a", "prompt A")) + "\n" + json.dumps(_traj("task_shared", "shared prompt")) + "\n"
    )
    (bundle_b / "trajectories.jsonl").write_text(
        json.dumps(_traj("task_b", "prompt B")) + "\n" + json.dumps(_traj("task_shared_dup", "shared prompt")) + "\n"
    )

    entries = [
        {
            "miner": "alice",
            "hf_url": "https://huggingface.co/datasets/org/a",
            "trajectories_sha256": "a" * 64,
            "rows_total": 2,
            "dataset_version": "triton-distill-v0.2",
            "gpu_architecture": "blackwell",
        },
        {
            "miner": "bob",
            "hf_url": "https://huggingface.co/datasets/org/b",
            "trajectories_sha256": "b" * 64,
            "rows_total": 2,
            "dataset_version": "triton-distill-v0.2",
            "gpu_architecture": "blackwell",
        },
    ]

    def fake_resolve(entry, proof_cache=None, download_proof=None):
        repo = entry["hf_url"].rsplit("/", 1)[-1]
        return tmp_path / repo / "proof"

    monkeypatch.setattr("eval.export_registry_snapshot.resolve_proof_dir", fake_resolve)

    accepted = collect_accepted_trajectories(entries, dedupe="exact", sparkproof_root=None, export_fn=_keep_all)
    task_ids = [((row.get("metadata") or {}).get("prompt_meta") or {}).get("task_id") for row in accepted]
    assert task_ids == ["task_a", "task_shared", "task_b"]


def test_collect_accepted_trajectories_keeps_same_prompt_across_architectures(tmp_path, monkeypatch):
    bundle_a = tmp_path / "a" / "proof"
    bundle_b = tmp_path / "b" / "proof"
    bundle_a.mkdir(parents=True)
    bundle_b.mkdir(parents=True)
    shared = "shared prompt"
    (bundle_a / "trajectories.jsonl").write_text(
        json.dumps(_traj("task_bw", shared, gpu_architecture="blackwell")) + "\n"
    )
    (bundle_b / "trajectories.jsonl").write_text(
        json.dumps(_traj("task_hp", shared, gpu_architecture="hopper-h100")) + "\n"
    )

    entries = [
        {
            "miner": "alice",
            "hf_url": "https://huggingface.co/datasets/org/a",
            "trajectories_sha256": "a" * 64,
            "rows_total": 1,
            "dataset_version": "triton-distill-v0.2",
            "gpu_architecture": "blackwell",
        },
        {
            "miner": "bob",
            "hf_url": "https://huggingface.co/datasets/org/b",
            "trajectories_sha256": "b" * 64,
            "rows_total": 1,
            "dataset_version": "triton-distill-v0.2",
            "gpu_architecture": "hopper",
        },
    ]

    def fake_resolve(entry, proof_cache=None, download_proof=None):
        repo = entry["hf_url"].rsplit("/", 1)[-1]
        return tmp_path / repo / "proof"

    monkeypatch.setattr("eval.export_registry_snapshot.resolve_proof_dir", fake_resolve)

    accepted = collect_accepted_trajectories(entries, dedupe="exact", sparkproof_root=None, export_fn=_keep_all)
    task_ids = [((row.get("metadata") or {}).get("prompt_meta") or {}).get("task_id") for row in accepted]
    assert task_ids == ["task_bw", "task_hp"]


def test_write_registry_snapshot_writes_task_id_index(tmp_path, monkeypatch):
    bundle = tmp_path / "a" / "proof"
    bundle.mkdir(parents=True)
    (bundle / "trajectories.jsonl").write_text(json.dumps(_traj("task_a", "prompt A")) + "\n")

    entry = {
        "miner": "alice",
        "hf_url": "https://huggingface.co/datasets/org/a",
        "trajectories_sha256": "a" * 64,
        "rows_total": 1,
        "dataset_version": "triton-distill-v0.2",
        "gpu_architecture": "blackwell",
    }

    monkeypatch.setattr(
        "eval.export_registry_snapshot.resolve_proof_dir",
        lambda entry, proof_cache=None, download_proof=None: bundle,
    )

    out = tmp_path / "snapshot.jsonl"
    task_ids = tmp_path / "task_ids.json"
    report = write_registry_snapshot(
        [entry], out_path=out, task_ids_path=task_ids, sparkproof_root=None, export_fn=_keep_all
    )
    assert report["rows_total"] == 1
    assert len(load_trajectories_jsonl(out)) == 1
    payload = json.loads(task_ids.read_text())
    assert payload["task_ids"] == ["task_a"]


def test_write_registry_snapshot_bytes_are_lf_only(tmp_path, monkeypatch):
    """Snapshot bytes must match the sha-pinned LF serialization on every OS.

    accepted_registry_snapshot_sha256 is computed over the exact file bytes; a
    platform-dependent newline translation (CRLF from text-mode writes on Windows)
    changes the digest and breaks pin verification for an honest export.
    """
    bundle = tmp_path / "a" / "proof"
    bundle.mkdir(parents=True)
    row = _traj("task_a", "prompt A")
    (bundle / "trajectories.jsonl").write_text(json.dumps(row) + "\n")

    entry = {
        "miner": "alice",
        "hf_url": "https://huggingface.co/datasets/org/a",
        "trajectories_sha256": "a" * 64,
        "rows_total": 1,
        "dataset_version": "triton-distill-v0.2",
        "gpu_architecture": "blackwell",
    }
    monkeypatch.setattr(
        "eval.export_registry_snapshot.resolve_proof_dir",
        lambda entry, proof_cache=None, download_proof=None: bundle,
    )

    out = tmp_path / "snapshot.jsonl"
    write_registry_snapshot(
        [entry], out_path=out, task_ids_path=tmp_path / "ids.json", sparkproof_root=None, export_fn=_keep_all
    )

    raw = out.read_bytes()
    assert raw.endswith(b"\n")
    assert b"\r" not in raw


def test_snapshot_manifest_pins_and_verify(tmp_path, monkeypatch):
    bundle = tmp_path / "a" / "proof"
    bundle.mkdir(parents=True)
    (bundle / "trajectories.jsonl").write_text(json.dumps(_traj("task_a", "prompt A")) + "\n")
    entry = {
        "miner": "alice",
        "hf_url": "https://huggingface.co/datasets/org/a",
        "trajectories_sha256": "a" * 64,
        "rows_total": 1,
        "dataset_version": "triton-distill-v0.2",
        "gpu_architecture": "blackwell",
    }
    monkeypatch.setattr(
        "eval.export_registry_snapshot.resolve_proof_dir",
        lambda entry, proof_cache=None, download_proof=None: bundle,
    )

    out = tmp_path / "snapshot.jsonl"
    task_ids = tmp_path / "task_ids.json"
    manifest_path = tmp_path / "mix_manifest.json"
    manifest_path.write_text("{}\n")

    report = write_registry_snapshot(
        [entry], out_path=out, task_ids_path=task_ids, sparkproof_root=None, export_fn=_keep_all
    )
    manifest = attach_snapshot_pins_to_manifest(manifest_path, report)
    assert manifest["accepted_registry_snapshot_sha256"] == report["sha256"]
    assert (
        verify_registry_snapshot_pins(
            [entry],
            manifest=manifest,
            snapshot_path=out,
            task_ids_path=task_ids,
            sparkproof_root=None,
            export_fn=_keep_all,
        )
        == []
    )

    manifest["accepted_registry_snapshot_sha256"] = "f" * 64
    issues = verify_registry_snapshot_pins(
        [entry],
        manifest=manifest,
        snapshot_path=out,
        task_ids_path=task_ids,
        sparkproof_root=None,
        export_fn=_keep_all,
    )
    assert any("does not match mix_manifest pin" in issue for issue in issues)


def _drop_task(*task_ids: str):
    """Stub exporter that drops (returns None for) the given task_ids, like SparkProof
    dropping empty/failed rows — mirrors mix_registry_datasets' unexportable filter."""
    dropped = set(task_ids)

    def export_fn(trajectory):
        return None if _task_id_from_trajectory(trajectory) in dropped else trajectory

    return export_fn


def test_collect_accepted_trajectories_skips_unexportable_rows(tmp_path, monkeypatch):
    # Rows the exporter drops (empty/failed) must not occupy the accepted snapshot,
    # matching mix_registry_datasets — otherwise accepted_task_ids over-reports and
    # miners skip regenerating tasks the mix never actually filled.
    bundle = tmp_path / "a" / "proof"
    bundle.mkdir(parents=True)
    (bundle / "trajectories.jsonl").write_text(
        json.dumps(_traj("task_a", "prompt A"))
        + "\n"
        + json.dumps(_traj("task_bad", "prompt BAD"))
        + "\n"
        + json.dumps(_traj("task_c", "prompt C"))
        + "\n"
    )
    entry = {
        "miner": "alice",
        "hf_url": "https://huggingface.co/datasets/org/a",
        "trajectories_sha256": "a" * 64,
        "rows_total": 3,
        "dataset_version": "triton-distill-v0.2",
        "gpu_architecture": "blackwell",
    }
    monkeypatch.setattr(
        "eval.export_registry_snapshot.resolve_proof_dir",
        lambda entry, proof_cache=None, download_proof=None: bundle,
    )

    accepted = collect_accepted_trajectories(
        [entry], dedupe="exact", sparkproof_root=None, export_fn=_drop_task("task_bad")
    )
    assert [_task_id_from_trajectory(row) for row in accepted] == ["task_a", "task_c"]


def test_unexportable_row_does_not_dedup_block_a_later_duplicate(tmp_path, monkeypatch):
    # An unexportable row must not enter the dedup state: a later good row with the same
    # prompt must still be accepted (the mix keeps it), or the snapshot would wrongly
    # dedup it away and disagree with the canonical mix.
    for name, task in (("a", "task_empty"), ("b", "task_real")):
        d = tmp_path / name / "proof"
        d.mkdir(parents=True)
        (d / "trajectories.jsonl").write_text(json.dumps(_traj(task, "shared prompt")) + "\n")

    entries = [
        {
            "miner": "alice",
            "hf_url": "https://huggingface.co/datasets/org/a",
            "trajectories_sha256": "a" * 64,
            "rows_total": 1,
            "dataset_version": "triton-distill-v0.2",
            "gpu_architecture": "blackwell",
        },
        {
            "miner": "bob",
            "hf_url": "https://huggingface.co/datasets/org/b",
            "trajectories_sha256": "b" * 64,
            "rows_total": 1,
            "dataset_version": "triton-distill-v0.2",
            "gpu_architecture": "blackwell",
        },
    ]
    monkeypatch.setattr(
        "eval.export_registry_snapshot.resolve_proof_dir",
        lambda entry, proof_cache=None, download_proof=None: tmp_path / entry["hf_url"].rsplit("/", 1)[-1] / "proof",
    )

    accepted = collect_accepted_trajectories(
        entries, dedupe="exact", sparkproof_root=None, export_fn=_drop_task("task_empty")
    )
    assert [_task_id_from_trajectory(row) for row in accepted] == ["task_real"]
