import fnmatch
import hashlib
import json
import os
import shutil
import types
from pathlib import Path

import pytest

from eval.mix_registry import (
    MIX_VERSION,
    _import_trajectory_exporter,
    load_registry,
    mix_registry_datasets,
    resolve_proof_dir,
    select_registry_entries,
    trajectory_to_sft_record,
    verify_mix_manifest,
)
from tests.test_dataset_verify import _write_proof_dir

SPARKPROOF_ROOT = Path(
    os.environ.get("SPARKPROOF_ROOT", Path(__file__).resolve().parents[1] / ".." / "SparkProof")
).resolve()


@pytest.fixture(scope="module")
def sparkproof_root():
    if not (SPARKPROOF_ROOT / "sparkproof" / "publish" / "hf_dataset.py").is_file():
        pytest.skip("SparkProof checkout required beside SparkDistill (or set SPARKPROOF_ROOT)")
    return SPARKPROOF_ROOT


def _registry_entry(miner: str, sha: str, *, rows: int = 2) -> dict:
    return {
        "miner": miner,
        "hf_url": f"https://huggingface.co/datasets/{miner}/sparkproof-{sha[:8]}",
        "trajectories_sha256": sha,
        "rows_total": rows,
        "dataset_version": "triton-distill-v0.2",
        "gpu_architecture": "blackwell",
    }


def _repair_trajectory(task_prompt: str, response: str, *, gpu_architecture: str = "hopper-h100") -> dict:
    return {
        "prompt": ("Your prior Triton 3.7.1 answer failed hardware validation.\nFailure: triton_api\nTrace tail:\n"),
        "response": response,
        "metadata": {
            "tier": "repair",
            "prompt_meta": {
                "task_id": task_prompt,
                "prompt": task_prompt,
                "origin": "torch_op",
                "split": "train",
                "gpu_architecture": gpu_architecture,
            },
        },
        "gpu_architecture": gpu_architecture,
    }


def _kernel_response(tag: str) -> str:
    """Distinct fenced Python so novelty dedupe does not collapse test fixtures."""
    return f"```python\n# {tag}\nprint('{tag}')\n```"


def _trajectory(prompt: str, response: str, *, gpu_architecture: str = "blackwell") -> dict:
    return {
        "prompt": prompt,
        "response": response,
        "metadata": {
            "prompt_meta": {
                "task_id": prompt,
                "prompt": prompt,
                "origin": "torch_op",
                "split": "train",
                "gpu_architecture": gpu_architecture,
            }
        },
        "gpu_architecture": gpu_architecture,
    }


def _write_registry(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")


def _fake_download(proof_roots: dict[str, Path]):
    def download(repo: str, _cache: Path | None) -> Path:
        return proof_roots[repo]

    return download


def test_select_registry_entries_by_sha256(tmp_path: Path):
    registry = [_registry_entry("alice", "a" * 64), _registry_entry("bob", "b" * 64)]
    selected = select_registry_entries(registry, sha256s=["b" * 64, "a" * 64])
    assert [row["miner"] for row in selected] == ["bob", "alice"]


def test_select_registry_entries_requires_match():
    with pytest.raises(ValueError, match="not found in registry"):
        select_registry_entries([], sha256s=["c" * 64])


def test_mix_registry_deduplicates_and_writes_manifest(tmp_path: Path, sparkproof_root: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    proof_a, _sha_a = _write_proof_dir(tmp_path / "a", rows=2)
    proof_b, _sha_b = _write_proof_dir(tmp_path / "b", rows=1)
    (proof_a / "trajectories.jsonl").write_text(
        json.dumps(_trajectory("shared prompt", "resp-a"))
        + "\n"
        + json.dumps(_trajectory("only-a", "resp-only-a"))
        + "\n",
        encoding="utf-8",
    )
    sha_a = hashlib.sha256((proof_a / "trajectories.jsonl").read_bytes()).hexdigest()
    (proof_a / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "passed": True,
                "blocked_rows": 0,
                "rows_total": 2,
                "trajectories_sha256": sha_a,
                "dataset_version": "triton-distill-v0.2",
                "gpu_architecture": "blackwell",
            }
        ),
        encoding="utf-8",
    )
    (proof_b / "trajectories.jsonl").write_text(
        json.dumps(_trajectory("shared prompt", "resp-b")) + "\n",
        encoding="utf-8",
    )
    sha_b = hashlib.sha256((proof_b / "trajectories.jsonl").read_bytes()).hexdigest()
    (proof_b / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "passed": True,
                "blocked_rows": 0,
                "rows_total": 1,
                "trajectories_sha256": sha_b,
                "dataset_version": "triton-distill-v0.2",
                "gpu_architecture": "blackwell",
            }
        ),
        encoding="utf-8",
    )

    registry_path = tmp_path / "registry.jsonl"
    entries = [_registry_entry("alice", sha_a, rows=2), _registry_entry("bob", sha_b, rows=1)]
    _write_registry(registry_path, entries)

    out_path = tmp_path / "mix_sft.jsonl"
    manifest_path = tmp_path / "mix_manifest.json"
    download = _fake_download(
        {
            "alice/sparkproof-" + sha_a[:8]: proof_a,
            "bob/sparkproof-" + sha_b[:8]: proof_b,
        }
    )

    result = mix_registry_datasets(
        entries,
        out_path=out_path,
        manifest_path=manifest_path,
        mix_id="mix-test",
        sparkproof_root=sparkproof_root,
        dedupe="exact",
        download_proof=download,
    )

    assert result.rows_total == 2
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["mix_version"] == MIX_VERSION
    assert manifest["rows_total"] == 2
    assert manifest["components"][0]["rows_selected"] == 2
    assert manifest["components"][1]["rows_selected"] == 0
    assert manifest["dedupe"]["exact_skipped"] == 1

    report = verify_mix_manifest(manifest_path, sft_path=out_path, registry_path=registry_path)
    assert report["verified"] is True


def test_mix_registry_keeps_same_prompt_across_gpu_architectures(tmp_path: Path, sparkproof_root: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    proof_a, _sha_a = _write_proof_dir(tmp_path / "a", rows=1)
    proof_b, _sha_b = _write_proof_dir(tmp_path / "b", rows=1)
    shared = "shared prompt across architectures"
    (proof_a / "trajectories.jsonl").write_text(
        json.dumps(_trajectory(shared, "resp-a", gpu_architecture="blackwell")) + "\n",
        encoding="utf-8",
    )
    sha_a = hashlib.sha256((proof_a / "trajectories.jsonl").read_bytes()).hexdigest()
    (proof_a / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "passed": True,
                "blocked_rows": 0,
                "rows_total": 1,
                "trajectories_sha256": sha_a,
                "dataset_version": "triton-distill-v0.2",
                "gpu_architecture": "blackwell",
            }
        ),
        encoding="utf-8",
    )
    (proof_b / "trajectories.jsonl").write_text(
        json.dumps(_trajectory(shared, "resp-b", gpu_architecture="hopper-h100")) + "\n",
        encoding="utf-8",
    )
    sha_b = hashlib.sha256((proof_b / "trajectories.jsonl").read_bytes()).hexdigest()
    (proof_b / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "passed": True,
                "blocked_rows": 0,
                "rows_total": 1,
                "trajectories_sha256": sha_b,
                "dataset_version": "triton-distill-v0.2",
                "gpu_architecture": "hopper-h100",
            }
        ),
        encoding="utf-8",
    )

    registry_path = tmp_path / "registry.jsonl"
    entries = [
        _registry_entry("alice", sha_a, rows=1),
        {**_registry_entry("bob", sha_b, rows=1), "gpu_architecture": "hopper"},
    ]
    _write_registry(registry_path, entries)

    out_path = tmp_path / "mix_sft.jsonl"
    manifest_path = tmp_path / "mix_manifest.json"
    download = _fake_download(
        {
            "alice/sparkproof-" + sha_a[:8]: proof_a,
            "bob/sparkproof-" + sha_b[:8]: proof_b,
        }
    )

    result = mix_registry_datasets(
        entries,
        out_path=out_path,
        manifest_path=manifest_path,
        mix_id="mix-arch-test",
        sparkproof_root=sparkproof_root,
        dedupe="exact",
        download_proof=download,
    )

    assert result.rows_total == 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["components"][0]["rows_selected"] == 1
    assert manifest["components"][1]["rows_selected"] == 1
    assert manifest["dedupe"]["exact_skipped"] == 0


def test_mix_registry_keeps_repair_rows_with_shared_wrapper_but_distinct_tasks(tmp_path: Path, sparkproof_root: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    proof_a, _sha_a = _write_proof_dir(tmp_path / "a", rows=1)
    proof_b, _sha_b = _write_proof_dir(tmp_path / "b", rows=1)
    (proof_a / "trajectories.jsonl").write_text(
        json.dumps(_repair_trajectory("translate matmul kernel task A", _kernel_response("matmul-a"))) + "\n",
        encoding="utf-8",
    )
    sha_a = hashlib.sha256((proof_a / "trajectories.jsonl").read_bytes()).hexdigest()
    (proof_a / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "passed": True,
                "blocked_rows": 0,
                "rows_total": 1,
                "trajectories_sha256": sha_a,
                "dataset_version": "triton-distill-v0.2",
                "gpu_architecture": "hopper-h100",
            }
        ),
        encoding="utf-8",
    )
    (proof_b / "trajectories.jsonl").write_text(
        json.dumps(_repair_trajectory("translate relu kernel task B", _kernel_response("relu-b"))) + "\n",
        encoding="utf-8",
    )
    sha_b = hashlib.sha256((proof_b / "trajectories.jsonl").read_bytes()).hexdigest()
    (proof_b / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "passed": True,
                "blocked_rows": 0,
                "rows_total": 1,
                "trajectories_sha256": sha_b,
                "dataset_version": "triton-distill-v0.2",
                "gpu_architecture": "hopper-h100",
            }
        ),
        encoding="utf-8",
    )

    registry_path = tmp_path / "registry.jsonl"
    entries = [
        {**_registry_entry("alice", sha_a, rows=1), "gpu_architecture": "hopper"},
        {**_registry_entry("bob", sha_b, rows=1), "gpu_architecture": "hopper"},
    ]
    _write_registry(registry_path, entries)

    out_path = tmp_path / "mix_sft.jsonl"
    manifest_path = tmp_path / "mix_manifest.json"
    download = _fake_download(
        {
            "alice/sparkproof-" + sha_a[:8]: proof_a,
            "bob/sparkproof-" + sha_b[:8]: proof_b,
        }
    )

    result = mix_registry_datasets(
        entries,
        out_path=out_path,
        manifest_path=manifest_path,
        mix_id="mix-repair-test",
        sparkproof_root=sparkproof_root,
        dedupe="exact",
        download_proof=download,
    )

    assert result.rows_total == 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["components"][0]["rows_selected"] == 1
    assert manifest["components"][1]["rows_selected"] == 1
    assert manifest["dedupe"]["exact_skipped"] == 0


def test_verify_mix_manifest_rejects_unknown_component(tmp_path: Path):
    manifest_path = tmp_path / "mix_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "mix_version": MIX_VERSION,
                "mix_id": "mix-test",
                "rows_total": 1,
                "sft_sha256": "deadbeef",
                "components": [_registry_entry("alice", "f" * 64)],
            }
        ),
        encoding="utf-8",
    )
    registry_path = tmp_path / "registry.jsonl"
    _write_registry(registry_path, [])

    report = verify_mix_manifest(manifest_path, registry_path=registry_path)
    assert report["verified"] is False
    assert any("not in registry" in issue for issue in report["issues"])


def _fake_hf_module(source_proof: Path):
    def snapshot_download(repo_id, repo_type=None, allow_patterns=None, cache_dir=None):
        dest = source_proof.parent / "downloaded"
        (dest / "proof").mkdir(parents=True, exist_ok=True)
        for f in source_proof.glob("*"):
            rel = f"proof/{f.name}"
            if allow_patterns and not any(fnmatch.fnmatch(rel, pat) for pat in allow_patterns):
                continue
            shutil.copy(f, dest / "proof" / f.name)
        return str(dest)

    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = snapshot_download
    return module


def test_resolve_proof_dir_downloads_full_bundle(tmp_path: Path, monkeypatch):
    proof, _ = _write_proof_dir(tmp_path, rows=2)
    sha = hashlib.sha256((proof / "trajectories.jsonl").read_bytes()).hexdigest()
    manifest = json.loads((proof / "dataset_manifest.json").read_text(encoding="utf-8"))
    manifest["trajectories_sha256"] = sha
    (proof / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", _fake_hf_module(proof))

    entry = _registry_entry("alice", sha, rows=2)
    resolved = resolve_proof_dir(entry)

    assert (resolved / "manifest.json").exists()
    assert (resolved / "gpu_attestation.json").exists()
    assert (resolved / "novelty_report.json").exists()
    assert (resolved / "trajectories.jsonl").exists()


def test_load_registry_validates_entries(tmp_path: Path):
    registry_path = tmp_path / "registry.jsonl"
    registry_path.write_text(json.dumps({"miner": "alice"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid registry entry"):
        load_registry(registry_path)


def test_trajectory_to_sft_record_skips_null_response(sparkproof_root: Path):
    export_fn = _import_trajectory_exporter(sparkproof_root)
    row = _trajectory("task prompt", "good code")
    row["response"] = None
    assert (
        trajectory_to_sft_record(row, component=_registry_entry("alice", "a" * 64), row_index=0, export_fn=export_fn)
        is None
    )


def test_trajectory_to_sft_record_skips_failed_validation(sparkproof_root: Path):
    export_fn = _import_trajectory_exporter(sparkproof_root)
    row = _trajectory("task prompt", "good code")
    row["sparkproof_validation"] = {"passed": False}
    assert (
        trajectory_to_sft_record(row, component=_registry_entry("alice", "a" * 64), row_index=0, export_fn=export_fn)
        is None
    )


def test_trajectory_to_sft_record_uses_prompt_meta_not_repair_wrapper(sparkproof_root: Path):
    export_fn = _import_trajectory_exporter(sparkproof_root)
    row = _repair_trajectory("translate relu kernel task A", _kernel_response("relu-a"))
    record = trajectory_to_sft_record(
        row,
        component=_registry_entry("alice", "a" * 64),
        row_index=0,
        export_fn=export_fn,
    )
    assert record is not None
    user_turn = next(m for m in record["messages"] if m["role"] == "user")
    assert user_turn["content"] == "translate relu kernel task A"
    assert "failed hardware validation" not in user_turn["content"]


def test_mix_registry_skips_unexportable_rows(tmp_path: Path, sparkproof_root: Path):
    proof, _ = _write_proof_dir(tmp_path, rows=2)
    good = _trajectory("good task", "kernel code")
    bad = _trajectory("bad task", "ignored")
    bad["response"] = None
    (proof / "trajectories.jsonl").write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n", encoding="utf-8")
    sha = hashlib.sha256((proof / "trajectories.jsonl").read_bytes()).hexdigest()
    (proof / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "passed": True,
                "blocked_rows": 0,
                "rows_total": 2,
                "trajectories_sha256": sha,
                "dataset_version": "triton-distill-v0.2",
                "gpu_architecture": "blackwell",
            }
        ),
        encoding="utf-8",
    )
    entry = _registry_entry("alice", sha, rows=2)
    out_path = tmp_path / "mix_sft.jsonl"
    manifest_path = tmp_path / "mix_manifest.json"
    result = mix_registry_datasets(
        [entry],
        out_path=out_path,
        manifest_path=manifest_path,
        mix_id="mix-skip-test",
        sparkproof_root=sparkproof_root,
        dedupe="none",
        download_proof=_fake_download({"alice/sparkproof-" + sha[:8]: proof}),
    )
    assert result.rows_total == 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["components"][0]["rows_skipped_unexportable"] == 1
    assert manifest["dedupe"]["unexportable_skipped"] == 1


def test_trajectory_to_sft_record_prefers_multi_turn_episode(sparkproof_root: Path):
    export_fn = _import_trajectory_exporter(sparkproof_root)
    row = _trajectory("write softmax", "final kernel")
    row["sparkproof_validation"] = {"passed": True}
    row["metadata"]["multi_turn"] = True
    row["metadata"]["episode_version"] = "sparkproof-episode-v1"
    row["metadata"]["episode"] = {
        "version": "sparkproof-episode-v1",
        "task_prompt": "write softmax",
        "system": "You are a kernel expert.",
        "provider": "anthropic",
        "turns": [
            {"role": "user", "kind": "task", "content": "write softmax"},
            {"role": "assistant", "kind": "attempt", "content": "failed attempt"},
            {"role": "user", "kind": "validator", "content": "[sparkproof-validator] FAILED"},
            {"role": "assistant", "kind": "repair", "content": "fixed kernel"},
        ],
        "accepted": True,
        "repairs_used": 1,
    }
    record = trajectory_to_sft_record(
        row,
        component=_registry_entry("alice", "a" * 64),
        row_index=0,
        export_fn=export_fn,
    )
    assert record is not None
    assert len(record["messages"]) == 5
    assert record["metadata"]["multi_turn"] is True
    assert record["metadata"]["repairs_used"] == 1


def test_load_trajectories_jsonl_rejects_non_object_row(tmp_path):
    # Miner-controlled non-object rows must raise a clean ValueError (issue #214).
    from eval.mix_registry import load_trajectories_jsonl

    path = tmp_path / "trajectories.jsonl"
    path.write_text('{"prompt":"p","response":"r"}\n[1, 2, 3]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="trajectory row must be a JSON object"):
        load_trajectories_jsonl(path)


def test_load_trajectories_jsonl_reports_invalid_json(tmp_path):
    from eval.mix_registry import load_trajectories_jsonl

    path = tmp_path / "trajectories.jsonl"
    path.write_text('{"prompt":"p"}\n{bad json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_trajectories_jsonl(path)
