import json
from pathlib import Path

from eval.mining_dataset import (
    DEFAULT_MINING_DATASET_REPO,
    aggregate_and_publish_mining_dataset,
    aggregate_registry_text,
    mining_dataset_repo,
)
from tests.test_dataset_verify import _write_proof_dir
from tests.test_mix_registry import _fake_download, _registry_entry, _trajectory


def test_mining_dataset_repo_default():
    assert mining_dataset_repo() == DEFAULT_MINING_DATASET_REPO


def test_mining_dedupe_mode_defaults_to_exact(monkeypatch):
    monkeypatch.delenv("SPARKDISTILL_MINING_DEDUPE", raising=False)
    from eval.mining_dataset import DEFAULT_MINING_DEDUPE, mining_dedupe_mode

    assert DEFAULT_MINING_DEDUPE == "exact"
    assert mining_dedupe_mode() == "exact"


def test_aggregate_registry_text_dedupes_by_sha():
    base = json.dumps(_registry_entry("alice", "a" * 64)) + "\n"
    head = base + json.dumps(_registry_entry("bob", "b" * 64)) + "\n"
    rows = aggregate_registry_text(base, head)
    assert len(rows) == 2
    assert rows[0]["miner"] == "alice"
    assert rows[1]["miner"] == "bob"


def test_aggregate_and_publish_mining_dataset(tmp_path: Path, monkeypatch, sparkproof_root):
    # Needs SparkProof's SFT exporter to build mining_sft.jsonl; skips when absent.
    (tmp_path / "a").mkdir()
    proof_a, _ = _write_proof_dir(tmp_path / "a", rows=1)
    (proof_a / "trajectories.jsonl").write_text(json.dumps(_trajectory("prompt-a", "resp-a")) + "\n", encoding="utf-8")
    sha_a = __import__("hashlib").sha256((proof_a / "trajectories.jsonl").read_bytes()).hexdigest()
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

    entries = [_registry_entry("alice", sha_a, rows=1)]
    download = _fake_download({"alice/sparkproof-" + sha_a[:8]: proof_a})

    def fake_publish(**kwargs):
        return {
            "published": True,
            "hf_url": "https://huggingface.co/datasets/org/mining",
            "rows_total": 1,
            "issues": [],
        }

    monkeypatch.setattr(
        "eval.export_registry_snapshot.publish_registry_snapshot",
        lambda *args, **kwargs: {
            "published": True,
            "hf_url": "https://huggingface.co/datasets/org/mining",
            "rows_total": 1,
        },
    )

    report = aggregate_and_publish_mining_dataset(
        entries,
        repo_id="org/mining",
        work_dir=tmp_path / "work",
        download_proof=download,
        publish_fn=fake_publish,
    )
    assert report["published"] is True
    assert (tmp_path / "work" / "mining_sft.jsonl").exists()
    assert (tmp_path / "work" / "mix_manifest.json").exists()


def test_gate_blocks_merge_when_mining_publish_fails(monkeypatch):
    import eval.registry_gate as registry_gate

    monkeypatch.setattr(
        registry_gate,
        "gate_registry_submission",
        lambda *args, **kwargs: {
            "verified": True,
            "label": "dataset:xs",
            "rows_total": 25,
            "issues": [],
        },
    )

    def fail_publish(*args, **kwargs):
        return {"published": False, "issues": ["hf upload failed"]}

    monkeypatch.setattr(
        registry_gate,
        "compute_rows_selected_for_entry",
        lambda *args, **kwargs: {"verified": True, "rows_selected": 25, "issues": []},
    )

    report = registry_gate.gate_registry_pr(
        base_registry_text="",
        head_registry_text=json.dumps(_registry_entry("alice", "a" * 64)) + "\n",
        sparkproof_root=Path("."),
        pr_body="- [x] **Dataset track submission**",
        changed_paths=["datasets/registry.jsonl"],
        mining_dataset_repo_id="org/mining",
        publish_mining_dataset=fail_publish,
    )
    assert report["reward_eligible"] is True
    assert report["merge_eligible"] is False
    assert any("hf upload failed" in issue for issue in report["issues"])


def test_gate_merges_when_mining_publish_succeeds(monkeypatch):
    import eval.registry_gate as registry_gate

    monkeypatch.setattr(
        registry_gate,
        "gate_registry_submission",
        lambda *args, **kwargs: {
            "verified": True,
            "label": "dataset:xs",
            "rows_total": 25,
            "issues": [],
        },
    )

    def ok_publish(*args, **kwargs):
        return {
            "published": True,
            "hf_url": "https://huggingface.co/datasets/org/mining",
            "rows_total": 25,
            "issues": [],
        }

    monkeypatch.setattr(
        registry_gate,
        "compute_rows_selected_for_entry",
        lambda *args, **kwargs: {"verified": True, "rows_selected": 25, "issues": []},
    )
    monkeypatch.setattr(
        "eval.export_registry_snapshot.verify_remote_registry_snapshot",
        lambda *args, **kwargs: [],
    )

    report = registry_gate.gate_registry_pr(
        base_registry_text="",
        head_registry_text=json.dumps(_registry_entry("alice", "a" * 64)) + "\n",
        sparkproof_root=Path("."),
        pr_body="- [x] **Dataset track submission**",
        changed_paths=["datasets/registry.jsonl"],
        mining_dataset_repo_id="org/mining",
        publish_mining_dataset=ok_publish,
    )
    assert report["merge_eligible"] is True
    assert report["mining_dataset"]["published"] is True
