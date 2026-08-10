import json


def test_export_mining_dpo_writes_chosen_rejected(tmp_path, monkeypatch):
    import datasets
    import eval.prepare_mining_dpo as prep

    monkeypatch.setattr("eval.prepare_mining_dpo.canonical_pref_repo_id", lambda *a, **k: "org/dpo")
    monkeypatch.setattr(
        "eval.prepare_mining_dpo.canonical_pref_hf_url", lambda *a, **k: "https://huggingface.co/datasets/org/dpo"
    )
    rows = [
        {"prompt": "p1", "chosen": "good kernel", "rejected": "bad kernel", "metadata": {"pair_type": "correctness"}}
    ]
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: rows)

    out = tmp_path / "dpo.jsonl"
    result = prep.export_mining_dpo(out_path=out, verify_pin=False)

    assert result["rows_written"] == 1
    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert row["prompt"] == "p1"
    assert row["chosen"] == "good kernel"
    assert row["rejected"] == "bad kernel"
    assert row["metadata"]["pair_type"] == "correctness"


def test_export_mining_dpo_rejects_incomplete_pairs(tmp_path, monkeypatch):
    import datasets
    import eval.prepare_mining_dpo as prep

    monkeypatch.setattr("eval.prepare_mining_dpo.canonical_pref_repo_id", lambda *a, **k: "org/dpo")
    monkeypatch.setattr("eval.prepare_mining_dpo.canonical_pref_hf_url", lambda *a, **k: "https://x")
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: [{"prompt": "p", "chosen": "c"}])  # no rejected

    import pytest

    with pytest.raises(ValueError, match="missing"):
        prep.export_mining_dpo(out_path=tmp_path / "dpo.jsonl", verify_pin=False)
