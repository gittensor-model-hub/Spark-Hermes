"""Tests for eval.prepare_mining_sft."""

import json
from pathlib import Path

from eval.prepare_mining_sft import export_mining_sft


def _stub_remote_manifest():
    return {
        "mix_id": "mining-gittensor-model-hub-sparkproof-mining",
        "rows_total": 2,
        "sft_sha256": "a" * 64,
        "components": [],
    }


def test_export_mining_sft_writes_canonical_jsonl_format(tmp_path: Path, monkeypatch):
    rows = [
        {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]},
        {
            "messages": [{"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}],
            "metadata": {"mix_source_miner": "alice"},
        },
    ]

    class _Split:
        def __iter__(self):
            return iter(rows)

        def __len__(self):
            return len(rows)

    def _fake_load_dataset(repo_id, split, token=None):
        assert repo_id == "gittensor-model-hub/sparkproof-mining"
        assert split == "train"
        return _Split()

    monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset)
    monkeypatch.setattr(
        "eval.prepare_mining_sft.fetch_remote_mix_manifest",
        lambda **kwargs: _stub_remote_manifest(),
    )

    out = tmp_path / "mining.jsonl"
    result = export_mining_sft(
        out_path=out,
        repo_id="gittensor-model-hub/sparkproof-mining",
        hf_token="tok",
        verify_pin=False,
    )
    assert result["rows_written"] == 2
    assert result["mix_manifest_path"] == str((tmp_path / "mix_manifest.json").resolve())
    assert result["mix_manifest_sft_sha256"] == "a" * 64

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert "metadata" not in first
    assert second["metadata"]["mix_source_miner"] == "alice"
    assert first["messages"][0]["content"] == "a"
    assert lines[0] == json.dumps(first, separators=(",", ":"))

    mix_manifest = json.loads((tmp_path / "mix_manifest.json").read_text(encoding="utf-8"))
    assert mix_manifest["sft_sha256"] == "a" * 64


def test_export_mining_sft_bytes_are_lf_only(tmp_path: Path, monkeypatch):
    """The export's raw bytes must match the sha-pinned LF serialization on every OS.

    mix_manifest.sft_sha256 is computed over the exact file bytes; a platform-dependent
    newline translation (e.g. CRLF from text-mode writes on Windows) silently changes the
    digest and fails the canonical pin check for an honest export.
    """
    rows = [
        {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]},
        {"messages": [{"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}]},
    ]

    class _Split:
        def __iter__(self):
            return iter(rows)

        def __len__(self):
            return len(rows)

    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: _Split())
    monkeypatch.setattr(
        "eval.prepare_mining_sft.fetch_remote_mix_manifest",
        lambda **kwargs: _stub_remote_manifest(),
    )

    out = tmp_path / "mining.jsonl"
    export_mining_sft(
        out_path=out,
        repo_id="gittensor-model-hub/sparkproof-mining",
        verify_pin=False,
    )

    raw = out.read_bytes()
    expected = b"".join(
        json.dumps({"messages": row["messages"]}, separators=(",", ":")).encode("utf-8") + b"\n" for row in rows
    )
    assert raw == expected
    assert b"\r" not in raw


def test_export_mining_sft_writes_custom_mix_manifest_path(tmp_path: Path, monkeypatch):
    class _Split:
        def __iter__(self):
            return iter([])

        def __len__(self):
            return 0

    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: _Split())
    monkeypatch.setattr(
        "eval.prepare_mining_sft.fetch_remote_mix_manifest",
        lambda **kwargs: _stub_remote_manifest(),
    )

    out = tmp_path / "nested" / "mining.jsonl"
    mix_out = tmp_path / "nested" / "custom_mix.json"
    with __import__("pytest").raises(ValueError, match="train split is empty"):
        export_mining_sft(
            out_path=out,
            repo_id="gittensor-model-hub/sparkproof-mining",
            verify_pin=False,
            mix_manifest_out=mix_out,
        )
    assert mix_out.exists()
