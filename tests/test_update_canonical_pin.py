"""Tests for eval.update_canonical_pin."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval import update_canonical_pin as pin


def _pin_payload(rows: int = 213) -> dict:
    return {
        "repo_id": "gittensor-model-hub/sparkproof-mining",
        "hf_url": "https://huggingface.co/datasets/gittensor-model-hub/sparkproof-mining",
        "training_dataset_path": "data/processed/sparkproof-mining_sft.jsonl",
        "mix_manifest": {
            "mix_id": "mining-gittensor-model-hub/sparkproof-mining",
            "rows_total": rows,
            "sft_sha256": "a" * 64,
        },
    }


def _write_pin(out: Path, rows: int = 213) -> None:
    out.write_text(json.dumps(_pin_payload(rows), indent=2) + "\n", encoding="utf-8")


def test_commit_canonical_pin_noop_when_already_current(monkeypatch, tmp_path: Path):
    out = tmp_path / "canonical.json"
    _write_pin(out)

    def refresh(**_):
        _write_pin(out)
        return _pin_payload()

    monkeypatch.setattr(pin, "refresh_canonical_pin_file", refresh)
    monkeypatch.setattr(
        pin,
        "_git_run",
        lambda *args, **kwargs: pytest.fail(f"unexpected git call: {args}"),
    )

    assert pin.commit_canonical_pin_to_main(out_path=out) == []


def test_commit_canonical_pin_falls_back_to_pr_on_branch_protection(monkeypatch, tmp_path: Path):
    out = tmp_path / "canonical.json"
    calls: list[list[str]] = []

    def refresh(**_):
        _write_pin(out, rows=999)
        return _pin_payload(999)

    def fake_git(args: list[str], *, check: bool = True):
        calls.append(args)
        if args[:2] == ["git", "push"] and args[-1] == "HEAD:main":
            return type(
                "R",
                (),
                {
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "GH013: Changes must be made through a pull request.",
                },
            )()
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(pin, "refresh_canonical_pin_file", refresh)
    monkeypatch.setattr(pin, "_git_run", fake_git)
    monkeypatch.setattr(pin, "publish_canonical_pin_via_pr", lambda **_: [])

    assert pin.commit_canonical_pin_to_main(out_path=out) == []
    assert any(args[:3] == ["git", "reset", "--hard"] for args in calls)


def test_publish_canonical_pin_via_pr_merges(monkeypatch, tmp_path: Path):
    out = tmp_path / "canonical.json"
    calls: list[list[str]] = []

    def refresh(**_):
        _write_pin(out, rows=213)
        return _pin_payload(213)

    def fake_git(args: list[str], *, check: bool = True):
        calls.append(args)
        return type("R", (), {"returncode": 0, "stdout": "https://github.com/org/repo/pull/9", "stderr": ""})()

    monkeypatch.setattr(pin, "refresh_canonical_pin_file", refresh)
    monkeypatch.setattr(pin, "_git_run", fake_git)

    assert pin.publish_canonical_pin_via_pr(out_path=out) == []
    assert calls[-2][:3] == ["gh", "pr", "create"]
    assert calls[-1][:4] == ["gh", "pr", "merge", "9"]
