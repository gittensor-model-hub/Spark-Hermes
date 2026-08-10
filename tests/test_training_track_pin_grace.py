"""Tests for canonical-pin grace window during training-track PRs (issue #118)."""

import json
from pathlib import Path

from eval.canonical_dataset import canonical_hf_url, sft_sha256_from_canonical_text
from eval.mix_registry import MIX_VERSION
from eval.training_track_gate import (
    _canonical_sft_sha256s_for_pr_window,
    validate_pr_body_canonical_pin,
    verify_remote_proof_bundle,
    verify_remote_proof_bundle_scores,
)
from eval.verify import verify_submission


def _canonical_json(sft_sha: str) -> str:
    return json.dumps(
        {
            "repo_id": "gittensor-model-hub/sparkproof-mining",
            "hf_url": "https://huggingface.co/datasets/gittensor-model-hub/sparkproof-mining",
            "mix_manifest": {"sft_sha256": sft_sha, "rows_total": 100},
        }
    )


SHA_HEAD = "a" * 64
SHA_BASE = "b" * 64
SHA_MID = "c" * 64

# A fixed blackwell frontier for gate-logic tests, so tiering assertions do not depend
# on the live runs/frontiers.json (which moves as runs are crowned, e.g. #301).
_PINNED_BLACKWELL_FRONTIER = {
    "gsm8k": 0.6,
    "triton": 0.42777777777777776,
    "triton_quick": 0.42777777777777776,
    "triton_exec_pass_rate": 0.0,
    "triton_correctness": 0.0,
    "triton_syntax_pass_rate": 1.0,
}


def test_sft_sha256_from_canonical_text():
    assert sft_sha256_from_canonical_text(_canonical_json(SHA_HEAD)) == SHA_HEAD
    assert sft_sha256_from_canonical_text("not json") is None


def test_canonical_sft_sha256s_for_pr_window(monkeypatch):
    def fake_show(ref: str, path: str) -> str | None:
        if path != "datasets/canonical.json":
            return None
        if ref in ("HEAD", "head"):
            return _canonical_json(SHA_HEAD)
        if ref == "merge-base":
            return _canonical_json(SHA_BASE)
        if ref == "mid-commit":
            return _canonical_json(SHA_MID)
        return None

    def fake_log(cmd, **kwargs):
        assert "merge-base..HEAD" in cmd

        class Result:
            returncode = 0
            stdout = "mid-commit\n"

        return Result()

    monkeypatch.setattr("eval.training_track_gate._git_show", fake_show)
    monkeypatch.setattr("eval.training_track_gate.subprocess.run", fake_log)

    shas = _canonical_sft_sha256s_for_pr_window(merge_base_ref="merge-base", head_ref="HEAD")
    assert shas == {SHA_HEAD, SHA_BASE, SHA_MID}


def test_validate_pr_body_accepts_merge_base_pin():
    url = "https://huggingface.co/datasets/gittensor-model-hub/sparkproof-mining"
    body = f"Canonical dataset URL: {url}\nPinned sft_sha256: `{SHA_BASE}`\n"
    issues = validate_pr_body_canonical_pin(body, acceptable_sft_shas={SHA_BASE, SHA_HEAD})
    assert issues == []


def test_validate_pr_body_rejects_stale_pin():
    url = "https://huggingface.co/datasets/gittensor-model-hub/sparkproof-mining"
    stale = "d" * 64
    body = f"Canonical dataset URL: {url}\nPinned sft_sha256: `{stale}`\n"
    issues = validate_pr_body_canonical_pin(body, acceptable_sft_shas={SHA_BASE, SHA_HEAD})
    assert issues
    assert any("merge-base window" in issue for issue in issues)


def test_verify_remote_proof_bundle_accepts_base_pin(monkeypatch):
    def fake_download(*, repo_id, repo_type, filename, token=None):
        tmp = Path("/tmp") / f"fake_{filename}"
        if filename == "manifest.json":
            tmp.write_text(
                json.dumps({"dataset_url": ("https://huggingface.co/datasets/gittensor-model-hub/sparkproof-mining")}),
                encoding="utf-8",
            )
        elif filename == "mix_manifest.json":
            tmp.write_text(json.dumps({"sft_sha256": SHA_BASE}), encoding="utf-8")
        return str(tmp)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

    issues = verify_remote_proof_bundle(
        "gittensor-model-hub/test-bundle",
        acceptable_sft_shas={SHA_BASE, SHA_HEAD},
    )
    assert issues == []


def test_verify_remote_proof_bundle_rejects_outside_window(monkeypatch):
    def fake_download(*, repo_id, repo_type, filename, token=None):
        tmp = Path("/tmp") / f"fake2_{filename}"
        if filename == "manifest.json":
            tmp.write_text(
                json.dumps({"dataset_url": ("https://huggingface.co/datasets/gittensor-model-hub/sparkproof-mining")}),
                encoding="utf-8",
            )
        elif filename == "mix_manifest.json":
            tmp.write_text(json.dumps({"sft_sha256": "e" * 64}), encoding="utf-8")
        return str(tmp)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

    issues = verify_remote_proof_bundle(
        "gittensor-model-hub/test-bundle",
        acceptable_sft_shas={SHA_BASE, SHA_HEAD},
    )
    assert any("accepted canonical pin" in issue for issue in issues)


def _bundle_with_mix_pin(tmp_path: Path, sft_sha: str) -> Path:
    """A proof bundle whose mix_manifest pins `sft_sha`, as published on HF."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"run_id": "r1", "dataset_url": canonical_hf_url()}), encoding="utf-8"
    )
    (bundle / "eval_scores.json").write_text(json.dumps({"scores": {"triton": 0.5}}), encoding="utf-8")
    (bundle / "mix_manifest.json").write_text(
        json.dumps({"mix_version": MIX_VERSION, "sft_sha256": sft_sha, "components": []}),
        encoding="utf-8",
    )
    return bundle


def test_verify_submission_accepts_pin_from_window(tmp_path):
    """eval.verify must honour the same grace window the gate computed.

    A weights-free bundle still ends at `checkpoint_required`; what matters is
    that the merge-base pin no longer fails the canonical-dataset claim.
    """
    bundle = _bundle_with_mix_pin(tmp_path, SHA_BASE)

    report = verify_submission(
        bundle,
        frontier=None,
        acceptable_sft_shas={SHA_BASE, SHA_HEAD},
    )
    assert report["reason"] == "checkpoint_required"


def test_verify_submission_rejects_pin_outside_window(tmp_path):
    bundle = _bundle_with_mix_pin(tmp_path, "e" * 64)

    report = verify_submission(
        bundle,
        frontier=None,
        acceptable_sft_shas={SHA_BASE, SHA_HEAD},
    )
    assert report["verified"] is False
    assert report["reason"] == "canonical_dataset_failed"


def test_verify_remote_proof_bundle_scores_threads_pin_window(tmp_path, monkeypatch):
    """A base-pin bundle must not be tiered eval:REJECT (which auto-closes the PR)."""
    import eval.training_track_gate as gate
    import eval.verify as verify_mod

    bundle = _bundle_with_mix_pin(tmp_path, SHA_BASE)

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *, repo_id, repo_type=None, token=None: str(bundle),
    )
    monkeypatch.setattr(
        gate,
        "_git_show",
        lambda ref, path: json.dumps({"passed": True, "token": "x"}) if path.endswith("attestation.json") else "",
    )
    monkeypatch.setattr(verify_mod, "check_attestation_integrity", lambda *a, **k: [])
    # Pin the frontier so the base-pin tiering assertion is independent of the live
    # runs/frontiers.json (a crowned frontier could turn the 0.5 claim into a regression).
    monkeypatch.setattr("eval.frontiers.load_frontier_scores", lambda *a, **k: _PINNED_BLACKWELL_FRONTIER)

    issues, eval_label = verify_remote_proof_bundle_scores(
        "org/repo",
        head_ref="HEAD",
        changed_paths=["recipes/foo.yaml", "runs/r1/attestation.json"],
        acceptable_sft_shas={SHA_BASE, SHA_HEAD},
    )
    assert issues == []
    assert eval_label != "eval:REJECT"
