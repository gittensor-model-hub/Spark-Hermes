"""Tests for eval.training_track_gate's no-GPU eval.verify wiring (issue: #120)."""

import json
from pathlib import Path

from eval.training_track_gate import (
    EVAL_LABELS,
    close_training_pr,
    find_attestation_path,
    record_merged_ledger_entry,
    score_claimed_eval_label,
    update_pr_eval_label,
    verify_remote_proof_bundle_scores,
)

_PR_BODY_WITH_BUNDLE = "Proof-bundle URL: https://huggingface.co/org/proof-repo"


def test_find_attestation_path_matches_run_dir():
    paths = ["recipes/foo.yaml", "runs/2026-07-15-magicrails-hopper-v2/attestation.json"]
    assert find_attestation_path(paths) == "runs/2026-07-15-magicrails-hopper-v2/attestation.json"


def test_find_attestation_path_absent():
    assert find_attestation_path(["recipes/foo.yaml"]) is None
    assert find_attestation_path(None) is None


def _fake_snapshot(bundle_dir: Path):
    def fake_snapshot_download(*, repo_id, repo_type, token=None):
        return str(bundle_dir)

    return fake_snapshot_download


# A fixed blackwell frontier for gate-logic tests, so tiering assertions do not depend
# on the live runs/frontiers.json (which moves as runs are crowned).
_PINNED_BLACKWELL_FRONTIER = {
    "gsm8k": 0.6,
    "triton": 0.42777777777777776,
    "triton_quick": 0.42777777777777776,
    "triton_exec_pass_rate": 0.0,
    "triton_correctness": 0.0,
    "triton_syntax_pass_rate": 1.0,
}


def test_verify_remote_proof_bundle_scores_rejects_forged_passed_true(tmp_path, monkeypatch):
    """{"passed": true} without JWKS/claim binding must not earn an eval tier."""
    import eval.training_track_gate as gate
    from eval.canonical_dataset import canonical_hf_url

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"run_id": "r1", "dataset_url": canonical_hf_url()}))
    (bundle / "eval_scores.json").write_text(json.dumps({"scores": {"triton": 0.5444}}))

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot(bundle))
    monkeypatch.setattr(
        gate,
        "_git_show",
        lambda ref, path: json.dumps({"passed": True}) if path.endswith("attestation.json") else "",
    )

    issues, eval_label = verify_remote_proof_bundle_scores(
        "org/repo",
        head_ref="HEAD",
        changed_paths=["recipes/foo.yaml", "runs/r1/attestation.json"],
    )
    assert eval_label == "eval:REJECT"
    assert any("JWKS" in i or "token" in i or "eat_nonce" in i for i in issues)


def test_verify_remote_proof_bundle_scores_tiers_claims_when_attestation_crypto_passes(tmp_path, monkeypatch):
    import eval.training_track_gate as gate
    import eval.verify as verify_mod
    from eval.canonical_dataset import canonical_hf_url

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"run_id": "r1", "dataset_url": canonical_hf_url()}))
    (bundle / "eval_scores.json").write_text(json.dumps({"scores": {"triton": 0.421}}))

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot(bundle))
    monkeypatch.setattr(
        gate,
        "_git_show",
        lambda ref, path: json.dumps({"passed": True, "token": "x"}) if path.endswith("attestation.json") else "",
    )
    monkeypatch.setattr(verify_mod, "check_attestation_integrity", lambda *a, **k: [])
    # Pin the frontier so this gate-logic test does not depend on runs/frontiers.json,
    # which legitimately moves as runs are crowned (e.g. #301 raised blackwell triton to
    # 0.606, which would turn the 0.421 claim below into an eval:REJECT regression).
    monkeypatch.setattr("eval.frontiers.load_frontier_scores", lambda *a, **k: _PINNED_BLACKWELL_FRONTIER)

    issues, eval_label = verify_remote_proof_bundle_scores(
        "org/repo",
        head_ref="HEAD",
        changed_paths=["recipes/foo.yaml", "runs/r1/attestation.json"],
    )
    # proof-only bundles defer checkpoint re-run; tier claimed scores only after crypto passes.
    assert issues == []
    assert eval_label == "eval:none"


def test_verify_remote_proof_bundle_scores_rejects_without_attestation(tmp_path, monkeypatch):
    from eval.canonical_dataset import canonical_hf_url

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"run_id": "r1", "dataset_url": canonical_hf_url()}))
    (bundle / "eval_scores.json").write_text(json.dumps({"scores": {"triton": 0.5444}}))

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot(bundle))

    issues, eval_label = verify_remote_proof_bundle_scores(
        "org/repo",
        head_ref="HEAD",
        changed_paths=["recipes/foo.yaml"],
    )
    assert eval_label == "eval:REJECT"
    assert any("attestation.json" in issue for issue in issues)


def test_verify_remote_proof_bundle_scores_surfaces_attested_mismatch(tmp_path, monkeypatch):
    import eval.verify as v

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"run_id": "r1"}))
    (bundle / "eval_scores.json").write_text(json.dumps({"scores": {"gsm8k": 0.64}}))

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot(bundle))
    monkeypatch.setattr(
        v,
        "verify_submission",
        lambda *a, **k: {
            "verified": False,
            "reason": "attested_eval_samples_failed",
            "issues": ["claimed gsm8k 0.64 diverges from attested regression sample 0.74"],
            "label": "eval:REJECT",
        },
    )

    issues, eval_label = verify_remote_proof_bundle_scores(
        "org/repo",
        head_ref="HEAD",
        changed_paths=None,
    )
    assert any("diverges from attested regression sample" in issue for issue in issues)
    assert eval_label == "eval:REJECT"


def test_verify_remote_proof_bundle_scores_passes_when_verified(tmp_path, monkeypatch):
    import eval.verify as v

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"run_id": "r1"}))
    (bundle / "eval_scores.json").write_text(json.dumps({"scores": {"gsm8k": 0.74}}))

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot(bundle))
    monkeypatch.setattr(v, "verify_submission", lambda *a, **k: {"verified": True, "label": "eval:BASELINE"})

    issues, eval_label = verify_remote_proof_bundle_scores(
        "org/repo",
        head_ref="HEAD",
        changed_paths=None,
    )
    assert issues == []
    assert eval_label == "eval:BASELINE"


def test_verify_remote_proof_bundle_scores_reads_attestation_from_head_ref(tmp_path, monkeypatch):
    import eval.training_track_gate as gate
    import eval.verify as v

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"run_id": "r1"}))
    (bundle / "eval_scores.json").write_text(json.dumps({"scores": {"gsm8k": 0.74}}))

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot(bundle))

    captured = {}

    def fake_show(ref, path):
        assert ref == "refs/remotes/origin/training-pr-head"
        assert path == "runs/r1/attestation.json"
        return json.dumps({"passed": True})

    def fake_verify_submission(bundle_dir, frontier, attestation=None, **kwargs):
        captured["attestation"] = attestation
        return {"verified": True, "label": "eval:BASELINE"}

    monkeypatch.setattr(gate, "_git_show", fake_show)
    monkeypatch.setattr(v, "verify_submission", fake_verify_submission)

    issues, eval_label = verify_remote_proof_bundle_scores(
        "org/repo",
        head_ref="refs/remotes/origin/training-pr-head",
        changed_paths=["runs/r1/attestation.json"],
    )
    assert issues == []
    assert eval_label == "eval:BASELINE"
    assert captured["attestation"] == {"passed": True}


def test_update_pr_eval_label_rejects_unknown_label():
    assert update_pr_eval_label(1, "eval:bogus") == ["refusing to apply unknown eval label 'eval:bogus'"]


def test_update_pr_eval_label_applies_known_label(monkeypatch):
    from types import SimpleNamespace

    import eval.training_track_gate as gate

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    assert update_pr_eval_label(120, "eval:BASELINE") == []
    assert ["gh", "pr", "edit", "120", "--add-label", "eval:BASELINE"] in calls
    # every other eval:* label gets removed so exactly one is ever active
    for other in EVAL_LABELS - {"eval:BASELINE"}:
        assert ["gh", "pr", "edit", "120", "--remove-label", other] in calls


def test_gate_training_pr_threads_eval_label(monkeypatch):
    import eval.training_track_gate as gate

    monkeypatch.setattr(gate, "should_enforce_training_gate", lambda *a, **k: True)
    monkeypatch.setattr(gate, "_canonical_sft_sha256s_for_pr_window", lambda **k: {"a" * 64})
    monkeypatch.setattr(gate, "validate_changed_paths", lambda *a, **k: [])
    monkeypatch.setattr(gate, "validate_recipe_paths_in_ref", lambda *a, **k: [])
    monkeypatch.setattr(gate, "validate_pr_body_canonical_pin", lambda *a, **k: [])
    monkeypatch.setattr(gate, "validate_pr_body_proof_bundle", lambda *a, **k: [])
    monkeypatch.setattr(gate, "is_training_track_pr", lambda *a, **k: True)
    monkeypatch.setattr(gate, "load_canonical", lambda: {})
    monkeypatch.setattr(gate, "parse_proof_bundle_hf_repo", lambda *a, **k: "org/repo")
    monkeypatch.setattr(gate, "verify_remote_proof_bundle", lambda *a, **k: [])
    monkeypatch.setattr(gate, "verify_remote_proof_bundle_scores", lambda *a, **k: ([], "eval:XL"))

    report = gate.gate_training_pr(
        head_ref="HEAD",
        changed_paths=["recipes/foo.yaml"],
        pr_body="- [x] Training/evaluation improvement",
        verify_hf_pin=False,
    )
    assert report["verified"] is True
    assert report["label"] == "training:valid"
    assert report["eval_label"] == "eval:XL"
    # A verified PR that earned a real reward tier is merge-eligible.
    assert report["merge_eligible"] is True


def test_gate_training_pr_not_merge_eligible_for_eval_none(monkeypatch):
    import eval.training_track_gate as gate

    monkeypatch.setattr(gate, "should_enforce_training_gate", lambda *a, **k: True)
    monkeypatch.setattr(gate, "_canonical_sft_sha256s_for_pr_window", lambda **k: {"a" * 64})
    monkeypatch.setattr(gate, "validate_changed_paths", lambda *a, **k: [])
    monkeypatch.setattr(gate, "validate_recipe_paths_in_ref", lambda *a, **k: [])
    monkeypatch.setattr(gate, "validate_pr_body_canonical_pin", lambda *a, **k: [])
    monkeypatch.setattr(gate, "validate_pr_body_proof_bundle", lambda *a, **k: [])
    monkeypatch.setattr(gate, "is_training_track_pr", lambda *a, **k: True)
    monkeypatch.setattr(gate, "load_canonical", lambda: {})
    monkeypatch.setattr(gate, "parse_proof_bundle_hf_repo", lambda *a, **k: "org/repo")
    monkeypatch.setattr(gate, "verify_remote_proof_bundle", lambda *a, **k: [])
    # verified format, but no reward tier -> must not auto-merge (it gets closed instead).
    monkeypatch.setattr(gate, "verify_remote_proof_bundle_scores", lambda *a, **k: ([], "eval:none"))

    report = gate.gate_training_pr(
        head_ref="HEAD",
        changed_paths=["recipes/foo.yaml"],
        pr_body="- [x] Training/evaluation improvement",
        verify_hf_pin=False,
    )
    assert report["verified"] is True
    assert report["merge_eligible"] is False


def test_record_merged_ledger_entry_no_op_without_proof_bundle():
    assert (
        record_merged_ledger_entry(
            pr_url="https://x/pull/1", pr_body="no bundle here", head_ref="HEAD", changed_paths=None
        )
        == []
    )


def test_record_merged_ledger_entry_appends_and_writes_result(tmp_path, monkeypatch):
    import eval.training_track_gate as gate

    report = {
        "verified": True,
        "label": "eval:BASELINE",
        "best_benchmark": None,
        "best_pct_delta": None,
        "regressions": [],
        "run_id": "run-1",
        "gpu_architecture": "hopper",
        "per_benchmark": {
            "gsm8k": {"candidate": 0.74, "frontier": None},
            "triton": {"candidate": 0.37, "frontier": None},
        },
    }
    monkeypatch.setattr(gate, "_download_and_verify_bundle", lambda *a, **k: (report, {"passed": True}, None, None))

    ledger_path = tmp_path / "ledger.jsonl"
    issues = record_merged_ledger_entry(
        pr_url="https://github.com/org/repo/pull/1",
        pr_body=_PR_BODY_WITH_BUNDLE,
        head_ref="HEAD",
        changed_paths=None,
        ledger_path=ledger_path,
    )
    assert issues == []

    lines = ledger_path.read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["run_id"] == "run-1"
    assert entry["label"] == "eval:BASELINE"
    assert entry["attested"] is True

    run_dir = tmp_path / "run-1"
    assert json.loads((run_dir / "result.json").read_text()) == report
    assert json.loads((run_dir / "attestation.json").read_text()) == {"passed": True}

    frontiers = json.loads((tmp_path / "frontiers.json").read_text())
    assert frontiers["hopper"]["run_id"] == "run-1"
    assert frontiers["hopper"]["scores"]["gsm8k"] == 0.74


def test_record_merged_ledger_entry_is_idempotent(tmp_path, monkeypatch):
    import eval.training_track_gate as gate

    report = {
        "verified": True,
        "label": "eval:BASELINE",
        "best_benchmark": None,
        "best_pct_delta": None,
        "regressions": [],
        "run_id": "run-1",
        "gpu_architecture": "hopper",
        "per_benchmark": {"triton": {"candidate": 0.37, "frontier": None}},
    }
    monkeypatch.setattr(gate, "_download_and_verify_bundle", lambda *a, **k: (report, None, None, None))

    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_text(json.dumps({"run_id": "run-1", "label": "eval:BASELINE"}) + "\n")

    issues = record_merged_ledger_entry(
        pr_url="https://github.com/org/repo/pull/1",
        pr_body=_PR_BODY_WITH_BUNDLE,
        head_ref="HEAD",
        changed_paths=None,
        ledger_path=ledger_path,
    )
    assert issues == []
    assert len(ledger_path.read_text().splitlines()) == 1  # no duplicate appended
    assert not (tmp_path / "run-1").exists()  # never even reached the write step
    # Frontier still heals on retry when ledger already had the run.
    assert json.loads((tmp_path / "frontiers.json").read_text())["hopper"]["scores"]["triton"] == 0.37


def test_record_merged_ledger_entry_never_overwrites_committed_attestation(tmp_path, monkeypatch):
    import eval.training_track_gate as gate

    report = {
        "verified": True,
        "label": "eval:BASELINE",
        "best_benchmark": None,
        "best_pct_delta": None,
        "regressions": [],
        "run_id": "run-1",
        "gpu_architecture": "blackwell",
        "per_benchmark": {"gsm8k": {"candidate": 0.6, "frontier": None}},
    }
    monkeypatch.setattr(
        gate, "_download_and_verify_bundle", lambda *a, **k: (report, {"passed": True, "new": True}, None, None)
    )

    ledger_path = tmp_path / "ledger.jsonl"
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "attestation.json").write_text('{"committed": true}')

    record_merged_ledger_entry(
        pr_url="https://github.com/org/repo/pull/1",
        pr_body=_PR_BODY_WITH_BUNDLE,
        head_ref="HEAD",
        changed_paths=None,
        ledger_path=ledger_path,
    )
    assert json.loads((run_dir / "attestation.json").read_text()) == {"committed": True}


def test_score_claimed_eval_label_empty_arch_bucket_is_baseline(tmp_path, monkeypatch):
    import eval.frontiers as frontiers_mod

    monkeypatch.setattr(frontiers_mod, "load_frontier_scores", lambda *a, **k: None)

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"run_id": "r1", "train_gpu": "hopper"}))
    (bundle / "eval_scores.json").write_text(json.dumps({"scores": {"triton": 0.5}}))

    assert score_claimed_eval_label(bundle, json.loads((bundle / "manifest.json").read_text())) == "eval:BASELINE"


def test_close_training_pr_eval_none(monkeypatch):
    from types import SimpleNamespace

    import eval.training_track_gate as gate

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    assert close_training_pr(150, eval_label="eval:none") == []
    assert calls == [
        [
            "gh",
            "pr",
            "close",
            "150",
            "--comment",
            "Closed automatically: claimed proof scores are below the merge threshold (eval:none).",
        ]
    ]


def test_main_auto_closes_training_valid_eval_none(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import eval.training_track_gate as gate

    report = {
        "verified": True,
        "label": "training:valid",
        "eval_label": "eval:none",
        "issues": [],
    }
    monkeypatch.setattr(gate, "gate_training_pr", lambda **k: report)

    out = tmp_path / "report.json"
    close_calls: list[dict] = []

    def fake_close(pr_number, **kwargs):
        close_calls.append({"pr_number": pr_number, **kwargs})
        return []

    monkeypatch.setattr(gate, "close_training_pr", fake_close)
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    rc = gate.main(
        [
            "--out",
            str(out),
            "--pr-number",
            "150",
            "--close-on-reject",
        ]
    )
    assert rc == 0
    assert close_calls == [{"pr_number": 150, "eval_label": "eval:none", "issues": []}]


def test_main_merges_when_merge_on_pass_and_eligible(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import eval.training_track_gate as gate

    monkeypatch.setattr(
        gate,
        "gate_training_pr",
        lambda **k: {
            "verified": True,
            "label": "training:valid",
            "eval_label": "eval:XS",
            "merge_eligible": True,
            "issues": [],
        },
    )
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    gh_output = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(gh_output))

    rc = gate.main(
        [
            "--out",
            str(tmp_path / "report.json"),
            "--skip-hf-pin-check",
            "--skip-proof-bundle-check",
            "--merge-on-pass",
            "--pr-number",
            "288",
        ]
    )
    assert rc == 0
    assert ["gh", "pr", "merge", "288", "--merge"] in calls
    # #309: a successful auto-merge signals the workflow to record the ledger +
    # crown the frontier in this same job (GITHUB_TOKEN merges raise no events).
    assert gh_output.read_text() == "merged=true\n"


def test_main_does_not_merge_when_not_eligible(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import eval.training_track_gate as gate

    monkeypatch.setattr(
        gate,
        "gate_training_pr",
        lambda **k: {
            "verified": True,
            "label": "training:valid",
            "eval_label": "eval:none",
            "merge_eligible": False,
            "issues": [],
        },
    )
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    gh_output = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(gh_output))

    gate.main(
        [
            "--out",
            str(tmp_path / "report.json"),
            "--skip-hf-pin-check",
            "--skip-proof-bundle-check",
            "--merge-on-pass",
            "--pr-number",
            "288",
        ]
    )
    assert not any(c[:3] == ["gh", "pr", "merge"] for c in calls)
    # Not merged -> no ledger/crown signal emitted for the workflow (#309).
    assert not gh_output.exists()


def test_emit_step_output_appends_when_github_output_set(tmp_path, monkeypatch):
    import eval.training_track_gate as gate

    out = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    gate._emit_step_output("merged", "true")
    assert out.read_text() == "merged=true\n"


def test_emit_step_output_noop_without_github_output(monkeypatch):
    import eval.training_track_gate as gate

    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    gate._emit_step_output("merged", "true")  # must not raise
