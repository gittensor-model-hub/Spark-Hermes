import hashlib
import json

from eval.dataset_verify import check_proof_dir, size_label, verify_dataset_submission


def _write_proof_dir(
    tmp_path, *, rows=3, attested=True, gate_passed=True, tamper_rows=False, gpu_architecture="blackwell"
):
    proof = tmp_path / "proof"
    proof.mkdir()

    traj_lines = "\n".join(json.dumps({"prompt": f"p{i}", "response": f"r{i}"}) for i in range(rows)) + "\n"
    (proof / "trajectories.jsonl").write_text(traj_lines)
    sha = hashlib.sha256(traj_lines.encode()).hexdigest()

    (proof / "manifest.json").write_text(json.dumps({"version": "sparkproof-2"}))
    (proof / "prompts.jsonl").write_text(json.dumps({"prompt": "p0"}) + "\n")
    (proof / "trajectories_raw.jsonl").write_text(traj_lines)
    (proof / "validation_report.jsonl").write_text(
        "\n".join(json.dumps({"index": i, "validation": {"passed": True}}) for i in range(rows)) + "\n"
    )
    (proof / "gpu_attestation.json").write_text(json.dumps({"passed": attested, "nonce": "a" * 64}))
    (proof / "novelty_report.json").write_text(json.dumps({"verified_rows": rows, "novel_verified_rows": rows}))
    (proof / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "passed": gate_passed,
                "blocked_rows": 0 if gate_passed else 2,
                "rows_total": rows,
                "trajectories_sha256": sha,
                "gpu_architecture": gpu_architecture,
            }
        )
    )

    if tamper_rows:
        (proof / "trajectories.jsonl").write_text(traj_lines + json.dumps({"prompt": "extra"}) + "\n")
    return proof, sha


def test_size_label_bands():
    assert size_label(200) == "dataset:xl"
    assert size_label(150) == "dataset:xl"
    assert size_label(100) == "dataset:l"
    assert size_label(75) == "dataset:m"
    assert size_label(50) == "dataset:s"
    assert size_label(25) == "dataset:xs"
    assert size_label(24) == "dataset:none"


def test_valid_proof_dir_passes(tmp_path):
    proof, sha = _write_proof_dir(tmp_path)
    issues, rows, gpu_architecture = check_proof_dir(proof, claimed_sha256=sha)
    assert issues == []
    assert rows == 3
    assert gpu_architecture == "blackwell"


def test_hopper_gpu_architecture_is_accepted(tmp_path):
    proof, sha = _write_proof_dir(tmp_path, gpu_architecture="hopper-h100")
    issues, rows, gpu_architecture = check_proof_dir(proof, claimed_sha256=sha)
    assert issues == []
    assert gpu_architecture == "hopper"


def test_unsupported_gpu_architecture_rejects(tmp_path):
    proof, sha = _write_proof_dir(tmp_path, gpu_architecture="ampere-a100")
    issues, rows, gpu_architecture = check_proof_dir(proof, claimed_sha256=sha)
    assert any("gpu_architecture" in issue for issue in issues)


def test_missing_gpu_architecture_defaults_to_blackwell(tmp_path):
    # Regression test: bundles published before the gpu_architecture field existed
    # (e.g. speedy00/sparkproof-miner-25-v1's live HF dataset_manifest.json) must
    # keep verifying — this field is re-checked on EVERY registry entry whenever
    # the canonical mining dataset re-aggregates, not just on new submissions, so
    # treating a missing key as a hard failure breaks every prior merge and blocks
    # the mining-dataset publish step for brand-new PRs too. Confirmed live: this
    # broke PR #105's auto-merge on 2026-07-15 before this fix.
    proof = tmp_path / "proof"
    proof.mkdir()
    traj_lines = json.dumps({"prompt": "p0", "response": "r0"}) + "\n"
    (proof / "trajectories.jsonl").write_text(traj_lines)
    sha = hashlib.sha256(traj_lines.encode()).hexdigest()
    (proof / "manifest.json").write_text(json.dumps({"version": "sparkproof-2"}))
    (proof / "prompts.jsonl").write_text(json.dumps({"prompt": "p0"}) + "\n")
    (proof / "trajectories_raw.jsonl").write_text(traj_lines)
    (proof / "validation_report.jsonl").write_text(json.dumps({"index": 0, "validation": {"passed": True}}) + "\n")
    (proof / "gpu_attestation.json").write_text(json.dumps({"passed": True, "nonce": "n" * 64}))
    (proof / "novelty_report.json").write_text(json.dumps({"verified_rows": 1, "novel_verified_rows": 1}))
    (proof / "dataset_manifest.json").write_text(
        json.dumps({"passed": True, "blocked_rows": 0, "rows_total": 1, "trajectories_sha256": sha})
    )
    issues, _rows, gpu_architecture = check_proof_dir(proof, claimed_sha256=sha)
    assert issues == []
    assert gpu_architecture == "blackwell"


def test_garbage_gpu_architecture_value_still_rejects(tmp_path):
    # Unlike a missing key, an explicitly-present-but-unrecognized value is a
    # real anomaly (corrupted or hand-edited manifest) and must still fail.
    proof, sha = _write_proof_dir(tmp_path, gpu_architecture="not-a-real-gpu")
    issues, _rows, gpu_architecture = check_proof_dir(proof, claimed_sha256=sha)
    assert gpu_architecture is None
    assert any("not recognized" in issue for issue in issues)


def test_failed_attestation_rejects(tmp_path):
    proof, _ = _write_proof_dir(tmp_path, attested=False)
    report = verify_dataset_submission(proof)
    assert report["label"] == "dataset:REJECT"
    assert any("gpu_attestation" in issue for issue in report["issues"])


def test_failed_release_gate_rejects(tmp_path):
    proof, _ = _write_proof_dir(tmp_path, gate_passed=False)
    report = verify_dataset_submission(proof)
    assert report["label"] == "dataset:REJECT"


def test_tampered_rows_after_gate_rejects(tmp_path):
    proof, _ = _write_proof_dir(tmp_path, tamper_rows=True)
    report = verify_dataset_submission(proof)
    assert report["label"] == "dataset:REJECT"
    assert any("sha256" in issue for issue in report["issues"])


def test_claimed_sha_mismatch_rejects(tmp_path):
    proof, _ = _write_proof_dir(tmp_path)
    report = verify_dataset_submission(proof, claimed_sha256="deadbeef")
    assert report["label"] == "dataset:REJECT"
    assert any("claimed in the PR" in issue for issue in report["issues"])


def test_missing_sparkproof_root_rejects(tmp_path):
    proof, sha = _write_proof_dir(tmp_path)
    report = verify_dataset_submission(proof, claimed_sha256=sha, sparkproof_root=None)
    assert report["label"] == "dataset:REJECT"
    assert any("sparkproof-root is required" in issue for issue in report["issues"])


def test_missing_artifact_rejects(tmp_path):
    proof, _ = _write_proof_dir(tmp_path)
    (proof / "gpu_attestation.json").unlink()
    report = verify_dataset_submission(proof, sparkproof_root=None)
    assert report["label"] == "dataset:REJECT"
    assert any("missing proof artifact" in issue for issue in report["issues"])


def test_missing_tdx_on_new_bundle_rejects(tmp_path):
    proof, sha = _write_proof_dir(tmp_path)
    (proof / "gpu_attestation.json").write_text(json.dumps({"passed": True, "nonce": "a" * 64, "tdx": None}))
    issues, _rows, _arch = check_proof_dir(proof, claimed_sha256=sha)
    assert any("tdx required" in issue for issue in issues)


def test_bound_tdx_passes(tmp_path):
    import base64

    from eval.attestation import _TDX_REPORT_DATA_OFFSET, tdx_report_data

    nonce = "ab" * 32
    proof, sha = _write_proof_dir(tmp_path)
    quote = b"\x00" * _TDX_REPORT_DATA_OFFSET + tdx_report_data(nonce) + b"\x00" * 64
    (proof / "gpu_attestation.json").write_text(
        json.dumps(
            {
                "passed": True,
                "nonce": nonce,
                "tdx": {
                    "quote_b64": base64.b64encode(quote).decode(),
                    "report_data": tdx_report_data(nonce).hex(),
                },
            }
        )
    )
    issues, rows, gpu_architecture = check_proof_dir(proof, claimed_sha256=sha)
    assert issues == []
    assert rows == 3


def test_tdx_rejects_forged_json_report_data(tmp_path):
    import base64

    from eval.attestation import _TDX_REPORT_DATA_OFFSET, tdx_report_data

    nonce_quote = "ab" * 32
    nonce_json = "cd" * 32
    proof, sha = _write_proof_dir(tmp_path)
    quote = b"\x00" * _TDX_REPORT_DATA_OFFSET + tdx_report_data(nonce_quote) + b"\x00" * 64
    (proof / "gpu_attestation.json").write_text(
        json.dumps(
            {
                "passed": True,
                "nonce": nonce_json,
                "tdx": {
                    "quote_b64": base64.b64encode(quote).decode(),
                    "report_data": tdx_report_data(nonce_json).hex(),
                },
            }
        )
    )
    issues, _rows, _arch = check_proof_dir(proof, claimed_sha256=sha)
    assert any("REPORTDATA" in i or "report_data" in i for i in issues)


def test_sparkproof_verify_runs_online_trust_anchors(monkeypatch, tmp_path):
    # Without --online, sparkproof-verify never checks the NRAS signature and the
    # gate would accept a hand-written gpu_attestation.json.
    import eval.dataset_verify as dv

    captured = {}

    class Result:
        returncode = 0
        stdout = '{"verified": true}'
        stderr = ""

    def fake_run(cmd, cwd=None, capture_output=None, text=None, timeout=None):
        captured["cmd"] = cmd
        return Result()

    monkeypatch.setattr(dv.subprocess, "run", fake_run)
    issues = dv.run_sparkproof_verify(tmp_path, tmp_path)
    assert issues == []
    assert "--online" in captured["cmd"]
