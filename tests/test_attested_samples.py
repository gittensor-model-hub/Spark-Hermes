import json

from eval.attested_samples import (
    ATTESTED_SAMPLES_FILENAME,
    build_attested_samples_document,
    build_gsm8k_regression_entry,
    build_lm_eval_entry,
    build_triton_entry,
    read_attested_samples,
    verify_attested_eval_samples,
)
from eval.regression_sample import build_regression_sample, load_regression_problems


def _bound_attestation(bundle_dir, digest: str, monkeypatch) -> dict:
    """GPU+TDX attestation whose *signed* JWT eat_nonce matches claim_sha256."""
    import base64
    import json

    import jwt
    from cryptography.hazmat.primitives.asymmetric import ec

    from eval.attestation import _TDX_REPORT_DATA_OFFSET, tdx_report_data

    key = ec.generate_private_key(ec.SECP384R1())

    def encode(payload):
        return jwt.encode(payload, key, algorithm="ES384", headers={"kid": "nv-eat-kid-test"})

    token = json.dumps(
        [
            ["JWT", jwt.encode({"sub": "overall"}, "k", algorithm="HS256")],
            {
                "REMOTE_GPU_CLAIMS": [
                    [
                        "JWT",
                        encode({"iss": "https://nras.attestation.nvidia.com", "sub": "platform"}),
                    ],
                    {
                        "GPU-0": encode(
                            {
                                "iss": "https://nras.attestation.nvidia.com",
                                "hwmodel": "GH100",
                                "eat_nonce": digest,
                            }
                        )
                    },
                ]
            },
        ]
    )

    class FakeKey:
        def __init__(self, k):
            self.key = k.public_key()

    class FakeJWKClient:
        def __init__(self, url):
            pass

        def get_signing_key_from_jwt(self, encoded):
            return FakeKey(key)

    monkeypatch.setattr(jwt, "PyJWKClient", FakeJWKClient)

    quote = b"\x00" * _TDX_REPORT_DATA_OFFSET + tdx_report_data(digest) + b"\x00" * 64
    return {
        "passed": True,
        "token": token,
        "claims": {"eat_nonce": "must-be-ignored"},
        "tdx": {
            "report_data": tdx_report_data(digest).hex(),
            "quote_b64": base64.b64encode(quote).decode(),
        },
    }


def _claim_binding(bundle_dir, attestation):
    from eval.verify import check_claim_binding

    return check_claim_binding(bundle_dir, attestation)


def _tdx_binding(bundle_dir, attestation):
    from eval.verify import check_tdx_binding

    return check_tdx_binding(bundle_dir, attestation)


def test_verify_attested_samples_requires_gpu_and_tdx_bindings(tmp_path, monkeypatch):
    responses = [
        {"problem_id": int(row["problem_id"]), "model_response": f"#### {row['answer'].split('####')[-1].strip()}"}
        for row in load_regression_problems()
    ]
    document = build_attested_samples_document({"gsm8k": build_gsm8k_regression_entry(responses)})

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / ATTESTED_SAMPLES_FILENAME).write_text(json.dumps(document, indent=2))
    (bundle / "manifest.json").write_text(json.dumps({"run_id": "r1"}))
    (bundle / "eval_scores.json").write_text(json.dumps({"scores": {"gsm8k": 1.0}}))

    verified, issues = verify_attested_eval_samples(
        bundle,
        {"gsm8k": 1.0},
        {"gsm8k": 0.5},
        None,
        claim_binding=_claim_binding,
        tdx_binding=_tdx_binding,
    )
    assert verified == set()
    assert any("GPU CC attestation" in issue for issue in issues)

    from proof.bundle import claim_sha256

    digest = claim_sha256(bundle)
    # JSON-only "binding" is no longer accepted.
    gpu_only = {"passed": True, "claims": {"eat_nonce": digest}}
    verified, issues = verify_attested_eval_samples(
        bundle,
        {"gsm8k": 1.0},
        {"gsm8k": 0.5},
        gpu_only,
        claim_binding=_claim_binding,
        tdx_binding=_tdx_binding,
    )
    assert verified == set()
    assert any("claim_sha256-bound GPU attestation" in issue for issue in issues)


def test_verify_attested_lm_eval_and_triton_entries(tmp_path, monkeypatch):
    lm_payload = {
        "results": {
            "humaneval": {"pass@1,none": 0.75},
        }
    }
    triton_report = {
        "summary": {"avg_composite": 0.5, "exec_pass_rate": 0.0, "avg_correctness": 0.0, "syntax_pass_rate": 1.0},
        "details": [
            {"level": 1, "composite_score": 0.5},
            {"level": "bugfix", "composite_score": 0.5},
        ],
    }
    document = build_attested_samples_document(
        {
            "humaneval": build_lm_eval_entry("humaneval", lm_payload, 0.75),
            "triton": build_triton_entry(triton_report),
        }
    )

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / ATTESTED_SAMPLES_FILENAME).write_text(json.dumps(document, indent=2))
    (bundle / "manifest.json").write_text(json.dumps({"run_id": "r1"}))
    (bundle / "eval_scores.json").write_text(
        json.dumps({"scores": {"humaneval": 0.75, "triton": 0.5, "triton_quick": 0.5}})
    )

    from proof.bundle import claim_sha256

    digest = claim_sha256(bundle)
    attestation = _bound_attestation(bundle, digest, monkeypatch)

    verified, issues = verify_attested_eval_samples(
        bundle,
        {"humaneval": 0.75, "triton": 0.5, "triton_quick": 0.5},
        None,
        attestation,
        claim_binding=_claim_binding,
        tdx_binding=_tdx_binding,
    )
    assert issues == []
    assert verified == {"humaneval", "triton"}


def test_read_attested_samples_wraps_legacy_gsm8k_file(tmp_path):
    sample = build_regression_sample(
        [
            {"problem_id": int(row["problem_id"]), "model_response": f"#### {row['answer'].split('####')[-1].strip()}"}
            for row in load_regression_problems()
        ]
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "gsm8k_regression_sample.json").write_text(json.dumps(sample, indent=2))

    wrapped = read_attested_samples(bundle)
    assert wrapped is not None
    assert "gsm8k" in wrapped["benchmarks"]


def _triton_report(avg_composite, details_composites):
    return {
        "summary": {
            "avg_composite": avg_composite,
            "exec_pass_rate": 0.0,
            "avg_correctness": 0.0,
            "syntax_pass_rate": 1.0,
            "avg_api_modernity": 0.0,
            "avg_perf_awareness": 0.0,
            "avg_gen_time_s": 0.0,
        },
        "details": [{"level": lvl, "composite_score": c} for lvl, c in details_composites],
    }


def test_verify_tritonbench_report_rejects_forged_full_composite():
    # Inflated summary.avg_composite (the tier value) with honest quick details must fail (#241).
    from eval.attested_samples import verify_tritonbench_report

    entry = build_triton_entry(_triton_report(0.90, [(1, 0.44), ("bugfix", 0.44)]))
    score, issues = verify_tritonbench_report(
        entry, claimed={"triton": 0.90, "triton_quick": 0.44}, frontier={"triton": 0.428}
    )
    assert score is None
    assert issues


def test_verify_tritonbench_report_accepts_honest_report():
    import pytest

    from eval.attested_samples import verify_tritonbench_report

    entry = build_triton_entry(_triton_report(0.44, [(1, 0.44), ("bugfix", 0.44)]))
    score, issues = verify_tritonbench_report(
        entry, claimed={"triton": 0.44, "triton_quick": 0.44}, frontier={"triton": 0.40}
    )
    assert issues == []
    assert score == pytest.approx(0.44)


def test_verify_tritonbench_report_rejects_report_without_details():
    from eval.attested_samples import verify_tritonbench_report

    score, issues = verify_tritonbench_report(
        {"report": {"summary": {"avg_composite": 0.44}}, "scores": {}}, claimed={"triton": 0.44}, frontier=None
    )
    assert score is None
    assert any("no per-problem details" in i for i in issues)
