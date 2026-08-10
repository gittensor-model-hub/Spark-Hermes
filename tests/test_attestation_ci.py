"""CI-level GPU + TDX attestation integrity checks (no GPU required).

Covers the three fail-closed signals training CI uses for proof-only bundles:
1. GPU NRAS token vs NVIDIA JWKS (`check_gpu_signature` / `verify_gpu_token`)
2. TDX REPORTDATA binding to claim_sha256 (`check_tdx_binding`)
3. TDX quote DCAP/PCS verification (`check_tdx_signature` / `verify_tdx_quote`)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

jwt = pytest.importorskip("jwt")


def _es384_token_fixture(
    *,
    eat_nonce: str | None = None,
    device_eat_nonce: str | None = None,
    hwmodel: str = "GH100 A01 GSP BROM",
):
    from cryptography.hazmat.primitives.asymmetric import ec

    key = ec.generate_private_key(ec.SECP384R1())
    encode = lambda payload: jwt.encode(  # noqa: E731
        payload, key, algorithm="ES384", headers={"kid": "nv-eat-kid-ci-test"}
    )
    platform_payload: dict = {"iss": "https://nras.attestation.nvidia.com", "sub": "platform"}
    if eat_nonce is not None:
        platform_payload["eat_nonce"] = eat_nonce
    device_payload: dict = {
        "iss": "https://nras.attestation.nvidia.com",
        "hwmodel": hwmodel,
    }
    if device_eat_nonce is not None:
        device_payload["eat_nonce"] = device_eat_nonce
    token = json.dumps(
        [
            ["JWT", jwt.encode({"sub": "overall"}, "k", algorithm="HS256")],
            {
                "REMOTE_GPU_CLAIMS": [
                    ["JWT", encode(platform_payload)],
                    {"GPU-0": encode(device_payload)},
                ]
            },
        ]
    )
    return key, token


def _patch_jwks(monkeypatch, key):
    class FakeKey:
        def __init__(self, k):
            self.key = k.public_key()

    class FakeJWKClient:
        def __init__(self, url):
            pass

        def get_signing_key_from_jwt(self, encoded):
            return FakeKey(key)

    monkeypatch.setattr(jwt, "PyJWKClient", FakeJWKClient)


def _bundle(tmp_path: Path, scores: dict | None = None) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"run_id": "ci-attest-1"}), encoding="utf-8")
    (bundle / "eval_scores.json").write_text(
        json.dumps({"scores": scores or {"gsm8k": 0.6, "triton": 0.5}}),
        encoding="utf-8",
    )
    return bundle


# --- 1. GPU attestation (JWKS) -------------------------------------------------


def test_ci_gpu_attestation_jwks_accepts_valid_token(monkeypatch):
    from eval.verify import check_gpu_signature

    key, token = _es384_token_fixture()
    _patch_jwks(monkeypatch, key)
    result = check_gpu_signature({"passed": True, "token": token})
    assert result is not None
    assert result["verified"] is True
    assert result["tokens_checked"] == 2


def test_ci_gpu_attestation_jwks_rejects_forged_token(monkeypatch):
    from cryptography.hazmat.primitives.asymmetric import ec

    from eval.verify import check_gpu_signature

    _, token = _es384_token_fixture()
    other = ec.generate_private_key(ec.SECP384R1())
    _patch_jwks(monkeypatch, other)
    result = check_gpu_signature({"passed": True, "token": token})
    assert result is not None
    assert result["verified"] is False


def test_ci_gpu_attestation_missing_token_is_none():
    from eval.verify import check_gpu_signature

    assert check_gpu_signature({"passed": True}) is None
    assert check_gpu_signature({"passed": True, "token": ""}) is None


# --- 2a. TDX binding (REPORTDATA ↔ claim_sha256) ------------------------------


def _fake_tdx_quote_b64(nonce_or_digest: str) -> str:
    """Minimal TDX-shaped quote embedding REPORTDATA at the v4 offset."""
    import base64

    from eval.attestation import _TDX_REPORT_DATA_OFFSET, tdx_report_data

    quote = b"\x00" * _TDX_REPORT_DATA_OFFSET + tdx_report_data(nonce_or_digest) + b"\x00" * 128
    return base64.b64encode(quote).decode()


def test_ci_tdx_binding_accepts_matching_report_data(tmp_path):
    from eval.attestation import tdx_report_data
    from eval.verify import check_tdx_binding
    from proof.bundle import claim_sha256

    bundle = _bundle(tmp_path)
    digest = claim_sha256(bundle)
    quote_b64 = _fake_tdx_quote_b64(digest)
    att = {
        "passed": True,
        "tdx": {"report_data": tdx_report_data(digest).hex(), "quote_b64": quote_b64},
    }
    assert check_tdx_binding(bundle, att) is True


def test_ci_tdx_binding_rejects_mismatched_report_data(tmp_path):
    from eval.verify import check_tdx_binding

    bundle = _bundle(tmp_path)
    att = {"passed": True, "tdx": {"report_data": "00" * 64, "quote_b64": _fake_tdx_quote_b64("11" * 32)}}
    assert check_tdx_binding(bundle, att) is False


def test_ci_tdx_binding_rejects_forged_json_with_genuine_quote(tmp_path):
    """Quote bound to digest A + JSON report_data for digest B must fail."""
    from eval.attestation import tdx_report_data
    from eval.verify import check_tdx_binding
    from proof.bundle import claim_sha256

    bundle = _bundle(tmp_path)
    digest = claim_sha256(bundle)
    other = "ab" * 32
    att = {
        "passed": True,
        "tdx": {
            "quote_b64": _fake_tdx_quote_b64(other),
            "report_data": tdx_report_data(digest).hex(),
        },
    }
    assert check_tdx_binding(bundle, att) is False


def test_ci_tdx_binding_rejects_json_sidecar_mismatch(tmp_path):
    from eval.attestation import tdx_report_data
    from eval.verify import check_tdx_binding
    from proof.bundle import claim_sha256

    bundle = _bundle(tmp_path)
    digest = claim_sha256(bundle)
    att = {
        "passed": True,
        "tdx": {
            "quote_b64": _fake_tdx_quote_b64(digest),
            "report_data": tdx_report_data("cd" * 32).hex(),
        },
    }
    assert check_tdx_binding(bundle, att) is False


# --- 2b. TDX signature (DCAP / Intel PCS) -------------------------------------


def test_ci_tdx_signature_accepts_up_to_date_quote(monkeypatch):
    import sys
    import types

    from eval.verify import check_tdx_signature

    class Report:
        status = "UpToDate"
        advisory_ids: list[str] = []

    fake = types.ModuleType("dcap_qvl")

    async def ok(quote, pccs_url=None):
        return Report()

    fake.get_collateral_and_verify = ok
    monkeypatch.setitem(sys.modules, "dcap_qvl", fake)

    result = check_tdx_signature({"passed": True, "tdx": {"quote_b64": "AAAA", "report_data": "ab" * 32}})
    assert result is not None
    assert result["verified"] is True
    assert result["status"] == "UpToDate"


def test_ci_tdx_signature_rejects_stale_quote(monkeypatch):
    import sys
    import types

    from eval.verify import check_tdx_signature

    class Report:
        status = "OutOfDate"
        advisory_ids = ["INTEL-SA-00837"]

    fake = types.ModuleType("dcap_qvl")

    async def stale(quote, pccs_url=None):
        return Report()

    fake.get_collateral_and_verify = stale
    monkeypatch.setitem(sys.modules, "dcap_qvl", fake)

    result = check_tdx_signature({"passed": True, "tdx": {"quote_b64": "AAAA", "report_data": "ab" * 32}})
    assert result is not None
    assert result["verified"] is False
    assert result["status"] == "OutOfDate"


# --- Integrity gate (GPU + both TDX checks) -----------------------------------


def test_check_attestation_integrity_requires_gpu_jwks_and_claim_binding(tmp_path, monkeypatch):
    from eval.verify import check_attestation_integrity
    from proof.bundle import claim_sha256

    bundle = _bundle(tmp_path)
    digest = claim_sha256(bundle)

    # Forged passed:true with no token.
    assert any("JWKS" in i or "token" in i for i in check_attestation_integrity(bundle, {"passed": True}))

    # Valid JWKS token but eat_nonce only in editable JSON claims — must NOT bind.
    key, unbound_token = _es384_token_fixture()
    _patch_jwks(monkeypatch, key)
    assert any(
        "eat_nonce" in i
        for i in check_attestation_integrity(
            bundle,
            {"passed": True, "token": unbound_token, "claims": {"eat_nonce": digest}},
        )
    )

    # Signed device JWT carries eat_nonce == claim_sha256 → bind.
    key2, bound_token = _es384_token_fixture(device_eat_nonce=digest)
    _patch_jwks(monkeypatch, key2)
    ok = check_attestation_integrity(
        bundle,
        {
            "passed": True,
            "token": bound_token,
            # Misleading JSON must be ignored when signed JWT binds correctly.
            "claims": {"eat_nonce": "deadbeef"},
        },
    )
    assert ok == []


def test_check_attestation_integrity_rejects_json_rebinding_of_stolen_token(tmp_path, monkeypatch):
    """Any valid NRAS token + edited JSON claims must not rebind to another bundle."""
    from eval.verify import check_attestation_integrity
    from proof.bundle import claim_sha256

    bundle = _bundle(tmp_path)
    digest = claim_sha256(bundle)
    # Token was issued for a different nonce (stolen/replayed).
    key, stolen = _es384_token_fixture(device_eat_nonce="ab" * 32)
    _patch_jwks(monkeypatch, key)
    issues = check_attestation_integrity(
        bundle,
        {"passed": True, "token": stolen, "claims": {"devices": {"GPU-0": {"eat_nonce": digest}}}},
    )
    assert any("eat_nonce" in i for i in issues)


def test_check_attestation_integrity_requires_both_tdx_checks_when_quote_present(tmp_path, monkeypatch):
    import sys
    import types

    from eval.attestation import tdx_report_data
    from eval.verify import check_attestation_integrity
    from proof.bundle import claim_sha256

    bundle = _bundle(tmp_path)
    digest = claim_sha256(bundle)
    key, token = _es384_token_fixture(device_eat_nonce=digest)
    _patch_jwks(monkeypatch, key)

    class Report:
        status = "UpToDate"
        advisory_ids: list[str] = []

    fake = types.ModuleType("dcap_qvl")

    async def ok(quote, pccs_url=None):
        return Report()

    fake.get_collateral_and_verify = ok
    monkeypatch.setitem(sys.modules, "dcap_qvl", fake)

    # Wrong REPORTDATA → fail TDX binding even if DCAP would pass.
    bad_bind = check_attestation_integrity(
        bundle,
        {
            "passed": True,
            "token": token,
            "tdx": {"report_data": "11" * 64, "quote_b64": _fake_tdx_quote_b64(digest)},
        },
    )
    assert any("REPORTDATA" in i for i in bad_bind)

    good = check_attestation_integrity(
        bundle,
        {
            "passed": True,
            "token": token,
            "tdx": {
                "report_data": tdx_report_data(digest).hex(),
                "quote_b64": _fake_tdx_quote_b64(digest),
            },
        },
    )
    assert good == []


def test_check_attestation_integrity_require_tdx_for_attested_samples_path(tmp_path, monkeypatch):
    from eval.verify import check_attestation_integrity
    from proof.bundle import claim_sha256

    bundle = _bundle(tmp_path)
    digest = claim_sha256(bundle)
    key, token = _es384_token_fixture(eat_nonce=digest)
    _patch_jwks(monkeypatch, key)
    att = {"passed": True, "token": token}
    assert check_attestation_integrity(bundle, att, require_tdx=False) == []
    assert any("TDX quote is required" in i for i in check_attestation_integrity(bundle, att, require_tdx=True))


def test_verify_submission_rejects_hwmodel_spoofed_via_claims_sidecar(tmp_path, monkeypatch):
    """A genuinely attested Hopper run must not claim train_gpu on the Blackwell frontier.

    Everything here is real except `attestation["claims"]`: the NRAS token is
    JWKS-valid and its signed device JWT reports hwmodel=GH100, while the unsigned
    sidecar the miner commits in `runs/<run-id>/attestation.json` says B300. Trusting
    the sidecar tiered the run against the wrong (speed-derived) frontier bucket.
    """
    from eval.canonical_dataset import canonical_hf_url
    from eval.verify import verify_submission
    from proof.bundle import claim_sha256

    bundle = _bundle(tmp_path)
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "spoof-1",
                "dataset_url": canonical_hf_url(),
                "train_hours": 3.0,
                "train_gpu": "NVIDIA B300",
            }
        ),
        encoding="utf-8",
    )
    digest = claim_sha256(bundle)
    key, token = _es384_token_fixture(device_eat_nonce=digest, hwmodel="GH100 A01 GSP BROM")
    _patch_jwks(monkeypatch, key)

    report = verify_submission(
        bundle,
        frontier={"gsm8k": 0.5, "triton": 0.4},
        attestation={
            "passed": True,
            "token": token,
            "claims": {"devices": {"GPU-0": {"hwmodel": "B300 A01 GSP BROM"}}},
        },
    )
    assert report["verified"] is False
    assert report["reason"] == "training_claims_failed"
    assert report["label"] == "eval:REJECT"
    assert any("corroborate" in i for i in report["issues"])


def test_verify_submission_accepts_signed_hwmodel_matching_train_gpu(tmp_path, monkeypatch):
    """The honest Hopper path still passes the training-claims gate."""
    from eval.canonical_dataset import canonical_hf_url
    from eval.verify import check_training_claims
    from proof.bundle import claim_sha256

    bundle = _bundle(tmp_path)
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "honest-1",
                "dataset_url": canonical_hf_url(),
                "train_hours": 3.0,
                "train_gpu": "NVIDIA H200",
            }
        ),
        encoding="utf-8",
    )
    key, token = _es384_token_fixture(device_eat_nonce=claim_sha256(bundle))
    _patch_jwks(monkeypatch, key)
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert check_training_claims(manifest, {"passed": True, "token": token}) == []


def test_verify_submission_rejects_forged_passed_true_attestation(tmp_path):
    """CI must not accept {"passed": true} without NRAS/JWKS evidence."""
    from eval.canonical_dataset import canonical_hf_url
    from eval.verify import verify_submission

    bundle = _bundle(tmp_path)
    (bundle / "manifest.json").write_text(
        json.dumps({"run_id": "r1", "dataset_url": canonical_hf_url()}),
        encoding="utf-8",
    )
    report = verify_submission(
        bundle,
        frontier={"gsm8k": 0.5, "triton": 0.4},
        attestation={"passed": True},
    )
    assert report["verified"] is False
    assert report["reason"] == "attestation_integrity_failed"
    assert report["label"] == "eval:REJECT"
    assert any("token" in i or "JWKS" in i for i in report["issues"])
