import json

import pytest

from eval.attestation import _decode_overall_claims
from eval.verify import check_training_claims

jwt = pytest.importorskip("jwt")


def _token(overall: dict, devices: dict[str, dict]) -> str:
    encode = lambda payload: jwt.encode(payload, "k", algorithm="HS256")  # noqa: E731
    return json.dumps(
        [
            ["JWT", encode(overall)],
            {"REMOTE_GPU_CLAIMS": [["JWT", encode({"sub": "platform"})], {k: encode(v) for k, v in devices.items()}]},
        ]
    )


def test_decode_includes_device_hardware_claims():
    token = _token(
        {"iss": "NRAS", "x-nvidia-overall-att-result": True},
        {"GPU-0": {"hwmodel": "GB20X", "x-nvidia-gpu-driver-version": "595.71.05"}},
    )
    claims = _decode_overall_claims(token)
    assert claims["iss"] == "NRAS"
    assert claims["devices"]["GPU-0"]["hwmodel"] == "GB20X"


def test_device_claims_corroborate_training_gpu(monkeypatch):
    # The overall JWT has no hardware fields; without device submodule claims the
    # verify-side corroboration check wrongly rejected genuinely attested bundles.
    # hwmodel must come from the JWKS-verified device JWT, not the JSON sidecar.
    key, token = _es384_token_fixture()
    _patch_jwks(monkeypatch, key)
    attestation = {"passed": True, "token": token, "claims": _decode_overall_claims(token)}
    manifest = {"train_hours": 0.1, "train_gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition"}
    assert check_training_claims(manifest, attestation) == []


def test_garbage_token_decodes_to_empty():
    assert _decode_overall_claims("not json") == {}


def test_extract_report_data_from_quote():
    import base64

    from eval.attestation import _TDX_REPORT_DATA_OFFSET, extract_report_data_from_quote, tdx_report_data

    digest = "ef" * 32
    quote = b"\x00" * _TDX_REPORT_DATA_OFFSET + tdx_report_data(digest) + b"\x00" * 32
    assert extract_report_data_from_quote(base64.b64encode(quote).decode()) == tdx_report_data(digest).hex()
    assert extract_report_data_from_quote("AAAA") is None


def test_tdx_report_data_pads_digest():
    from eval.attestation import tdx_report_data

    digest = "ab" * 32
    data = tdx_report_data(digest)
    assert len(data) == 64
    assert data[:32] == bytes.fromhex(digest)
    assert data[32:] == b"\x00" * 32


def test_tdx_quote_via_provisioned_node(tmp_path):
    from eval.attestation import _TDX_REPORT_DATA_OFFSET, tdx_quote, tdx_report_data

    digest = "cd" * 32
    node = tmp_path / "report"
    node.mkdir()
    (node / "provider").write_text("tdx_guest\n")
    # Emulate the kernel: outblob holds a quote embedding the report data at the
    # v4 offset (in reality it is regenerated on every inblob write).
    fake_quote = b"\x00" * _TDX_REPORT_DATA_OFFSET + tdx_report_data(digest) + b"\x00" * 128
    (node / "outblob").write_bytes(fake_quote)

    result = tdx_quote(digest, report_path=node)
    assert result is not None
    assert result["provider"] == "tdx_guest"
    assert result["report_data"] == tdx_report_data(digest).hex()
    assert (node / "inblob").read_bytes() == tdx_report_data(digest)


def test_tdx_quote_absent_on_non_tdx_host(tmp_path):
    from eval.attestation import tdx_quote

    # mkdir fails inside a nonexistent parent -> None, never raises.
    assert tdx_quote("ab" * 32, report_path=tmp_path / "no" / "tsm" / "node") is None


def test_verify_tdx_quote_reports_missing_library(monkeypatch):
    import builtins
    import sys

    from eval.attestation import verify_tdx_quote

    monkeypatch.setitem(sys.modules, "dcap_qvl", None)
    real_import = builtins.__import__

    def no_dcap(name, *args, **kwargs):
        if name == "dcap_qvl":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "dcap_qvl")
    monkeypatch.setattr(builtins, "__import__", no_dcap)
    result = verify_tdx_quote("AAAA")
    assert result["verified"] is False
    assert "not installed" in result["status"]


def test_verify_tdx_quote_maps_status(monkeypatch):
    import sys
    import types

    from eval.attestation import verify_tdx_quote

    class Report:
        def __init__(self, status, advisories):
            self.status = status
            self.advisory_ids = advisories

    fake = types.ModuleType("dcap_qvl")

    async def fake_verify(quote, pccs_url=None):
        return Report("UpToDate", [])

    fake.get_collateral_and_verify = fake_verify
    monkeypatch.setitem(sys.modules, "dcap_qvl", fake)
    result = verify_tdx_quote("AAAA")
    assert result == {"verified": True, "status": "UpToDate", "advisory_ids": []}

    async def stale_verify(quote, pccs_url=None):
        return Report("OutOfDate", ["INTEL-SA-00837"])

    fake.get_collateral_and_verify = stale_verify
    result = verify_tdx_quote("AAAA")
    assert result["verified"] is False
    assert result["status"] == "OutOfDate"
    assert result["advisory_ids"] == ["INTEL-SA-00837"]


def _es384_token_fixture(*, device_eat_nonce: str | None = None):
    from cryptography.hazmat.primitives.asymmetric import ec

    key = ec.generate_private_key(ec.SECP384R1())
    encode = lambda payload: jwt.encode(  # noqa: E731
        payload, key, algorithm="ES384", headers={"kid": "nv-eat-kid-test"}
    )
    device = {"iss": "https://nras.attestation.nvidia.com", "hwmodel": "GB20X"}
    if device_eat_nonce is not None:
        device["eat_nonce"] = device_eat_nonce
    token = json.dumps(
        [
            ["JWT", jwt.encode({"sub": "overall"}, "k", algorithm="HS256")],
            {
                "REMOTE_GPU_CLAIMS": [
                    ["JWT", encode({"iss": "https://nras.attestation.nvidia.com", "sub": "platform"})],
                    {"GPU-0": encode(device)},
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


def test_verify_gpu_token_accepts_valid_signatures(monkeypatch):
    from eval.attestation import verify_gpu_token

    key, token = _es384_token_fixture()
    _patch_jwks(monkeypatch, key)
    result = verify_gpu_token(token)
    assert result["verified"] is True
    assert result["tokens_checked"] == 2
    assert result["issues"] == []
    assert result["claims"]["devices"]["GPU-0"]["hwmodel"] == "GB20X"


def test_verify_gpu_token_expected_nonce_checks_signed_eat_nonce(monkeypatch):
    from eval.attestation import verify_gpu_token

    key, token = _es384_token_fixture(device_eat_nonce="aa" * 32)

    class FakeKey:
        def __init__(self, k):
            self.key = k.public_key()

    class FakeJWKClient:
        def __init__(self, url):
            pass

        def get_signing_key_from_jwt(self, encoded):
            return FakeKey(key)

    monkeypatch.setattr(jwt, "PyJWKClient", FakeJWKClient)
    assert verify_gpu_token(token, expected_nonce="aa" * 32)["verified"] is True
    bad = verify_gpu_token(token, expected_nonce="bb" * 32)
    assert bad["verified"] is False
    assert any("eat_nonce" in i for i in bad["issues"])


def test_verify_gpu_token_rejects_wrong_key(monkeypatch):
    from cryptography.hazmat.primitives.asymmetric import ec

    from eval.attestation import verify_gpu_token

    _, token = _es384_token_fixture()
    other = ec.generate_private_key(ec.SECP384R1())

    class FakeKey:
        key = other.public_key()

    class FakeJWKClient:
        def __init__(self, url):
            pass

        def get_signing_key_from_jwt(self, encoded):
            return FakeKey()

    monkeypatch.setattr(jwt, "PyJWKClient", FakeJWKClient)
    result = verify_gpu_token(token)
    assert result["verified"] is False
    assert result["tokens_checked"] == 0
    assert len(result["issues"]) == 2


def test_verify_gpu_token_garbage_is_unverified():
    from eval.attestation import verify_gpu_token

    result = verify_gpu_token("not json")
    assert result["verified"] is False


# --- MRTD: which guest image ran, as distinct from whether the quote is genuine ----------


def _quote_with_mrtd(mrtd_hex: str) -> str:
    """A quote long enough to slice, carrying `mrtd_hex` at the MRTD offset."""
    import base64

    from eval.attestation import _TDX_MRTD_OFFSET, _TDX_REPORT_DATA_OFFSET

    quote = bytearray(_TDX_REPORT_DATA_OFFSET + 64)
    quote[_TDX_MRTD_OFFSET : _TDX_MRTD_OFFSET + 48] = bytes.fromhex(mrtd_hex)
    return base64.b64encode(bytes(quote)).decode()


def test_mrtd_is_sliced_from_the_quote_not_read_from_the_sidecar():
    """The `mrtd` field in a bundle is written by the submitter; a measurement read from
    the document being checked establishes nothing."""
    from eval.attestation import extract_mrtd_from_quote

    assert extract_mrtd_from_quote(_quote_with_mrtd("ab" * 48)) == "ab" * 48


def test_a_short_quote_yields_no_mrtd():
    import base64

    from eval.attestation import extract_mrtd_from_quote

    assert extract_mrtd_from_quote(base64.b64encode(b"tiny").decode()) is None
    assert extract_mrtd_from_quote("not base64 !!!") is None


def test_an_unpinned_allowlist_is_reported_not_passed():
    """A genuine, correctly-bound quote from an arbitrary guest image passes tdx_bound and
    tdx_signature. Reporting the gap is the point."""
    from eval.verify import check_tdx_measurement

    att = {"tdx": {"quote_b64": _quote_with_mrtd("cd" * 48)}}
    verdict, reason = check_tdx_measurement(att)
    assert verdict is None
    assert "not checked" in reason
    assert "cd" in reason


def test_an_approved_measurement_passes():
    from eval.verify import check_tdx_measurement

    att = {"tdx": {"quote_b64": _quote_with_mrtd("cd" * 48)}}
    assert check_tdx_measurement(att, allowed=["CD" * 48]) == (True, "")


def test_an_unapproved_measurement_is_refused():
    from eval.verify import check_tdx_measurement

    att = {"tdx": {"quote_b64": _quote_with_mrtd("cd" * 48)}}
    verdict, reason = check_tdx_measurement(att, allowed=["ab" * 48])
    assert verdict is False
    assert "not in the approved set" in reason


def test_no_tdx_quote_is_neither_pass_nor_fail():
    from eval.verify import check_tdx_measurement

    assert check_tdx_measurement(None)[0] is None
    assert check_tdx_measurement({})[0] is None
