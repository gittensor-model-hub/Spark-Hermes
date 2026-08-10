"""Tests for eval.training_gpus."""

from eval.training_gpus import attestation_corroborates_training_gpu, is_accepted_training_gpu


def test_accepted_training_gpus():
    assert is_accepted_training_gpu("NVIDIA RTX PRO 6000 Blackwell Server Edition")
    assert is_accepted_training_gpu("NVIDIA B200")
    assert is_accepted_training_gpu("NVIDIA B300")
    assert is_accepted_training_gpu("NVIDIA H100 SXM")
    assert is_accepted_training_gpu("NVIDIA H200")
    assert not is_accepted_training_gpu("NVIDIA A100")


def test_attestation_corroboration_by_family():
    assert attestation_corroborates_training_gpu(
        "NVIDIA H100",
        {"passed": True, "claims": {"hwmodel": "NVIDIA GH100"}},
    )
    assert not attestation_corroborates_training_gpu(
        "NVIDIA H100",
        {"passed": True, "claims": {"hwmodel": "GB202 RTX PRO 6000"}},
    )


def _attestation(hwmodel: str, **claims) -> dict:
    """Attestation shaped like a real NRAS decode: hwmodel on the device submodule."""
    return {"passed": True, "claims": {**claims, "devices": {"GPU-0": {"hwmodel": hwmodel}}}}


def test_attestation_corroboration_reads_per_device_hwmodel():
    assert attestation_corroborates_training_gpu("NVIDIA B200", _attestation("B200 A01 GSP BROM"))
    assert attestation_corroborates_training_gpu(
        "NVIDIA RTX PRO 6000 Blackwell Server Edition", _attestation("GB202 A01 GSP BROM")
    )
    assert not attestation_corroborates_training_gpu("NVIDIA B200", _attestation("GH100 A01 GSP BROM"))


def test_h200_corroborates_against_the_gh100_die():
    # A genuine H200 attests as the GH100 die it shares with H100 (#120).
    assert attestation_corroborates_training_gpu("NVIDIA H200", _attestation("GH100 A01 GSP BROM"))


def test_hex_claim_fields_are_not_hardware_evidence():
    # 'b200' is itself valid hex, so it occurs by chance in sha256-shaped fields.
    # The nonce is the miner's own claim_sha256 and is grindable, so a Hopper node
    # must not corroborate a Blackwell claim just by carrying a matching nonce.
    nonce = "87acb1e183a2b0f74c3b2008b8ef6975a95269bc490a8886f317fa4bd714b085"
    assert "b200" in nonce
    assert not attestation_corroborates_training_gpu("NVIDIA B200", _attestation("GH100 A01 GSP BROM", eat_nonce=nonce))


def test_claims_without_hwmodel_do_not_corroborate():
    assert not attestation_corroborates_training_gpu(
        "NVIDIA B200",
        {"passed": True, "claims": {"eat_nonce": "deadbeef"}},
    )
    assert not attestation_corroborates_training_gpu(
        "NVIDIA B200",
        {"passed": True, "claims": {"eat_nonce": "deadbeefb200cafe"}},
    )


def test_absent_attestation_or_claims_still_skips_corroboration():
    # Attestation is optional; absence is handled by the caller, not here.
    assert attestation_corroborates_training_gpu("NVIDIA B200", None)
    assert attestation_corroborates_training_gpu("NVIDIA B200", {"passed": True, "claims": {}})


def test_signed_claims_override_the_editable_sidecar():
    # attestation["claims"] is committed by the miner and can say anything; the
    # JWKS-verified NRAS claims are what must decide.
    spoofed = _attestation("B200 A01 GSP BROM")
    hopper = {"devices": {"GPU-0": {"hwmodel": "GH100 A01 GSP BROM"}}}
    assert not attestation_corroborates_training_gpu("NVIDIA B200", spoofed, signed_claims=hopper)
    assert attestation_corroborates_training_gpu("NVIDIA H100", spoofed, signed_claims=hopper)


def test_signed_claims_without_hwmodel_are_fail_closed():
    # Unlike the legacy sidecar path, empty signed claims corroborate nothing —
    # otherwise deleting attestation["claims"] would still buy a free pass.
    spoofed = _attestation("B200 A01 GSP BROM")
    assert not attestation_corroborates_training_gpu("NVIDIA B200", spoofed, signed_claims={})
    assert not attestation_corroborates_training_gpu(
        "NVIDIA B200", spoofed, signed_claims={"eat_nonce": "deadbeefb200cafe"}
    )
