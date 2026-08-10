"""GPU confidential-computing attestation for proof-of-training submissions.

Wraps NVIDIA's `nv-attestation-sdk` (https://github.com/NVIDIA/nvtrust) so a miner's
proof-of-training claim can be backed by real evidence that the GPU it ran on was
inside a Confidential Computing (CC) trust boundary — e.g. a Blackwell RTX PRO 6000
Server Edition CC node — rather than just an unverifiable checkpoint + score claim.

Deprecation note: `nv-attestation-sdk` is EOL 2026-09-15. Its replacement
(https://github.com/NVIDIA/attestation-sdk) is C++/Rust-only today with no Python
bindings, so this remains the only real, currently-installable Python attestation
path. Install with `uv sync --extra proof` (needs a real NVIDIA CC-capable GPU +
driver stack to actually attest) — this module stays importable without it since the
SDK is only imported inside `attest_gpu`, not at module load time.

    python -m eval.attestation --out runs/<run-id>/attestation.json

Default remote-attestation endpoints below are NVIDIA's own public NRAS/RIM/OCSP
services, taken from `nv-attestation-sdk`'s own test fixtures (not guessed).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_POLICY_PATH = Path(__file__).parent / "policies" / "gpu_remote_v3.json"

DEFAULT_NRAS_GPU_URL = "https://nras.attestation.nvidia.com/v3/attest/gpu"
DEFAULT_RIM_URL = "https://rim.attestation.nvidia.com/v1/rim/"
DEFAULT_OCSP_URL = "https://ocsp.ndis.nvidia.com/"
DEFAULT_NRAS_ISSUER = "https://nras.attestation.nvidia.com"
DEFAULT_NRAS_JWKS_URL = "https://nras.attestation.nvidia.com/.well-known/jwks.json"


@dataclass(frozen=True)
class AttestationResult:
    passed: bool
    environment: str
    token: str
    claims: dict


def attest_gpu(
    environment: str = "REMOTE",
    policy_path: Path = DEFAULT_POLICY_PATH,
    service_key: str | None = None,
    nras_gpu_url: str = DEFAULT_NRAS_GPU_URL,
    rim_url: str = DEFAULT_RIM_URL,
    ocsp_url: str = DEFAULT_OCSP_URL,
    nonce: str | None = None,
) -> AttestationResult:
    """Collect local GPU evidence and verify it (by default against NVIDIA's Remote
    Attestation Service — NRAS) against the appraisal policy at `policy_path`.

    `nonce` (a hex string) binds the attestation to specific content — pass a proof
    bundle's `claim_sha256` so the NRAS-signed EAT commits that exact claim to this
    GPU (`eat_nonce` in the token; `eval.verify` recomputes and compares it).

    Requires `nv-attestation-sdk` (`uv sync --extra proof`) and, for `environment="REMOTE"`,
    a GPU SKU that supports Confidential Computing (Hopper H100+, including the Blackwell
    RTX PRO 6000 Server Edition CC node this project targets) with a CC-enabled driver.
    """
    from nv_attestation_sdk import attestation as nv_attestation

    env = getattr(nv_attestation.Environment, environment.upper())

    client = nv_attestation.Attestation()
    client.set_name("sparkdistill-proof")
    if nonce is not None:
        client.set_nonce(nonce)
    if service_key is not None:
        client.set_service_key(service_key)
    client.add_verifier(nv_attestation.Devices.GPU, env, nras_gpu_url, "", ocsp_url=ocsp_url, rim_url=rim_url)

    evidence_list = client.get_evidence()
    passed = bool(client.attest(evidence_list))

    token = client.get_token() if passed else ""
    policy_text = Path(policy_path).read_text()
    validated = bool(client.validate_token(policy_text)) if passed else False

    claims = _decode_overall_claims(token) if token else {}
    nv_attestation.Attestation.reset()
    return AttestationResult(passed=passed and validated, environment=environment.upper(), token=token, claims=claims)


def _decode_overall_claims(token: str) -> dict:
    """Best-effort, unverified decode of the EAT's claims for display.

    Not the trust boundary — `validate_token` above (which checks signatures against
    the appraisal policy) is what actually decides `passed`.

    Besides the overall JWT, per-device submodule tokens are decoded under a
    `devices` key: they carry the hardware identity (`hwmodel`, driver/vbios
    versions) that `eval.verify.check_training_claims` corroborates the claimed
    training GPU against — the overall JWT alone has no hardware fields.
    """
    import jwt  # PyJWT, a transitive dep of nv-attestation-sdk

    def _decode(encoded: str) -> dict:
        return jwt.decode(encoded, options={"verify_signature": False})

    try:
        parsed = json.loads(token)
        claims = _decode(parsed[0][1])
        devices: dict = {}
        for section in parsed[1:]:
            if not isinstance(section, dict):
                continue
            # e.g. {"REMOTE_GPU_CLAIMS": [["JWT", <platform>], {"GPU-0": <jwt>, ...}]}
            for entries in section.values():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    for device, device_jwt in entry.items():
                        try:
                            devices[device] = _decode(device_jwt)
                        except Exception:
                            continue
        if devices:
            claims["devices"] = devices
        return claims
    except Exception:
        return {}


# TDX quote layout (Intel TDX DCAP quote v4): 48-byte header, then the 584-byte
# TD report body whose final 64 bytes are REPORTDATA — where the claim digest goes.
_TDX_QUOTE_HEADER_LEN = 48
_TDX_BODY_LEN = 584
_TDX_REPORT_DATA_OFFSET = _TDX_QUOTE_HEADER_LEN + _TDX_BODY_LEN - 64
_TDX_MRTD_OFFSET = _TDX_QUOTE_HEADER_LEN + 16 + 48 + 48 + 8 + 8 + 8  # MRTD follows TCB/MRSEAM/MRSIGNERSEAM/attrs


def tdx_report_data(nonce_hex: str) -> bytes:
    """64-byte TDX REPORTDATA for a claim digest: sha256 bytes + zero padding."""
    digest = bytes.fromhex(nonce_hex)
    if len(digest) > 64:
        raise ValueError("nonce too long for TDX report data (max 64 bytes)")
    return digest.ljust(64, b"\x00")


def extract_report_data_from_quote(quote_b64: str) -> str | None:
    """Hex REPORTDATA sliced from a TDX quote — never trust JSON ``report_data`` alone."""
    import base64

    try:
        quote = base64.b64decode(quote_b64, validate=False)
    except Exception:
        return None
    if len(quote) < _TDX_REPORT_DATA_OFFSET + 64:
        return None
    return quote[_TDX_REPORT_DATA_OFFSET : _TDX_REPORT_DATA_OFFSET + 64].hex()


def extract_mrtd_from_quote(quote_b64: str) -> str | None:
    """Hex MRTD sliced from a TDX quote — *which guest image* the TD booted.

    Sliced from the quote bytes for the same reason REPORTDATA is: the ``mrtd`` field
    that ``tdx_quote`` writes into the bundle is a sidecar the submitter can edit, and a
    measurement read from the document being checked establishes nothing.

    REPORTDATA answers "does this quote commit to this bundle" and the DCAP chain answers
    "is this quote genuine". Neither answers "was this the image we approved" — a genuine,
    correctly-bound quote can come from any guest the operator chose to boot. That is the
    question MRTD answers, and it is the one that matters when the machine belongs to the
    party being measured.
    """
    import base64

    try:
        quote = base64.b64decode(quote_b64, validate=False)
    except Exception:
        return None
    if len(quote) < _TDX_MRTD_OFFSET + 48:
        return None
    return quote[_TDX_MRTD_OFFSET : _TDX_MRTD_OFFSET + 48].hex()


def tdx_quote(nonce_hex: str, report_path: Path | None = None) -> dict | None:
    """Capture an Intel TDX quote binding `nonce_hex` via configfs-tsm, or None.

    The VM userland (serving stack, harness) is only *declared* in a proof bundle;
    a TDX quote adds the measured-VM half: MRTD/RTMRs cover the guest image and
    kernel, signed by Intel, with the claim digest in REPORTDATA. Returns None on
    hosts without TDX or without a provisioned report node.

    The kernel's configfs-tsm report directory is root-owned; provision a
    persistent node once per boot (needs sudo) and point
    SPARKDISTILL_TSM_REPORT_PATH at it:

        sudo chmod 0777 /sys/kernel/config/tsm/report
        mkdir /sys/kernel/config/tsm/report/sparkdistill
        sudo chmod 0666 /sys/kernel/config/tsm/report/sparkdistill/inblob
        export SPARKDISTILL_TSM_REPORT_PATH=/sys/kernel/config/tsm/report/sparkdistill
    """
    import base64
    import os

    node = report_path or Path(
        os.environ.get("SPARKDISTILL_TSM_REPORT_PATH") or "/sys/kernel/config/tsm/report/sparkdistill"
    )
    created = False
    try:
        if not node.is_dir():
            node.mkdir()  # works only as root; a pre-provisioned node skips this
            created = True
        (node / "inblob").write_bytes(tdx_report_data(nonce_hex))
        quote = (node / "outblob").read_bytes()
        provider = (node / "provider").read_text().strip() if (node / "provider").exists() else ""
    except OSError:
        return None
    finally:
        if created:
            try:
                node.rmdir()
            except OSError:
                pass
    if len(quote) < _TDX_REPORT_DATA_OFFSET + 64:
        return None
    return {
        "provider": provider,
        "quote_b64": base64.b64encode(quote).decode(),
        "report_data": quote[_TDX_REPORT_DATA_OFFSET : _TDX_REPORT_DATA_OFFSET + 64].hex(),
        "mrtd": quote[_TDX_MRTD_OFFSET : _TDX_MRTD_OFFSET + 48].hex(),
    }


def verify_gpu_token(
    token: str,
    jwks_url: str = DEFAULT_NRAS_JWKS_URL,
    issuer: str = DEFAULT_NRAS_ISSUER,
    *,
    expected_nonce: str | None = None,
) -> dict:
    """Verify the NRAS-signed JWTs in a GPU attestation token against NVIDIA's JWKS.

    The EAT's platform and per-device tokens are ES384-signed by NRAS (`kid`
    resolved from NVIDIA's published JWKS); signature, issuer, and expiry are
    all enforced. The overall JWT is HS256 and SDK-local — it is intentionally
    NOT counted as evidence. Without this check a validator would be trusting
    the committed attestation JSON on the miner's word.

    When ``expected_nonce`` is set (typically ``claim_sha256(bundle)``), also
    require that a *signed* ``eat_nonce`` on the platform JWT or a per-device JWT
    matches — never the editable ``attestation["claims"]`` JSON sidecar.

    Returns ``{"verified": bool, "tokens_checked": int, "issues": [...], "claims": {...}}``
    where ``claims`` are decoded from JWKS-verified JWTs only (plus ``devices``).
    """
    import jwt

    issues: list[str] = []
    checked = 0
    platform_claims: dict = {}
    devices: dict[str, dict] = {}
    try:
        parsed = json.loads(token)
        client = jwt.PyJWKClient(jwks_url)
        for section in parsed[1:]:
            if not isinstance(section, dict):
                continue
            for entries in section.values():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    signed_tokens: list[tuple[str, str]] = []
                    if isinstance(entry, list) and len(entry) == 2 and entry[0] == "JWT":
                        signed_tokens.append(("platform", entry[1]))
                    elif isinstance(entry, dict):
                        signed_tokens.extend((str(k), str(v)) for k, v in entry.items())
                    for name, encoded in signed_tokens:
                        try:
                            key = client.get_signing_key_from_jwt(encoded)
                            decoded = jwt.decode(encoded, key.key, algorithms=["ES384"], issuer=issuer)
                            checked += 1
                            if name == "platform":
                                platform_claims = decoded
                            else:
                                devices[name] = decoded
                        except Exception as exc:
                            issues.append(f"{name}: {exc}")
    except Exception as exc:
        issues.append(f"token unparseable: {exc}")
    if checked == 0 and not issues:
        issues.append("no NRAS-signed tokens found in attestation")

    signed_claims = dict(platform_claims)
    if devices:
        signed_claims["devices"] = devices

    if expected_nonce is not None:
        expected = str(expected_nonce).lower().removeprefix("0x")
        nonces = [signed_claims.get("eat_nonce")]
        nonces += [device.get("eat_nonce") for device in devices.values()]
        if not any(str(nonce).lower().removeprefix("0x") == expected for nonce in nonces if nonce):
            issues.append(
                "signed eat_nonce does not match the expected claim_sha256-bound nonce — "
                "token was not produced for this bundle's content"
            )

    return {
        "verified": checked > 0 and not issues,
        "tokens_checked": checked,
        "issues": issues,
        "claims": signed_claims,
    }


def verify_tdx_quote(quote_b64: str, pccs_url: str | None = None) -> dict:
    """Verify a TDX quote's Intel signature chain via DCAP (dcap-qvl + Intel PCS).

    Validates the quote's ECDSA signature, PCK certificate chain up to Intel's
    root CA, QE identity, and platform TCB status against live PCS collateral.
    Returns {"verified": bool, "status": str, "advisory_ids": [...]} —
    `status "UpToDate"` with no advisories is the clean pass; stale-TCB statuses
    are surfaced for validator policy rather than silently accepted.

    Requires `dcap-qvl` (`uv sync --extra proof`) and network access to Intel
    PCS (or a PCCS mirror via `pccs_url`).
    """
    import asyncio
    import base64

    try:
        import dcap_qvl
    except ImportError:
        return {"verified": False, "status": "dcap-qvl not installed", "advisory_ids": []}

    try:
        quote = base64.b64decode(quote_b64)
        report = asyncio.run(dcap_qvl.get_collateral_and_verify(quote, pccs_url))
    except Exception as exc:
        return {"verified": False, "status": f"verification error: {exc}", "advisory_ids": []}
    return {
        "verified": report.status == "UpToDate" and not report.advisory_ids,
        "status": report.status,
        "advisory_ids": list(report.advisory_ids),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--environment", default="REMOTE", choices=["LOCAL", "REMOTE", "TEST"])
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--service-key", default=None, help="NGC service key with NRAS access (optional)")
    parser.add_argument(
        "--nonce",
        default=None,
        help="hex content-binding nonce, e.g. the claim_sha256 printed by proof.bundle",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    result = attest_gpu(
        environment=args.environment, policy_path=args.policy, service_key=args.service_key, nonce=args.nonce
    )

    # Best-effort measured-VM evidence: present on TDX guests with a provisioned
    # configfs-tsm node, null elsewhere (GPU attestation alone still decides passed).
    tdx = tdx_quote(args.nonce) if args.nonce else None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "passed": result.passed,
                "environment": result.environment,
                "token": result.token,
                "claims": result.claims,
                "tdx": tdx,
            },
            indent=2,
        )
    )
    tdx_note = "with TDX quote" if tdx else "no TDX quote"
    print(f"attestation {'PASSED' if result.passed else 'FAILED'} ({result.environment}, {tdx_note})", file=sys.stderr)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
