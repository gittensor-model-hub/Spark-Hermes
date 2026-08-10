"""Cheap verification of a submitted proof-of-training bundle.

Instead of a full retrain, a proof bundle is checked by: (1) optionally requiring a
passed GPU CC (+ TDX) attestation with fail-closed JWKS/DCAP crypto and claim
binding (CPU-only, no GPU), (2) re-checking attested eval samples on CPU when
present, else re-running each claimed benchmark on a small held-out sample against
the bundle's checkpoint and comparing to the claimed scores within a tolerance,
and only if both pass, (3) scoring the (now-trusted) claimed scores against the
frontier via `eval.score`. A mismatch beyond tolerance is treated as a fabricated
or stale claim and rejected outright — cheap verification does not re-run the full
basket, so it must not silently trust an unverified number either.

    python -m eval.verify --bundle-repo <hf-repo-id> --frontier eval/results/frontier.json \\
        [--attestation runs/<run-id>/attestation.json] --limit 50 --tolerance-pct 2.0 \\
        --out eval/results/report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

from eval.attested_samples import (
    ATTESTED_VERIFY_LIMIT,
    has_attested_samples,
    verify_attested_eval_samples,
)
from eval.benchmarks import BENCHMARKS, assert_fraction_scores
from eval.canonical_dataset import (
    canonical_hf_url,
    canonical_pref_hf_url,
    canonical_sha256_for_track,
)
from eval.dataset_verify import _sha256_file
from eval.frontiers import load_frontier_scores, training_track_of
from eval.gpu_architecture import DEFAULT_GPU_ARCHITECTURE, GpuArchitecture, normalize_gpu_architecture
from eval.harness import run_harness
from eval.hf_pin import pinned_download
from eval.mix_registry import REGISTRY_PATH, verify_mix_manifest
from eval.regression_sample import REGRESSION_BENCHMARK_KEY
from eval.score import score
from eval.training_gpus import (
    accepted_training_gpu_label,
    attestation_corroborates_training_gpu,
    is_accepted_training_gpu,
)

# Training-track budget (see docs/miner-guide.md): a proof-of-training claim must have
# been produced within this wall-clock budget on an accepted CC GPU.
MAX_TRAIN_HOURS = 5.0


def resolve_bundle_gpu_architecture(manifest: dict) -> GpuArchitecture:
    """Best-effort GPU architecture for this bundle, used to pick its frontier
    bucket (`eval.frontiers`) and pass to `eval.score` for tiering.

    Prefers an explicit `gpu_architecture` field, then the training-track
    `train_gpu` claim, else Blackwell — every bundle predating both fields was
    generated before Hopper support existed, so the legacy default is safe.
    """
    for value in (manifest.get("gpu_architecture"), manifest.get("train_gpu")):
        if value:
            arch = normalize_gpu_architecture(str(value))
            if arch is not None:
                return arch
    return DEFAULT_GPU_ARCHITECTURE


def check_training_claims(
    manifest: dict,
    attestation: dict | None,
    max_train_hours: float = MAX_TRAIN_HOURS,
    *,
    gpu_sig: dict | None = None,
) -> list[str]:
    """Validate the bundle's training-track claims (train_hours / train_gpu).

    Older bundles without these fields are not failed here — they simply don't
    qualify for the training track and fall back to full retrain-verification.

    The `train_gpu` corroboration reads hwmodel from the *JWKS-verified* NRAS
    claims, never from `attestation["claims"]` — that sidecar is miner-editable
    (same rule as `check_claim_binding` for `eat_nonce`). Pass `gpu_sig` to reuse
    an already-computed `check_gpu_signature` result.
    """
    issues: list[str] = []
    train_hours = manifest.get("train_hours")
    if train_hours is not None and float(train_hours) > max_train_hours:
        issues.append(f"train_hours {train_hours} exceeds the {max_train_hours}h budget")

    train_gpu = manifest.get("train_gpu")
    if train_gpu is not None and not is_accepted_training_gpu(str(train_gpu)):
        issues.append(f"train_gpu {train_gpu!r} is not an accepted training GPU ({accepted_training_gpu_label()})")

    if train_gpu is not None and attestation is not None:
        signed_claims = signed_attestation_claims(attestation, gpu_sig=gpu_sig)
        if signed_claims is None:
            issues.append(
                "attestation GPU token is unverified, so no signed hwmodel claim can "
                "corroborate the claimed training GPU"
            )
        elif not attestation_corroborates_training_gpu(str(train_gpu), attestation, signed_claims=signed_claims):
            issues.append("attestation claims do not corroborate the claimed training GPU")

    # `gpu_architecture` outranks `train_gpu` in resolve_bundle_gpu_architecture and picks
    # both the frontier a run is tiered against and the bucket its scores merge into, but
    # `proof.bundle` never writes it — an explicit value is hand-added. Bind it to the
    # attested `train_gpu` (verified above against the signed hwmodel) so it can't re-point
    # a run at the other architecture's frontier (issue #237).
    gpu_architecture = manifest.get("gpu_architecture")
    if gpu_architecture is not None and attestation is not None:
        claimed_arch = normalize_gpu_architecture(str(gpu_architecture))
        train_arch = normalize_gpu_architecture(str(train_gpu)) if train_gpu is not None else None
        if claimed_arch is None:
            issues.append(f"manifest gpu_architecture {gpu_architecture!r} is not recognized")
        elif train_gpu is None:
            issues.append(
                "manifest gpu_architecture is set but no train_gpu claim evidences it; "
                "the architecture cannot be corroborated against the attestation"
            )
        elif train_arch != claimed_arch:
            issues.append(
                f"manifest gpu_architecture {claimed_arch!r} does not match the attested "
                f"training GPU architecture {train_arch!r}"
            )
    return issues


def check_canonical_dataset_claim(
    manifest: dict,
    *,
    bundle_dir: Path | None = None,
    acceptable_sft_shas: set[str] | None = None,
    acceptable_pref_shas: set[str] | None = None,
) -> list[str]:
    """Training-track bundles must cite the pinned canonical dataset for their track.

    SFT bundles (the default) check the canonical mining SFT mix — behavior unchanged. A
    bundle that declares ``train_objective: "dpo"`` is checked against the canonical
    *preference* dataset pin instead. The two pins are disjoint, so an SFT bundle can never
    satisfy the DPO pin or vice-versa. A DPO bundle submitted before a canonical preference
    pin exists **fails closed** (the SFT path keeps its bootstrap fail-open-when-unpinned).
    """
    issues: list[str] = []
    is_dpo = str(manifest.get("train_objective") or "").strip().lower() == "dpo"
    track = "dpo" if is_dpo else "sft"
    dataset_url = manifest.get("dataset_url")

    if is_dpo:
        try:
            expected_url = canonical_pref_hf_url().rstrip("/")
        except (FileNotFoundError, ValueError, KeyError):
            return ["no canonical preference dataset pin is configured; the DPO track is unavailable"]
    else:
        try:
            expected_url = canonical_hf_url().rstrip("/")
        except (FileNotFoundError, ValueError):
            return issues

    if not dataset_url:
        issues.append(f"training bundle must set dataset_url to the canonical {track} dataset ({expected_url})")
        return issues

    if str(dataset_url).rstrip("/") != expected_url:
        issues.append(
            f"dataset_url must be canonical {expected_url}, got {dataset_url!r}; "
            "training-track submissions may not use private or synthetic datasets"
        )

    if bundle_dir is not None:
        mix_path = bundle_dir / "mix_manifest.json"
        if mix_path.exists():
            mix_data = json.loads(mix_path.read_text(encoding="utf-8"))
            sha_field = "pref_sha256" if is_dpo else "sft_sha256"
            remote_sha = mix_data.get(sha_field)
            allowed = acceptable_pref_shas if is_dpo else acceptable_sft_shas
            if allowed is None:
                try:
                    allowed = {canonical_sha256_for_track(track)}
                except ValueError:
                    return issues
            if remote_sha not in allowed:
                issues.append(
                    f"bundle mix_manifest.{sha_field} does not match an accepted canonical {track} pin "
                    f"(allowed {len(allowed)} pin(s) for this PR window)"
                )
    return issues


def check_mix_provenance(
    bundle_dir: Path,
    manifest: dict,
    *,
    registry_path: Path = REGISTRY_PATH,
) -> list[str]:
    """Validate a cross-miner mix copied into the proof bundle."""
    mix_path = bundle_dir / "mix_manifest.json"
    if not mix_path.exists():
        if manifest.get("mix_manifest_sha256"):
            return ["bundle manifest references mix_manifest_sha256 but mix_manifest.json is missing"]
        return []

    expected_sha = manifest.get("mix_manifest_sha256")
    if expected_sha and _sha256_file(mix_path) != expected_sha:
        return ["mix_manifest.json sha256 does not match bundle manifest"]

    report = verify_mix_manifest(mix_path, registry_path=registry_path)
    return list(report.get("issues") or [])


def check_claim(claimed: dict[str, float], rerun: dict[str, float], tolerance_pct: float = 2.0) -> list[str]:
    """Return the benchmark keys where the claimed score diverges from the cheap
    re-run by more than the tolerance (percentage points, absolute).

    A benchmark's `claim_tolerance_pct` overrides the global `tolerance_pct`
    (e.g. triton's tiny problem set drifts more across serving instances than
    sample-based benchmarks do). The `triton` re-run is level-1-only, so it is
    compared against the claim's `triton_quick` (the same problem subset) when
    present — a full-run composite covers harder levels and would mismatch an
    honest claim systematically.

    Both score maps must be fractions in [0, 1]; the `* 100.0` below turns the gap into
    percentage points, so a 0-100 percentage would make the tolerance 100x too tight.
    """
    assert_fraction_scores(claimed, "claimed (eval_scores.json)")
    assert_fraction_scores(rerun, "re-run (harness)")
    mismatches = []
    for key, rerun_value in rerun.items():
        claimed_value = claimed.get(key)
        if key == "triton" and "triton_quick" in claimed:
            claimed_value = claimed["triton_quick"]
        if claimed_value is None:
            continue
        benchmark = BENCHMARKS.get(key)
        tolerance = (
            tolerance_pct
            if benchmark is None or benchmark.claim_tolerance_pct is None
            else benchmark.claim_tolerance_pct
        )
        if abs(claimed_value - rerun_value) * 100.0 > tolerance:
            mismatches.append(key)
    return mismatches


@contextmanager
def _no_student_endpoint_env():
    """Force the re-run to serve the bundle's own checkpoint.

    SPARKDISTILL_STUDENT_ENDPOINT is a miner convenience; during verification a
    stale value would silently score whatever model that endpoint serves instead
    of the checkpoint under verification.
    """
    saved = os.environ.pop("SPARKDISTILL_STUDENT_ENDPOINT", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["SPARKDISTILL_STUDENT_ENDPOINT"] = saved


def check_claim_binding(
    bundle_dir: Path,
    attestation: dict | None,
    *,
    gpu_sig: dict | None = None,
) -> bool | None:
    """Whether a *JWKS-signed* NRAS ``eat_nonce`` commits to this exact bundle.

    Returns True when a signed platform or per-device JWT carries ``eat_nonce``
    equal to the bundle's ``claim_sha256`` (see ``proof.bundle``), False when an
    attestation is present but unbound / unsigned / mismatched, and None when
    there is no attestation.

    Critical: do **not** trust ``attestation["claims"]`` — that JSON is miner-
    editable. Binding must come from JWTs verified against NVIDIA's JWKS (the
    same rule SparkProof enforces in ``verify_nras_token(..., expected_nonce=)``).
    NRAS v3 often places the nonce on per-device submodule tokens rather than
    the platform JWT.
    """
    if attestation is None:
        return None
    token = attestation.get("token")
    if not token:
        return False

    from proof.bundle import claim_sha256

    expected = claim_sha256(bundle_dir)
    if gpu_sig is None:
        from eval.attestation import verify_gpu_token

        gpu_sig = verify_gpu_token(token)
    if not gpu_sig.get("verified"):
        return False

    claims = gpu_sig.get("claims") or {}
    nonces = [claims.get("eat_nonce")]
    nonces += [device.get("eat_nonce") for device in (claims.get("devices") or {}).values()]
    return any(str(nonce).lower().removeprefix("0x") == expected for nonce in nonces if nonce)


def check_tdx_binding(bundle_dir: Path, attestation: dict | None) -> bool | None:
    """Whether the attestation's Intel TDX *quote bytes* commit to this bundle.

    Extracts 64-byte REPORTDATA from ``tdx.quote_b64`` at the TDX v4 offset and
    requires it equal ``tdx_report_data(claim_sha256)``. Does **not** trust the
    miner-editable ``tdx.report_data`` JSON field alone (a genuine quote could
    otherwise be rebound by forging that sidecar). When JSON ``report_data`` is
    present it must also match the quote slice.

    Returns None when no TDX blob was captured; GPU binding remains the minimum bar.
    Quote signature (DCAP/PCS) is checked separately by ``check_tdx_signature``.
    """
    if attestation is None or not attestation.get("tdx"):
        return None
    from eval.attestation import extract_report_data_from_quote, tdx_report_data
    from proof.bundle import claim_sha256

    tdx = attestation["tdx"]
    quote_b64 = tdx.get("quote_b64") or ""
    if not quote_b64:
        return False
    quote_report_data = extract_report_data_from_quote(quote_b64)
    if quote_report_data is None:
        return False
    expected = tdx_report_data(claim_sha256(bundle_dir)).hex()
    if quote_report_data.lower() != expected:
        return False
    json_report_data = str(tdx.get("report_data") or "").lower()
    if json_report_data and json_report_data != quote_report_data.lower():
        return False
    return True


def check_gpu_signature(attestation: dict | None) -> dict | None:
    """Verify the attestation's NRAS-signed GPU tokens against NVIDIA's JWKS.

    The GPU counterpart of `check_tdx_signature`: without it, the committed
    attestation JSON's `passed` flag and claims are taken on the miner's word.
    Returns None when there is no attestation.
    """
    if attestation is None or not attestation.get("token"):
        return None
    from eval.attestation import verify_gpu_token

    return verify_gpu_token(attestation["token"])


def signed_attestation_claims(attestation: dict | None, *, gpu_sig: dict | None = None) -> dict | None:
    """Claims decoded from JWKS-verified NRAS tokens, or None when unverifiable.

    The trusted counterpart of the miner-editable `attestation["claims"]` sidecar:
    anything that decides a reward (hwmodel, eat_nonce) must come from here.
    """
    if gpu_sig is None:
        gpu_sig = check_gpu_signature(attestation)
    if not gpu_sig or not gpu_sig.get("verified"):
        return None
    return gpu_sig.get("claims") or {}


def check_tdx_signature(attestation: dict | None, pccs_url: str | None = None) -> dict | None:
    """DCAP-verify the attestation's TDX quote against Intel PCS.

    Complements `check_tdx_binding` (which proves the quote commits to this
    bundle): this proves the quote itself is genuine — ECDSA signature, PCK
    chain to Intel's root CA, QE identity, and TCB status. Without it, a
    fabricated `tdx` blob with a matching report_data would pass binding.
    Returns None when no TDX quote is present.
    """
    if attestation is None or not attestation.get("tdx"):
        return None
    from eval.attestation import verify_tdx_quote

    return verify_tdx_quote(attestation["tdx"].get("quote_b64") or "", pccs_url)


def check_tdx_measurement(attestation: dict | None, allowed: Sequence[str] = ()) -> tuple[bool | None, str]:
    """Whether the TD booted a guest image we approved. Returns `(verdict, reason)`.

    The third and missing leg of TDX verification. `check_tdx_binding` proves the quote
    commits to this bundle; `check_tdx_signature` proves the quote is genuine. Neither
    asks **which guest ran** -- so a genuine, correctly-bound quote from an arbitrary
    image the operator chose to boot passes both. On hardware the submitter owns, that is
    the whole question.

    `None` means unpinned, and unpinned is reported rather than passed. There is no
    allowlist in this repository yet: the expected MRTD for a reproducible guest image has
    to be obtained by building that image, not by reading it off a submission. Until one
    exists, a TDX quote here establishes "some TD, on genuine Intel hardware, committing
    to this bundle" -- which is worth something and is not the same claim as a measured VM.

    Deliberately not raising. A caller that needs the stronger property must treat `None`
    as failure; `eval.verify`'s report carries the verdict and the reason so a reviewer
    sees the gap instead of an absent field.
    """
    if attestation is None or not attestation.get("tdx"):
        return None, "no TDX quote captured"
    from eval.attestation import extract_mrtd_from_quote

    quote_b64 = str(attestation["tdx"].get("quote_b64") or "")
    measured = extract_mrtd_from_quote(quote_b64)
    if measured is None:
        return False, "TDX quote too short to contain an MRTD"
    if not allowed:
        return None, (
            f"MRTD {measured[:16]}... is recorded but not checked: no approved guest measurements are "
            "pinned in this repository, so the quote shows some TD ran, not that it ran the image we "
            "approved. Pin them by building the reproducible guest image and recording its MRTD."
        )
    if measured.lower() not in {a.strip().lower() for a in allowed}:
        return False, f"MRTD {measured} is not in the approved set of {len(allowed)} guest measurement(s)"
    return True, ""


def check_attestation_integrity(
    bundle_dir: Path,
    attestation: dict | None,
    *,
    require_tdx: bool = False,
    gpu_sig: dict | None = None,
) -> list[str]:
    """CPU-only fail-closed checks for GPU CC + optional Intel TDX attestation.

    Designed for CI (no GPU): forged ``{"passed": true}`` JSON must not pass.
    Always requires a verifiable NRAS GPU token (JWKS) and claim_sha256 nonce
    binding. When ``require_tdx`` is set or a ``tdx`` blob is present, also
    requires TDX REPORTDATA binding and DCAP/PCS quote verification.
    """
    if attestation is None:
        return ["attestation is required for integrity verification"]
    if not attestation.get("passed"):
        return ["attestation must report passed: true"]

    issues: list[str] = []
    if gpu_sig is None:
        gpu_sig = check_gpu_signature(attestation)
    if gpu_sig is None:
        issues.append("attestation missing NRAS GPU token for JWKS signature verification")
    elif not gpu_sig.get("verified"):
        detail = "; ".join(str(item) for item in (gpu_sig.get("issues") or [])) or "unverified"
        issues.append(f"GPU attestation JWKS signature failed: {detail}")

    # Binding must use signed JWT eat_nonce from gpu_sig, never editable JSON claims.
    if check_claim_binding(bundle_dir, attestation, gpu_sig=gpu_sig) is not True:
        issues.append("GPU attestation signed eat_nonce does not bind claim_sha256 for this bundle")

    has_tdx = bool(attestation.get("tdx"))
    if require_tdx or has_tdx:
        if not has_tdx:
            issues.append("TDX quote is required for no-GPU attested verification")
        else:
            if check_tdx_binding(bundle_dir, attestation) is not True:
                issues.append("TDX REPORTDATA does not bind claim_sha256 for this bundle")
            tdx_sig = check_tdx_signature(attestation)
            if tdx_sig is None:
                issues.append("TDX quote missing for DCAP/PCS signature verification")
            elif not tdx_sig.get("verified"):
                status = tdx_sig.get("status") or "unverified"
                advisories = tdx_sig.get("advisory_ids") or []
                suffix = f" (advisories={advisories})" if advisories else ""
                issues.append(f"TDX quote DCAP/PCS verification failed: {status}{suffix}")
    return issues


# Headroom the gate requires between an NRAS attestation's expiry and submission
# time. NRAS GPU tokens are valid ~1h. The pre-merge gate verifies the attestation
# while it is live, but the post-merge ledger (training_track_ledger.yml) re-verifies
# it and crowns runs/frontiers.json only on a non-REJECT result. A token valid at gate
# time but expired by the time the ledger runs merges as a reward tier yet never raises
# the frontier — the silent divergence behind #288 and #301. Requiring headroom at
# submission time keeps the proof valid across the (auto-merge-fast) gate -> merge ->
# ledger window.
GATE_MIN_ATTESTATION_HEADROOM_SECONDS = 20 * 60


def _min_signed_exp(gpu_sig: dict | None) -> int | None:
    """Earliest exp (unix seconds) across the JWKS-verified platform + device tokens.

    Reads only ``gpu_sig`` (JWKS-verified claims), never the miner-editable
    ``attestation["claims"]`` sidecar — same trust rule as ``signed_attestation_claims``.
    Returns None when the signature is unverified or no token carries an ``exp``.
    """
    if not gpu_sig or not gpu_sig.get("verified"):
        return None
    claims = gpu_sig.get("claims") or {}
    exps: list[int] = []
    candidates = [claims.get("exp")]
    candidates += [device.get("exp") for device in (claims.get("devices") or {}).values() if isinstance(device, dict)]
    for value in candidates:
        # bool is an int subclass; a JSON true/false is never a timestamp.
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            exps.append(int(value))
    return min(exps) if exps else None


def attestation_freshness_issues(
    attestation: dict | None,
    *,
    min_validity_seconds: float,
    gpu_sig: dict | None = None,
    now: float | None = None,
) -> list[str]:
    """Gate-only: flag a proof whose NRAS token expires before the ledger re-verifies it.

    Returns ``[]`` when there is no attestation, its signature is unverifiable (that is
    reported by ``check_attestation_integrity``, not here), no token carries an ``exp``,
    or the token has at least ``min_validity_seconds`` of validity left. Otherwise
    returns one issue telling the miner to re-attest.

    Never call this at ledger time: there an expired token already fails JWKS, and a
    still-valid one must crown the frontier regardless of remaining headroom. It exists
    only so the pre-merge gate rejects a proof that would pass now but be dead by the
    time the post-merge ledger re-runs the same verification (see #288 / #301).
    """
    if attestation is None:
        return []
    if gpu_sig is None:
        gpu_sig = check_gpu_signature(attestation)
    exp = _min_signed_exp(gpu_sig)
    if exp is None:
        return []
    remaining = exp - (time.time() if now is None else now)
    if remaining >= min_validity_seconds:
        return []
    return [
        f"GPU attestation expires in {int(remaining)}s but the post-merge ledger "
        f"re-verifies it before crowning the frontier; require >= "
        f"{int(min_validity_seconds)}s of headroom so the proof is still valid then "
        f"(NRAS tokens live ~1h). Re-attest with the same --nonce immediately before "
        f"opening the PR."
    ]


def check_checkpoint_manifest(manifest: dict, checkpoint_path: Path | None) -> bool | None:
    """Compare a local checkpoint against the bundle's per-file sha256 manifest.

    Returns None when the bundle predates checkpoint manifests or no local
    checkpoint was reproduced (attested GSM8K-only verification).
    """
    expected = manifest.get("checkpoint_manifest")
    if not expected or checkpoint_path is None or not checkpoint_path.is_dir():
        return None
    from proof.bundle import checkpoint_manifest

    return checkpoint_manifest(checkpoint_path) == expected


def verify_submission(
    bundle_dir: Path,
    frontier: dict[str, float] | None,
    limit: int = ATTESTED_VERIFY_LIMIT,
    tolerance_pct: float = 2.0,
    attestation: dict | None = None,
    *,
    registry_path: Path = REGISTRY_PATH,
    checkpoint: Path | None = None,
    acceptable_sft_shas: set[str] | None = None,
    acceptable_pref_shas: set[str] | None = None,
) -> dict:
    """Verify a proof bundle; `frontier=None` is the BASELINE case.

    When no frontier exists yet (first verified run on a student/phase, per
    `.gittensor/weights.json`), every proof and claim check still runs, but
    instead of tier scoring the submission is labeled `eval:BASELINE` — its
    scores then seed `runs/frontier.json` for the next submission to beat.

    `acceptable_sft_shas` is the canonical-pin grace window ([#121]) for the
    calling PR: the pins valid from its merge-base through HEAD. Leave it None
    (validator default) to require the pin at the current `datasets/canonical.json`.
    """
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    claimed = json.loads((bundle_dir / "eval_scores.json").read_text())["scores"]
    gpu_architecture = resolve_bundle_gpu_architecture(manifest)

    if attestation is not None and not attestation.get("passed"):
        return {
            "verified": False,
            "reason": "attestation_failed",
            "label": "eval:REJECT",
            "run_id": manifest.get("run_id"),
        }

    # Fail-closed CPU crypto when attestation is present. Attested-sample
    # bundles require TDX as well (GPU nonce alone is not enough for no-GPU
    # verification); bare attestation without samples still needs a real NRAS
    # token + claim binding so forged {"passed": true} cannot pass CI.
    gpu_sig = check_gpu_signature(attestation)
    if attestation is not None:
        integrity_issues = check_attestation_integrity(
            bundle_dir,
            attestation,
            require_tdx=has_attested_samples(bundle_dir),
            gpu_sig=gpu_sig,
        )
        if integrity_issues:
            return {
                "verified": False,
                "reason": "attestation_integrity_failed",
                "issues": integrity_issues,
                "label": "eval:REJECT",
                "run_id": manifest.get("run_id"),
            }

    training_issues = check_training_claims(manifest, attestation, gpu_sig=gpu_sig)
    if training_issues:
        return {
            "verified": False,
            "reason": "training_claims_failed",
            "issues": training_issues,
            "label": "eval:REJECT",
            "run_id": manifest.get("run_id"),
        }

    mix_issues = check_mix_provenance(bundle_dir, manifest, registry_path=registry_path)
    if mix_issues:
        return {
            "verified": False,
            "reason": "mix_provenance_failed",
            "issues": mix_issues,
            "label": "eval:REJECT",
            "run_id": manifest.get("run_id"),
        }

    canonical_issues = check_canonical_dataset_claim(
        manifest,
        bundle_dir=bundle_dir,
        acceptable_sft_shas=acceptable_sft_shas,
        acceptable_pref_shas=acceptable_pref_shas,
    )
    if canonical_issues:
        return {
            "verified": False,
            "reason": "canonical_dataset_failed",
            "issues": canonical_issues,
            "label": "eval:REJECT",
            "run_id": manifest.get("run_id"),
        }

    attested_keys, attested_issues = verify_attested_eval_samples(
        bundle_dir,
        claimed,
        frontier,
        attestation,
        claim_binding=check_claim_binding,
        tdx_binding=check_tdx_binding,
    )
    if attested_issues and has_attested_samples(bundle_dir):
        return {
            "verified": False,
            "reason": "attested_eval_samples_failed",
            "issues": attested_issues,
            "label": "eval:REJECT",
            "run_id": manifest.get("run_id"),
        }

    claimed_benchmarks = sorted(key for key in claimed if key in BENCHMARKS)
    claimed_benchmarks = [key for key in claimed_benchmarks if key not in attested_keys]

    checkpoint_path: Path | None = bundle_dir / "checkpoint"
    if not checkpoint_path.is_dir():
        if claimed_benchmarks:
            if checkpoint is None:
                return {
                    "verified": False,
                    "reason": "checkpoint_required",
                    "issues": [
                        "proof-only bundle: reproduce the checkpoint from the recipe + dataset "
                        "and pass it via --checkpoint"
                    ],
                    "label": "eval:REJECT",
                    "run_id": manifest.get("run_id"),
                }
            checkpoint_path = checkpoint
        else:
            checkpoint_path = None

    rerun: dict[str, float] = {}
    if claimed_benchmarks:
        with _no_student_endpoint_env():
            rerun = run_harness(
                str(checkpoint_path),
                claimed_benchmarks,
                Path("eval/results/_verify"),
                limit=limit,
            )
        mismatches = check_claim(claimed, rerun, tolerance_pct)
        if mismatches:
            return {
                "verified": False,
                "reason": "claim_mismatch",
                "mismatches": mismatches,
                "label": "eval:REJECT",
                "run_id": manifest.get("run_id"),
            }

    if frontier:
        report = score(claimed, frontier, gpu_architecture=gpu_architecture)
    else:
        report = {
            "label": "eval:BASELINE",
            "best_benchmark": None,
            "best_pct_delta": None,
            "regressions": [],
            "per_benchmark": {key: {"candidate": claimed[key], "frontier": None} for key in claimed},
            "frontier_updates": sorted(claimed),
            "frontier_scores": dict(claimed),
        }
    report["verified"] = True
    report["reason"] = None
    # The full claimed score set, authoritative for frontier merging. `per_benchmark`
    # is a display projection: `eval.score` only emits keys present in *both* the
    # candidate and the frontier, so a benchmark the frontier does not carry yet
    # would otherwise never reach `runs/frontiers.json` — and a benchmark absent
    # from the frontier is never regression-guarded.
    report["scores"] = dict(claimed)
    report["run_id"] = manifest.get("run_id")
    report["gpu_architecture"] = gpu_architecture
    # Carry the declared track so the ledger merges into the right frontier bucket
    # (SFT arch bucket vs `<arch>::dpo`) — the first DPO run seeds its own baseline.
    report["train_objective"] = manifest.get("train_objective")
    report["attested_eval_benchmarks"] = sorted(attested_keys)
    report["attested_gsm8k_regression"] = REGRESSION_BENCHMARK_KEY in attested_keys
    # Trust signals — also fail-closed earlier via check_attestation_integrity
    # when attestation is present; retained here for ledger / human review.
    report["claim_bound"] = check_claim_binding(bundle_dir, attestation, gpu_sig=gpu_sig)
    report["gpu_signature"] = gpu_sig
    report["tdx_bound"] = check_tdx_binding(bundle_dir, attestation)
    report["tdx_signature"] = check_tdx_signature(attestation)
    # Which guest image ran, as opposed to whether the quote is genuine and bound. Reported
    # with its reason so an unpinned allowlist reads as a stated gap rather than a blank.
    measured, measured_reason = check_tdx_measurement(attestation)
    report["tdx_measured_vm"] = measured
    report["tdx_measured_vm_reason"] = measured_reason
    report["checkpoint_hash_match"] = check_checkpoint_manifest(manifest, checkpoint_path)
    return report


def _resolve_bundle_dir(bundle_repo: str | None, bundle_path: Path | None, revision: str | None = None) -> Path:
    if bundle_path is not None:
        return bundle_path
    if bundle_repo is None:
        raise ValueError("one of --bundle-repo or --bundle-path is required")
    from huggingface_hub import snapshot_download

    return Path(pinned_download(snapshot_download, revision=revision, repo_id=bundle_repo))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle-repo", default=None, help="HF hub repo id to download the proof bundle from")
    parser.add_argument(
        "--bundle-revision",
        default=None,
        help="exact 40-char commit sha of --bundle-repo. Required with it: a branch name resolves "
        "at download time, so the bundle verified here is not the bundle anyone fetches later",
    )
    parser.add_argument(
        "--bundle-path", type=Path, default=None, help="local bundle dir (alternative to --bundle-repo)"
    )
    parser.add_argument(
        "--frontier",
        type=Path,
        default=None,
        help="frontier scores json, as a flat {benchmark: score} dict. Default: resolve "
        "the bundle's GPU architecture from its manifest and load that architecture's "
        "bucket from runs/frontiers.json (falling back to the legacy runs/frontier.json "
        "for Blackwell); an unset bucket means no frontier exists yet -> eval:BASELINE",
    )
    parser.add_argument("--attestation", type=Path, default=None, help="attestation json from eval.attestation")
    parser.add_argument(
        "--limit",
        type=int,
        default=ATTESTED_VERIFY_LIMIT,
        help="examples per benchmark for the cheap re-run",
    )
    parser.add_argument("--tolerance-pct", type=float, default=2.0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="locally reproduced checkpoint dir, required for proof-only bundles (no weights on HF)",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    bundle_dir = _resolve_bundle_dir(args.bundle_repo, args.bundle_path, args.bundle_revision)
    if args.frontier is not None:
        frontier = json.loads(args.frontier.read_text())["scores"] if args.frontier.exists() else None
    else:
        manifest = json.loads((bundle_dir / "manifest.json").read_text())
        frontier = load_frontier_scores(resolve_bundle_gpu_architecture(manifest), track=training_track_of(manifest))
    attestation = json.loads(args.attestation.read_text()) if args.attestation else None

    report = verify_submission(
        bundle_dir,
        frontier,
        limit=args.limit,
        tolerance_pct=args.tolerance_pct,
        attestation=attestation,
        checkpoint=args.checkpoint,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"{report['label']} (verified={report['verified']}, reason={report['reason']})", file=sys.stderr)
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
