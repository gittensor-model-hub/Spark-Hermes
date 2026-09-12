"""Automated gate for training/evaluation improvement pull requests.

Training-track PRs must compete on recipe/hyperparameter changes against the
single pinned canonical mining dataset. Local generators, private blends, and
non-registry dataset paths are rejected.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import yaml

from eval.canonical_dataset import (
    CANONICAL_PATH,
    CANONICAL_TRAINING_DATASET_PATH,
    TRAINING_TRACK_DPO,
    TRAINING_TRACK_SFT,
    assert_recipe_uses_canonical_dataset,
    canonical_hf_url,
    canonical_pref_hf_url,
    canonical_pref_sha256,
    canonical_sft_sha256,
    load_canonical,
    pref_sha256_from_canonical_text,
    recipe_training_track,
    sft_sha256_from_canonical_text,
    verify_remote_matches_pin,
)
from eval.hf_pin import download_with_optional_pin
from eval.verify import check_canonical_dataset_claim

_TRAINING_TRACK_CHECKBOX_RE = re.compile(
    r"^\s*-\s*\[[xX]\]\s+\*{0,2}Training/evaluation improvement\*{0,2}\s*$",
    re.MULTILINE,
)
_DATASET_TRACK_CHECKBOX_RE = re.compile(
    r"^\s*-\s*\[[xX]\]\s+\*{0,2}Dataset track submission\*{0,2}\s*$",
    re.MULTILINE,
)
_CANONICAL_SHA_IN_BODY_RE = re.compile(r"`([0-9a-f]{64})`")
_PROOF_BUNDLE_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?Proof[- ]bundle URL(?:\s*\([^)]*\))?\s*:\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_HF_MODEL_REPO_URL_RE = re.compile(r"https://huggingface\.co/(?!datasets/)([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)")
_PROOF_BUNDLE_PLACEHOLDER_RE = re.compile(
    r"^(?:n/a|na|none|pending(?:\s+after.*)?|tbd|todo|optional|-|\.)$",
    re.IGNORECASE,
)
# attestation.json lives in the PR itself (runs/<run_id>/attestation.json), never on HF —
# the proof bundle only carries claim-bound artifacts, not the attestation that binds them.
_ATTESTATION_PATH_RE = re.compile(r"^runs/[^/]+/attestation\.json$")
_FORBIDDEN_CHANGED_GLOBS = (
    "eval/gen_*.py",
    "scripts/prepare_triton*.sh",
    "scripts/prepare_sft_data.sh",
    "data/processed/*",
)
_ALLOWED_ALWAYS = frozenset(
    {
        "datasets/canonical.json",
    }
)
TRAINING_LABELS = frozenset({"training:valid", "training:REJECT", "training:skipped"})
_LABEL_COLORS = {
    "training:valid": "0e8a16",
    "training:REJECT": "b60205",
}

# eval:* is the reward-tier label eval.verify computes (separate from training:* above,
# which only gates format/provenance) — applied when verify_remote_proof_bundle_scores
# can compute one from CPU-verifiable claims or deferred checkpoint tiering.
EVAL_LABELS = frozenset(
    {"eval:XL", "eval:L", "eval:M", "eval:S", "eval:XS", "eval:none", "eval:BASELINE", "eval:REJECT"}
)
_AUTO_CLOSE_EVAL_LABELS = frozenset({"eval:none", "eval:REJECT"})
# A verified training PR whose eval tier is a real reward (XS..XL, or BASELINE for the
# first verified checkpoint) is merge-eligible; eval:none/REJECT are closed instead.
_REWARDED_EVAL_LABELS = EVAL_LABELS - _AUTO_CLOSE_EVAL_LABELS
_EVAL_LABEL_COLORS = {
    "eval:XL": "1d76db",
    "eval:L": "0e8a16",
    "eval:M": "2cbe4e",
    "eval:S": "7bd88f",
    "eval:XS": "c5def5",
    "eval:none": "d4c5f9",
    "eval:BASELINE": "fbca04",
    "eval:REJECT": "b60205",
}


def is_training_track_pr(pr_body: str | None) -> bool:
    body = pr_body or ""
    if _DATASET_TRACK_CHECKBOX_RE.search(body):
        return False
    return bool(_TRAINING_TRACK_CHECKBOX_RE.search(body))


def is_dataset_track_pr(pr_body: str | None) -> bool:
    return bool(_DATASET_TRACK_CHECKBOX_RE.search(pr_body or ""))


def _matches_forbidden(path: str) -> str | None:
    for pattern in _FORBIDDEN_CHANGED_GLOBS:
        if fnmatch(path, pattern):
            return pattern
    return None


def validate_changed_paths(changed_paths: list[str] | None) -> list[str]:
    if changed_paths is None:
        return []
    issues: list[str] = []
    for path in changed_paths:
        if path in _ALLOWED_ALWAYS:
            continue
        if path == "datasets/registry.jsonl":
            issues.append("training-track PRs must not modify datasets/registry.jsonl; use the dataset track")
            continue
        forbidden = _matches_forbidden(path)
        if forbidden:
            issues.append(
                f"training-track PRs may not change {path!r} "
                f"(matches forbidden pattern {forbidden!r}); "
                "contribute new rows through SparkProof + the dataset registry first"
            )
    return issues


def _git_show(ref: str, path: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def validate_recipe_paths_in_ref(head_ref: str, recipe_paths: list[str]) -> list[str]:
    issues: list[str] = []
    for recipe_path in recipe_paths:
        if not recipe_path.startswith("recipes/") or not recipe_path.endswith((".yaml", ".yml")):
            continue
        text = _git_show(head_ref, recipe_path)
        if text is None:
            continue
        try:
            recipe = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            issues.append(f"{recipe_path}: invalid YAML: {exc}")
            continue
        if not isinstance(recipe, dict):
            issues.append(f"{recipe_path}: root must be a mapping")
            continue
        issues.extend(f"{recipe_path}: {issue}" for issue in assert_recipe_uses_canonical_dataset(recipe))
    return issues


def pr_training_track(head_ref: str, recipe_paths: list[str]) -> str:
    """Track a training PR competes on: ``dpo`` if any changed recipe is ``rl: dpo``, else ``sft``."""
    for recipe_path in recipe_paths:
        if not recipe_path.startswith("recipes/") or not recipe_path.endswith((".yaml", ".yml")):
            continue
        text = _git_show(head_ref, recipe_path)
        if not text:
            continue
        try:
            recipe = yaml.safe_load(text)
        except yaml.YAMLError:
            continue
        if isinstance(recipe, dict) and recipe_training_track(recipe) == TRAINING_TRACK_DPO:
            return TRAINING_TRACK_DPO
    return TRAINING_TRACK_SFT


def _canonical_sha256s_for_pr_window(
    *,
    merge_base_ref: str | None,
    head_ref: str = "HEAD",
    extractor=sft_sha256_from_canonical_text,
) -> set[str]:
    """Canonical pins valid for a training PR while dataset-track merges advance HEAD.

    ``extractor`` reads the digest out of a ``datasets/canonical.json`` text — the SFT
    ``sft_sha256`` by default, or ``pref_sha256_from_canonical_text`` for the DPO track.
    """
    shas: set[str] = set()

    def add_from_ref(ref: str) -> None:
        text = _git_show(ref, CANONICAL_PATH.as_posix())
        if not text:
            return
        sha = extractor(text)
        if sha:
            shas.add(sha)

    add_from_ref(head_ref)
    if not merge_base_ref:
        return shas

    add_from_ref(merge_base_ref)
    log = subprocess.run(
        [
            "git",
            "log",
            "--format=%H",
            f"{merge_base_ref}..{head_ref}",
            "--",
            CANONICAL_PATH.as_posix(),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if log.returncode == 0:
        for commit in log.stdout.splitlines():
            commit = commit.strip()
            if commit:
                add_from_ref(commit)
    return shas


def _canonical_sft_sha256s_for_pr_window(*, merge_base_ref: str | None, head_ref: str = "HEAD") -> set[str]:
    return _canonical_sha256s_for_pr_window(
        merge_base_ref=merge_base_ref, head_ref=head_ref, extractor=sft_sha256_from_canonical_text
    )


def _canonical_pref_sha256s_for_pr_window(*, merge_base_ref: str | None, head_ref: str = "HEAD") -> set[str]:
    return _canonical_sha256s_for_pr_window(
        merge_base_ref=merge_base_ref, head_ref=head_ref, extractor=pref_sha256_from_canonical_text
    )


def validate_pr_body_canonical_pin(
    pr_body: str | None,
    *,
    track: str = TRAINING_TRACK_SFT,
    acceptable_sft_shas: set[str] | None = None,
    acceptable_pref_shas: set[str] | None = None,
) -> list[str]:
    is_dpo = track == TRAINING_TRACK_DPO
    sha_field = "pref_sha256" if is_dpo else "sft_sha256"
    if not pr_body:
        return [f"training-track PR body must cite the pinned canonical dataset URL and {sha_field}"]
    issues: list[str] = []
    try:
        expected_url = canonical_pref_hf_url() if is_dpo else canonical_hf_url()
    except (FileNotFoundError, ValueError, KeyError):
        return [
            "no canonical preference dataset pin is configured; the DPO track is unavailable"
            if is_dpo
            else "canonical dataset pin is not configured"
        ]
    if expected_url not in pr_body:
        issues.append(f"PR body must cite canonical {'preference ' if is_dpo else ''}dataset URL {expected_url}")

    allowed = acceptable_pref_shas if is_dpo else acceptable_sft_shas
    if allowed is None:
        try:
            allowed = {canonical_pref_sha256() if is_dpo else canonical_sft_sha256()}
        except ValueError as exc:
            issues.append(str(exc))
            return issues

    cited_shas = set(_CANONICAL_SHA_IN_BODY_RE.findall(pr_body))
    if not cited_shas:
        issues.append(
            f"PR body must cite the canonical {sha_field} used for training (one of: {', '.join(sorted(allowed))})"
        )
    elif not cited_shas & allowed:
        issues.append(
            f"PR body {sha_field} must match a canonical pin from this PR's merge-base window "
            f"(allowed: {', '.join(sorted(allowed))})"
        )
    return issues


def parse_proof_bundle_hf_repo(pr_body: str | None) -> str | None:
    """Return org/repo for the cited Hugging Face proof bundle model repo, if any."""
    body = pr_body or ""
    field_match = _PROOF_BUNDLE_LINE_RE.search(body)
    if field_match:
        value = field_match.group(1).strip().strip("`")
        if value and not _PROOF_BUNDLE_PLACEHOLDER_RE.match(value):
            repo_match = _HF_MODEL_REPO_URL_RE.search(value)
            if repo_match:
                return repo_match.group(1)
    return None


def validate_pr_body_proof_bundle(pr_body: str | None) -> list[str]:
    """Training-track PRs must publish and cite a Hugging Face proof bundle."""
    if not pr_body:
        return ["training-track PR body must cite a Hugging Face proof-bundle URL"]

    issues: list[str] = []
    field_match = _PROOF_BUNDLE_LINE_RE.search(pr_body)
    if not field_match:
        issues.append(
            "PR body must include a Proof-bundle URL field with a published "
            "https://huggingface.co/<user>/<repo> model repo link"
        )
        return issues

    value = field_match.group(1).strip().strip("`")
    if not value or _PROOF_BUNDLE_PLACEHOLDER_RE.match(value):
        issues.append("Proof-bundle URL must be a published Hugging Face model repo URL (not pending, n/a, or empty)")
        return issues

    if not parse_proof_bundle_hf_repo(pr_body):
        issues.append(
            "Proof-bundle URL must be a Hugging Face model repo "
            "(https://huggingface.co/<user>/<repo>), not a datasets URL"
        )
    return issues


def verify_remote_proof_bundle(
    repo_id: str,
    *,
    hf_token: str | None = None,
    acceptable_sft_shas: set[str] | None = None,
    acceptable_pref_shas: set[str] | None = None,
) -> list[str]:
    """Download the cited bundle manifest and verify canonical dataset claims."""
    from huggingface_hub import hf_hub_download

    issues: list[str] = []
    try:
        manifest_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename="manifest.json",
            token=hf_token,
        )
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"failed to download proof bundle manifest from {repo_id}: {exc}"]

    if not isinstance(manifest, dict):
        return [f"{repo_id}/manifest.json must be a JSON object"]

    issues.extend(check_canonical_dataset_claim(manifest))

    # Track-aware pin pre-check (full pin binding is re-checked in verify_submission):
    # a DPO bundle carries pref_sha256 and is matched against the canonical preference pin.
    is_dpo = str(manifest.get("train_objective") or "").strip().lower() == TRAINING_TRACK_DPO
    sha_field = "pref_sha256" if is_dpo else "sft_sha256"
    allowed = acceptable_pref_shas if is_dpo else acceptable_sft_shas
    if allowed is None:
        try:
            allowed = {canonical_pref_sha256() if is_dpo else canonical_sft_sha256()}
        except ValueError:
            allowed = set()

    try:
        mix_manifest_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename="mix_manifest.json",
            token=hf_token,
        )
        mix_data = json.loads(Path(mix_manifest_path).read_text(encoding="utf-8"))
        remote_sha = mix_data.get(sha_field)
        if remote_sha not in allowed:
            issues.append(
                f"proof bundle mix_manifest.{sha_field} does not match an accepted canonical pin "
                f"for this PR window (allowed {len(allowed)} pin(s))"
            )
    except Exception:
        pass

    try:
        hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename="eval_scores.json",
            token=hf_token,
        )
    except Exception as exc:
        issues.append(f"proof bundle missing eval_scores.json on {repo_id}: {exc}")

    return issues


def find_attestation_path(changed_paths: list[str] | None) -> str | None:
    for path in changed_paths or []:
        if _ATTESTATION_PATH_RE.match(path):
            return path
    return None


def score_claimed_eval_label(bundle_dir: Path, manifest: dict) -> str | None:
    """Tier claimed scores when full verify is deferred (checkpoint_required)."""
    from eval.frontiers import load_frontier_scores, training_track_of
    from eval.score import score
    from eval.verify import resolve_bundle_gpu_architecture

    eval_path = bundle_dir / "eval_scores.json"
    if not eval_path.exists():
        return None
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    scores = payload.get("scores") if isinstance(payload.get("scores"), dict) else payload
    if not isinstance(scores, dict):
        return None
    arch = resolve_bundle_gpu_architecture(manifest)
    frontier = load_frontier_scores(arch, track=training_track_of(manifest))
    if frontier is None:
        return "eval:BASELINE"
    report = score(scores, frontier, gpu_architecture=arch)
    return report["label"]


def _download_and_verify_bundle(
    repo_id: str,
    *,
    revision: str | None = None,
    head_ref: str,
    changed_paths: list[str] | None,
    hf_token: str | None = None,
    acceptable_sft_shas: set[str] | None = None,
    acceptable_pref_shas: set[str] | None = None,
) -> tuple[dict | None, dict | None, str | None, Path | None]:
    """Download the cited proof bundle and run eval.verify against it.

    Shared by verify_remote_proof_bundle_scores (PR-time gating) and
    record_merged_ledger_entry (merge-time ledger write) — both need the same
    attestation lookup, frontier resolution, and verify_submission call, just
    consuming the result differently (issues+label vs. the full report).

    `acceptable_sft_shas` carries the PR's canonical-pin grace window ([#121])
    into `verify_submission`; without it the bundle's `mix_manifest.sft_sha256`
    is re-checked against live HEAD only and an honest submission whose pin
    advanced mid-training is rejected — the exact race issue #118 describes.

    Returns `(report, attestation, error, bundle_dir)`: `error` is set (and the other
    fields None) only when the bundle couldn't be read at all (bad attestation JSON,
    download failure, or files missing) — genuine verify_submission outcomes
    (including REJECT) come back as a populated `report`, never as `error`.
    """
    from huggingface_hub import snapshot_download

    from eval.frontiers import load_frontier_scores, training_track_of
    from eval.verify import resolve_bundle_gpu_architecture, verify_submission

    attestation = None
    attestation_path = find_attestation_path(changed_paths)
    if attestation_path is not None:
        try:
            from hermes.evidence_json import evidence_object

            text = _git_show(head_ref, attestation_path)
            if text is None:
                raise ValueError("attestation could not be read")
            attestation = evidence_object(text)
        except ValueError:
            return None, None, f"{attestation_path}: invalid JSON", None

    try:
        # `revision` is used when the submission carries one and left to the Hub's default
        # otherwise. NOT required here, unlike the rollout track: this gate has live miners,
        # and making a pin mandatory would reject every submission already in flight.
        #
        # Passed only when set. `revision=None` is equivalent to omitting it for the real
        # client but not for every wrapper standing in for it, and on a live path a kwarg a
        # caller's double does not know is a breaking change dressed as a default.
        bundle_dir = Path(
            download_with_optional_pin(
                snapshot_download, revision=revision, repo_id=repo_id, repo_type="model", token=hf_token
            )
        )
    except Exception as exc:
        return None, None, f"failed to download proof bundle from {repo_id} for verification: {exc}", None

    if not (bundle_dir / "manifest.json").exists() or not (bundle_dir / "eval_scores.json").exists():
        return None, None, None, bundle_dir  # already flagged by verify_remote_proof_bundle

    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    frontier = load_frontier_scores(resolve_bundle_gpu_architecture(manifest), track=training_track_of(manifest))

    report = verify_submission(
        bundle_dir,
        frontier,
        attestation=attestation,
        acceptable_sft_shas=acceptable_sft_shas,
        acceptable_pref_shas=acceptable_pref_shas,
    )
    return report, attestation, None, bundle_dir


def _attestation_issues_for_eval_tiering(
    changed_paths: list[str] | None,
    attestation: dict | None,
    bundle_dir: Path | None,
) -> list[str]:
    """Training-track eval tiering requires a cryptographically valid PR attestation.

    Presence of ``passed: true`` is not enough — CI verifies the NRAS GPU token
    against NVIDIA's JWKS and the claim_sha256 nonce binding without a GPU. When
    a TDX blob is present, both REPORTDATA binding and DCAP/PCS verification are
    required as well.
    """
    if find_attestation_path(changed_paths) is None:
        return [
            "training-track eval tiering requires runs/<run-id>/attestation.json in the PR "
            "(proof-only HF bundles without attestation cannot earn eval:XL/L/M/S/XS labels)"
        ]
    if attestation is None:
        return ["could not read attestation.json from the PR head for eval tiering"]
    if bundle_dir is None:
        return ["proof bundle directory missing for attestation integrity checks"]
    from eval.verify import check_attestation_integrity

    return check_attestation_integrity(bundle_dir, attestation, require_tdx=False)


def verify_remote_proof_bundle_scores(
    repo_id: str,
    *,
    head_ref: str,
    changed_paths: list[str] | None,
    hf_token: str | None = None,
    acceptable_sft_shas: set[str] | None = None,
    acceptable_pref_shas: set[str] | None = None,
) -> tuple[list[str], str | None]:
    """Re-run eval.verify's no-GPU attested checks against the cited proof bundle.

    This is the exact CPU-only check a validator otherwise has to run by hand:
    claim_sha256/TDX binding, GPU-attestation JWKS signature, and recomputing any
    attested gsm8k/triton samples from their own bundled responses. It catches
    claim/attested-sample mismatches and GPU-corroboration issues (e.g. an H200
    run whose attestation reports hwmodel=GH100, the same die as H100) at PR time
    instead of only when a human happens to check manually — that gap is what let
    gittensor-model-hub/SparkDistill#120 sit with a stale gsm8k claim (and no
    eval:* label at all) until review.

    Never needs a GPU: proof bundles are weights-free (no checkpoint on HF). When
    attested eval samples cover every claimed benchmark and the PR commits a
    cryptographically valid GPU (+ TDX when present/required) attestation bound
    to ``claim_sha256``, ``eval.verify`` completes entirely on CPU. Otherwise
    verification ends in ``checkpoint_required``; reward-tier labels may only be
    derived from deferred claims when the PR attestation passes JWKS + claim
    binding (and TDX checks when a quote is present) — forged ``passed: true``
    JSON is ``eval:REJECT``.

    `acceptable_sft_shas` must be the same canonical-pin grace window
    `verify_remote_proof_bundle` was given, so the two stages agree on which pins
    this PR may have trained against ([#121]).

    Returns `(issues, eval_label)` — `eval_label` is the reward-tier label
    eval.verify computed (e.g. "eval:BASELINE", "eval:XL", "eval:REJECT"), or
    a tier derived from claimed scores when verify is deferred *and* attestation
    crypto passes ("checkpoint_required"), or None when nothing could be computed
    (download failure).
    """
    report, attestation, error, bundle_dir = _download_and_verify_bundle(
        repo_id,
        head_ref=head_ref,
        changed_paths=changed_paths,
        hf_token=hf_token,
        acceptable_sft_shas=acceptable_sft_shas,
        acceptable_pref_shas=acceptable_pref_shas,
    )
    if error is not None:
        return [error], None
    if report is None:
        return [], None

    # Gate-only freshness guard: a proof valid now but whose NRAS token expires before
    # the post-merge ledger re-verifies it would merge as a reward tier yet never crown
    # the frontier (the #288 / #301 divergence). Require headroom at submission time so
    # a fresh attestation survives the gate -> merge -> ledger window; a stale one lands
    # as training:REJECT (not auto-closed) and the miner re-attests and pushes. Not
    # applied at ledger time — there an expired token already fails JWKS.
    from eval.verify import GATE_MIN_ATTESTATION_HEADROOM_SECONDS, attestation_freshness_issues

    freshness_issues = attestation_freshness_issues(
        attestation, min_validity_seconds=GATE_MIN_ATTESTATION_HEADROOM_SECONDS
    )

    if report.get("reason") == "checkpoint_required":
        if bundle_dir is not None:
            attestation_issues = _attestation_issues_for_eval_tiering(
                changed_paths,
                attestation,
                bundle_dir,
            )
            if attestation_issues:
                return attestation_issues, "eval:REJECT"
            manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
            return freshness_issues, score_claimed_eval_label(bundle_dir, manifest)
        return [], None
    if report.get("verified"):
        return freshness_issues, report.get("label")
    detail = report.get("issues") or report.get("mismatches") or []
    reason = report.get("reason") or "verification failed"
    issues = [f"eval.verify {reason}: {issue}" for issue in detail] if detail else [f"eval.verify: {reason}"]
    return issues, report.get("label")


def record_merged_ledger_entry(
    *,
    pr_url: str,
    pr_body: str | None,
    head_ref: str,
    changed_paths: list[str] | None,
    merge_base_ref: str | None = None,
    hf_token: str | None = None,
    ledger_path: Path = Path("runs/ledger.jsonl"),
    frontiers_path: Path | None = None,
) -> list[str]:
    """Append the merged training-track PR's result to runs/ledger.jsonl.

    Meant to run once, after merge (head_ref should be the merged commit, so
    attestation.json is read from its final on-disk location). Idempotent:
    skips silently if run_id is already present in the ledger, so a re-run
    (e.g. workflow retry) never double-appends. Non-training-track PRs (no
    cited HF proof-bundle repo) are a silent no-op — nothing to log.

    Also seeds / raises ``runs/frontiers.json`` for verified non-REJECT runs
    (including ``eval:BASELINE``). Frontier updates still run when the ledger
    entry already exists, so a retry can heal a previously missed frontier write.

    ``merge_base_ref`` reuses the PR's canonical-pin grace window ([#121]) so a
    run that was merge-eligible at gate time is not re-rejected here just because
    the pin advanced again before the ledger job ran — that would record
    ``eval:REJECT`` for an already-merged PR and skip its frontier update.
    """
    from eval.frontiers import apply_verified_report_to_frontiers
    from eval.ledger import append_entry, build_entry

    repo_id = parse_proof_bundle_hf_repo(pr_body)
    if repo_id is None:
        return []

    acceptable_sft_shas = (
        _canonical_sft_sha256s_for_pr_window(merge_base_ref=merge_base_ref, head_ref="HEAD") if merge_base_ref else None
    )
    report, attestation, error, _bundle_dir = _download_and_verify_bundle(
        repo_id,
        head_ref=head_ref,
        changed_paths=changed_paths,
        hf_token=hf_token,
        acceptable_sft_shas=acceptable_sft_shas,
    )
    if error is not None:
        return [error]
    if report is None:
        return ["proof bundle missing manifest.json or eval_scores.json — nothing to log"]

    run_id = report.get("run_id")
    if not run_id:
        return ["proof bundle manifest.json is missing run_id — nothing to log"]

    proof_bundle = f"https://huggingface.co/{repo_id}"
    existing_run_ids = set()
    if ledger_path.exists():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing_run_ids.add(json.loads(line).get("run_id"))
    if run_id not in existing_run_ids:
        entry = build_entry(run_id, pr_url, proof_bundle, report, attestation)
        append_entry(ledger_path, entry)

        run_dir = ledger_path.parent / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        # attestation.json for a training-track PR is already committed by the miner as part
        # of the PR itself (runs/<run_id>/attestation.json) — never rewrite that file here,
        # only add it if this run somehow doesn't already have one on disk.
        attestation_dest = run_dir / "attestation.json"
        if attestation is not None and not attestation_dest.exists():
            attestation_dest.write_text(json.dumps(attestation, indent=2) + "\n", encoding="utf-8")

    apply_verified_report_to_frontiers(
        report,
        proof_bundle=proof_bundle,
        path=frontiers_path if frontiers_path is not None else ledger_path.parent / "frontiers.json",
    )
    return []


def _is_training_submission_path(path: str) -> bool:
    """Whether a changed path signals a training-track submission.

    A recipe *file* (YAML under ``recipes/``) or a local generator does. Docs or
    other non-recipe files under ``recipes/`` (e.g. a README) do not — a
    documentation or tooling PR that merely touches ``recipes/`` is not a
    submission, and must not be gated (and, with ``--close-on-reject``, auto-closed)
    as one. See issue #283 / PR #284, closed by this exact misclassification.
    """
    if path.startswith(("eval/gen_", "scripts/prepare_triton", "scripts/prepare_sft_data")):
        return True
    return path.startswith("recipes/") and path.endswith((".yaml", ".yml"))


def should_enforce_training_gate(
    pr_body: str | None,
    changed_paths: list[str] | None,
) -> bool:
    if pr_body is not None and is_dataset_track_pr(pr_body):
        return False
    if pr_body is not None and is_training_track_pr(pr_body):
        return True
    if not changed_paths:
        return False
    return any(_is_training_submission_path(path) for path in changed_paths)


def gate_training_pr(
    *,
    head_ref: str,
    changed_paths: list[str] | None,
    pr_body: str | None,
    merge_base_ref: str | None = None,
    verify_hf_pin: bool = True,
    verify_proof_bundle: bool = True,
    hf_token: str | None = None,
) -> dict[str, Any]:
    if not should_enforce_training_gate(pr_body, changed_paths):
        return {
            "verified": True,
            "label": "training:skipped",
            "issues": [],
            "merge_eligible": False,
            "canonical": load_canonical(),
            "training_dataset_path": CANONICAL_TRAINING_DATASET_PATH,
        }

    acceptable_sft_shas = _canonical_sft_sha256s_for_pr_window(
        merge_base_ref=merge_base_ref,
        head_ref="HEAD",
    )
    acceptable_pref_shas = _canonical_pref_sha256s_for_pr_window(
        merge_base_ref=merge_base_ref,
        head_ref="HEAD",
    )

    issues: list[str] = []
    if pr_body is not None and not is_training_track_pr(pr_body):
        issues.append("check 'Training/evaluation improvement' in the pull request template")
    issues.extend(validate_changed_paths(changed_paths))

    recipe_paths = sorted({path for path in (changed_paths or []) if path.startswith("recipes/")})
    issues.extend(validate_recipe_paths_in_ref(head_ref, recipe_paths))

    # A `rl: dpo` recipe competes on the preference track: its PR body cites the canonical
    # preference pin and its bundle is verified against pref_sha256 / the DPO frontier.
    track = pr_training_track(head_ref, recipe_paths)

    issues.extend(
        validate_pr_body_canonical_pin(
            pr_body,
            track=track,
            acceptable_sft_shas=acceptable_sft_shas,
            acceptable_pref_shas=acceptable_pref_shas,
        )
    )
    issues.extend(validate_pr_body_proof_bundle(pr_body))

    if verify_hf_pin:
        issues.extend(verify_remote_matches_pin(hf_token=hf_token))

    eval_label: str | None = None
    if verify_proof_bundle:
        repo_id = parse_proof_bundle_hf_repo(pr_body)
        if repo_id is not None:
            bundle_issues = verify_remote_proof_bundle(
                repo_id,
                hf_token=hf_token,
                acceptable_sft_shas=acceptable_sft_shas,
                acceptable_pref_shas=acceptable_pref_shas,
            )
            issues.extend(bundle_issues)
            if not bundle_issues:
                score_issues, eval_label = verify_remote_proof_bundle_scores(
                    repo_id,
                    head_ref=head_ref,
                    changed_paths=changed_paths,
                    hf_token=hf_token,
                    acceptable_sft_shas=acceptable_sft_shas,
                    acceptable_pref_shas=acceptable_pref_shas,
                )
                issues.extend(score_issues)

    label = "training:valid" if not issues else "training:REJECT"
    return {
        "verified": not issues,
        "label": label,
        "eval_label": eval_label,
        # Only a verified PR that earned a real reward tier auto-merges (mirrors the
        # dataset gate's merge_eligible); eval:none/REJECT are closed, not merged.
        "merge_eligible": (not issues) and eval_label in _REWARDED_EVAL_LABELS,
        "issues": issues,
        "track": track,
        "canonical": load_canonical(),
        "acceptable_sft_shas": sorted(acceptable_sft_shas),
        "acceptable_pref_shas": sorted(acceptable_pref_shas),
        "training_dataset_path": CANONICAL_TRAINING_DATASET_PATH,
    }


def update_pr_training_label(pr_number: int, label: str) -> list[str]:
    if label not in TRAINING_LABELS:
        return [f"refusing to apply unknown training label {label!r}"]

    subprocess.run(
        ["gh", "label", "create", label, "--color", _LABEL_COLORS[label], "--force"],
        capture_output=True,
        text=True,
        check=False,
    )
    for existing in TRAINING_LABELS:
        if existing != label:
            subprocess.run(
                ["gh", "pr", "edit", str(pr_number), "--remove-label", existing],
                capture_output=True,
                text=True,
                check=False,
            )
    result = subprocess.run(
        ["gh", "pr", "edit", str(pr_number), "--add-label", label],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return [result.stderr.strip() or result.stdout.strip() or "failed to apply label"]
    return []


def update_pr_eval_label(pr_number: int, label: str) -> list[str]:
    """Apply the eval.verify reward-tier label, replacing any prior eval:* label."""
    if label not in EVAL_LABELS:
        return [f"refusing to apply unknown eval label {label!r}"]

    subprocess.run(
        ["gh", "label", "create", label, "--color", _EVAL_LABEL_COLORS[label], "--force"],
        capture_output=True,
        text=True,
        check=False,
    )
    for existing in EVAL_LABELS:
        if existing != label:
            subprocess.run(
                ["gh", "pr", "edit", str(pr_number), "--remove-label", existing],
                capture_output=True,
                text=True,
                check=False,
            )
    result = subprocess.run(
        ["gh", "pr", "edit", str(pr_number), "--add-label", label],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return [result.stderr.strip() or result.stdout.strip() or "failed to apply label"]
    return []


def close_training_pr(
    pr_number: int,
    *,
    training_label: str = "training:REJECT",
    eval_label: str | None = None,
    issues: list[str] | None = None,
) -> list[str]:
    """Close a training-track PR that is not merge-eligible."""
    if eval_label == "eval:none":
        summary = "Closed automatically: claimed proof scores are below the merge threshold (eval:none)."
        result = subprocess.run(
            ["gh", "pr", "close", str(pr_number), "--comment", summary],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return [f"could not close PR #{pr_number}: {(result.stderr or result.stdout).strip()}"]
        return []

    if eval_label == "eval:REJECT":
        summary = "Closed automatically: eval verification rejected this submission."
        if issues:
            bullets = "\n".join(f"- {issue}" for issue in issues[:8])
            summary = f"{summary}\n\n{bullets}"
        result = subprocess.run(
            ["gh", "pr", "close", str(pr_number), "--comment", summary],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return [f"could not close PR #{pr_number}: {(result.stderr or result.stdout).strip()}"]
        return []

    body = "\n".join(f"- {issue}" for issue in (issues or [])) or "- training-track gate rejected this PR"
    comment = subprocess.run(
        ["gh", "pr", "comment", str(pr_number), "--body", f"## {training_label}\n\n{body}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if comment.returncode != 0:
        return [comment.stderr.strip() or comment.stdout.strip() or "failed to comment"]
    close = subprocess.run(
        ["gh", "pr", "close", str(pr_number)],
        capture_output=True,
        text=True,
        check=False,
    )
    if close.returncode != 0:
        return [close.stderr.strip() or close.stdout.strip() or "failed to close PR"]
    return []


def _emit_step_output(name: str, value: str) -> None:
    """Append a GitHub Actions step output; a no-op off Actions (no GITHUB_OUTPUT)."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{name}={value}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head-ref", default="HEAD")
    parser.add_argument(
        "--merge-base-ref",
        default=None,
        help="git ref for merge-base with the PR base; enables canonical-pin grace window",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pr-body-file", type=Path, default=None)
    parser.add_argument("--changed-paths-file", type=Path, default=None)
    parser.add_argument("--skip-hf-pin-check", action="store_true")
    parser.add_argument("--skip-proof-bundle-check", action="store_true")
    parser.add_argument("--apply-label", action="store_true")
    parser.add_argument("--close-on-reject", action="store_true")
    parser.add_argument("--merge-on-pass", action="store_true")
    parser.add_argument("--pr-number", type=int, default=None)
    args = parser.parse_args(argv)

    pr_body = args.pr_body_file.read_text(encoding="utf-8") if args.pr_body_file else None
    changed_paths = None
    if args.changed_paths_file:
        changed_paths = [
            line.strip() for line in args.changed_paths_file.read_text(encoding="utf-8").splitlines() if line.strip()
        ]

    report = gate_training_pr(
        head_ref=args.head_ref,
        changed_paths=changed_paths,
        pr_body=pr_body,
        merge_base_ref=args.merge_base_ref,
        verify_hf_pin=not args.skip_hf_pin_check,
        verify_proof_bundle=not args.skip_proof_bundle_check,
        hf_token=os.environ.get("HF_TOKEN"),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        f"{report['label']} verified={report['verified']} issues={len(report.get('issues') or [])}",
        file=sys.stderr,
    )
    for issue in report.get("issues") or []:
        print(f"  - {issue}", file=sys.stderr)

    if args.apply_label:
        if args.pr_number is None:
            print("--apply-label requires --pr-number", file=sys.stderr)
            return 1
        if report["label"] == "training:skipped":
            print("training gate skipped", file=sys.stderr)
        else:
            label_issues = update_pr_training_label(args.pr_number, report["label"])
            if label_issues:
                for issue in label_issues:
                    print(f"  - {issue}", file=sys.stderr)
                return 1
        eval_label = report.get("eval_label")
        if eval_label is not None:
            eval_label_issues = update_pr_eval_label(args.pr_number, eval_label)
            if eval_label_issues:
                for issue in eval_label_issues:
                    print(f"  - {issue}", file=sys.stderr)
                return 1

    if args.close_on_reject and args.pr_number is not None and report.get("label") == "training:REJECT":
        close_issues = close_training_pr(
            args.pr_number,
            training_label="training:REJECT",
            issues=list(report.get("issues") or []),
        )
        if close_issues:
            for issue in close_issues:
                print(f"  - {issue}", file=sys.stderr)
            return 1
        print(f"closed PR #{args.pr_number}", file=sys.stderr)

    eval_label = report.get("eval_label")
    if (
        args.close_on_reject
        and args.pr_number is not None
        and report.get("label") == "training:valid"
        and eval_label in _AUTO_CLOSE_EVAL_LABELS
    ):
        close_issues = close_training_pr(
            args.pr_number,
            eval_label=eval_label,
            issues=list(report.get("issues") or []),
        )
        if close_issues:
            for issue in close_issues:
                print(f"  - {issue}", file=sys.stderr)
            return 1
        print(f"closed PR #{args.pr_number}", file=sys.stderr)

    # Auto-merge a fully verified, reward-earning training PR (the ledger + frontier
    # update run post-merge via training_track_ledger.yml). Mirrors the dataset gate's
    # --merge-on-pass so the training track isn't left waiting on a manual merge.
    if args.merge_on_pass and report.get("merge_eligible") and args.pr_number is not None:
        merge = subprocess.run(
            ["gh", "pr", "merge", str(args.pr_number), "--merge"],
            capture_output=True,
            text=True,
            check=False,
        )
        if merge.returncode != 0:
            print(merge.stderr or merge.stdout, file=sys.stderr)
            return 1
        print(f"merged PR #{args.pr_number}", file=sys.stderr)
        # A GITHUB_TOKEN merge raises no events, so training_track_ledger.yml never
        # fires for an auto-merge (#309). Signal the workflow to record the ledger and
        # crown the frontier in this same job, right after the merge.
        _emit_step_output("merged", "true")

    return 0 if report.get("verified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
