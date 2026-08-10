"""Validator check for a dataset-track submission (SparkProof bundle on Hugging Face).

The dataset track is the first half of the SN74 economy: a miner runs SparkProof on a
Blackwell CC VM, publishes the verified dataset to Hugging Face (rows + `proof/`
artifacts), and opens a text-only PR here linking that HF repo plus the bundle's
`trajectories_sha256`. The validator runs this tool, which checks:

1. The `proof/` artifacts exist in the HF repo (dataset_manifest.json from SparkProof's
   release gate, manifest.json, gpu_attestation.json, trajectories.jsonl, ...).
2. The GPU CC attestation passed — the data really came from an attested Blackwell node.
3. **Intel TDX** (production): `gpu_attestation.tdx` binds the measured VM to the same
   dataset nonce as NRAS — closes the userland-trust gap GPU attestation alone leaves.
4. The release gate passed (decontamination + provenance) and `trajectories.jsonl`
   matches the sha256 the release gate recorded, so the rows can't be swapped after
   gating.
5. Re-runs full production `sparkproof-verify` (pinned generator, pinned teachers,
   raw/verified consistency, merkle, attestation nonce) when a SparkProof checkout is
   available — required for merge.
6. Sizes the dataset into a reward label from verified row count (bundle sizing).
   SparkDistill's registry gate then **downgrades to a fair label** from
   ``mix_manifest.components[].rows_selected`` after cross-registry dedupe — see
   ``eval.fair_dataset_label``.

   | label | rows |
   |---|---|
   | `dataset:xl` | >= 150 |
   | `dataset:l` | >= 100 |
   | `dataset:m` | >= 75 |
   | `dataset:s` | >= 50 |
   | `dataset:xs` | >= 25 |
   | `dataset:none` | < 25 (proof may verify, but below merge/reward threshold) |
   | `dataset:REJECT` | any check above failed |

    python -m eval.dataset_verify --hf-repo <user>/sparkproof-triton-v0 \\
        [--claimed-sha256 <trajectories_sha256 from the PR>] \\
        [--sparkproof-root ../SparkProof] --out eval/results/dataset_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from eval.gpu_architecture import DEFAULT_GPU_ARCHITECTURE, dataset_architecture_allowed, normalize_gpu_architecture
from eval.hf_pin import download_with_optional_pin

# (min_rows, label) — first match wins, checked largest-to-smallest.
_SIZE_BANDS = [
    (150, "dataset:xl"),
    (100, "dataset:l"),
    (75, "dataset:m"),
    (50, "dataset:s"),
    (25, "dataset:xs"),
]
MERGE_THRESHOLD_ROWS = _SIZE_BANDS[-1][0]
REWARDED_DATASET_LABELS = frozenset(label for _, label in _SIZE_BANDS)

REQUIRED_PROOF_FILES = (
    "manifest.json",
    "dataset_manifest.json",
    "gpu_attestation.json",
    "trajectories.jsonl",
    "trajectories_raw.jsonl",
    "validation_report.jsonl",
    "prompts.jsonl",
)


def size_label(rows: int) -> str:
    for min_rows, label in _SIZE_BANDS:
        if rows >= min_rows:
            return label
    return "dataset:none"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_dataset_tdx_attestation(attestation: dict) -> list[str]:
    """Production dataset bundles must include TDX bound to the dataset nonce."""
    if "tdx" not in attestation:
        return []  # legacy bundle predating TDX requirement
    tdx = attestation.get("tdx")
    if not tdx:
        return [
            "gpu_attestation.tdx required for production dataset bundles — "
            "regenerate on an Intel TDX guest with configfs-tsm provisioned"
        ]
    nonce = attestation.get("nonce") or ""
    if not nonce:
        return ["gpu_attestation.tdx present but dataset nonce missing"]
    from eval.attestation import extract_report_data_from_quote, tdx_report_data

    quote_b64 = tdx.get("quote_b64") or ""
    if not quote_b64:
        return ["gpu_attestation.tdx missing quote_b64"]
    quote_report_data = extract_report_data_from_quote(quote_b64)
    if quote_report_data is None:
        return ["gpu_attestation.tdx quote_b64 is too short or unparseable to extract REPORTDATA"]
    expected = tdx_report_data(nonce).hex()
    if quote_report_data.lower() != expected:
        return ["gpu_attestation.tdx quote REPORTDATA does not match dataset attestation nonce"]
    json_report_data = str(tdx.get("report_data") or "").lower()
    if json_report_data and json_report_data != quote_report_data.lower():
        return ["gpu_attestation.tdx report_data JSON does not match REPORTDATA inside quote_b64"]
    return []


def check_proof_dir(proof_dir: Path, claimed_sha256: str | None = None) -> tuple[list[str], int, str | None]:
    """Return (issues, verified_row_count, gpu_architecture) for a bundle's proof directory."""
    issues: list[str] = []
    for name in REQUIRED_PROOF_FILES:
        if not (proof_dir / name).exists():
            issues.append(f"missing proof artifact: {name}")
    if issues:
        return issues, 0, None

    attestation = json.loads((proof_dir / "gpu_attestation.json").read_text())
    if not attestation.get("passed"):
        issues.append("gpu_attestation.passed is false")
    if not attestation.get("nonce"):
        issues.append("gpu_attestation.nonce missing — bundle predates content-bound attestation")
    issues.extend(_check_dataset_tdx_attestation(attestation))

    dataset_manifest = json.loads((proof_dir / "dataset_manifest.json").read_text())
    if not dataset_manifest.get("passed"):
        issues.append("release gate did not pass (dataset_manifest.passed is false)")
    if dataset_manifest.get("blocked_rows"):
        issues.append(f"release gate blocked {dataset_manifest['blocked_rows']} rows")

    actual_sha = _sha256_file(proof_dir / "trajectories.jsonl")
    gated_sha = dataset_manifest.get("trajectories_sha256")
    if gated_sha and actual_sha != gated_sha:
        issues.append("trajectories.jsonl sha256 does not match dataset_manifest — rows changed after release gate")
    if claimed_sha256 and actual_sha != claimed_sha256:
        issues.append("trajectories.jsonl sha256 does not match the hash claimed in the PR")

    rows = int(dataset_manifest.get("rows_total") or 0)
    actual_rows = sum(1 for line in (proof_dir / "trajectories.jsonl").read_text().splitlines() if line.strip())
    if actual_rows != rows:
        issues.append(f"rows_total mismatch: manifest={rows} trajectories.jsonl={actual_rows}")

    novelty_path = proof_dir / "novelty_report.json"
    if not novelty_path.exists():
        issues.append("missing novelty_report.json from release gate")

    raw_gpu_architecture = dataset_manifest.get("gpu_architecture")
    if raw_gpu_architecture is None:
        # Legacy bundle predating Hopper support (field didn't exist yet) —
        # every dataset accepted before this field existed was Blackwell-only,
        # so defaulting here doesn't let anything actually unsupported through.
        gpu_architecture = DEFAULT_GPU_ARCHITECTURE
    else:
        gpu_architecture = normalize_gpu_architecture(raw_gpu_architecture)
        if gpu_architecture is None:
            issues.append(f"dataset_manifest.gpu_architecture {raw_gpu_architecture!r} is not recognized")
    if gpu_architecture is not None and not dataset_architecture_allowed(gpu_architecture):
        issues.append(f"gpu_architecture {gpu_architecture!r} is not an accepted dataset-generation architecture")

    return issues, rows, gpu_architecture


def run_sparkproof_verify(proof_dir: Path, sparkproof_root: Path, *, production: bool = True) -> list[str]:
    """Re-run full SparkProof policy verification via the sibling SparkProof checkout.

    --online activates the cryptographic trust anchors: without it, the stored
    NRAS attestation token's NVIDIA signature is never verified and the gate
    would accept a hand-written gpu_attestation.json. Requires network access
    to NVIDIA's JWKS (production gates run in CI, which has it).
    """
    cmd = ["uv", "run", "sparkproof-verify", "--bundle", str(proof_dir), "--online"]
    if not production:
        cmd.append("--dev")
    result = subprocess.run(
        cmd,
        cwd=sparkproof_root,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        tail = (result.stdout + result.stderr).strip().splitlines()[-8:]
        return [f"sparkproof-verify failed: {' | '.join(tail)}"]
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    if not report.get("verified"):
        return list(report.get("issues") or ["sparkproof-verify returned verified=false"])
    return []


def verify_dataset_submission(
    proof_dir: Path | None = None,
    *,
    claimed_sha256: str | None = None,
    sparkproof_root: Path | None = None,
    hf_repo: str | None = None,
    hf_revision: str | None = None,
    production: bool = True,
) -> dict:
    if proof_dir is None:
        proof_dir = _resolve_proof_dir(hf_repo, None, hf_revision)
    issues, rows, gpu_architecture = check_proof_dir(proof_dir, claimed_sha256)
    if sparkproof_root is None:
        issues.append("sparkproof-root is required for production dataset verification")
    elif not issues:
        issues.extend(run_sparkproof_verify(proof_dir, sparkproof_root, production=production))

    label = "dataset:REJECT" if issues else size_label(rows)
    return {
        "verified": not issues,
        "label": label,
        "rows_total": rows,
        "gpu_architecture": gpu_architecture,
        "issues": issues,
    }


def _resolve_proof_dir(hf_repo: str | None, proof_path: Path | None, revision: str | None = None) -> Path:
    if proof_path is not None:
        return proof_path
    if hf_repo is None:
        raise ValueError("one of --hf-repo or --proof-path is required")
    from huggingface_hub import snapshot_download

    snapshot = Path(
        # Same as mix_registry: the live dataset registry predates the field, so a
        # revision is used when supplied and not demanded when it is not.
        download_with_optional_pin(
            snapshot_download,
            revision=revision,
            repo_id=hf_repo,
            repo_type="dataset",
            allow_patterns=["proof/*"],
        )
    )
    return snapshot / "proof"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-repo", default=None, help="HF datasets repo id from the miner's PR")
    parser.add_argument(
        "--hf-revision",
        default=None,
        help="exact 40-char commit sha of --hf-repo; a branch name is not a pin",
    )
    parser.add_argument(
        "--proof-path", type=Path, default=None, help="local proof/bundle dir (alternative to --hf-repo)"
    )
    parser.add_argument("--claimed-sha256", default=None, help="trajectories_sha256 claimed in the PR text")
    parser.add_argument(
        "--sparkproof-root",
        type=Path,
        required=True,
        help="path to a SparkProof checkout to re-run full production policy verification",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    proof_dir = _resolve_proof_dir(args.hf_repo, args.proof_path, args.hf_revision)
    report = verify_dataset_submission(
        proof_dir,
        claimed_sha256=args.claimed_sha256,
        sparkproof_root=args.sparkproof_root,
        hf_repo=args.hf_repo,
        production=True,
    )
    if args.hf_repo:
        report["hf_repo"] = args.hf_repo

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"{report['label']} (rows={report['rows_total']}, verified={report['verified']})", file=sys.stderr)
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
