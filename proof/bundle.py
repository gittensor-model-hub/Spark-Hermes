"""Assemble a proof-of-training bundle: the claim, not the weights.

A bundle holds the eval scores, the training claims, and a per-file sha256
manifest of the checkpoint — NOT the checkpoint itself. Trained weights are
never the artifact of a submission (see README): the validator reproduces the
checkpoint locally from the recipe + dataset and verifies against the recorded
hashes and scores, instead of downloading multi-GB weights from Hugging Face.
Pass `--include-checkpoint` only for legacy full-weight bundles.

The printed `claim_sha256` binds the whole claim: pass it as the attestation
nonce (`python -m eval.attestation --nonce <claim_sha256>`) so the NRAS-signed
EAT cryptographically commits this exact bundle to the attested GPU. An
attestation result belongs in the PR's `runs/<run-id>/` record via
`eval.ledger`, not inside the published bundle.

    python -m proof.bundle --checkpoint outputs/qwen3.5-4b-phase1 --scores eval/results/candidate.json --run-id 2026-07-09-qwen3.5-4b-001 --out proof/_bundles/2026-07-09-qwen3.5-4b-001
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from eval.attested_samples import ATTESTED_SAMPLES_FILENAME
from eval.dataset_verify import _sha256_file
from eval.regression_sample import REGRESSION_SAMPLE_FILENAME


@dataclass(frozen=True)
class ProofBundle:
    run_id: str
    bundle_dir: Path
    base_model: str
    created_at: str
    claim_sha256: str = ""


def checkpoint_manifest(checkpoint_dir: Path) -> dict[str, str]:
    """Per-file sha256 of a checkpoint directory (relative path -> digest)."""
    return {
        path.relative_to(checkpoint_dir).as_posix(): _sha256_file(path)
        for path in sorted(checkpoint_dir.rglob("*"))
        if path.is_file()
    }


def claim_sha256(bundle_dir: Path) -> str:
    """Digest binding the bundle's claim files — used as the attestation nonce.

    Covers eval_scores.json, manifest.json, and attested eval sample files when
    present, so attested artifacts cannot change after GPU + TDX attestation.
    """
    digest = hashlib.sha256()
    for name in ("eval_scores.json", "manifest.json", ATTESTED_SAMPLES_FILENAME, REGRESSION_SAMPLE_FILENAME):
        path = bundle_dir / name
        if path.exists():
            digest.update(_sha256_file(path).encode())
    return digest.hexdigest()


def build_bundle(
    checkpoint_dir: Path,
    scores_path: Path,
    out_dir: Path,
    run_id: str,
    base_model: str,
    train_hours: float | None = None,
    train_gpu: str | None = None,
    dataset_url: str | None = None,
    mix_manifest: Path | None = None,
    attested_eval_samples: Path | None = None,
    gsm8k_regression_sample: Path | None = None,
    include_checkpoint: bool = False,
) -> ProofBundle:
    """Record `checkpoint_dir`'s hashes and `scores_path`'s scores into `out_dir`.

    The checkpoint itself stays local: the manifest carries a per-file sha256
    manifest so a validator's locally reproduced checkpoint can be compared,
    without anyone shipping weights. `include_checkpoint=True` restores the
    legacy full-weight copy. `out_dir` is created fresh; call with an out_dir
    that doesn't already hold an unrelated bundle. `train_hours`/`train_gpu`/
    `dataset_url` are the training-track claims (see docs/miner-guide.md);
    `eval.verify` enforces the wall-clock budget and GPU requirement against them.
    """
    # Fail loudly on a missing/typo'd checkpoint path. On the default proof-only
    # path nothing else reads the checkpoint contents, so without this check a
    # wrong --checkpoint yields an *empty* checkpoint_manifest and a vacuous
    # claim_sha256 that still looks well-formed.
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(
            f"checkpoint directory {checkpoint_dir} does not exist; refusing to build a "
            "proof bundle whose checkpoint_manifest would be empty"
        )
    # Reject an empty checkpoint *before* creating out_dir or writing any bundle file, so a
    # rejected build leaves nothing behind — matching the missing-checkpoint path above.
    # Computing the manifest after out_dir.mkdir / the eval_scores.json write leaked a partial,
    # manifest-less bundle on rejection.
    ckpt_manifest = checkpoint_manifest(checkpoint_dir)
    if not ckpt_manifest:
        raise ValueError(
            f"checkpoint directory {checkpoint_dir} contains no files; the checkpoint_manifest "
            "would be empty and the proof claim vacuous"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    if include_checkpoint:
        shutil.copytree(checkpoint_dir, out_dir / "checkpoint", dirs_exist_ok=True)

    scores = json.loads(scores_path.read_text())
    (out_dir / "eval_scores.json").write_text(json.dumps(scores, indent=2))

    created_at = datetime.now(UTC).isoformat()
    manifest: dict = {"run_id": run_id, "base_model": base_model, "created_at": created_at}
    manifest["checkpoint_manifest"] = ckpt_manifest
    if train_hours is not None:
        manifest["train_hours"] = train_hours
    if train_gpu is not None:
        manifest["train_gpu"] = train_gpu
    if dataset_url is not None:
        manifest["dataset_url"] = dataset_url
    if mix_manifest is not None:
        if not mix_manifest.exists():
            raise FileNotFoundError(mix_manifest)
        shutil.copy(mix_manifest, out_dir / "mix_manifest.json")
        manifest["mix_manifest_sha256"] = _sha256_file(out_dir / "mix_manifest.json")
        mix_data = json.loads(mix_manifest.read_text(encoding="utf-8"))
        manifest["mix_id"] = mix_data.get("mix_id")
        manifest["mix_rows_total"] = mix_data.get("rows_total")
        manifest["mix_component_count"] = len(mix_data.get("components") or [])
    if attested_eval_samples is not None:
        if not attested_eval_samples.exists():
            raise FileNotFoundError(attested_eval_samples)
        shutil.copy(attested_eval_samples, out_dir / ATTESTED_SAMPLES_FILENAME)
        manifest["attested_eval_samples_sha256"] = _sha256_file(out_dir / ATTESTED_SAMPLES_FILENAME)
        sample_doc = json.loads(attested_eval_samples.read_text(encoding="utf-8"))
        manifest["attested_eval_benchmarks"] = sorted((sample_doc.get("benchmarks") or {}).keys())
    if gsm8k_regression_sample is not None:
        if not gsm8k_regression_sample.exists():
            raise FileNotFoundError(gsm8k_regression_sample)
        shutil.copy(gsm8k_regression_sample, out_dir / REGRESSION_SAMPLE_FILENAME)
        manifest["gsm8k_regression_sample_sha256"] = _sha256_file(out_dir / REGRESSION_SAMPLE_FILENAME)
        sample_data = json.loads(gsm8k_regression_sample.read_text(encoding="utf-8"))
        manifest["gsm8k_regression_exact_match"] = sample_data.get("exact_match")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return ProofBundle(
        run_id=run_id,
        bundle_dir=out_dir,
        base_model=base_model,
        created_at=created_at,
        claim_sha256=claim_sha256(out_dir),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True, help="scores json from eval.harness")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--train-hours", type=float, default=None, help="claimed wall-clock training time (budget: 5h)")
    parser.add_argument("--train-gpu", default=None, help="claimed training GPU, e.g. 'NVIDIA RTX PRO 6000 Blackwell'")
    parser.add_argument("--dataset-url", default=None, help="HF datasets URL the checkpoint was trained on")
    parser.add_argument(
        "--mix-manifest",
        type=Path,
        default=None,
        help="committed mix_manifest.json from eval.mix_registry (cross-miner training mix)",
    )
    parser.add_argument(
        "--attested-eval-samples",
        type=Path,
        default=None,
        help="attested eval artifacts from eval.export_attested_samples (GPU+TDX no-verify path)",
    )
    parser.add_argument(
        "--gsm8k-regression-sample",
        type=Path,
        default=None,
        help="legacy gsm8k-only sample from eval.export_gsm8k_regression_sample",
    )
    parser.add_argument(
        "--include-checkpoint",
        action="store_true",
        help="legacy: also copy the full checkpoint weights into the bundle",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    bundle = build_bundle(
        args.checkpoint,
        args.scores,
        args.out,
        args.run_id,
        args.base_model,
        train_hours=args.train_hours,
        train_gpu=args.train_gpu,
        dataset_url=args.dataset_url,
        mix_manifest=args.mix_manifest,
        attested_eval_samples=args.attested_eval_samples,
        gsm8k_regression_sample=args.gsm8k_regression_sample,
        include_checkpoint=args.include_checkpoint,
    )
    print(f"wrote proof bundle {bundle.run_id} to {bundle.bundle_dir}", file=sys.stderr)
    print(f"claim_sha256 (use as attestation nonce): {bundle.claim_sha256}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
