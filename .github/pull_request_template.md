# Spark-Hermes contribution

> **Not a training or dataset submission?** Community PRs that don't check one of the
> two boxes below are automatically closed — please open an **issue** instead (bug
> report, idea, or harness/tooling/docs suggestion). See `CONTRIBUTING.md`.

## Track

Select the one track this PR belongs to.

- [ ] **Dataset track submission**
- [ ] **Training/evaluation improvement**

> Dataset PRs must append exactly one line to `datasets/registry.jsonl` and must
> not change any other file. The dataset workflow reads the checked box above,
> verifies the Hugging Face `proof/` bundle, assigns a `dataset:*` label, and
> merges only submissions with at least 25 verified rows (`dataset:xs` or above).

## Dataset submission

Complete this section only when **Dataset track submission** is checked.

- Hugging Face dataset URL:
- Verified row count:
- `trajectories_sha256`:
- SparkProof dataset version:
- GPU architecture used to generate/validate the dataset (`blackwell` or `hopper`):

Registry line:

```json
{"miner": "<github-handle>", "hf_url": "https://huggingface.co/datasets/<org>/<repo>", "trajectories_sha256": "<64-character hash from dataset_manifest.json>", "rows_total": 25, "dataset_version": "triton-distill-v0.2", "gpu_architecture": "blackwell"}
```

### Dataset checklist

- [ ] I generated this dataset with an unmodified, pinned SparkProof checkout.
- [ ] The release gate and production `sparkproof-verify` pass (including `gpu_attestation.tdx` on TDX guests).
- [ ] The Hugging Face repository contains the complete `proof/` directory.
- [ ] The submitted rows are training data, not `test`, `eval`, or `held_out` data.
- [ ] The dataset does not contain TritonBench or other protected evaluation material.
- [ ] I understand that fewer than 25 verified rows receives `dataset:none` and is not merged.

## Training/evaluation improvement

Complete this section only when **Training/evaluation improvement** is checked.

Training-track PRs compete on **recipe and hyperparameter changes only**. Every miner
trains on the same pinned canonical dataset — no local generators, private blends, or
alternate `data/processed/` paths.

- Canonical dataset URL (required): `https://huggingface.co/datasets/gittensor-model-hub/sparkproof-mining`
- Pinned `sft_sha256` (required, cite the pin you trained on — any pin from merge-base through HEAD is accepted; see #121):
- Recipe changed:
- Frontier benchmark delta:
- Proof-bundle URL (required): `https://huggingface.co/<user>/sparkdistill-<run-id>`

### Training checklist

- [ ] My recipe uses only `data/processed/sparkproof-mining_sft.jsonl` (from `scripts/prepare_mining_sft.sh`).
- [ ] I did **not** add or modify `eval/gen_*.py`, `scripts/prepare_triton*.sh`, or `datasets/registry.jsonl`.
- [ ] New training rows were contributed through the **dataset track** first (SparkProof + registry PR).
- [ ] My PR cites a published Hugging Face proof-bundle URL with canonical `dataset_url` and `mix_manifest.json`.

CI applies `training:valid` or `training:REJECT` and may auto-close rejected training PRs.

## Summary

Explain what changed, why it should improve the student, and how you tested it.
