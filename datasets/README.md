# `datasets/`

The in-repo registry of every verified SparkProof dataset that was merged — the
dataset-track counterpart of `runs/` (which records proof-of-training runs).

> **This is the Triton distillation track, not the Spark-Hermes rollout competition.**
> The product direction is the loop described in the [root README](../README.md): the
> frozen model attempts real tasks, miners optimise the rollout, and verified improvements
> train the next checkpoint. This registry and its gate remain live and unchanged — 23
> submissions merged under it and the corpus they compose is what every training-track
> bundle is still required to cite — but new work belongs to the rollout track.
>
> Nothing here is rewritten when the direction changes. A ledger of paid work is history,
> and history that gets edited to match the current plan is not evidence of anything.

## How a dataset gets in here (miner flow)

1. Run SparkProof on a Blackwell or Hopper H100/H200 CC VM and pass the release gate
   (`sparkproof-publish-dataset` refuses to publish otherwise). SparkProof detects the
   GPU automatically and stamps prompts, mutation/failure-mining templates, and the
   dataset manifest with the matching architecture — an unsupported GPU (Ampere, Ada,
   ...) fails fast before any generation starts.
2. Publish to Hugging Face. The publisher uploads the dataset rows **and** the proof
   artifacts under `proof/` in the same HF repo (`manifest.json`,
   `dataset_manifest.json`, `gpu_attestation.json` with optional `tdx` Intel quote,
   `trajectories.jsonl`, ...).
3. Build the registry line and open a **text-only PR** that appends it to
   `datasets/registry.jsonl`:

```bash
scripts/registry_line.sh --bundle <sparkproof-bundle-dir> --miner <github-handle> \
  --repo-id <user>/<repo> --append datasets/registry.jsonl
```

   In the PR template, check **Dataset track submission**. Dataset PRs may not modify any
   other file:

```json
{"miner": "<github-handle>", "hf_url": "https://huggingface.co/datasets/<user>/<repo>", "trajectories_sha256": "<from dataset_manifest.json>", "rows_total": 128, "dataset_version": "triton-distill-v0.2", "gpu_architecture": "blackwell"}
```

`gpu_architecture` is `"blackwell"` or `"hopper"` (`scripts/registry_line.sh` reads it
straight from the bundle's `dataset_manifest.json` — no need to set it by hand). The
gate cross-checks the claimed value against what the re-verified bundle actually
attested; a mismatch is `dataset:REJECT`.

No dataset files are committed here — the PR is the link plus the hash that pins the
exact gated rows.

### Intel TDX (production required on new bundles)

GPU CC attestation proves the GPU; **Intel TDX** proves the measured VM that ran
SparkProof. Provision configfs-tsm once per boot on TDX guests before `sparkproof-prove`:

```bash
sudo chmod 0777 /sys/kernel/config/tsm/report
mkdir /sys/kernel/config/tsm/report/sparkproof
sudo chmod 0666 /sys/kernel/config/tsm/report/sparkproof/inblob
export SPARKPROOF_TSM_REPORT_PATH=/sys/kernel/config/tsm/report/sparkproof
```

`gpu_attestation.json` then includes `tdx` with `quote_b64` and `report_data` bound to
the dataset nonce. Bundles merged **before** #122
lack a `tdx` key and remain grandfathered; republish with TDX for strongest trust.

## What the validator does

Registry PRs are gated automatically by `.github/workflows/dataset_registry.yml`.
The workflow reads the dataset-track checkbox, rejects changes outside
`datasets/registry.jsonl`, verifies the proof, **aggregates every merged registry line
(including the proposed submission) into the canonical mining dataset on Hugging Face**
(default: [`gittensor-model-hub/sparkproof-mining`](https://huggingface.co/datasets/gittensor-model-hub/sparkproof-mining)),
replaces any stale `dataset:*` label with the computed result, and merges only when
verification, aggregation publish, and the `dataset:xs` threshold all pass.
Rejected PRs are labeled `dataset:REJECT` and closed automatically. Sub-threshold
valid proofs (`dataset:none`) are also closed automatically.

The gate runs `eval.registry_gate`, which for each appended registry line:

1. Validates JSON schema and rejects duplicate `hf_url` / `trajectories_sha256`.
2. Downloads `proof/` from Hugging Face.
3. Runs `eval.dataset_verify` with a pinned SparkProof checkout.

`dataset_verify` checks, in order: required proof artifacts (including
`trajectories_raw.jsonl`, `validation_report.jsonl`, `novelty_report.json`);
GPU CC attestation passed with a content-bound nonce; **Intel TDX** (`gpu_attestation.tdx`
`report_data` bound to that nonce — required on new bundles; legacy entries without a
`tdx` key are grandfathered); release gate passed and rows
still match the gated sha256; `dataset_manifest.gpu_architecture` is `blackwell` or
`hopper` (anything else is `dataset:REJECT`); and full production `sparkproof-verify`
(pinned generator, Fable 5 / GPT 5.6 Sol at `xhigh`, raw→verified consistency, merkle,
attestation nonce). The registry gate then cross-checks the PR's claimed
`gpu_architecture` against this re-verified value — a mismatch rejects the same way a
`rows_total` mismatch does. Any failure is `dataset:REJECT` and the PR is not merged.

Manual re-check:

```bash
python -m eval.dataset_verify --hf-repo <user>/<repo> \
    --claimed-sha256 <trajectories_sha256 from the PR> \
    --sparkproof-root ../SparkProof --out eval/results/dataset_report.json
```

| label | verified rows |
|---|---|
| `dataset:xl` | >= 150 |
| `dataset:l` | >= 100 |
| `dataset:m` | >= 75 |
| `dataset:s` | >= 50 |
| `dataset:xs` | >= 25 |
| `dataset:none` | < 25 (proof may be valid, but not merged/rewarded) |
| `dataset:REJECT` | attestation, release-gate, hash, or policy failure |

**SN74 payout:** `fixed_base_score (1.0) × label_multiplier`. Training-track `eval:*`
tiers pay **2×** the `dataset:*` multiplier at the same letter (e.g. `dataset:l` = 2.5,
`eval:L` = 5.0). See [`.gittensor/weights.json`](../.gittensor/weights.json).

Merged datasets feed the **single canonical mining dataset** used by every training-track
PR. After each registry merge, CI refreshes [`canonical.json`](canonical.json) with the
pinned `mix_manifest.sft_sha256` and row count.

## Canonical mining dataset (training track)

**HF repo:** [`gittensor-model-hub/sparkproof-mining`](https://huggingface.co/datasets/gittensor-model-hub/sparkproof-mining)

**In-repo pin:** [`canonical.json`](canonical.json) — `repo_id`, `hf_url`,
`training_dataset_path`, and `mix_manifest.sft_sha256`.

**Training-track pin grace ([#121]):** dataset merges can advance the pin while a miner
is still training. The training gate accepts proof bundles whose `sft_sha256` matches
any canonical pin from the PR merge-base through `main` HEAD (not only live HEAD).
Miners cite the pin they trained on in the PR body; stale pins outside that window reject.

Before a registry PR merges, CI:

1. Verifies the miner's `proof/` bundle
2. Mixes **all** registry lines (existing + proposed) with deduplication
3. Publishes the union to the mining dataset repo (`train` split + `mix_manifest.json`)
4. Merges the PR only if publish succeeds

Registry aggregation uses `SPARKDISTILL_MINING_DEDUPE` when republishing sparkproof-mining
(default **`exact`**). **Quality** is enforced by SparkProof before merge (release gate,
decontamination, GPU validation, `sparkproof-verify`). **Dedupe** only removes redundant
copies at mix time — `exact` drops identical prompts; the older `near` mode also dropped
structurally similar rows and shrank nghetienhiep's 161-row submission to 77 when mixed
with speedy00. Set `SPARKDISTILL_MINING_DEDUPE=none` only for local debugging.

**Fair reward labels:** after mixing, the gate labels each submission from
`mix_manifest.components[].rows_selected` (canonical contribution), not raw bundle
`rows_total`. A 159-row bundle that only adds 25 novel mix rows earns `dataset:xs`, not
`dataset:xl`. **Exact dedupe is architecture-scoped:** the same prompt on Blackwell vs
Hopper counts as a fresh row (not a duplicate).

**Miner-side dedupe prevention:** after each registry merge, CI publishes
`accepted_registry_snapshot.jsonl` on the canonical mining HF repo and pins
`accepted_registry_snapshot_sha256` in `mix_manifest.json`. SparkProof **v0.1.2+**
downloads and verifies that snapshot automatically:

```bash
sparkproof-publish-dataset --bundle <dir> --repo-id <you>/<repo> --release-gate --mining-repo
```

Or download first via `scripts/download_registry_snapshot.sh` (SparkProof repo) and pass
`--registry-snapshot`. See [SparkProof `docs/MINER_GUIDE.md`](https://github.com/gittensor-model-hub/SparkProof/blob/main/docs/MINER_GUIDE.md).
The registry gate recomputes and checks the pin after every merge.

After each eligible registry merge, CI refreshes [`canonical.json`](canonical.json) from
the live HF `mix_manifest.json` (also triggered by `.github/workflows/update_canonical_pin.yml`).

**Training-track rule:** recipes must use `data/processed/sparkproof-mining_sft.jsonl`
only. Prepare with `scripts/prepare_mining_sft.sh` (verifies HF matches `canonical.json` and
writes `data/processed/mix_manifest.json` for `proof.bundle --mix-manifest`).
PRs that add local generators, private blends, or alternate recipe paths are rejected by
`eval.training_track_gate`. Proof bundles must set `dataset_url` to the canonical `hf_url`.

Override the mining publish target with `SPARKDISTILL_MINING_DATASET_REPO` in CI or locally.

Local dry-run without HF upload:

```bash
uv run python -m eval.registry_gate ... --skip-mining-publish
```

Refresh the pin manually after a mining republish:

```bash
scripts/update_canonical_pin.sh
```

## Cross-miner mixing (maintainer / registry CI only)

Registry aggregation uses `scripts/mix_registry.sh` internally. **Training miners must not**
build private mixes for competition PRs — train on the canonical snapshot above instead.

- **`registry.jsonl`** — append-only, one line per merged dataset PR. Never edited or
  reordered; corrections are appended, not rewritten (same convention as
  `runs/ledger.jsonl`).

## Verified smoke test (2026-07-11)

End-to-end run on a Blackwell RTX PRO 6000 CC VM (`ssh -p 20004 ubuntu@<host>`):

> **Maintainer-only flow.** The vendored TritonBench tree this path depends on was never part
> of the public `Spark-Hermes` repository. It lives in the pre-rename archive
> (`Spark-Hermes-3.8-27B`, private) at `archive/tritonbench`, so the commands below are
> reproducible only with access to that archive. Directory names are kept as written because
> they match the maintainer checkout layout.

```bash
# SparkProof on the CC VM (sibling SparkDistill required for decontamination + SFT)
cd SparkProof
# .env: YUNWU_API_KEY or OPENROUTER_API_KEY, HF_TOKEN (org write access)
# SparkDistill/tritonbench must exist (gitignored — rsync or clone beside SparkProof)

scripts/run_triton_pipeline.sh \
  --run-id triton-cc-hf-001 \
  --limit 2 \
  --release-gate \
  --publish gittensor-model-hub/sparkproof-triton-v0
```

**Published:** [gittensor-model-hub/sparkproof-triton-v0](https://huggingface.co/datasets/gittensor-model-hub/sparkproof-triton-v0)

| check | result |
|---|---|
| rows published | 2 (both silver tier) |
| duplicate prompts / task_ids / responses | none — `api_tl_tensor`, `api_tl_tensor_descriptor` |
| release gate | `passed: true`, `blocked_rows: 0` |
| `trajectories_sha256` | `a746fa812fb098737cded713daf0f58b8ff59e485c9bdf8fd94f6b5cc1d5c846` |
| `proof/` artifacts on HF | yes (`manifest.json`, `dataset_manifest.json`, `gpu_attestation.json`, `trajectories.jsonl`, ...) |

**Validator re-check** (any machine with SparkProof + SparkDistill checkouts):

```bash
cd SparkDistill
python -m eval.dataset_verify \
  --hf-repo gittensor-model-hub/sparkproof-triton-v0 \
  --claimed-sha256 a746fa812fb098737cded713daf0f58b8ff59e485c9bdf8fd94f6b5cc1d5c846 \
  --sparkproof-root ../SparkProof \
  --out eval/results/dataset_report.json
# → verified=true, label=dataset:none (2 rows < 25 reward threshold)
```

**CC VM gotchas observed during the smoke test:**

- SSH port can change when the VM is reprovisioned (e.g. `20004` not `20002`).
- `SparkDistill/tritonbench/` is gitignored — decontamination and the release gate fail
  without it (`decontamination requires a TritonBench problem corpus`). Sync from a dev
  machine: `rsync -az SparkDistill/tritonbench/ ubuntu@<host>:~/SparkDistill/tritonbench/`.
- `HF_TOKEN` must be in SparkProof `.env` with write access to the target org/repo.

