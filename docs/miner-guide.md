# SparkDistill Miner Guide

This guide is for SN74 miners and contributors who want to earn rewards by improving
SparkDistill. The rule is simple: rewards come from verified improvements, not from
claims, formatting, or duplicated ideas.

> **Scope.** This guide covers the **live** SN74 mining tracks, which are unchanged: the
> dataset track and the training track, both scored on reasoning/Triton distillation. The
> Hermes-native agent work described in
> [`docs/roadmap-hermes.md`](roadmap-hermes.md) is the project's forward direction and is
> **not yet a mining track** — nothing in `hermes/` or `hermesbench/` is scored or
> rewarded today. Mine against what is documented below.

There are **two mining tracks**, each with its own labels and rewards:

1. **Dataset track** (`dataset:xs/s/m/l/xl`) — run [SparkProof](https://github.com/gittensor-model-hub/SparkProof)
   on a Blackwell or Hopper H100/H200 CC VM to generate verified Triton training data,
   publish it to Hugging Face, and open a text-only registry PR here. See *Dataset
   Track* below.
2. **Training track** (`eval:xs/s/m/l/xl`) — train the student on the **pinned canonical
   mining dataset** with an improved recipe, beat the frontier eval, and prove it (RTX PRO
   6000 CC attestation, ≤ 5h wall-clock). The rest of this guide covers this track.

The same person can mine both: contribute new rows on the dataset track, then compete on
the training track once those rows are merged into the canonical pin. Every training PR uses
the same `sparkproof-mining` snapshot — fair comparison is by recipe quality, not private
data.

### SN74 payout multipliers (Gittensor)

Live payout is `fixed_base_score (1.0) × label_multiplier × time_decay` from
[`master_repositories.json`](https://github.com/entrius/gittensor/blob/test/gittensor/validator/weights/master_repositories.json)
(`gittensor-model-hub/SparkDistill-Hermes`). **Training-track `eval:*` tiers pay 2× the
`dataset:*` tier at the same letter** — a verified frontier win is weighted higher than
adding training rows at the same size band:

| tier | `dataset:*` | `eval:*` |
|---|---|---|
| XL | 4.0 | **8.0** |
| L | 2.5 | **5.0** |
| M | 1.5 | **3.0** |
| S | 1.0 | **2.0** |
| XS | 0.5 | **1.0** |
| BASELINE | — | **2.0** |

`eval:BASELINE` (first verified checkpoint on a new student/phase) pays **2.0**.
`eval:none`, `dataset:none`, and `*:REJECT` pay **0**.

Reference copy: [`.gittensor/weights.json`](../.gittensor/weights.json).

## Dataset Track (`dataset:xs/s/m/l/xl`)

The full flow, end to end:

1. **Generate** on a Blackwell RTX PRO 6000 or Hopper H100/H200 **Intel TDX** CC VM
   with an unmodified SparkProof checkout: `scripts/run_triton_pipeline.sh` (teacher calls go through an
   approved gateway to the pinned teachers — GPT 5.6 Sol / Fable 5 at `xhigh` — and every
   kernel is compiled and executed on the GPU that's actually present; SparkProof detects
   Blackwell vs. Hopper automatically and rejects anything else before generation starts).
2. **Publish** with `sparkproof-publish-dataset --bundle <dir> --repo-id <you>/<repo>`.
   The release gate (decontamination + provenance) must pass; the publisher uploads the
   dataset rows **and** the proof artifacts under `proof/` in the same HF repo.
3. **Open a text-only PR** here appending one line to `datasets/registry.jsonl` — build it
   with `scripts/registry_line.sh --bundle <dir> --miner <github-handle> --repo-id <you>/<repo>`
   (reads `trajectories_sha256` from `dataset_manifest.json`). See
   [`datasets/README.md`](../datasets/README.md). Check **Dataset track submission** in
   the PR template. No dataset files are committed, and a dataset PR may not change any
   file other than the registry.
4. **The validator** runs `python -m eval.dataset_verify`, then aggregates every registry
   line (including yours) into the canonical mining dataset on Hugging Face
   ([`gittensor-model-hub/sparkproof-mining`](https://huggingface.co/datasets/gittensor-model-hub/sparkproof-mining))
   **before** merging. Publish failure blocks merge. The workflow applies the computed
   `dataset:*` label and merges only at the `dataset:xs` threshold (25 rows) or above:

| label | verified rows |
|---|---|
| `dataset:xl` | >= 150 |
| `dataset:l` | >= 100 |
| `dataset:m` | >= 75 |
| `dataset:s` | >= 50 |
| `dataset:xs` | >= 25 |
| `dataset:none` | proof valid but below 25 rows; not merged or rewarded |
| `dataset:REJECT` | attestation, decontamination, hash, or policy failure |

**Verified smoke test (2026-07-11):** 2 rows published to
[gittensor-model-hub/sparkproof-triton-v0](https://huggingface.co/datasets/gittensor-model-hub/sparkproof-triton-v0),
`eval.dataset_verify` returned `verified=true` / `dataset:none` (below the 25-row
`dataset:xs` threshold). No duplicate prompts, task IDs, or responses. Full commands and
the pinned `trajectories_sha256` are in [`datasets/README.md`](../datasets/README.md).

### Avoid registry dedupe before you publish

SparkProof downloads and verifies the pinned snapshot for you (SparkProof v0.1.2+):

```bash
# Recommended: download + verify + publish in one step
uv sync --extra publish --frozen
sparkproof-publish-dataset \
  --bundle <dir> \
  --repo-id <you>/<repo> \
  --release-gate --mining-repo

# Or download first, then publish with an explicit path
scripts/download_registry_snapshot.sh --out-dir ./snapshots   # from SparkProof repo
sparkproof-publish-dataset \
  --bundle <dir> \
  --repo-id <you>/<repo> \
  --release-gate \
  --registry-snapshot ./snapshots/accepted_registry_snapshot.jsonl
```

See [SparkProof `docs/MINER_GUIDE.md`](https://github.com/gittensor-model-hub/SparkProof/blob/main/docs/MINER_GUIDE.md#avoid-registry-dedupe-surprises).

`novelty_report.json` will then include registry duplicates, not just
intra-bundle duplicates. Target `novel_verified_rows` ≥ 25 before publishing.
Pin the snapshot by comparing your file's sha256 to
`mix_manifest.accepted_registry_snapshot_sha256` on the canonical mining repo
(or `datasets/canonical.json` after the next pin refresh).
The lightweight `accepted_task_ids.json` on the same HF repo is useful for
pre-generation filtering so you do not burn GPU on tasks already accepted.

SparkDistill sizes rewards from canonical-mix `rows_selected` (fair label), so
miners should treat the snapshot as the source of truth for expected credit.

**Before you generate on a CC VM:** decontamination needs the TritonBench problem set. It
is **no longer in this repository** — it moved to the `archive/tritonbench` branch, so
check it out beside your working copy and point SparkProof at it:

```bash
git worktree add ../tritonbench archive/tritonbench
export SPARKPROOF_TRITONBENCH_PROBLEMS=../tritonbench/tritonbench
```

Without it, decontamination aborts — which is the correct direction to fail, since a
dataset that was never checked against the benchmark it will be scored on is not a
verified dataset. (Earlier revisions of this guide called the tree gitignored. It was
tracked, all 1,118 files of it.) Set `HF_TOKEN` in `.env` before `--publish`.

**Intel TDX (required on new dataset bundles):** provision configfs-tsm once per boot so
`sparkproof-prove` captures `gpu_attestation.tdx` bound to the dataset nonce — see
[`datasets/README.md`](../datasets/README.md#intel-tdx-production-required-on-new-bundles).
Without TDX, production verification records `"tdx": null` and rejects the bundle.

### Can miners submit an evaluation dataset?

**No — the current dataset track accepts verified training trajectories only.** There is
no miner evaluation-dataset track, registry schema, reward label, or merge path. A miner
can technically upload any files to a personal Hugging Face repository, but publishing a
repository does not make it an accepted SparkDistill dataset. A registry submission
containing eval material is expected to fail the release gate or production verification,
receive `dataset:REJECT`, and are closed automatically by CI. Sub-threshold valid
proofs (`dataset:none`) are also closed automatically.

This distinction is important because the word `eval` is used in two different ways:

- `dataset:xs/s/m/l/xl` rewards miners for contributing **training data** that passed
  SparkProof generation, GPU validation, decontamination, attestation, and registry
  verification.
- `eval:XS/S/M/L/XL` rewards a **training recipe or checkpoint improvement** measured by
  the validator on its held-out benchmark basket. It does not mean the miner contributed
  the benchmark or an evaluation dataset.

#### What miners may submit

Miners may create novel prompts, tasks, reference operations, mutations, and resulting
teacher trajectories, provided they are intended for training and pass every dataset
gate. Miner-generated does not automatically mean eval data. A novel task remains
eligible training data when, among other requirements:

- its metadata identifies a trainable origin and a `train` or `dev` split;
- it was generated by the pinned SparkProof version and pinned teachers;
- its kernel was validated on required CC hardware (Blackwell or Hopper H100/H200);
- it does not duplicate or structurally overlap a protected benchmark;
- it passes the release gate, novelty checks, and full production verification.

The accepted artifact is the verified trajectory dataset and its `proof/` directory. It
is not a proposed benchmark, answer key, evaluator implementation, or replacement
held-out split.

#### What is rejected as eval leakage

SparkProof applies multiple independent controls rather than trusting a miner-provided
filename or description:

1. **Origin policy.** Tasks whose origin/source is `tritonbench`,
   `kernelbench_eval`, `private_eval`, or `yaml` are forbidden from training generation.
2. **Split policy.** Rows marked `test`, `eval`, or `held_out` are blocked by the
   pre-publish release gate.
3. **Prompt fingerprints.** Exact normalized prompt matches against the loaded eval
   corpus are rejected.
4. **Semantic fingerprints.** Tasks matching protected combinations such as operation,
   target API, reference operation, dtype, shape class, and layout are rejected.
5. **Code-structure fingerprints.** Generated code is AST-canonicalized (user-defined
   names removed while operators, control flow, and Triton APIs are preserved) and
   compared with protected benchmark implementations. Renaming variables is therefore
   not enough to bypass decontamination.
6. **Required eval corpus.** Production decontamination aborts if the protected
   TritonBench corpus is unavailable; it does not silently continue without checking.
7. **Validator replay.** SparkDistill downloads `proof/` from Hugging Face, checks the
   release-gate result and pinned trajectory hash, and re-runs SparkProof production
   verification before the registry PR can merge.

Examples that are not accepted:

- copying, translating, paraphrasing, or mutating a TritonBench problem for training;
- labeling benchmark-derived rows as `train` to disguise their origin;
- publishing `eval`, `test`, or `held_out` rows through the dataset registry;
- submitting benchmark prompts, expected answers, private evaluator data, or an answer
  key as a rewardable dataset;
- editing rows after proving or replacing the Hugging Face dataset while retaining the
  old registry hash.

Some attempts are rejected during prompt construction, while others are caught during
decontamination, release gating, hash verification, or validator replay. Passing one
layer does not bypass the later layers.

#### Why miners cannot define the scored evaluation

Allowing a miner to submit both training data and the evaluation used to reward that
submission creates a direct conflict of interest. The miner could train on the proposed
questions, choose tasks tailored to its model, include answer-equivalent variants, or
design a metric that favors its own outputs. Public evaluation data also stops being
meaningfully held out once every miner can optimize against it.

For that reason, scoring uses the validator-controlled frozen benchmark basket and
held-out prompts. Miners may improve **evaluation tooling** through an ordinary code PR,
but such a change is reviewed as evaluator code and does not become a rewardable dataset;
documentation-, test-, or harness-only changes also do not receive an eval quality label
without a separately verified model improvement.

#### Could an evaluation-dataset track be added later?

Yes, but it would be a separate protocol, not an extension of
`datasets/registry.jsonl`. At minimum it would need independent benchmark governance,
private or commit-reveal test content, duplicate and contamination checks against both
training and existing evaluations, answer-key verification, difficulty and coverage
measurement, adversarial review, immutable versioning, and rules preventing the
submitter from being scored on its own contribution. None of those mechanisms currently
constitutes a supported miner submission path.

If your data is meant to train the student, submit it through the dataset track. If it is
meant to measure or rank students, do not publish it as a dataset-track contribution;
propose the evaluation design separately for maintainer review.

## Training Track: What Scores

A PR can score when it does all of the following:

- Includes the **recipe** used to produce the checkpoint and trains on the **pinned
  canonical mining dataset** only (`data/processed/sparkproof-mining_sft.jsonl` from
  [`datasets/canonical.json`](../datasets/canonical.json)). See *Sharing Your Dataset And
  Recipe* below. This is not optional: no canonical dataset + recipe, no score, no matter
  how good your local eval numbers look.
- Trains / regenerates trajectories from source on the evaluator's hardware.
- Preserves correctness against the frozen benchmark reference (no format breakage, no
  garbled outputs).
- Improves at least one benchmark in the basket by **the current eval threshold or more**.
- Avoids unacceptable regressions in the other guarded benchmarks.
- Changes code that is actually used by the training/eval path for the current phase.

The measured benchmarks are:

| benchmark | target |
|---|---|
| BFCL | function/tool-calling accuracy |
| GSM8K | grade-school math reasoning |
| HumanEval | code generation correctness |
| IFEval | instruction-following accuracy |
| MMLU-Pro | broad knowledge/reasoning |
| AIME | competition-level multi-step math reasoning |
| GPQA-Diamond | graduate-level science reasoning |
| TritonBench (`triton`) | Triton kernel expertise — generated kernels are compiled and executed on the GPU |

Small gains are not aggregated across benchmarks. A PR must clear the threshold on at
least one benchmark without dropping others below their floor.

**`triton` is the improvement signal for Triton-focused recipes** (the ones trained on
dataset-track data) — the general basket cannot measure kernel skill and only guards
against catastrophic forgetting there. It runs through the vendored TritonBench harness
instead of lm-eval: the student is served behind an OpenAI-compatible endpoint, each
generated kernel is compiled and executed on the GPU (Blackwell or Hopper H100/H200 —
`eval.triton_bench` records which one it ran on, since the composite is speed-derived
and hardware-sensitive), and the headline score is the average composite (correctness +
execution + API modernity). TritonBench problems
are quarantined from training data by SparkProof's release-gate decontamination, which
is what keeps this a legitimate held-out eval — a dataset row that structurally matches
a TritonBench problem is blocked before it can ever be trained on.

```bash
# serve + score in one step (requires vllm), or point --endpoint at a running server
uv run python -m eval.triton_bench --checkpoint outputs/qwen3.5-4b-phase1 --serve \
    --out eval/results/triton.json

# or as part of the basket
uv run python -m eval.harness --checkpoint outputs/qwen3.5-4b-phase1 --benchmark triton \
    --out eval/results/candidate.json
```

## What Does Not Score

These changes may be useful, but they do not earn a quality label unless they also
produce a verified frontier improvement:

- Documentation-only changes.
- Refactors with no benchmark improvement.
- Test-only changes.
- Eval harness changes that do not improve measured checkpoint quality.
- Copying an already-merged trajectory set or recipe without a new measurable improvement.
- Changes that improve one synthetic eval path but are unused by the phase's scoring target.

## Quality Gate

The evaluator compares your resulting checkpoint against the current frontier checkpoint
on the frozen benchmark basket. A PR is rejected if it degrades correctness too much on
any guarded benchmark, even if it improves another.

The gate checks:

- Per-benchmark accuracy vs. the frontier checkpoint.
- Held-out prompts not seen during trajectory generation or training.
- Stable, well-formed outputs (no truncation/format collapse from a bad recipe change).

Do not trade breadth for a single benchmark's score. Quality is measured across the whole
basket.

## Regression Labels

A PR can improve Triton and regress another benchmark. The bot makes this explicit with
benchmark-specific labels:

| label | meaning |
|---|---|
| `regression-bfcl` | BFCL accuracy regressed |
| `regression-gsm8k` | GSM8K accuracy regressed beyond its floor |
| `regression-humaneval` | HumanEval accuracy regressed |
| `regression-ifeval` | IFEval accuracy regressed |
| `regression-mmlu-pro` | MMLU-Pro accuracy regressed |
| `regression-aime24` | AIME accuracy regressed |
| `regression-gpqa-diamond` | GPQA-Diamond accuracy regressed |
| `regression-triton` | TritonBench composite regressed (Triton-track recipes) |

**GSM8K floor:** default **1%** max drop vs frontier. When Triton improves by **≥ 2%**
relative to frontier (enough to earn at least `eval:XS`), GSM8K may regress up to **2%**
instead — a deliberate trade for Triton-focused recipes.

**Frontier updates:** on any verified non-`eval:REJECT` run, the frontier is merged per
benchmark: any score that beats the current frontier high is updated — including GSM8K
when a miner improves math reasoning even if Triton is flat. Triton tiers still come only
from the Triton composite.

**Per-architecture frontiers:** Blackwell and Hopper each keep their own frontier bucket
in `runs/frontiers.json` (`eval.frontiers`) — TritonBench composites are hardware-sensitive,
so a Hopper run is only ever tiered against other Hopper runs, never against Blackwell's.
`eval.verify` resolves which bucket applies to a bundle from its manifest's
`gpu_architecture` (dataset track) or `train_gpu` (training track) claim; legacy bundles
with neither field default to Blackwell. Both architectures tier on Triton the same way —
`--frontier` still accepts an explicit flat scores file when you want to override the
resolved bucket.

If no benchmark improves by at least the eval threshold and any guarded benchmark
regresses, the PR is rejected and may be auto-closed.

## Quality Labels

The reward label (`eval:XS` through `eval:XL`) comes from **TritonBench only** — the
domain improvement signal for mining recipes. The general basket (GSM8K, BFCL,
HumanEval, …) is regression-guarded: any drop beyond its floor yields `eval:REJECT`
and a `regression-*` label, but **GSM8K/BFCL improvements alone do not earn a tier**.

Given the current frontiers (`runs/frontiers.json` → `blackwell` `triton ≈ 0.428` /
`gsm8k = 0.6`; `hopper` `triton ≈ 0.372` / `gsm8k = 0.74`), approximate **Blackwell
triton** scores needed when GSM8K holds at ≥ 0.588 (2% relaxed floor) or
≥ 0.594 (1% default floor):

| label | triton composite (approx.) | relative gain vs frontier |
|---|---|---|
| `eval:L` | ≥ 0.471 | +10% |
| `eval:M` | ≥ 0.454 | +6% |
| `eval:S` | ≥ 0.443 | +3.5% |
| `eval:XS` | ≥ 0.437 | +2% |

| label | meaning |
|---|---|
|---|---|
| `eval:XL` | very large verified quality improvement |
| `eval:L` | large verified quality improvement |
| `eval:M` | medium verified quality improvement |
| `eval:S` | small verified quality improvement |
| `eval:XS` | minimum accepted verified quality improvement |
| `eval:none` | correct, but no significant improvement |
| `eval:REJECT` | correctness failure, training failure, or unacceptable regression |

The exact label is deterministic from the evaluator output. The bot does not use AI
judgment to decide rewards.

## Sharing Your Dataset And Recipe (Required)

**No trained weights are ever merged.** What actually gets merged — and what the evaluator
actually trusts — is your **recipe (the Axolotl YAML) and the dataset it trained on**,
because those are what the evaluator reproduces from source to verify your claim. This is
also what makes the whole system fair: because the recipe and dataset behind the current
frontier are always public, anyone can fork the leader and try to beat it with one more
optimization. Nobody — including whoever currently holds the frontier ("the king") — can
permanently dominate by keeping a checkpoint secret; there's no way to merge a PR without
its recipe and dataset becoming public too.

In practice today:

- Small recipe changes: just include the changed `sft.yaml` (or new recipe file) in your
  PR as normal.
- Datasets: `data/processed/` is git-ignored (these files are large). **Training-track
  PRs must use the pinned canonical mining dataset**
  ([`gittensor-model-hub/sparkproof-mining`](https://huggingface.co/datasets/gittensor-model-hub/sparkproof-mining),
  pin in [`datasets/canonical.json`](../datasets/canonical.json)). Prepare locally with
  `scripts/prepare_mining_sft.sh` (verifies the Hugging Face manifest matches the pin) and
  point your recipe at `data/processed/sparkproof-mining_sft.jsonl`. Cite the canonical
  URL and pinned `sft_sha256` in your PR body and via `proof.bundle --dataset-url`.
  **Do not** generate private blends (`eval/gen_*.py`, `scripts/prepare_triton*.sh`) for
  competition PRs — CI rejects them. New rows enter through SparkProof + the dataset
  registry first; registry merges refresh the pin on `main` automatically. See
  [`datasets/README.md`](../datasets/README.md) and `scripts/registry_line.sh`.

**Canonical pin grace window ([#121]):** a full train→eval cycle can take ~60 minutes.
If a dataset-track PR merges while you are training, `main` HEAD may advance to a newer
pin before you open your training PR. CI accepts your proof bundle when
`mix_manifest.sft_sha256` matches **any canonical pin from your PR's merge-base through
current HEAD** — cite that same pin in the PR body. Pins outside that window still reject.
Prepare with `scripts/prepare_mining_sft.sh` at train time; you do not need to retrain
just because the pin moved after you started, as long as your bundle matches a pin in
that window.

## Proof Of Training (Skip Full Retrain-Verification)

By default, the evaluator retrains/re-evals your PR from source — accurate, but slow.
This section is a shortcut for verifying your **eval claim** only — it has nothing to do
with the dataset-sharing requirement above, which always applies. If your checkpoint beats
the frontier, you can prove your claimed numbers instead of just asserting them, and get a
much cheaper verification pass:

```bash
# install the attestation + Hugging Face publishing extras
uv sync --extra proof

# prepare canonical training data + mix_manifest.json for the proof bundle
scripts/prepare_mining_sft.sh

# 1. export attested eval artifacts on your GPU CC + Intel TDX guest (cheap 50-example set)
uv run python -m eval.export_attested_samples \
    --checkpoint outputs/<your-checkpoint> \
    --out eval/results/attested_eval_samples.json

# 2. package the CLAIM into a bundle — eval scores + training claims + a per-file
#    sha256 manifest of your checkpoint. The weights themselves are NOT uploaded.
#    Include --attested-eval-samples so every claimed benchmark can verify on CPU.
python -m proof.bundle --checkpoint outputs/<your-checkpoint> --scores eval/results/candidate.json \
    --run-id <run-id> --out proof/_bundles/<run-id> \
    --train-hours 4.2 --train-gpu "NVIDIA RTX PRO 6000 Blackwell" \
    --dataset-url https://huggingface.co/datasets/gittensor-model-hub/sparkproof-mining \
    --mix-manifest data/processed/mix_manifest.json \
    --attested-eval-samples eval/results/attested_eval_samples.json
# note the printed claim_sha256

# 3. attest the GPU + measured TDX guest (nonce = claim_sha256; captures NRAS + TDX quote)
python -m eval.attestation --nonce <claim_sha256> --out runs/<run-id>/attestation.json

# 4. publish the (small, weights-free) bundle
python -m proof.publish --bundle proof/_bundles/<run-id> --repo-id <your-hf-username>/sparkdistill-<run-id>
```

**Score units:** every benchmark score in `candidate.json` / the bundle is a **fraction in
`[0, 1]`** (an accuracy / pass-rate / composite), matching `runs/frontier.json` — never a
`0`–`100` percentage. `eval.harness` already emits fractions; if you hand-assemble scores,
keep them in `[0, 1]` or verification rejects the claim outright.

Put the printed Hugging Face URL — and, if you ran it, your attestation.json — in your
PR. The validator runs `eval.verify`: when you include `--attested-eval-samples` and a
**GPU CC + Intel TDX** attestation bound to `claim_sha256`, it re-checks every bundled
benchmark artifact on CPU and **skips the harness re-run entirely** — no local
checkpoint reproduction and no validator GPU (`attested_eval_benchmarks` in the
report). Without attested samples (or without both GPU nonce binding and TDX
`report_data` binding), the validator still reproduces your checkpoint locally and
re-runs claimed benchmarks on GPU. The attestation is authenticated end-to-end:
NVIDIA JWKS (`gpu_signature`) and Intel PCS (`tdx_signature`). If your claim doesn't
hold up within tolerance, the PR is rejected outright —
a proof bundle that misrepresents its scores is treated as worse than no bundle at
all, not just "unverified."

For the `triton` domain score, serve your checkpoint with the **pinned** vLLM stack so
your numbers are comparable with the validator's re-run — `scripts/install_serve.sh`
installs it in a dedicated venv, and the eval configs use greedy decoding
(`temperature: 0.0`) for the same reason. Do not change either.

### Intel TDX measured-VM proof (optional, strongest tier)

GPU CC attestation proves the GPU/driver; it does not measure the VM your serving
stack and harness ran in. On a TDX guest (Targon CC VMs are), `eval.attestation`
additionally captures an **Intel TDX quote** with the same `claim_sha256` in its
64-byte REPORTDATA, signed by Intel. `eval.verify` reports this as `tdx_bound`,
and separately checks the quote's own signature chain as `tdx_signature`.

**What that does and does not establish.** Together those two say: *a genuine TD,
on genuine Intel hardware, committed to this bundle.* They do **not** say which
guest image the TD booted. The quote's MRTD measures that, and this repository
pins no approved MRTD values yet — so `eval.verify` records the measurement,
reports `tdx_measured_vm: null`, and states the reason rather than leaving the
field absent. A run on an arbitrary guest image passes `tdx_bound` and
`tdx_signature` today.

Pinning approved measurements requires building the reproducible guest image and
recording its MRTD out of band; a measurement read off a submission is written by
the submitter and establishes nothing. Until that exists, treat the TDX tier as
"genuine TD, bound to this bundle" and not as a measured-VM proof.

The kernel's configfs-tsm interface is root-owned; provision a persistent report
node **once per boot** (needs sudo), then attest as usual:

```bash
sudo chmod 0777 /sys/kernel/config/tsm/report
mkdir /sys/kernel/config/tsm/report/sparkdistill
sudo chmod 0666 /sys/kernel/config/tsm/report/sparkdistill/inblob
export SPARKDISTILL_TSM_REPORT_PATH=/sys/kernel/config/tsm/report/sparkdistill
```

Without a provisioned node (or on non-TDX hosts) the attestation simply records
`"tdx": null` — GPU claim binding remains the minimum bar. The validator checks
both the *binding* (REPORTDATA == claim digest, `tdx_bound`) and the quote's
authenticity via full **Intel DCAP verification** (`tdx_signature`): ECDSA
signature, PCK certificate chain to Intel's root CA, QE identity, and TCB
status against live Intel PCS collateral — `"UpToDate"` with no advisories is
the clean pass.

Training-track claims are enforced too: `--train-hours` beyond the **5-hour wall-clock
budget** is `eval:REJECT`, `--train-gpu` must be an accepted CC node (**RTX PRO 6000
Blackwell**, **B200/B300**, or **H100/H200**), and when attestation is included the
you attach a CC attestation, its attested hardware model must corroborate the claimed
GPU (a mismatched attestation is worse than none). The attestation itself is
authenticated end-to-end: the validator verifies the NRAS-signed GPU tokens against
NVIDIA's JWKS (`gpu_signature`) and the TDX quote against Intel PCS
(`tdx_signature`) — a hand-crafted attestation JSON fails both. `--dataset-url` should point at a
dataset merged through the dataset track (`datasets/registry.jsonl`), which is what
makes the training result reproducible by anyone.

| label | meaning |
|---|---|
| `proof:attested` | GPU CC + TDX attestation passed; CPU re-check of bundled eval artifacts only |
| `proof:unattested` | HF bundle submitted, no attestation; cheap re-verification only |
| `proof:none` | no bundle submitted; full retrain-verification applies |

Merged proof-of-training runs are appended to [`runs/ledger.jsonl`](../runs/ledger.jsonl)
— see [`runs/README.md`](../runs/README.md).

## Local Checklist Before Opening A PR

### Triton / SparkProof path (Blackwell or Hopper CC VM — recommended)

Run from **SparkProof** on the CC VM (sibling **SparkDistill** repo required — including
the gitignored `SparkDistill/tritonbench/` tree for decontamination). Works on Blackwell
RTX PRO 6000 / B100 / B200 or Hopper H100 / H200 — SparkProof detects which one and
rejects anything else:

```bash
cd SparkProof
cp .env.example .env   # YUNWU_API_KEY or OPENROUTER_API_KEY; HF_TOKEN for publish

scripts/install.sh              # first boot only
scripts/cc_check.sh           # Blackwell/Hopper + Python.h + GPU CC attestation

# smoke: generate + prove + verify (no HF)
scripts/run_triton_pipeline.sh --limit 2

# smoke + publish (verified 2026-07-11 — see datasets/README.md)
scripts/run_triton_pipeline.sh --run-id triton-cc-hf-001 --limit 2 \
  --release-gate --publish gittensor-model-hub/sparkproof-triton-v0

# validator re-check from SparkDistill (any machine)
cd ../SparkDistill
python -m eval.dataset_verify --hf-repo gittensor-model-hub/sparkproof-triton-v0 \
  --claimed-sha256 <from dataset_manifest.json> --sparkproof-root ../SparkProof \
  --out eval/results/dataset_report.json
```

Then eval from SparkDistill (training track, once a student checkpoint exists):

```bash
cd ../SparkDistill
scripts/eval.sh --checkpoint outputs/qwen3.5-4b-phase1 --compare-frontier
# Triton domain score (requires vLLM or --endpoint):
uv run python -m eval.triton_bench --checkpoint outputs/qwen3.5-4b-phase1 --serve \
  --quick --out eval/results/triton.json
```

### Legacy / local teacher path (no SparkProof)

Run these from the SparkDistill repo root:

```bash
# Trajectory generation (if your PR touches teacher/)
scripts/generate_trajectories.sh --prompts data/prompts/phase1.jsonl --out data/processed/phase1_trajectories.jsonl

# Fold captured reasoning into <think>-tagged SFT records (messages for qwen3_5)
scripts/prepare_sft_data.sh --in data/processed/phase1_trajectories.jsonl --out data/processed/phase1_sft.jsonl --format messages

# Training (if your PR touches recipes/)
scripts/install_train.sh
scripts/prepare_mining_sft.sh
scripts/train.sh recipes/qwen3.5-4b-phase1/sft-mining.yaml

# Or train on local phase-1 trajectories:
# scripts/train.sh recipes/qwen3.5-4b-phase1/sft.yaml

# Quality eval — always run before opening a PR
scripts/eval.sh --checkpoint outputs/qwen3.5-4b-phase1 --compare-frontier
```

## PR Requirements

A good PR includes:

- **A link to the dataset you trained on**, if it isn't small enough to commit directly
  (see *Sharing Your Dataset And Recipe* above). Required, not optional.
- A short description of what changed and why (trajectory prompt set, data mix,
  hyperparameter, eval coverage).
- The files and recipes changed.
- Local eval numbers, including which benchmarks moved and by how much.
- Any expected benchmark-specific effect: `bfcl`, `gsm8k`, `humaneval`, `ifeval`,
  `mmlu-pro`, `aime24`, or `gpqa-diamond`.
- Proof-bundle URL (required): published Hugging Face model repo from `proof.publish`
  (and attestation.json, if collected). This is in addition to, not instead of, the
  dataset link above.

Keep PRs narrow. A small recipe or trajectory-prompt PR with a clear eval delta is easier
to verify and merge than a broad rewrite.

## Current Target

The current frontier is Phase 1: **Qwen3.5-4B**, distilled from the teacher basket
(Claude Fable 5, GPT 5.6). The project is especially interested in:

- Higher-quality / more diverse reasoning trajectories, especially for underrepresented
  task types in the benchmark basket — reasoning-heavy prompts (multi-step math, logic,
  proof-style code correctness) matter most since the goal is reasoning distillation.
- Data mix and hyperparameter improvements in `recipes/qwen3.5-4b-phase1/sft.yaml`.
- Eval basket coverage that catches regressions the current benchmarks miss.

## Do Not Game The Eval

The evaluator uses held-out prompts, frozen benchmark data, immutable logs, and
path-aware labels. Attempts to tune for the harness instead of the checkpoint's real
quality can be rejected or ignored.

The best way to earn is to make the shipped student checkpoint genuinely better and keep
it honest.
