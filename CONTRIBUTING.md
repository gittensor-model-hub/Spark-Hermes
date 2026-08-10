# Contributing to SparkDistill

SparkDistill is the model-quality arm of **SN74 on [Gittensor](https://gittensor.io/)**, the same subnet that funds
[`sparkinfer`](https://github.com/gittensor-ai-lab/sparkinfer). Contributions are rewarded
for **real, verified distillation quality improvements** — not benchmark gaming. This guide
is how to make a contribution that counts.

## Built through [Gittensor](https://gittensor.io/)

[Gittensor](https://gittensor.io/) helps power SPARKDISTILL through SN74: the project receives subnet emissions,
contributors submit source PRs, the evaluator retrains or re-scores those PRs against a
frozen reference, and rewards are assigned from verified marginal quality improvements
that keep the checkpoint honest. You do not need to be in Discord or understand the
subnet internals to contribute, but the source of the incentive loop is clear: SPARKDISTILL
is built through **SN74 on [Gittensor](https://gittensor.io/)**.

## Principles

- **The submission is the recipe + dataset, never the weights.** The evaluator retrains
  from your recipe and dataset from source on its own hardware — that retrain is the
  source of truth. A shipped checkpoint (via the proof-of-training fast path) is a
  verification convenience for the eval *numbers* only; it is never merged or trusted
  directly, and it never substitutes for including the recipe and dataset in your PR.
- **Quality first, not just loss.** A recipe change that lowers training loss but degrades
  held-out benchmark quality is worth zero. Every change is gated against a frozen eval
  basket (see *Quality gate* below).
- **Reasoning, not just answers.** SparkDistill's goal is reasoning distillation: a
  trajectory or recipe change should be judged on whether it makes the student better at
  *reasoning through* a problem, not just at matching a final answer — that's why the
  benchmark basket includes hard-reasoning tasks (AIME, GPQA-Diamond) alongside the
  broader basket.
- **General, not overfit.** Improvements must hold across the benchmark basket — BFCL,
  GSM8K, HumanEval, IFEval, MMLU-Pro, AIME, GPQA-Diamond — not just one task. A win on
  one benchmark but a loss elsewhere is overfitting.
- **Phase-scoped.** Phase 1 targets Qwen3.5-4B. Recipes and eval changes should target the
  current phase's student model unless a PR is explicitly opening the next phase.
- **Fair by construction, not by policy.** Because every merged PR's recipe and dataset
  are public, the current frontier ("the king") can always be forked and improved on by
  anyone — a miner can't permanently dominate by hoarding a secret checkpoint. "Copy the
  frontier and add one optimization" is a normal, expected way to compete, not a violation
  — see *How rewards work* below for why it still only pays for your marginal delta.

## Found a bug or have an idea?

**Open an issue, not a PR.** Community pull requests here are limited to the two
rewarded tracks (training or dataset — see *How rewards work* below). A community PR
that is neither is automatically commented on with this policy and closed by CI
(`.github/workflows/community_pr_policy.yml` / `eval/community_pr_policy.py`) — it is
not queued for manual review. Maintainer PRs (and bots, e.g. dependabot) are exempt.

This isn't a rejection of the contribution — harness/tooling/docs bugs and ideas are
genuinely welcome, just as an **issue**: describe what happens, its impact, and (if you
have one) your suggested fix. Maintainers triage issues and make the harness/tooling/docs
changes themselves, since the eval/scoring harness is maintainer-owned (see
*Maintainer-owned paths* below) — a PR changing it from a community account is closed
regardless of quality.

## Before you open a PR

```bash
# 1. install + sanity checks
uv sync
ruff check .
pyright

# 2. trajectory / recipe change — does it actually train?
scripts/train.sh recipes/<your-recipe>/sft.yaml --dry-run

# 3. quality — did the resulting checkpoint get better?
scripts/eval.sh --checkpoint outputs/<your-checkpoint> --compare-frontier
```

**Quality gate.** Run `scripts/eval.sh` on the checkpoint *before* and *after* your change.
A correct improvement must:

- improve at least one benchmark in the basket by the current eval threshold, and
- not regress any other guarded benchmark below its floor.

Do not trade breadth for a single benchmark's score. Quality is measured across the whole
basket.

## Sharing your dataset (required on every PR)

Whatever dataset you trained on must be reproducible from your PR. Small files can be
committed; Triton training data uses the **dataset track** instead of `data/processed/`
in git.

**Dataset track (Triton, shipped):**

1. Generate and prove on a Blackwell CC VM with [SparkProof](https://github.com/gittensor-model-hub/SparkProof)
   (`scripts/run_triton_pipeline.sh --release-gate --publish <org>/<repo>`).
2. Publish verified rows + `proof/` artifacts to Hugging Face (`sparkproof-publish-dataset`).
3. Append one line to [`datasets/registry.jsonl`](datasets/registry.jsonl) via a text-only PR.
   Build the line with `scripts/registry_line.sh --bundle <dir> --miner <handle> --repo-id <org>/<repo>`.
4. Registry CI verifies the bundle and merges at `dataset:xs` (≥25 verified rows).

Training PRs train on the **single pinned canonical mining dataset** only. Cite
[`datasets/canonical.json`](datasets/canonical.json) in your PR body (`hf_url` +
`mix_manifest.sft_sha256`) and set `proof.bundle --dataset-url` to the same URL. Recipes
must reference `data/processed/sparkproof-mining_sft.jsonl` — prepare it locally with
`scripts/prepare_mining_sft.sh` (verifies the Hugging Face pin). Training-track PRs that
add local generators (`eval/gen_*.py`), private blends, or non-canonical recipe paths are
rejected by CI. New rows enter through the dataset track first; registry merges refresh
the pin on `main` automatically.

**Pin grace ([#121]):** if the canonical pin advances on `main` while you are training,
your training PR still passes when the proof bundle's `mix_manifest.sft_sha256` matches
any pin from the PR merge-base through HEAD — cite that pin in the PR body.

A PR whose recipe can't be reproduced because it uses unpublished or non-canonical data
should be treated as incomplete, even if local eval numbers look good.

## Proof of training (optional fast path for eval verification)

This fast path is a shortcut for verifying your **eval claim** — it has nothing to do with
sharing your dataset above, which is always required. If your checkpoint beats the
frontier, you can skip full retrain-verification by proving your claimed numbers instead
of just asserting them:

```bash
# 4. (optional) attest the GPU you trained/evaluated on
uv sync --extra proof
python -m eval.attestation --out runs/<run-id>/attestation.json

# 5. package + publish a proof bundle (checkpoint + eval scores only)
python -m proof.bundle --checkpoint outputs/<your-checkpoint> --scores eval/results/candidate.json \
    --run-id <run-id> --out proof/_bundles/<run-id>
python -m proof.publish --bundle proof/_bundles/<run-id> --repo-id <your-hf-username>/sparkdistill-<run-id>
```

Put the resulting Hugging Face URL in your PR description. The evaluator then runs
`eval.verify` — a small held-out re-run of your claimed scores, plus attestation
validation if you included it — instead of retraining from scratch. **The checkpoint
bundle itself is never merged or trusted as the deliverable** — it only lets verification
skip a full retrain; the recipe + dataset in your PR remain the actual submission and the
thing the evaluator can always fall back to reproducing from source. A PR without a proof
bundle still gets the full from-source retrain-verification; proof of training only makes
verification cheaper, it never replaces the quality gate above or the dataset-sharing
requirement above.

`nv-attestation-sdk` (the attestation library `eval.attestation` wraps) is the current
NVIDIA Python SDK for GPU confidential-computing attestation but is marked EOL
2026-09-15; see the module docstring for the migration note. Miners without CC-capable
hardware (e.g. a Blackwell RTX PRO 6000 Server Edition node) can skip attestation
entirely and still use the HF bundle + cheap re-run path — attestation only affects how
much trust the ledger records for that run, not whether the fast path is available.

## How rewards work (SN74 on [Gittensor](https://gittensor.io/))

**Quality-only.** You're paid for the **verified marginal quality improvement** your PR
adds over the current best ("frontier") checkpoint, not your rank — so "copy the leader +
ε" pays ≈ ε. Both the current frontier checkpoint and your PR's resulting checkpoint are
evaluated on the same held-out benchmark basket in one run and scored on the delta between
them, so eval-run variance can't inflate or hide your result.

**Non-quality PRs score 0** — SN74 emits only for verified quality improvements. Maintainer
PRs of this kind (bug fixes, refactors, tests, tooling, docs) are still welcome and
reviewed normally. From a **community** contributor, though, a PR that is neither a
training-track nor a dataset-track submission is **auto-closed** by CI, not queued for
review — see *Found a bug or have an idea?* above for the actual path. (The eval/scoring
harness is maintainer-owned — see *Maintainer-owned paths* below.)

The eval loop labels each PR **XL / L / M / S / XS** from the measured delta (or
**BASELINE** for the first verified checkpoint on a new student/phase) — never by hand —
and that tier is the payout. **Training-track `eval:*` multipliers are 2× the `dataset:*`
tier at the same letter** (e.g. `eval:L` = 5.0 vs `dataset:l` = 2.5); see
[`.gittensor/weights.json`](.gittensor/weights.json) and
[`docs/miner-guide.md`](docs/miner-guide.md#sn74-payout-multipliers-gittensor). A quality
improvement is scored the same wherever it lands (`teacher/`, `recipes/`, `eval/`); there
is **no per-subsystem budget**.

## Maintainer-owned paths

The eval/scoring harness itself (`eval/harness.py`, `eval/score.py`, `eval/verify.py`,
`eval/attestation.py`, `eval/ledger.py`, and the frozen benchmark data they read) is
maintainer-owned. This protects the eval loop from being tuned to inflate scores rather
than to improve real quality. In particular, only the eval bot appends to
`runs/ledger.jsonl` — it is not an append-anything log.

A maintainer PR that touches the harness in ways that change scoring or verification
behavior is held for manual review, same as always. A **community** PR touching the same
paths is auto-closed by the policy above, not held for review — file an issue instead.

## Canonical mining dataset (training track)

Registry CI aggregates all merged datasets into one Hugging Face repo
([`gittensor-model-hub/sparkproof-mining`](https://huggingface.co/datasets/gittensor-model-hub/sparkproof-mining)).
The active pin lives in [`datasets/canonical.json`](datasets/canonical.json) and is refreshed
on `main` after each registry merge.

**Training-track rule:** every miner trains on exactly this pinned dataset. Prepare locally:

```bash
scripts/prepare_mining_sft.sh
# writes data/processed/sparkproof-mining_sft.jsonl after verifying the HF pin
```

Do not re-mix registry rows into a private training file for competition PRs. To add new
rows, run SparkProof and open a dataset-track registry PR first.

## SparkProof: Blackwell-verified Triton datasets

**Implementation:** [SparkProof](https://github.com/gittensor-model-hub/SparkProof) (sibling repo).

**Production Triton datasets (`sparkproof-2`) must be validated on NVIDIA Blackwell**
(RTX PRO 6000 Server Edition CC, `SPARKPROOF_BLACKWELL_PROFILE=workstation`). SparkProof:

- Calls teachers via **OpenRouter** (Fable 5 + GPT 5.6)
- **Compiles and executes** Triton 3.7.1 kernels on Blackwell; publishes verified-only trajectories
- Seals manifest + Merkle root + **GPU CC attestation** (`gpu_attestation.json`)
  + **Intel TDX measured-VM quote** (`gpu_attestation.tdx`, production required)

Run SparkProof **on the Blackwell or Hopper CC VM inside an Intel TDX guest** (SSH) — no Polaris. Teacher calls use OpenRouter
with **`reasoning.effort: xhigh`** on `anthropic/claude-fable-5` and `openai/gpt-5.6-sol`.

Provision TDX once per boot (see [`datasets/README.md`](datasets/README.md#intel-tdx-production-required-on-new-bundles)):

```bash
sudo chmod 0777 /sys/kernel/config/tsm/report
mkdir /sys/kernel/config/tsm/report/sparkproof
sudo chmod 0666 /sys/kernel/config/tsm/report/sparkproof/inblob
export SPARKPROOF_TSM_REPORT_PATH=/sys/kernel/config/tsm/report/sparkproof
```

The key insight: SparkProof does not try to prove that Claude or GPT *itself* executed
honestly — commercial APIs don't expose signed inference attestations, so that's not
achievable today. Instead it proves that the **miner's dataset-generation and Blackwell
validation process** was honest, reproducible, and tamper-proof. That's a much narrower,
achievable claim, and notably for the Triton track it requires **GPU confidential computing
on Blackwell** for compile/execute validation.

**What `sparkproof-2` cryptographically proves:** teacher outputs were captured via the
approved OpenRouter gateway; each kept sample passed Triton validation **on the attested
Blackwell GPU**; the verified dataset manifest and Merkle root were not modified afterward;
GPU CC attestation binds the validation run to confidential-computing hardware; **Intel TDX**
(`gpu_attestation.tdx`) binds the measured VM that ran SparkProof to the same dataset nonce.

**What it explicitly does not prove (and says so):** that Anthropic/OpenAI/OpenRouter
internally executed the model honestly, or that OpenRouter routed to the correct provider.
Those remain ordinary API-trust assumptions — the same ones every developer already makes
when calling these APIs today. SparkProof only records the provider/model identifiers the
API returned; it can't independently verify what happened on the other side of the HTTPS
call.

**Pipeline sketch:** run on **Blackwell CC VM** → OpenRouter calls with pinned xhigh slugs →
hash each request/response → Triton compile+execute on GPU → filter verified samples →
GPU CC attestation → Merkle root + `sparkproof-2` manifest → publish bundle to Hugging Face.
Validators replay `request_sha256`, check Blackwell validation proofs, and verify GPU attestation.

**Security properties this buys:** a miner can't forge or edit a dataset after generation
(hashes/Merkle root change), can't skip Blackwell validation (verified samples only in Merkle),
and can't substitute cheaper models (request body hash binds xhigh + slug).

**Feasibility:** GPU CC attestation uses `nv-attestation-sdk` (see SparkDistill `eval/attestation.py`
and SparkProof `gpu_attestation.json`). No Polaris / CPU TDX required when using the CC VM.

**Future directions once this exists:** bind provider-signed responses into each record if
Anthropic/OpenAI start signing API responses; bind confidential-inference attestations into
the manifest if providers start exposing them; publish manifests to an append-only
transparency log; optional zero-knowledge proofs of dataset properties (e.g. licensing,
policy compliance) without revealing prompts/responses.

Validated datasets live on Hugging Face and are indexed in-repo via `datasets/registry.jsonl`.
CI maintains the canonical mining dataset at `gittensor-model-hub/sparkproof-mining`.
SparkProof proves each component's integrity; `mix_manifest.json` on HF proves composition.

## What Does Not Score

- Documentation-only changes.
- Refactors with no benchmark improvement.
- Test-only changes.
- Eval harness changes that do not improve measured checkpoint quality.
- Copying an already-merged trajectory set or recipe without a new measurable improvement.
- Changes that improve one synthetic eval path but are unused by the scoring target.

## Current Target

Phase 1: **Qwen3.5-4B**, distilled from the teacher basket (Claude Fable 5, GPT 5.6)
in [`teacher/providers.py`](teacher/providers.py), trained via
[`recipes/qwen3.5-4b-phase1/sft.yaml`](recipes/qwen3.5-4b-phase1/sft.yaml).

The project is especially interested in:

- Higher-quality / more diverse teacher trajectories.
- Data mix and hyperparameter improvements in the Phase 1 recipe.
- Eval basket coverage that catches regressions the current benchmarks miss.

## Do Not Game The Eval

The evaluator uses held-out prompts, frozen benchmark data, and reproducible training
seeds. Attempts to tune for the harness instead of real checkpoint quality can be rejected
or ignored.

The best way to earn is to make the shipped student checkpoint genuinely better and keep
it honest.
