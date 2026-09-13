# Spark-Hermes Miner Guide

How to compete, what wins, and what the labels are worth.

There are three ways to contribute, each with its own gate and its own label:

| track | what you submit | registry line | label |
|---|---|---|---|
| **Strategy** | a private surface that makes the frozen model solve a challenge | `datasets/strategies.jsonl` | crown |
| **Dataset** | verified training rows, published to Hugging Face | `datasets/registry.jsonl` | `dataset:xs…xl` |
| **Training** | a better recipe on the pinned canonical dataset | — | `eval:XS…XL` |

Rollout datasets generated against the pinned teachers use the same shape as the dataset track
with `datasets/rollouts.jsonl` as the registry.

The strategy track is the competition proper and the project's forward direction. **Start there.**

Rights, licensing and what is public versus private are in [`CONTRIBUTING.md`](../CONTRIBUTING.md).
Rewards are set by the SN74 subnet from merged, labelled pull requests; the label tables below
are this repository's declared intent, not a payout guarantee.

---

## Strategy track

### What you are competing on

The validator runs the pinned model against real tasks. When it fails one — or passes it
expensively — that run becomes a **challenge** in [`datasets/challenges/`](../datasets/challenges).
Your job is to make the *same frozen model* do better on it.

You do not change weights, and you do not choose the model, environment or runtime. Those are
pinned. You control the **surface**: `SOUL.md`, `skills/*/SKILL.md` and
`skills/*/references/*.md` — prose the model reads before every decision.

Read the packet before writing anything. It names the failure class:

```bash
python -c "import json;d=json.load(open('datasets/challenges/tc-log-rotation-order.json'));\
print(d['failure_class'], d['baseline'])"
```

That matters more than it sounds. The first surface written for this repository targeted step
budgets on a task whose failure class was `malformed_protocol`, and made things **58.6% and
92.6% worse** across two paired runs. A later one aimed at the real failure class cut median
tokens by 24.7%.

### 1. Build a surface

```bash
python -m miner init  --dir ./my-surface --skill protocol-discipline
python -m miner check --dir ./my-surface
```

`check` costs nothing — no model, no GPU, no pull request. It calls the validator's own
contract and prompt-assembly code, so a surface that fails it produces no run at all rather
than a bad one. It refuses executables, shell scripts under `skills/`, symlinks, and an empty
directory (which would run as the unmodified baseline).

`check` says *"admissible, not that it helps"* — and means it. The first surface here passed
`check` cleanly and then raised median tokens by 58.6% and 92.6%.

### 2. Rehearse before you commit

```bash
python -m miner evaluate --dir ./my-surface --task tc-log-rotation-order \
  --base-url http://127.0.0.1:8001/v1 --model qwen3.8-27b --repeats 10
```

Two arms, paired: with your surface and without, on your machine. Pairing removes hardware
variance, which is why both run locally instead of comparing against the packet's baseline.

`--repeats` defaults to 10 and you will usually want more. One attempt tells you almost nothing
and feels like it tells you everything: a surface here scored **3 of 3** on its first measurement
and **6 of 10** on the next — a real improvement over the 4/10 baseline, and still refused,
because the gate requires every attempt to pass. When the interval does not clear zero, the
report prints how many paired repeats would settle evidence like yours.

### 3. Find out which rule is doing the work

```bash
python -m miner search --dir ./my-surface --task tc-log-rotation-order \
  --base-url http://127.0.0.1:8001/v1 --model qwen3.8-27b --repeats 10
```

Leave-one-out ablation: the full surface, then one candidate per rule with that rule removed.
A rule whose removal changes nothing is costing context for free.

The leader is re-measured on **fresh episodes**, and that second number is the one to believe.
On candidates drawn from an identical distribution — nothing to find, by construction — a
screening leader appeared in **36 of 40** searches and **none** survived confirmation. Picking
the best of *k* manufactures winners.

### 4. Upload privately, commit publicly

Upload the bundle to the validator. It is a JSON map of path to text — not an archive — and it
is validated before a byte is stored:

```bash
curl -X POST http://<validator>/v1/round/<round_id>/submission \
  -H 'Content-Type: application/json' \
  -d '{"miner_id": "<your-github-login>", "files": {"SOUL.md": "...", "skills/p/SKILL.md": "..."}}'
```

You get a receipt:

```json
{"submission_id": "...", "bundle_sha256": "sha256:...", "status": "pending"}
```

Then open a pull request appending **one line** to `datasets/strategies.jsonl`:

```json
{"schema_version": 2, "round_id": "r-001", "miner_id": "<your-github-login>",
 "bundle_sha256": "sha256:...", "task_ids": ["tc-log-rotation-order"]}
```

Your surface stays private — it is what you are competing with. The digest is public,
timestamped and attributable, which stops the validator evaluating a different bundle than the
one you committed to and stops you revising after the fact.

**Admission is authenticated.** The validator fetches your pull request from GitHub itself;
nothing in the upload or the line confers authority. Receipts are public, so three identities
must agree — the PR author as GitHub reports it, the `miner_id` on the line, and the `miner_id`
on the receipt — or committing to someone else's digest would be evaluated under your name.
`datasets/strategies.jsonl` is append-only.

Upload as often as you like; the response says nothing about merit. **The digest in your pull
request decides which bundle is judged**, not the most recent upload. One commitment per miner
per round.

### 5. What happens then

```text
pending → evaluating → result
```

The validator runs your surface on the pinned model against the withheld verifiers you never
see, scores it against the challenge's baseline, and once an hour crowns exactly one winner.
Every other pull request in that round closes with its reason. A barren hour keeps the crown
where it is; two barren hours rotate the task.

### What it takes to win

**Correctness is a gate, not a term.** Your surface must pass **every** attempt over at least
ten, and a pass means the published check *and* the withheld one. A surface fitted to the
visible assertions scores as the failure it is.

Only then does efficiency count, and the margin must clear zero on a bootstrap interval rather
than on a point estimate. Fewer tool calls too: a token win bought by collapsing thirty
operations into one helper script does not take the crown. And **verification cannot regress**
— fewer checks count only when the task's required verification is still fully satisfied.

An hour can pass with no crown. Nothing is awarded when nothing beats its own baseline.

### Checking the validator

After a round settles the validator publishes the challenge, the ledger, the per-task salt, the
episode logs and the scorecards. With the withheld check:

```bash
python -m validator.audit verify --bundle <bundle> --withheld-check ./hidden.sh
```

confirms the validator graded against the check it committed to before submissions opened. It
does not prove the episodes came from the pinned model; the manifest says so in its own
`does_not_prove` field.

---

## Dataset track (`dataset:xs` … `dataset:xl`)

Verified training rows, generated with [SparkProof](https://github.com/gittensor-model-hub/SparkProof)
and published to Hugging Face.

1. **Generate** with an unmodified SparkProof checkout: `scripts/run_triton_pipeline.sh`. Teacher
   calls go through an approved gateway to the pinned teachers, and every kernel is compiled and
   executed on the GPU present.
2. **Publish** with `sparkproof-publish-dataset --bundle <dir> --repo-id <you>/<repo> --release-gate --mining-repo`.
   The release gate (decontamination + provenance) must pass; rows and `proof/` land in the same
   HF repo.
3. **Open a text-only PR** appending one line to `datasets/registry.jsonl`, built with
   `scripts/registry_line.sh --bundle <dir> --miner <handle> --repo-id <you>/<repo>`. A dataset PR
   changes nothing else.
4. **The validator** runs `python -m eval.dataset_verify`, aggregates every line into the canonical
   mining dataset, and merges only at `dataset:xs` or above:

| label | verified rows |
|---|---|
| `dataset:xl` | ≥ 150 |
| `dataset:l` | ≥ 100 |
| `dataset:m` | ≥ 75 |
| `dataset:s` | ≥ 50 |
| `dataset:xs` | ≥ 25 |
| `dataset:none` | valid proof, below 25 — not merged, not rewarded |
| `dataset:REJECT` | decontamination, hash or policy failure |

Rows are sized from the canonical mix's `rows_selected` after cross-registry deduplication, so
check `novel_verified_rows` ≥ 25 against the registry snapshot **before** publishing — the
snapshot on the canonical mining repo is the source of truth for expected credit.

**Decontamination is mandatory and aborts if the protected benchmark corpus is unavailable.**
Rows whose origin is a protected benchmark, rows marked `test` / `eval` / `held_out`, exact and
semantic prompt matches, and AST-canonicalized code matches are all rejected — renaming variables
does not pass. A registry submission containing eval material gets `dataset:REJECT` and is closed
automatically.

**Miners cannot submit an evaluation dataset.** The track accepts training trajectories only.
Letting a miner define the evaluation that rewards their own submission is a direct conflict of
interest, so scoring uses the validator-controlled held-out basket. Evaluation *tooling*
improvements are welcome as ordinary code PRs and are reviewed as evaluator code, not rewarded
as data.

---

## Training track (`eval:XS` … `eval:XL`)

A better recipe on the **pinned canonical mining dataset** — never a private blend.

A PR scores when it includes the recipe, trains only on the canonical dataset
(`data/processed/sparkproof-mining_sft.jsonl` from [`datasets/canonical.json`](../datasets/canonical.json)),
preserves correctness against the frozen benchmark reference, improves at least one benchmark
by the current threshold, and avoids unacceptable regressions elsewhere. Small gains are not
aggregated across benchmarks.

| benchmark | target |
|---|---|
| TritonBench (`triton`) | Triton kernel expertise — **the tier signal** |
| BFCL | tool-calling accuracy |
| GSM8K | math reasoning — regression floor |
| HumanEval | code correctness |
| IFEval | instruction following |
| MMLU-Pro | broad reasoning |
| AIME | competition math |
| GPQA-Diamond | graduate-level science |

The `eval:*` tier comes from **TritonBench only**. The general basket is regression-guarded:
any drop beyond its floor yields `eval:REJECT` and a `regression-<benchmark>` label, but
general-basket improvements alone do not earn a tier. GSM8K may drop at most 1% (2% when
Triton improves by ≥ 2%).

| label | meaning |
|---|---|
| `eval:XL` … `eval:XS` | verified quality improvement, very large → minimum accepted |
| `eval:BASELINE` | first verified checkpoint |
| `eval:none` | correct, no significant improvement |
| `eval:REJECT` | correctness failure, training failure, or unacceptable regression |

Prepare with `scripts/prepare_mining_sft.sh` and cite the canonical URL and pinned `sft_sha256`
in the PR. A train → eval cycle takes about an hour, so CI accepts a bundle matching **any**
canonical pin from your merge-base through current HEAD.

**Proof bundle (optional fast path).** By default the evaluator retrains and re-evaluates from
source. A published proof bundle — eval scores, training claims, and a per-file checkpoint
manifest, never the weights — lets the validator verify your claim more cheaply:

```bash
python -m proof.bundle --checkpoint outputs/<ckpt> --scores eval/results/candidate.json \
    --run-id <id> --out proof/_bundles/<id> --train-hours 4.2 --train-gpu "<gpu>" \
    --dataset-url https://huggingface.co/datasets/gittensor-model-hub/sparkproof-mining \
    --mix-manifest data/processed/mix_manifest.json
python -m proof.publish --bundle proof/_bundles/<id> --repo-id <you>/spark-hermes-<id>
```

Scores are fractions in `[0, 1]`, never percentages. A bundle that misrepresents its scores is
treated as worse than no bundle. `--train-hours` beyond the 5-hour budget is `eval:REJECT`.

Trained weights are not merged into the repository.

---

## What does not score

Useful, but no quality label without a verified frontier improvement:

- documentation-only, test-only, or refactor-only changes
- eval-harness changes that do not improve measured checkpoint quality
- copying an already-merged dataset or recipe without a new measurable improvement
- improving a path the current phase's scoring target does not use

The eval harness, tooling and docs are maintainer-owned. Community PRs that are not track
submissions are closed automatically.

---

## Labels and payout

This repository's declared intent, from [`.gittensor/weights.json`](../.gittensor/weights.json):

| tier | `dataset:*` | `eval:*` |
|---|---|---|
| XL | 4.0 | **8.0** |
| L | 2.5 | **5.0** |
| M | 1.5 | **3.0** |
| S | 1.0 | **2.0** |
| XS | 0.5 | **1.0** |
| BASELINE | — | **2.0** |

`*:none` and `*:REJECT` are 0. `area:*` labels are categorisation only and carry no weight.

Labels are computed, not clicked: `eval/pr_labels.py` derives `area:*` from the diff, and each
track's gate applies its own tier. The subnet reads merged, labelled pull requests under its own
eligibility and reward policy.

---

## Do not game the eval

Held-out prompts, frozen benchmark data, withheld verifiers, immutable logs, path-aware labels.
Attempts to tune for the harness instead of for real quality are rejected or ignored. Contribute
reproducible evidence of useful changes.
