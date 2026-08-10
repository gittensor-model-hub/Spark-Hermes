# Spark-Hermes-Agent-3.8-27B

**The directory is named for the target, not for what it currently trains on.**
`Spark-Hermes-3.8-27B` is the model this pipeline exists to produce; Qwen3.8-27B is
announced and not yet published, so the development baseline is the pinned Qwen3.6-27B.
The name changes nothing about the pin — [`hermes/base_model.json`](../../base_model.json)
is what these recipes are checked against.

**One model, one personality: a worker that is exceptionally good at operating inside
Hermes Agent.** The name carries the claim — this is not "Qwen but smarter", it is Qwen
optimised for Hermes execution, and it should be judged only on that. Not a CUDA specialist, not an SWE specialist — those branch from this one
later, and branching before the general worker is good produces four mediocre models
instead of one useful one.

It is judged on whether it can drive the harness, not on what it knows:

- does it pick the right tool, with the right arguments, without flailing?
- does it hold an objective across a long task without drifting?
- does it recover when a tool fails, rather than repeating the command?
- **does it verify its own work before declaring success?**

Chat benchmarks measure the brain. These measure the worker. See
[`hermesbench/README.md`](../../../hermesbench/README.md) and
[`docs/roadmap-hermes.md`](../../../docs/roadmap-hermes.md).

**Nothing here has been trained.** No checkpoint exists, and the corpus it would train on
does not exist either, because no teacher tournament has run yet.

**Base model, pinned 2026-08-10.** `Qwen/Qwen3.6-27B` at revision
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, recorded in
[`hermes/base_model.json`](../../base_model.json) and kept in step with these recipes by
`tests/test_base_model.py`. Qwen3.8-27B — the model this directory is named for — is the
Phase 1 target and **is not published**: the only repositories under that name are
third-party derivatives with no official base, which cannot be pinned or verified. Phase 0
validates the pipeline on 3.6, which exists.

The pin is a revision, not a name. A name resolves to whatever the repository holds when
someone runs it, so two runs could agree on every other digest this project computes and
still have trained on different weights.

Worth knowing before reading a memory estimate: 3.6-27B is a
`Qwen3_5ForConditionalGeneration` with a vision tower, so its text hyperparameters live
under `text_config` and anything reading the top level of `config.json` gets `None` for
every field. Its own chat template emits `<tool_call>`, `<tools>`, `<tool_response>` and
`<think>` and does not emit `<scratch_pad>` — which is why the pin records `hermes-4`, and
records the evidence rather than the choice.

**Why these live outside `recipes/`.** Any `.yaml` under the top-level `recipes/`
directory marks a PR as an SN74 training-track submission, which must train on the
canonical mining dataset and carry a proof bundle — the gate auto-closes PRs that do not.
These are Phase 0 scaffolding, not mining submissions, so they sit here instead of
weakening a gate that scores live on-chain work.

## Stages

Run in order — each resumes from the previous stage's adapter.

| Stage | Config | Teaches | Data |
|---|---|---|---|
| A | `stage-a-reasoning.yaml` | planning, decomposition, debugging | reasoning traces (Bespoke-Stratos, NuminaMath, OpenThoughts) |
| B | `stage-b-hermes.yaml` | observe → plan → act → verify → recover | agent trajectories, mixed simulated + executed |
| C | `stage-c-tools.yaml` | tool choice, well-formed calls, no invented tools | **executed** trajectories only |

The order matters. Stage B on a model that cannot plan produces a model that calls tools
confidently and wrongly.

## Building the data

```bash
# 1. synthetic trajectories from a pinned teacher (simulated tool results)
python -m hermes.generate \
  --tasks hermes/tasks/phase0.jsonl \
  --out data/processed/hermes_trajectories.jsonl \
  --provider anthropic

# 2. render to Axolotl chat_template messages
#    no --require-success: failure/recovery rows are the point of stage B
python -m hermes.format \
  --in data/processed/hermes_trajectories.jsonl \
  --out data/processed/hermes_trajectories_sft.jsonl
```

Stage C consumes `hermes_executed_sft.jsonl` — the same rendering restricted to
trajectories whose tool results came from real execution (HermesBench runs,
`metadata.executed = true`):

```bash
python -m hermes.format \
  --in data/processed/hermes_trajectories.jsonl \
  --out data/processed/hermes_executed_sft.jsonl \
  --executed-only
```

Simulated tool results are a teacher's guess about what a command would print; training
the reliability stage on guesses teaches the model that its predictions about the world
are the world.

Both invocations also drop episodes whose closing turn was written by the bench runner
rather than the agent (`metadata.harness_final` — "step budget exhausted", setup
failures). Pass `--keep-harness-finals` to retain them; you almost never want to, since
rendered as-is they train the student to produce that sentence as its answer.

## Hardware

QLoRA (4-bit base + adapters on `q,k,v,o`) sized for a single RTX PRO 6000 Blackwell
(96 GB). H100/H200 and B200 work with the same configs; raise `micro_batch_size` before
`sequence_len` if you have headroom, since Stage B/C episodes are long and truncating
mid-episode teaches the model that episodes simply stop.

## Evaluating

MMLU and HumanEval do not measure agents. Score with
[HermesBench](../../../hermesbench/README.md), whose success signal is each task's own
verification command rather than the model's closing summary.
