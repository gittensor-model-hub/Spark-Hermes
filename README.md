![Spark-Hermes-3.8-27B banner](docs/images/spark-hermes-3.8-27b.png)

# Spark-Hermes

**An open agent model that gets better through verified competition.**

A frozen model attempts real tasks. When it fails, loops, or burns tokens, that run becomes a
challenge. Miners compete to make the *same frozen model* do better. The validator runs every
entry itself, against checks the miner never sees. Winning rollouts become training data. The
maintainer trains the next model. Repeat.

```text
RUN → OPTIMIZE → VERIFY → LEARN → REPEAT
```

| | |
|---|---|
| **Target** | `Spark-Hermes-3.8-27B` |
| **Base** | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) @ `1d4bf0f2`, served bf16 |
| **Proof of concept** | `Qwen/Qwen3.5-4B` @ `851bf6e8` on an RTX 5090 32 GB (`rtx5090-poc`) |
| **Production hardware** | RTX PRO 6000 Blackwell 96 GB, bf16 |
| **Runtime** | Hermes 4 agent, over the `qwen35` wire format the base natively speaks |
| **Competition** | SN74 Gittensor — verified rollout optimization |

- **Miners:** [`docs/miner-guide.md`](docs/miner-guide.md)
- **Operators:** [`docs/train-spark-hermes.md`](docs/train-spark-hermes.md)
- **Validators:** [`docs/competition-ingress.md`](docs/competition-ingress.md) · [`docs/competition-settlement.md`](docs/competition-settlement.md)
- **Contributing, rights and licensing:** [`CONTRIBUTING.md`](CONTRIBUTING.md)

---

## The loop

```text
TASK
  │
  ▼
FROZEN SPARK-HERMES MODEL                 the validator runs the baseline
  │
  ├── verified + efficient ────────────────┐
  │                                        │
  └── fail / loop / expensive              │
             │                             │
             ▼                             │
       CHALLENGE PACKET                    │  datasets/challenges/
             │                             │
             ▼                             │
       SN74 MINERS                         │  improve a private surface
             │                             │  SOUL.md, skills, references
             ▼                             │
   PRIVATE BUNDLE ──► validator API        │  the surface stays the miner's edge
   PUBLIC DIGEST  ──► pull request         │  timestamped, attributable, binding
             │                             │
             ▼                             │
     VALIDATOR RUNS IT                     │  pinned model, environment, runtime
   pinned everything, its own hardware     │  withheld verifiers never leave
             │                             │
             ▼                             │
       VERIFY + SCORE                      │  correctness gates efficiency
             │                             │
             ▼                             │
     ONE CROWN PER HOUR                    │  every other pull request closes
             │                             │
             ├──► ROUND PROOF PUBLISHED    │  the commitment is opened
             │                             │
             ▼                             │
  VERIFIED SFT / PREFERENCE DATA           │
             │                             │
             ▼                             │
    TRAIN NEXT SPARK-HERMES                │
             │                             │
             └─────────────────────────────┘
                         REPEAT
```

This is the product: not a one-time fine-tune, and not a race to use the biggest hidden model.

---

## How a task becomes a challenge

The pinned model runs each task with the canonical strategy. A challenge opens when the baseline
fails verification, times out, loops, exceeds its token / turn / tool budget, repeats expensive
actions, finishes without enough verification, or succeeds far above the expected resource
envelope.

```text
BASELINE                        MINER CANDIDATE
result      FAIL                result      PASS
tokens      57,000              tokens      22,000
tool calls  38                  tool calls  11
```

Failure → verified success is the highest-value improvement. Tasks come from real agent
workloads, public repositories at pinned commits, regression cases, generated tool-use
scenarios, and mutation-generated repository failures. Generated tasks are classified against
the current baseline — too easy becomes SFT coverage, the frontier becomes competition, and very
hard becomes future curriculum.

---

## What miners compete on

For each challenge the model, tokenizer, runtime, system prompt, tool schemas, workspace,
budgets and seeds are **fixed**. What a miner controls is the **surface** — `SOUL.md`, skills
and references — which shapes planning, recovery, verification, stopping and tool-use policy.

Miners do **not** win by changing weights. They win by making the same model behave better.

The full flow — build, rehearse, upload privately, commit the digest publicly, what wins, and
what the labels are worth — is in [`docs/miner-guide.md`](docs/miner-guide.md).

---

## How work is scored

**Correctness is a gate, not a term.** No efficiency reward exists until the entry passes the
correct model epoch, the Hermes protocol, allowed tools only, a complete trace, the published
verifier, the **withheld** verifier, the security checks, minimum verification strength and the
budget. It must pass every attempt over at least ten.

Then, and only then, efficiency:

```text
gates the crown          verified success
                         overfit rate   (published pass, withheld fail)
                         tokens
                         model turns
                         weighted tool use
                         repeated actions

reported, never gates    wall time
```

Wall time is excluded on purpose. Tokens and tool calls are exact and recomputable from the
trace; wall time depends on the card and the scheduler, so rewarding it would pay for hardware.
And **verification cannot regress** — the cheapest rollout is to stop checking, and that is not
optimization.

**The withheld check.** Every task publishes a salted commitment to a second verifier whose body
lives only on the validator. Both run on every attempt. A strategy that passes the published check
and fails the withheld one is counted as `overfit` — it learned the benchmark, not the job. The
salt is per task, so opening one round unseals nothing else.

**The round proof.** After settle, the validator publishes the challenge, the ledger, the per-task
salt, the episode logs and the scorecards. `validator.audit verify` recomputes the commitment and
confirms the validator graded against the check it committed to *before* submissions opened. It
does not prove the episodes came from the pinned model — nothing in a published bundle can, and
the manifest says so in a `does_not_prove` field rather than implying more.

---

## What accepted rollouts train

A winning rollout is valuable precisely because it came from the model that originally failed.

- **SFT** — verified trajectories: observe → plan → act → read result → recover → verify → stop.
- **Preference pairs** — matched states: `patch exists, tests not run` → chosen *run the test*,
  rejected *claim completion*.
- **Recovery data** — failed baselines teach loop detection, premature completion, bad tool
  choice, weak verification.

Trajectories are executed action/observation traces, never chat. A tool call with no observed
result is not a trajectory, and a row that does not record the prompt that produced it is not
trained on.

---

## The model

```text
Qwen/Qwen3.8-27B  @ 1d4bf0f2            (bf16, 96 GB card)
    ↓
Hermes 4 agent over the qwen35 wire format
    ↓
verified rollout corpus
    ↓
Spark-Hermes-3.8-27B                    (bf16 — what the competition scores)
```

The base is pinned to a revision in [`hermes/base_model.json`](hermes/base_model.json), because
a name resolves to whatever a repository holds when someone runs it. Four things about it are
easy to get wrong, and all four are recorded with the pin:

**It does not speak Hermes — and it looks like it does.** Its template writes `<tool_call>`,
`<tool_response>` and `<think>`, every one the Hermes spelling. Only the payload differs — one
element per parameter where Hermes puts a JSON object. Read as Hermes, every turn scores as
*malformed*, not silent. [`hermes/qwen35.py`](hermes/qwen35.py) exists for that reason; see
[`docs/wire-dialects.md`](docs/wire-dialects.md).

**It is multimodal.** Text hyperparameters live under `text_config`; reading the top level of
`config.json` returns `None` and computes a memory budget out of nothing.

**Its KV cache lives in 16 of 64 layers.** 48 are linear attention with fixed per-sequence
state. Counting all 64 overstates per-token cost 4×. Derived: 32,768 B/token at fp8 — **not
yet measured** against a live server.

**Serving has not been brought up against this pin.** [`docs/serving-qwen3.8.md`](docs/serving-qwen3.8.md)
records what is known and the checklist that would make it trustworthy.

The 4B proof of concept is the same architecture one size down — identical KV heads, head_dim,
context, and a byte-identical tokenizer vocabulary — so token-efficiency numbers measured on it
are numbers about the 27B.

---

## Status

The software loop runs end to end on CPU: authenticated admission, exact-bundle judging, durable
settlement, replay, corpus construction, preparation, crossed evaluation and activation. CPU
fixtures establish software behaviour only.

What is real, and what is not yet:

```text
exercised on a live model     announce, open a window, check / rehearse / upload a surface,
                              validator run, score, record, publish the round proof
tests only                    crown (nothing has reached 10/10), aggregate
not yet                       real training, trusted live serving, SN74 registration
```

**The binding constraint is challenge supply.** The suite is 19 tasks; **4** qualify as
challenges. Power to detect a 20-point paired improvement:

```text
tasks    power
    4      0.0%     <- challenges open today
   19     15.3%
   30     57.2%
   60     99.0%
```

Every number this project could publish rests on the first row. Growing the suite is the launch
gate; suite-to-challenge yield has been 21%.

---

## Quickstart

**Miner** — no model, no GPU, no pull request needed to start:

```bash
uv sync
python -m miner init  --dir ./my-surface --skill protocol-discipline
python -m miner check --dir ./my-surface
```

Then rehearse, search, upload and commit: [`docs/miner-guide.md`](docs/miner-guide.md).

**Operator** — CPU demonstration, no GPU:

```bash
scripts/install.sh && source .venv/bin/activate
spark-hermes doctor --software-only --profile rtx5090-poc
spark-hermes cycle demo --root /tmp/spark-cycle-demo --mode fixture
```

Real training: [`docs/train-spark-hermes.md`](docs/train-spark-hermes.md).

**Validator:**

```bash
scripts/serve_agent.sh /path/to/model my-model 8001          # SGLang, -> :8001/v1
python -m hermesbench.runner --suite all --model my-model --base-url http://127.0.0.1:8001/v1 \
  --episodes-out base.jsonl --keep-trajectories --allow-unsandboxed
python -m hermes.challenge --episodes base.jsonl --model-revision <rev> --harness-digest <d> \
  --out datasets/challenges/
```

Rounds, admission and settlement: [`docs/competition-ingress.md`](docs/competition-ingress.md),
[`docs/competition-settlement.md`](docs/competition-settlement.md). The withheld checks live in a
private repository; without them every run says on stderr which tasks it cannot score.

**Everyone:** `scripts/check.sh` runs the full software gate — ruff, pyright, selfcheck, tests.

---

## Repository layout

| Path | What |
|---|---|
| [`hermes/`](hermes) | trajectories, wire dialects, prompts, training formatting, base-model pin |
| [`hermesbench/`](hermesbench) | task execution, mutation supply, objective verification |
| [`validator/`](validator) | competition service: admission, judging, crown, settlement, round proofs |
| [`miner/`](miner) | the competitor's surface: init, check, rehearse, ablation search |
| [`admin/`](admin) | the operator's `spark-hermes` command: corpus, prepare, train, merge, release |
| [`eval/`](eval) | evaluation, submission gates, PR labels |
| [`teacher/`](teacher) | teacher interfaces for corpus candidates |
| [`proof/`](proof) | round-proof bundles |
| [`recipes/`](recipes) | the separate Qwen3.5-4B student line (agent recipes are under `hermes/recipes/`) |
| [`runs/`](runs) | immutable verified run records |
| [`scripts/`](scripts) | install, serve, train, check |
| [`docs/`](docs) | runbooks, serving evidence, design notes |

---

## Engineering rules

1. **Same competition model.**
2. **Correctness before efficiency.**
3. **Verification cannot regress.**
4. **Executed trajectories beat imagined trajectories.**
5. **Hidden evaluation prevents visible-task hard-coding.**
6. **Every important artifact is content-addressed.**
7. **A proof claims only what it actually binds.**
8. **Policy is not evidence.**
9. **Failures are training opportunities.**
10. **The loop matters more than one checkpoint.**

---

## Security

HermesBench executes model-authored tool actions and shell commands. Run it only inside an
approved sandbox or a disposable VM.

## License

Repository code is MIT — see [`LICENSE`](LICENSE). Upstream models, runtimes and datasets keep
their own licenses; contribution and derived-artifact policy is in [`CONTRIBUTING.md`](CONTRIBUTING.md).
