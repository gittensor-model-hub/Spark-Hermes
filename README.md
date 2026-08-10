![Spark-Hermes banner](docs/images/banner.png)

<sub>Two things in the artwork above do not match the system and are being redrawn. The rollout
host is an **RTX PRO 6000 Blackwell Server Edition (96 GB)**, not a GeForce RTX 5090 32 GB. And
"lower latency" is not a thing a miner is scored on — wall time is reported and never gates the
crown, for the reason given under [Pareto frontier](#pareto-frontier).</sub>

# SPARK-HERMES

### Verified agent intelligence, continuously improved by SN74 Gittensor

**Target:** `Spark-Hermes-3.8-27B`
**Development baseline:** pinned `Qwen3.6-27B` @ `6a9e13bd`, served bf16
**Runtime:** Hermes 4
**Competition:** verified rollout optimization
**Execution:** NVIDIA Confidential Computing enabled; Intel TDX quote verification implemented,
no approved guest measurement pinned yet

> **Same model. Better rollout. Verified improvement. Better next model.**

Spark-Hermes is an open-weight Hermes-native agent model improved through continuous,
verifiable competition.

```text
RUN → OPTIMIZE → VERIFY → LEARN → REPEAT
```

The current model attempts real tasks. When it fails, loops, uses excessive tokens or tool
calls, or takes too long, that execution becomes a challenge.

SN74 miners compete to make **the same frozen model** perform the task better. Accepted
improvements are hardware-attested, independently verified, added to the training corpus,
and used by the maintainer to train the next open Spark-Hermes checkpoint.

---

## The continuous improvement loop

```text
TASK
  │
  ▼
FROZEN SPARK-HERMES MODEL
  │
  ├── verified + efficient ────────────────┐
  │                                        │
  └── fail / loop / expensive              │
             │                             │
             ▼                             │
       CHALLENGE PACKAGE                   │
             │                             │
             ▼                             │
       SN74 MINERS                         │
   optimize agent strategy                 │
             │                             │
             ▼                             │
  INTEL TDX + NVIDIA GPU CC                │
             │                             │
             ▼                             │
       VERIFY + SCORE                      │
   correctness first                       │
   efficiency second                       │
             │                             │
             ▼                             │
  VERIFIED SFT / PREFERENCE / RL DATA      │
             │                             │
             ▼                             │
    TRAIN NEXT SPARK-HERMES                │
             │                             │
             └─────────────────────────────┘
                         REPEAT
```

This is the product: not a one-time fine-tune, and not a miner race to use the biggest
hidden model.

---

## What miners compete on

For each challenge, the model and execution environment are fixed.

### Fixed by the epoch

What the code pins is [`PINNED_CONFIG_KEYS`](hermes/profile.py):

```text
model                     terminal.timeout
provider                  toolsets
agent.reasoning_effort    tool_output
terminal.backend          compression
                          context.engine
```

alongside the model digest, the tokenizer, the Hermes commit, the system prompt and tool
schemas, the initial workspace, the verification contract, the budgets and the replay seeds.

### Miner-controlled engineering space

```text
system / SOUL prompt      recovery policy
skill selection           verification policy
planning policy           stopping policy
context management        tool-use policy
```

Miners do **not** win by changing model weights. They win by making the same model behave
better.

**Sampling is not pinned, and was described in two contradictory ways.** This list used to end
with "sampling policy" as miner-controlled while the section below claimed the epoch fixes
"generation settings". Both cannot hold, and neither matches the code: the pinned list fixes
`agent.reasoning_effort` and says nothing about temperature or top-p. Read it as unpinned — a
pin that is only described is not a pin, and a miner tuning temperature today is inside the
rules as written.

---

## One canonical model epoch

Every miner runs the same content-addressed deployment artifact for a model epoch.

```text
open base / Spark-Hermes checkpoint
        ↓
canonical inference runtime
        ↓
model epoch digest
        ↓
same bytes on every miner
```

"Same model" means identical weights, tokenizer, runtime configuration and Hermes protocol —
not merely the same model name.

**Nothing is quantized.** This section was titled "One canonical quantized model" and put a
quantization step in that chain. [`hermes/base_model.json`](hermes/base_model.json) pins
`Qwen/Qwen3.6-27B` at revision `6a9e13bd` with no quantization field, and the measured baseline
served it in bf16 — which is why the KV figure in that file is `kv_bytes_measured` rather than
one computed from an assumed precision. Quantization and kernel work would belong in a separate
inference-optimization track; no such track exists.

---

## Confidential miner execution

The target rollout track uses:

```text
NVIDIA RTX PRO 6000 Blackwell Server Edition
+
NVIDIA Confidential Computing
+
Intel TDX confidential VM
```

The proof chain binds:

```text
model epoch
task + challenge lease
miner strategy
runtime
trajectory/result root
hardware state
```

The validator verifies the Intel and NVIDIA evidence, exact artifact digests, task scope,
resource accounting, replay protection, and the public/hidden task verdicts.

The goal is to prove that **this exact approved model and strategy produced this exact
result inside the declared confidential environment**.

---

## How a task becomes a challenge

The canonical model first runs the task with the canonical strategy.

A challenge is created when the baseline:

```text
fails verification
times out
enters a no-progress loop
uses excessive tokens
uses excessive model turns
uses excessive tool calls
repeats expensive actions
finishes without enough verification
or succeeds far above the expected resource envelope
```

Example:

```text
BASELINE
result      FAIL
tokens      57,000
tool calls  38
wall time   300 s

MINER CANDIDATE
result      PASS
tokens      22,000
tool calls  11
wall time   74 s
```

Failure → verified success is the highest-value improvement.

---

## Endless task generation

The validator does not hand-write an infinite benchmark. Task supply comes from:

```text
real user / agent workloads
public repositories at pinned commits
issues and regression cases
existing agent benchmarks
generated tool-use scenarios
mutation-generated repository failures
failure clusters from the current Spark-Hermes model
```

### Mutation supply

```text
known-good repository
      ↓
controlled mutation
      ↓
confirm the original verifier now fails
      ↓
new repair task
```

Mutation families can vary logic, boundary conditions, configuration, dependency state,
multi-file consistency, error handling, tool availability, generated artifacts and
performance constraints.

Equivalent mutations are discarded.

### Difficulty mining

Generated tasks are classified against the current baseline:

```text
too easy   → coverage / SFT
frontier   → miner competition / preference signal
very hard  → failure discovery / future curriculum
```

As Spark-Hermes improves, the challenge generator must continuously find harder failures.

---

## A withheld check prevents hard-coding

**Hidden sibling tasks do not exist.** This section described each visible challenge as the
head of a family of hidden variants — A, B, C — varying repositories, filenames, constants and
error order. No task in the corpus declares a sibling or a family; grep for either and the only
hit is a line in [`hermes/miner_contract.json`](hermes/miner_contract.json) citing "the
hidden-sibling defence" as though it were already in place. The mechanism further down this
file has always listed sibling verification as a *planned* boundary, and this section
contradicted it.

What exists, on all 19 tasks, is a **withheld check on the same task**:

```text
visible task
├── published verifier      the miner can read it, so it can be optimised against
└── withheld verifier       committed by salted digest, held privately
```

Both run on every attempt. A strategy that passes the published check and fails the withheld
one is counted in `overfit_rate` — it learned the benchmark rather than the job. That is a
narrower defence than a task family: it catches a strategy fitted to the *published assertions*,
and it does not catch one fitted to this repository's filenames and constants. Sibling tasks
would catch that, which is why they are still on the roadmap.

Each task's salt is derived per task, `HMAC(master, task_id)`, so publishing one task's salt
after its round settles says nothing about any other. Under a single shared salt the first
audit would unseal every unspent task in the corpus.

---

## How miner work is scored

### Correctness is a gate

No efficiency reward exists until all required gates pass:

```text
correct model epoch
valid Intel TDX + NVIDIA CC evidence
valid Hermes protocol
allowed tools only
complete trace
public verifier pass
hidden verifier pass
security checks pass
minimum verification strength
budget respected
```

### Two kinds of win

**1. Failure → success**

```text
baseline:  fail
candidate: pass
```

**2. Expensive success → efficient success**

Both pass, but the candidate improves:

```text
model turns ↓            the axis a strategy actually moves
input tokens ↓           follows from turns; see below
weighted tool cost ↓
repeated actions ↓
output tokens ↓          5.1% of the bill
```

while maintaining equal or stronger verification.

**Input and output tokens are not two comparable axes.** Measured over the 190-episode baseline:

```text
input   9,271,813   94.9%      re-sent context
output    493,190    5.1%
cache hit rate          0.00%  prefix caching was off
```

Input dominates by nearly twenty to one, and almost all of it is the same conversation re-sent
on every turn with no prefix cache behind it. So a strategy does not reduce input tokens by
writing less — it reduces them by taking fewer turns, and input follows. Listing the two side
by side invites a miner to optimise the 5% and read the 95% as separately winnable. Wall time
is deliberately absent from this list; the section below says why.

### Verification cannot regress

The cheapest possible rollout is to stop checking. That is not optimization.

Fewer tests or verification calls count only when the task's required verification remains
fully satisfied.

---

## Pareto frontier

There is no universal exchange rate between one token, one tool call and one second.

After correctness gates, candidates are compared as a vector:

```text
gates the crown          verified success
                         overfit rate (published pass, withheld fail)
                         tokens
                         model turns
                         weighted tool use
                         repeated actions

reported, never gates    wall time
```

Non-dominated candidates remain on the challenge's Pareto frontier. A pre-registered SN74
reward policy distributes emissions across that verified frontier.

### Wall time is reported and never gates the crown

This section claimed wall time was a frontier axis, normalized as
`normalized_time = candidate_time / baseline_time`, "so hardware variance does not masquerade as
strategy quality". [`hermes.acceptance.dominates`](hermes/acceptance.py) does the opposite, and
deliberately:

> Tokens and tool calls are exact and recomputable from the trace. Wall time is not, and a
> king-of-the-hill bar that only ever rises would lock in whichever run got favourable
> scheduling — permanently, because no later run could legitimately beat it.

Dividing by a baseline on the same node removes the node's *average* speed. It does not remove
run-to-run variance, and the crown is a ratchet: once a lucky measurement sets the bar, no
honest strategy can clear it and the bar never decays. A ratio does not fix a metric that is not
recomputable from the trace, so latency is reported beside the crown and gates nothing.

---

## What accepted rollouts train

The successful candidate is especially valuable because it comes from the same model that
originally failed.

### SFT

Verified trajectories teach:

```text
observe → plan → act → read result → recover → verify → stop
```

### Preference data

Matched states can create strong preferences:

```text
state:
  patch exists
  tests have not run

chosen:
  run targeted test

rejected:
  claim completion
```

Policy-only ties are not quality preferences.

### Critic / recovery data

Failed baselines teach:

```text
loop detection
premature completion
bad tool choice
failed recovery
redundant context
weak verification
```

### Later RL

Repeated rollouts from the same student policy can later support verifier-reward RL / GRPO.

---

## The SN74 learning flywheel

```text
M0 = current model
H0 = current strategy

miners improve:
(M0, H0) → (M0, H1)

verified rollouts become:
D1

maintainer trains:
(M0, H1, D1) → M1

M1 becomes the next frozen model
        ↓
new failure frontier
        ↓
new miner competition
        ↓
new verified dataset
        ↓
M2
        ↓
repeat
```

> **Miners improve behavior. Verified behavior becomes data. Data improves the weights.
> Better weights create a harder frontier for miners.**

---

## Why trajectories, not chat

Chat teaches an answer:

```text
User: How do I fix this bug?
Assistant: Check memory allocation.
```

A real rollout teaches behavior:

```text
inspect repository
→ run failing test
→ inspect evidence
→ patch
→ run targeted test
→ run full verifier
→ stop
```

[`hermes/trajectory.py`](hermes/trajectory.py) represents executed action/observation traces.
A tool call with no observed result is not a real tool-use trajectory.

### A trajectory is only half a row

The other half is the prompt that caused it. Assistant turns are a response to specific
text, so a row that does not record which prompt it ran under cannot honestly be trained
on, and one exported under a *different* prompt teaches instructed behaviour as though it
were spontaneous. Executed rows carry their prompt, and the exporter refuses one that does
not.

That makes the question answerable rather than assumable: the model is served under some
prompt, and if it is not the one that produced the data, the difference has a price.
[`hermes/leakage.py`](hermes/leakage.py) measures it — on the current mining export, 83 of
133 rows carry assistant text whose only source is the system prompt.
`hermes.format --system-policy` is where that is decided. Measure first: once the prompt is
gone there is nothing left to trace the text back to, and the rate reads zero because it is
unmeasurable, not because the corpus is clean.

---

## The stack

| Component | Role |
|---|---|
| **Spark-Hermes** (this repo) | challenge generation, rollout competition, corpus construction and training |
| [**SparkProof**](https://github.com/gittensor-model-hub/SparkProof) | verified Triton training corpus, and its export/dedupe primitives |
| [**SparkInfer**](https://github.com/gittensor-ai-lab/sparkinfer) | canonical fast inference for the frozen model |
| **Hermes Agent** | agent runtime, tools, sessions and execution semantics |
| **SN74 Gittensor** | open competition and rewards for verified marginal improvement |

---

## Model path

### Development

```text
Qwen3.6-27B
    ↓
Hermes 4 baseline
    ↓
verified rollout-evolution corpus
    ↓
Spark-Hermes development checkpoint
```

### Target

```text
open Qwen3.8-27B base when available and pinned
    ↓
same verified pipeline
    ↓
Spark-Hermes-3.8-27B
```

The exact revision and artifact digest—not the marketing name—define a model epoch.

The development baseline is pinned in [`hermes/base_model.json`](hermes/base_model.json):
`Qwen/Qwen3.6-27B` at `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. A name resolves to
whatever a repository holds when someone runs it, so two runs could agree on every other
digest here and still have trained on different weights.

Two things about that baseline are easy to get wrong and are recorded with the pin. It is a
`Qwen3_5ForConditionalGeneration` with a vision tower, so "27B" is not 27B of text
parameters and the text hyperparameters live under `text_config`. And its own chat template
emits `<tool_call>`, `<tools>`, `<tool_response>` and `<think>` and does not emit
`<scratch_pad>` — which is how the Hermes 4 dialect was established here: by reading the
model, not by choosing for it. Nothing forks Hermes; [`hermes/templates/`](hermes/templates)
pins the rendered wire format as bytes so a change to it has to appear as a diff.

Qwen3.8-27B is announced and **not yet published**.

---

## Current status

Already present in the repository:

```text
Hermes trajectory representation
Hermes protocol handling
task execution + objective verification
dataset generation
SFT / DPO export
Pareto selection
cost accounting
mutation task supply
hardware-attestation utilities
submission gates
training recipes
```

Next production boundary:

```text
miner StrategySpec
TDX + GPU CC generation proof
approved guest measurement pinned
paired baseline/candidate execution
hidden sibling tasks
improvement scoring
strategy frontier
continuous model retraining
separate inference-optimization track (quantization, kernels)
```

Baseline failure mining and challenge packaging have moved out of this list: the first four
challenges are in [`datasets/challenges/`](datasets/challenges), opened by
`python -m hermes.challenge` from the baseline's own episode log.

Planned rollout-evolution components are not claimed as live until their production path is
merged and exercised.

**The withheld verifiers are not in this repository.** Each task carries a salted
commitment to its withheld check instead; the checks themselves are held privately, and
`hermesbench.withheld.overlay` attaches them and verifies each against its commitment, so
the private half cannot drift from what was published here. A checkout without that tree is
a public checkout -- a legitimate state, and `suitecheck` reports what it cannot score
rather than reporting a clean zero. A verifier a model can read is one it can be optimised
against, which is the whole reason `overfit_rate` means anything.

Two limits worth stating rather than discovering later.

**Challenge supply is the binding constraint, and it is tighter than task supply.** The suite is
19 tasks, not the 16 this file used to claim (16 in `v1`, 3 in `v0`). Simulated power to detect a
20-point paired improvement under the exact McNemar test in
[`hermesbench/repeats.py`](hermesbench/repeats.py):

```text
tasks    power
    4      0.0%     <- challenges actually open
   19     15.3%     <- the whole suite
   30     57.2%
   60     99.0%
```

But a task only becomes a challenge if the baseline reliably fails it, and of the 19 exactly
**4** qualified — the rest the baseline either handles or passes too often to be worth a round.
So the number that governs every claim above is 4, where the test has no power at all. Growing
the suite is necessary and not sufficient; the yield from suite to challenge was 21%.

**Attestation is verified but not yet policed.** [`eval/verify.py`](eval/verify.py) checks NRAS
tokens and extracts TDX `REPORTDATA` from the quote itself rather than trusting the
miner-editable JSON field, and [`eval/rollout_track.py`](eval/rollout_track.py) refuses a
submission whose digests disagree with the attested manifest. What is missing is the policy: no
approved guest measurement is pinned, so a quote proves a genuine confidential VM ran and
committed to this bundle, not that it ran an image we approved. The GPU side has CC enabled but
`nvtrust`/NRAS has not been exercised end to end on the rollout host.

---

## Repository layout

| Path | What |
|---|---|
| [`hermes/`](hermes) | trajectories, protocol, prompts and training formatting |
| [`hermesbench/`](hermesbench) | task execution, mutation supply and objective verification |
| [`eval/`](eval) | evaluation and submission gates |
| [`proof/`](proof) | proof and attestation utilities |
| [`recipes/`](recipes) | SN74 training recipes |
| [`runs/`](runs) | immutable verified run/frontier records |

---

## Existing pipeline quickstart

```bash
uv sync

cat hermes/base_model.json

python -m hermesbench.runner --suite v0 --workspace-root /tmp/hermesbench --list

python -m hermes.generate \
  --tasks hermes/tasks/phase0.jsonl \
  --out data/processed/hermes_trajectories.jsonl \
  --provider anthropic

python -m hermes.format \
  --in data/processed/hermes_trajectories.jsonl \
  --out data/processed/hermes_trajectories_sft.jsonl

# before choosing what happens to the system prompt, measure what removing it costs
python -m hermes.leakage --in data/processed/hermes_trajectories_sft.jsonl
```

The rollout-evolution CLI will be documented when the attested miner path is production-ready.

---

## Engineering rules

1. **Same competition model.**
2. **Correctness before efficiency.**
3. **Verification cannot regress.**
4. **Executed trajectories beat imagined trajectories.**
5. **Hidden evaluation prevents visible-task hard-coding.**
6. **Every important artifact is content-addressed.**
7. **Attestation claims only what the proof actually binds.**
8. **Policy is not evidence.**
9. **Failures are training opportunities.**
10. **The loop matters more than one checkpoint.**

---

## SN74 × Hermes

```text
SN74 competition
      ×
Hermes execution
      ×
confidential verification
      ×
open-weight training
      =
continuously improving verified agent intelligence
```

---

## Security

HermesBench executes model-authored tool actions and shell commands.

Run it only inside an approved sandbox, disposable VM or confidential workload environment.

---

## License

MIT — see [`LICENSE`](LICENSE).
