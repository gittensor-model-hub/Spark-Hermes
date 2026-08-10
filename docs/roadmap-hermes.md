# SparkDistill — Hermes-native agent roadmap

SparkDistill's product is not a model. It is a **verified worker**: a model that can be
handed a real task, will use tools to do it, and whose claim about having done it is
checkable by someone who does not trust it.

The stack:

| Component | Role |
|---|---|
| **SparkDistill** | creates intelligence specialization (data + recipes + eval) |
| **SparkProof** | provides verifiable training signals (attested, re-checkable data) |
| **SparkInfer** | makes deployment cheap and fast (local Blackwell serving) |
| **Hermes Agent** | provides the execution environment (tools the model acts through) |

A normal fine-tune is copyable in a weekend. The combination above is not, because the
moat is the *loop* — execution data feeding verification feeding training — not any single
checkpoint.

---

## Phase 0 — Qwen3.6-27B Hermes adapter (pipeline validation)

**Goal:** not the final model. Prove that SparkDistill can convert a strong general model
into a Hermes-native worker.

```
Qwen3.6-27B  +  Hermes specialization  ->  Spark-Hermes-Agent-3.8-27B
```

The question Phase 0 answers is *"does this work?"* — not *"can this compete?"*

### 0.1 Dataset: execution trajectories, not chat

The biggest failure mode available to us is collecting chat data.

Not enough — teaches an answer:

```
User:      How do I fix this bug?
Assistant: You should check memory allocation.
```

Enough — teaches a behavior:

```
User: Fix this repository bug.

<thinking>Need to inspect the repository.</thinking>
tool: terminal      command: git clone <repo>     result: repository downloaded
tool: terminal      command: pytest               result: 3 failures
<thinking>Need to inspect the failing file.</thinking>
tool: file_read     path: src/alloc.c             result: <contents>
tool: edit          patch: <diff>                 result: applied
tool: terminal      command: pytest               result: PASS

Final: Fixed bug. Tests pass.
```

The second form is what [`hermes/trajectory.py`](../hermes/trajectory.py) encodes: an
ordered list of steps, each one `thinking` / `tool_call` / `tool_result` / `final`, with
the observable outcome of every action preserved. A trajectory whose tool calls have no
results is not a trajectory — it is a transcript of a model guessing, and the schema
rejects it.

### 0.2 Data sources

Start with existing Hermes-family and agentic corpora:

- Hermes-3 dataset
- OpenHermes
- Hermes function-calling
- OpenThoughts-Agent
- SWE trajectories

Then add our own synthetic trajectories from frontier teachers (GPT-5.x, Claude, Kimi K3,
Qwen-Max) under the standing agent brief:

```
You are an expert Hermes agent. Solve this task.
Must: use tools, verify results, recover from failures, explain the final state.
```

Target scale: **10k–100k trajectories**. Generation is driven by
[`hermes/generate.py`](../hermes/generate.py); every row lands in the same schema
regardless of source, so a Hermes-3 row and a freshly generated row are interchangeable
downstream.

**Recovery matters more than success.** A corpus of only-successful trajectories teaches a
model that tools never fail. Failure-and-recovery segments — a command that errors, a test
that stays red, a retry that works — are the part that survives contact with reality, and
the schema tracks them explicitly (`recovery_steps`).

### 0.3 Training

QLoRA first, on `q,k,v,o` projections. Hardware: RTX PRO 6000 Blackwell / H100 / B200.

Three stages, in order:

| Stage | Teaches | Recipe |
|---|---|---|
| **A — reasoning repair** | planning, decomposition, debugging | `stage-a-reasoning.yaml` |
| **B — Hermes behavior** | observe → plan → act → verify → recover | `stage-b-hermes.yaml` |
| **C — tool reliability** | correct tool choice, well-formed calls, no hallucinated tools | `stage-c-tools.yaml` |

Configs live in [`hermes/recipes/spark-hermes-agent-3.8-27b/`](../hermes/recipes/spark-hermes-agent-3.8-27b/).
The order is load-bearing: stage B on a model that cannot plan produces a model that calls
tools confidently and wrongly.

### 0.4 Evaluation

MMLU and HumanEval do not measure agents. They measure whether a model knows things, and a
worker that knows everything and verifies nothing is exactly the failure mode we are
training out.

**HermesBench-v0** ([`hermesbench/`](../hermesbench/)) measures work:

```
Task: clone a repository, find the bug, write a patch, run the tests.
```

Scored on:

| Metric | Question it answers |
|---|---|
| `success_rate` | did the task actually get done (verified, not claimed) |
| `tool_efficiency` | ok calls / total calls — did it flail |
| `recovery_rate` | of episodes where a tool failed, how many still succeeded |
| `self_check_rate` | of episodes that changed something, how many looked afterwards |
| `mean_tokens` | what did the answer cost |
| `mean_wall_time_s` | how long did it take |
| `objective_completion` | long-horizon: how many sub-objectives held at the end |
| `objective_regression_rate` | long-horizon: how many it satisfied and then broke |

**Long horizon (`v1`).** Short episodes cannot exhibit goal drift or context corruption,
so `v1` tasks declare *checkpoints* — sub-objectives sampled repeatedly during the run.
An objective observed passing and later failing is drift, made concrete: not "the agent
lost the plot" but "checkpoint B passed at step 12 and failed at step 30". A single
end-of-episode check scores "finished A then destroyed it" identically to "never did A".

Long-horizon capability is a *model × harness* property, so the harness must not fight
the model: `max_verification_steps` is separate from `max_steps`, because charging an
agent's action budget for re-running its own tests penalises exactly the behavior
`self_check_rate` rewards.

Success is decided by a task's own verification command (`pytest`, a benchmark, a diff
check) — never by the model's final message. See
[`hermesbench/README.md`](../hermesbench/README.md).

---

## Phase 1 — Spark-Hermes-Agent-3.8-27B

**The first product is one model with one personality: a worker that is exceptionally
good at operating inside Hermes Agent.** Not a CUDA specialist, not an SWE specialist.

Branching into `Spark-Hermes-CUDA` / `-SWE` / `-Firmware` before the general worker is
good produces four mediocre models instead of one useful one — each trained on a fraction
of the trajectory data, each inheriting whatever the base does badly at driving the
harness. The specialists in Phase 3 all branch from this checkpoint, so its quality is
their floor.

It is judged on whether it can *drive the harness*, not on what it knows:

| Question | Measured by |
|---|---|
| Does it pick the right tool, with the right arguments? | `tool_calling` |
| Can it work a multi-command shell problem to completion? | `terminal_agent` |
| Can it hold an objective across a hundred steps without drifting? | `long_horizon` |
| **Does it verify its own work before declaring success?** | `self_verification` |

MMLU, HumanEval and GSM8K measure the brain. These measure the worker, and a model that
scores well on the first set and badly on the second is not shippable as an agent.

The positioning follows from that: not "a Qwen fine-tune", but **the first open-weight
model optimised specifically for Hermes Agent execution**. The model is one component; the
durable asset is the corpus of verified Hermes behaviour behind it.

When Qwen3.8-27B lands:

```
Qwen3.8-27B  ->  SparkDistill  ->  Spark-Hermes-3.8-27B
```

Phase 0 asked "does this work?". Phase 1 asks **"can this compete?"** — same pipeline,
full fine-tune budget, real corpus scale, HermesBench-v1 with adversarial tasks.

### Training recipe

**Stage 1 — reasoning foundation.** Bespoke-Stratos, NuminaMath, OpenThoughts, DeepSeek
reasoning traces. Improves planning, decomposition, debugging.

**Stage 2 — Hermes behavior.** Hermes trajectories, OpenHands trajectories, SWE-agent
trajectories, computer-use traces. Teaches observe → plan → act → verify → recover.

**Stage 3 — tool specialization.** Python execution, browser, terminal, git, docker, CUDA
tools. The model learns routing reflexes:

```
need numerical verification?  -> python
need a current API?           -> browser
need a code change?           -> terminal
```

---

## Phase 2 — native tools layer

Most agent stacks are:

```
LLM -> tools
```

Ours interposes a layer that makes claims checkable:

```
LLM -> Tool Intelligence Layer -> Verified Execution
```

### Python verification tool

The important one. A model says:

```
Matrix multiplication optimization gives 2x speedup
```

Instead of trusting that, the verifier runs it:

```
python benchmark.py
before: 100ms   after: 45ms   PASS
```

and the result becomes a training reward. That closes the loop:

```
claim -> execute -> verify -> reward
```

This is the same trust posture SparkProof already applies to datasets, moved to runtime
behavior: nothing counts because the model said it.

### Web reasoning tool

Not "always search" — the model estimates its own uncertainty and searches only when it
is low-confidence.

```
"Explain CUDA streams"          -> high confidence -> answer directly
"Latest CUDA 13 API changes"    -> low confidence  -> search, then answer
```

Always-search burns tokens and latency on things the model already knows; never-search
produces confident staleness. The estimate is the feature.

---

## Phase 3 — specialized Hermes workers

Instead of one giant model, a family of specialists fine-tuned from
`Spark-Hermes-3.8-27B`:

| Worker | Trained on | Does |
|---|---|---|
| **Spark-Hermes-CUDA** | CUDA repos, NVIDIA docs, CUTLASS, TensorRT, Triton, Nsight traces | optimize kernels, analyze profiles, write CUDA, debug performance |
| **Spark-Hermes-Firmware** | ESP32, STM32, Zephyr, FreeRTOS, Linux drivers, U-Boot | write drivers, debug UART, optimize memory, OTA updates |
| **Spark-Hermes-Cyber** | CyberGym, CVE patches, kernel vulnerabilities, fuzzing | find vulnerabilities, build PoCs, verify patches |
| **Spark-Hermes-SWE** | GitHub history, PR discussions, issues, commits | understand repo philosophy, fix issues, prepare PRs |

`Spark-Hermes-CUDA` is the direct descendant of this repo's existing Triton work — the
kernel-specialist corpus and TritonBench harness become that worker's domain eval rather
than the whole project's purpose.

---

## Phase 4 — intelligent router

```
                    User
                     |
              Hermes Router
                     |
      +--------------+--------------+
      |              |              |
  CUDA Agent    Firmware Agent   Cyber Agent
```

The router decides *"this is a CUDA optimization"* and dispatches. It can be a small
**3B–7B** model, because choosing a specialist is a far easier problem than being one.

Implemented in [`hermes/router/`](../hermes/router/).

### Abstention is the design, not a shortfall

The two routing errors do not cost the same:

| Mistake | Cost |
|---|---|
| firmware task → `general` | some quality; the generalist tries, and the same verification loop checks it |
| firmware task → `cyber` | a specialist works **confidently outside its training** |

Specialists are tuned to act, so one acting outside its domain is the expensive failure.
The router therefore falls back to the generalist rather than guess, and every decision
records whether it *abstained* or genuinely chose `general` — identical `target`, very
different meaning. `evaluate()` reports `misroute_rate` and `abstention_rate` separately,
because a router that trades misroutes for abstentions has improved even if its accuracy
has not moved.

Cross-domain work goes to the generalist by rule: *"profile the CUDA kernel, fuzz the
parser, open a PR"* is not a CUDA task with noise, it is one task spanning three
specialists, and giving it to any one of them means two thirds gets done out-of-domain.

### The cascade: rules first, tiny LLM only when unsure

```
task -> KeywordRouter --confident--> specialist                      (free)
             |
             +--unsure--> tiny router (3B-7B) --> specialist / generalist   (paid)
```

Most real tasks name their domain somewhere, and matching a word costs nothing. Routing
everything through an LLM pays full price for the easy majority; routing nothing through
one sends every ambiguous task to the generalist. `CascadeRouter` pays only for the hard
minority — on `routing_v0` the free tier settles **74.2%** and the tiny router sees
**25.8%**.

Only *uncertainty* escalates. `no_evidence` and `too_close` mean the rules cannot tell,
and a model might; `cross_domain` means the task provably spans three specialists, so the
generalist already is the answer and a second opinion is a wasted call. That distinction
rides on `reason_code`, not on matching a message string. `evaluate()` reports
`escalation_rate` so the saving is measurable, and a cascade with no tiny router deployed
degrades to the free tier rather than failing.

### The baseline exists to keep the model honest

`KeywordRouter` is deterministic, explainable and free. It is shipped so that a learned
router has a floor to clear — without one, *"the 3B router is 71% accurate"* is an
unreadable number. `compare()` reports a candidate that merely matches the baseline as
**not justifying its inference cost**, and one that buys accuracy with extra misroutes as
a regression.

### What the shipped routing suite does and does not prove

`tasks/routing_v0.jsonl` (31 labeled tasks) and the keyword lists in `domains.py` were
**authored together**, so the baseline scores 100% on it by construction. That makes it a
**regression harness, not validation**, and it leaves a learned router no headroom to
demonstrate anything.

Validating a real router needs `routing_v1`: tasks written by someone who has not seen
`domains.py`, ideally sampled from real requests, labeled by more than one person, with
disagreements preserved rather than resolved into false clarity. Cross-domain labels are
judgment calls — `cross-02` is labeled `firmware` and a reasonable person could say
`cyber`.

### Still to build

A served 3B–7B checkpoint behind `ModelRouter` (the interface, prompt, guards, and the
cascade that calls it all exist; the model does not), and the `routing_v1` held-out set
above.

---

## Phase 5 — SparkInfer deployment

The whole stack becomes practical on one desk:

```
RTX 5090 -> SparkInfer -> Spark-Hermes-3.8-27B-Q4 -> 300+ tok/s
```

```
Laptop -> Hermes -> CUDA specialist -> tools -> verified result
```

---

## The product

Not "another LLM":

```
                SparkOS AI Worker
                       |
                 Hermes Runtime
                       |
        ------------------------------
        |            |               |
   CUDA Expert  SWE Expert   Firmware Expert
        |            |               |
        ------------------------------
                       |
                SparkInfer Runtime
                       |
              Local Blackwell GPU
```

Qwen3.8-27B is only the brain substrate. The moat is:

1. Hermes execution data
2. SparkProof verification loop
3. SparkDistill pipeline
4. SparkInfer optimized deployment

**The model is not the product. The verified worker is the product.**

---

## Status

Phase 0 scaffolding is in the tree and runnable end to end on synthetic input; no
27B training run has happened yet, and no Phase 0 checkpoint exists. Base-model
availability (Qwen3.6-27B / Qwen3.8-27B) and frontier-teacher generation budget are
the two external gates on starting the real run.

| Phase | State |
|---|---|
| 0 — pipeline validation | scaffolding landed; awaiting base model + generation budget |
| 1 — full specialization | blocked on Phase 0 + Qwen3.8-27B release |
| 2 — native tools layer | verification interface designed (`hermesbench.verify`), tools not built |
| 3 — specialist workers | not started; CUDA corpus + TritonBench already serve as the CUDA worker's eval |
| 4 — router | **cascade, baseline and eval landed** (`hermes/router/`); no served tiny router, no held-out set |
| 5 — SparkInfer deployment | not started |

### What "landed" means for Phase 4

Real, tested code: the taxonomy, the abstention rules, the free keyword baseline, the
model-router prompt/parse/guards, and the evaluation harness. **No model is served**, and
the shipped routing suite cannot validate one (see above). The router is usable today as
a deterministic dispatcher and as the scaffolding a small model drops into.

---

## Where the Triton work sits now

TritonBench is **not** legacy. It is load-bearing in three places, and removing it would
break live on-chain rewards:

1. **The frontier is a triton composite.** `runs/frontiers.json` scores both
   architectures on `triton` / `triton_exec_pass_rate` / `triton_correctness`. Every
   miner's `eval:XS–XL` label is a delta against those numbers — delete the harness and
   there is nothing left to score against.
2. **Anti-forgery depends on it.** `eval.attested_samples.verify_tritonbench_report`
   validates attested samples so a miner cannot fabricate scores.
3. **Phase 3 inherits it.** It becomes **Spark-Hermes-CUDA**'s domain evaluation.

What changes across this roadmap is its *scope*, not its existence: kernel skill stops
being the whole project's purpose and becomes one specialist's measured competence.
