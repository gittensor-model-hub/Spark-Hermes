# Training pipeline history and current operator entry points

For current operator commands and readiness limits, use the [training runbook](train-spark-hermes.md).
The measurements below are historical, preceding the current CPU delivery; they do not prove
learning or serving compatibility for the current 4B or 27B pins. The installed CLI now connects
generation, verified corpus/replay, approved-parent preparation, candidate identity, crossed
agent/model evaluation and resumable release cycles. See [cycles](cycles.md),
[learning ingress](learning-boundary.md) and [crossed release](crossed-release.md).

The contribution stack is public and reusable; derived Spark Hermes artifacts and licensed
data remain private unless explicitly released. Preserve upstream Qwen Apache-2.0 and Hermes
MIT notices, modification attribution and separate data/contribution grants as described in
[CONTRIBUTING](../CONTRIBUTING.md). Local quality scores and CPU fixtures do not establish
model novelty, learned improvement, SN74 registration or live payout.

Run `spark-hermes doctor --software-only`, `spark-hermes selfcheck` and
`spark-hermes cycle demo --root /tmp/spark-cycle-demo --mode fixture` for the CPU path.
Use `spark-hermes status --root WORKSPACE --profile rtx5090-poc` to inspect a generated cycle
workspace. The selected first training target is pinned Qwen3.5-4B on RTX 5090; the final
`bf16` target stays pinned Qwen3.8-27B on PRO 6000 96 GB. Real training, licensed corpus and
fresh private checks, trusted serving, hardware/optional attestation and external SN74
onboarding remain separate prerequisites in both empty and populated fixture roots.

The original report below covered one operator and one GPU without miners, between 2026-08-11
and 2026-08-12. Its measurements are retained as historical observations.

```
  seeds ──► DNA ──► synth ──► GATE ──► tasks ──► probe ──► rollout ──► corpus ──► train ──► evaluate
 public    shape   a model   9 executed        measure    N attempts   best-of-N   axolotl   held-out
 traces            writes    checks            difficulty  per task     + pairs               suite
```

The historical branch was `admin/training-pipeline`. The competition machinery — submissions, commit–reveal,
receipts, the board, the crown — is not in this path. What is kept from it, and why, is in
[`admin/__init__.py`](../admin/__init__.py).

---

## 1. Seeds → task DNA

    python -c "from hermes.taskgen.seeds import read; list(read('lambda', limit=400))"

`lambda/hermes-agent-reasoning-traces` (apache-2.0) holds 7,646 real agent traces. **7,608 of them map
onto this harness's four tools.** Browser Automation (1,048 rows) is dropped rather than remapped: a
browser task performed with curl is a different task, and calling it the same puts a capability in the
corpus that no seed demonstrated.

Only the **shape** crosses over — domain, specialism, skills, horizon band, failure mode, tools.
`dna.assert_abstract` fails a DNA that shares a 40-character run with its seed, because the licence of
a public dataset governs what may be copied from it and the failure is otherwise silent.

Two classifier bugs were found here by printing distributions rather than reading examples: every DNA
in a 400-seed sample came out `repository_engineering`, and `error_recovery` fired on 93% of seeds.
Both were fields that looked measured and were constant.

Seed order is **stratified**. The parquet is grouped by category, so reading it in order gives one
bucket: 2 domains sequential versus 5 stratified over the same 40 rows.

## 2. DNA → a candidate task

    # driven by hermes/taskgen/cli.py; see stage 3

A model receives the abstraction and nothing of the seed, and writes seven delimited sections: the
prompt, the setup script, the published check, the withheld check, a reference solution, a **second
reference solution by a different route**, and a cheat.

Delimited blocks rather than JSON. Every artefact is a shell script, and asking a model to escape
heredocs inside JSON strings turns one missed backslash into an unparseable blob — or worse, a script
that parses and does something other than what it reads like.

## 3. The gate — nine executed checks

    python -m hermes.taskgen.cli --count 150 --concurrency 8 \
        --out var/tasks/gen-1 --salt-file <master salt> \
        --base-url http://127.0.0.1:8001/v1 --model qwen3.8-27b \
        --request-timeout 900 --max-tokens 20000

| # | check | why |
|---|---|---|
| 1 | setup exits 0 | |
| 2 | setup is **deterministic** — run twice, byte-identical | a withheld check cannot be pinned to values that move |
| 3 | published check **fails** an untouched workspace | otherwise every episode is a success |
| 4 | withheld check **fails** an untouched workspace | same |
| 5 | reference solution runs | without it a failed episode is unattributable |
| 6 | published check **passes** the reference | |
| 7 | withheld check **passes** the reference | else the task is unpassable and every episode is `overfit` forever |
| 8 | the two checks **disagree on a cheat** | a withheld check that agrees everywhere withholds nothing |
| 9 | the withheld check **accepts a different method** | see below |

Check 9 exists because the first task the gate ever accepted demanded a *symlink specifically*. An
agent that set an environment variable or copied the file would have fixed the service for real and
still failed, scoring as `overfit` while being correct. Checks 6 and 7 are structurally blind to it:
the reference solution comes from the same reply as the check and satisfies it by construction. **12
tasks were rejected by check 9** in one 160-task run.

Nothing is written before it is accepted. An accepted task gets its YAML, its withheld check, its
reference solution, and a real salted commitment under a per-task salt derived from the master.

**Rejected attempts keep all five scripts and the shell's own complaint.** `setup_exits_zero: 4` says a
script failed and nothing about why; the first diagnosis that mattered — every failure was an absolute
path like `mkdir /workspace` — took thirty seconds once the scripts were on disk.

### Measured

| | |
|---|---|
| acceptance, first run | 12% |
| acceptance after the relative-path fix | **81%** |
| tasks generated | 160 |
| duplicates | **none** — closest pair scores 0.056 against a 0.5 threshold |
| domains | 5 |

Duplicate rejection is character-shingle Jaccard on the prompt, calibrated against the real suite: two
hand-written ordering tasks score 0.018, a task against itself 1.000. A corpus wants many tasks per
skill and no task twice.

## 4. The probe — measure difficulty before paying for it

    python -m hermesbench.runner --suite generated --task-root var/tasks/gen-1 \
        --repeats 2 --concurrency 12 --keep-trajectories --allow-unsandboxed \
        --episodes-out var/probe/episodes.jsonl --out var/probe/manifest.json

Withheld checks live. This exists because the gate proves a task *executes correctly*, not that it is
*worth solving*, and eight rollouts of a trivial task cost the same as eight of a hard one.

**Run it at `--repeats 8`, not 2, and classify with `hermes.taskgen.triage`:**

    python -m hermes.taskgen.triage --episodes var/probe/episodes.jsonl \
        --repeats 8 --report var/probe/triage.json --keep-list var/probe/keep.txt

Keep `1 <= passes < repeats`. An all-pass task supplies no gradient and an all-fail task supplies no
reachable example. The measurements below were taken at `--repeats 2`, and two attempts do not
support a verdict: through this repo's own `hermesbench.repeats.wilson`, 2/2 leaves the true pass
rate anywhere in [0.34, 1.00] and 0/2 anywhere in [0.00, 0.66]. `hermes.challenge` already refuses to
open a challenge on a count for that reason; the probe that feeds it was never held to the same rule.
So read the `easy 117 (75%)` line below as undersampled rather than established.

### Measured — 320 episodes over 157 tasks

```
easy   2/2 pass         117  (75%)
hard   0/2 pass          27   of which 22 were ALL-truncated
mixed  1/2 pass          13   of which 10 were ALL-truncated

genuinely hard              5
genuinely mixed             3
truncated             108/314  (34%)
overfit                48  (15%)
malformed               1
withheld ran        314/314
```

Two findings, both instrument failures rather than model failures:

**32 of the 40 non-easy tasks failed only because they ran out of actions.** The old budget was
`max(6, horizon[1])` — the seed's own upper estimate as a ceiling, leaving no room to look around, be
wrong once, or check work. `hermes/taskgen/rebudget.py` raises it where a probe showed it binding, and
re-probing moved tasks `hard → easy` and `mixed → easy`, never the other way. The tasks were not hard.

**48 "overfit" episodes were mostly wrong answers.** On `gen-data-0142` the model ran
`awk -F, 'NR>1 {sum+=$2}'`, wrote 150, and the answer was 80 — but the published check was
`grep -Eq '^[0-9]+$'`, which asks *is it an integer*. A plainly wrong answer passed it and failed the
withheld one, and the episode was recorded as gaming. `overfit_rate` is the one metric whose job is to
detect a strategy fitting visible assertions, so this corrupts the instrument you would rely on to
notice a trained model gaming the benchmark.

## 5. Rollout

    python -m admin.cli rollout --root var/admin/run-1 --repeats 8

N attempts per surviving task through the same harness. `admin/split.py` refuses if any evaluation task
reached the set — the 19 hand-written tasks are what every published number is measured on, and
`overfit_rate`, the check that would notice the mistake, is measured on them too.

## 6. Corpus

    python -m admin.cli corpus --root var/admin/run-1

`validator/aggregate.py` renders rows through `hermes.format.to_messages_record`, which is dialect-aware:
reasoning goes to `reasoning_content` and tool arguments to a mapping for ATEM, `<think>` in content and
a JSON string for Hermes. That parameterisation is what makes a corpus trainable at all — and it is also
what lets a **teacher trajectory captured in Hermes render into ATEM**.

**SFT is best-of-N**, one row per task. Keeping every verified attempt is not more data: on a real
8-repeat run one task contributed 8 rows spanning 17,779 to 38,507 tokens — the same task solved the
same way, eight times. SFT is imitation, so that teaches the 38k path as often as the 17k one.
`125 rows → 19` on the same episodes.

**Pairs come in two kinds.**

*Correctness* — verified against unverified, same task, same round.

*Efficiency* — for a task where every attempt passed, cheapest correct against most expensive. At a
94.7% suite pass rate the correctness rule produces **zero** pairs, and the discarded signal is the one
the promotion gate scores: measured spreads on identical tasks run 1.03× to 3.10×. The bar is the
task's own median, not a fixed ratio: an earlier version compared min-to-max range against the
interquartile spread and fired on a task whose attempts all cost within 2% of each other.

Three guards, each from a measurement:

- a **truncated** attempt is never `chosen` — it stopped mid-work, and the cheapest way to finish is not to finish
- a truncated failure should not be `rejected` either — 34% of probe episodes truncated, so you would teach the model to avoid what was never its fault
- **unpriced** episodes yield nothing — every token count 0 makes every gap 0 and every pair look infinitely good, and `mean_tokens` was a column of zeros in this repo once

## 7. Evaluate

    python -m admin.cli evaluate --root var/admin/run-1 --print-only

Prints the command rather than running it. That number is what every claim rests on, so it gets launched
deliberately, with the withheld environment set, on a quiet machine — not as a side effect of a stage
that was really about something else.

The eval suite is the 19 hand-written tasks and is **never trained on**.

---

## What this pipeline cannot do

**Self-rollouts convert pass@N into pass@1. They cannot teach a task the model fails 0/N.** There is no
trajectory to imitate. So the band where this adds capability is the *middle* — tasks the model
sometimes solves.

Measured over 160 generated tasks, after every binding budget was raised and the affected tasks
re-probed (442 episodes in total):

```
easy   144
hard    12
mixed    4     <-- the band DPO consumes
```

**Four tasks.** The re-probe moved 12 tasks `hard → easy` and 13 `mixed → easy`, and moved **none**
toward harder. A batch of 160 gated, non-duplicated, deterministic tasks yielded four that produce a
preference pair.

Which means DPO from failures, the natural design, has a hole: a pair needs a `chosen`, and an all-fail
task has none. Eight rejected sides and nothing to prefer them against.

Two sources can fill it:

**A teacher.** `hermesbench.runner --base-url --model --api-key-env` drives any OpenAI-compatible
endpoint through the *same* harness, with real execution and the same withheld verification, and the
dialect-aware renderer converts the result to ATEM. This is config, not code. It is emphatically not
`hermes/generate.py`, which produces *simulated* trajectories stamped `executed=false` — a corpus of
those teaches a model that its predictions about the world are the world.

**The reference solutions.** Every generated task ships one, gate-proven to pass both checks — 160
correct answers to tasks the model fails, with no teacher involved. Usable as the `chosen` side of a
pair. **Not** usable as an SFT target: a reference solution is an answer without the inspection that
found it, and training on it would teach the model to skip looking, which is the behaviour the whole
suite exists to measure.

## The recurring defect

Every substantial problem found while building this was **a resource limit or an instrument reported as
a capability measurement**:

| what looked wrong | what was wrong |
|---|---|
| suite success 73.7% | reasoning charged against the action budget; **every** failing episode was a truncation. 94.7% once split |
| 27 "hard" generated tasks | 22 were budget-limited; re-probing moved them to easy |
| 48 "overfit" episodes | published checks too weak to catch a wrong answer |
| 12% generator acceptance | absolute paths in generated setup scripts |
| `parse` and timeout spikes | four generator processes competing for one GPU |
| `rejected_by: {'': 3}` | accepted-past-target tasks filed as unnamed rejections |
| `acceptance_rate: 3.265` | resumed tasks counted in a rate whose denominator was attempts |

The thing being measured kept turning out to be fine. That is the argument for keeping trajectories,
saving rejects, and printing distributions rather than examples.
