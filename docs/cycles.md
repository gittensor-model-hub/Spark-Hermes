# Durable cycle controller

`spark-hermes cycle` uses the **same root and SQLite database as release**. A cycle
does not introduce another incumbent pointer. Model, agent and epoch are resolved
from one accepted candidate; activation updates the pointer, ordered history and
operation receipt in a single transaction. `generation` increases on activation
and rollback, and `epoch_id` binds that generation to the original authority issuer.

Production is the default when a root is created. Fixture mode and namespace must
be selected at creation and cannot subsequently be changed. Fixture checkpoints,
scripted serving and fixture parent handoffs prove software behavior only. They
never authorize production promotion, measured learning, live delivery or payment.

## Installed CPU demonstration

After installing the wheel and its CPU dependencies, run from any directory:

```sh
spark-hermes cycle demo --root /tmp/spark-cycle-example --mode fixture
```

Use a fresh disposable root. The explicit `--mode fixture` is required; omitting
it or requesting production refuses before creating state. Repeating the command
on a completed root revalidates the original jobs and retains the incumbent.
The command executes local scripts and subprocesses and makes no network requests,
GPU calls, model-weight downloads or external mutations. The directory contains
private fixture checks and answers and should be treated as disposable test data.

The demonstration uses the pinned **Qwen3.5-4B** `rtx5090-poc` profile. It first
generates a task through `synthesise`, executes the real task gate and publishes
its salted withheld commitment. Baseline and contribution episodes run through
`validator.judge.runner_for` and `hermesbench.runner.main`, with real policy parsing,
local tools, public/private verifiers and the trajectory sink. Authenticated
GitHub response fixtures enter through `GitHubSource.collect/admit`; actual
judging and settlement produce scorecards and a pending action outbox. One admitted
contributor passes all ten withheld checks and wins the actual scoring gate; a
separate admitted contributor has two private-check failures that supply measured
curriculum feedback without receiving crown credit.

Bootstrap replay, SFT preparation, train, merge and candidate registration create
the initial configured fixture pair. Only the initial parent handoff uses the
explicit fixture approval interface. The production controller then settles the
first new round, freezes replay, computes curriculum, prepares from that parent,
supervises train/merge and executes all four crossed cells on six sealed fixture
families with ten attempts each. Its strict numerical decision activates the first
fixture release. Separate installed CLI processes resume the original cycle before
and after preparation. The committed feedback request is consumed by actual task
synthesis/gating/publication; the new baseline and admitted execution use the
activated model **and agent**. Second SFT uses the first strict release parent,
retains historical experience and evaluates six unused confirmation families. Its
negative joint result refuses activation, preserving the first pair and history.

Only external boundaries are substituted: scripted GitHub responses, synthesis
completions, serving completions/resource counts, fixture tokenizer and guarded
`FixtureTraining`. Real train/merge producers own completion and merged manifests;
the fixture writes only explicitly labelled adapter/checkpoint/tokenizer bytes.
The fixture tokenizer counts UTF-8 bytes, so this example uses a 65,536-byte fixture
preparation allowance. This is not a 4B GPU context/memory recommendation; the real
POC profile keeps its 2,048-token preparation default.

Inspect `summary.json` for exact first/second cycle IDs, original workspace paths,
the approved second parent, consumed feedback and remaining real-world prerequisites.
`commands.jsonl` records child argv, exit codes and complete output; exit 3 is the
expected second release refusal. Generated task `runner.log`, `github.jsonl`,
original baseline and candidate JSONL, receipts, SQLite outbox, replay mixtures,
prepared recipes, supervised job records, matrix files and incumbent history stay
under this root. These are software evidence, not measured model learning or rewards.
An interruption before a cycle has been created may leave bootstrap artifacts;
retain those diagnostics and use a new root. Once `demo-progress.json` is written,
the controller's original durable jobs are the recovery authority.

Use the printed workspace paths with `spark-hermes doctor --software-only --profile
rtx5090-poc --root WORKSPACE` and `spark-hermes status --profile rtx5090-poc --root WORKSPACE`. A populated fixture still
reports real corpus licensing, model training, serving/hardware and SN74 prerequisites.

## First use

Create the candidate, release and replay authorities using the producers described
in [crossed-release.md](crossed-release.md) and [learning-boundary.md](learning-boundary.md).
Keep the original private source roots, prepared recipes, corpora, checkpoints and
evaluation artifacts. The controller revalidates them on resume and status.

```sh
spark-hermes cycle init --root private/releases --config private/controller.json
spark-hermes cycle start --root private/releases --name cycle-001 --spec private/cycle-001.json
spark-hermes cycle status --root private/releases --id sha256:CYCLE_ID
spark-hermes cycle resume --root private/releases --id sha256:CYCLE_ID --through prepare
spark-hermes cycle dry-run --root private/releases --id sha256:CYCLE_ID
spark-hermes cycle resume --root private/releases --id sha256:CYCLE_ID \
  --execute-training --allow-unsandboxed
spark-hermes cycle active --root private/releases
```

`controller.json` contains `replay` (configured authority root), optional `github`
(`repository`, `credential_env`), and optional `bootstrap_parent` (`root`, `id`).
For first use, it may also contain a `release` object with the unchanged release
initialization fields: `candidates`, `incumbent`, `data_policy`, `policy`.
The bootstrap approval must name the exact configured initial model and agent.
Production bootstrap approvals require strict original release authority; a fixture
handoff is allowed only inside an explicitly created fixture namespace. Subsequent
cycles automatically use the active strict release as the SFT parent.

Each cycle specification requires these fields, with an optional `replay` path to
a new configured authority for that cycle's reviewed rights/data version:

```json
{
  "rounds": [{
    "source": "competition", "round_id": "round-001",
    "prs": [{"number": 7, "head": "EXACT_HEAD_SHA", "author": "contributor"}],
    "scorecards": "/private/competition/cards",
    "episodes": "/private/competition/episodes"
  }],
  "agent": "/private/new-agent.json",
  "workload": "/private/confirmation-workload.json",
  "training": {"profile": "bf16", "sequence_len": 8192, "max_steps": 100, "execution": "local"},
  "curriculum": {"version": "spark-curriculum-v1", "count": 6, "min_families": 3, "max_per_family": 2},
  "evaluation": {
    "schedule": [{"attempt_id": "0", "seed": 0}],
    "budget": {"max_steps": 16, "max_tokens": 4096, "tool_timeout_s": 30},
    "sampling": {"temperature": 0.0, "top_p": 1.0},
    "serving": {"old": {}, "new": {}}
  }
}
```

The abbreviated schedule above must be expanded to at least ten distinct attempts
for promotable evidence. Both serving objects require the complete trusted HTTPS
configuration from the release runbook. The controller maps their roles to the
exact resulting model IDs before freezing the experiment. It never accepts score
JSON as an execution substitute. Production credentials remain environment lookups;
they are not copied into cycle specifications.

Optional `history_ids` pins original `settled-experience` IDs in that cycle's replay
root. Re-import retained historical settlements into a new reviewed replay version,
then list the returned IDs. This preserves history without presenting old competition
rounds as the active epoch. Optional `confirmation_policy` names a newly reviewed
catalog file for the experiment: it must retain every known relationship,
membership/partition and exposure from the initial and previously used catalogs.
Its original bytes are frozen with the plan. Numerical release policy stays immutable.

`admin.task_feedback.generate_from_feedback` consumes an authoritative curriculum ID
and request ID, rechecks settled source evidence, runs the existing synthesis/gate
and commits the exact published task files and full request lineage. It requires a
disposable execution host and withheld salt. `verify_feedback_task` rechecks the
original committed request and published bytes before next-round baseline execution.

For derived competitions, `admin.competition_pair.build_epoch` binds the active
release, exact model/agent, installed evaluator, task/verifier fingerprints and
attempt schedule. `runner_for` accepts `release_root` and `serving_config`, and the
runner CLI exposes `--release-root` / `--serving-config`. Production uses the same
trusted serving identity protocol as crossed evaluation. An alias-only service,
old base, wrong agent or stale epoch cannot stand in for the released pair. Initial
base execution can select `--profile rtx5090-poc`; global upstream pins are preserved.

`execution=local` runs the existing Axolotl train and merge producers under a local
durable supervisor. It requires the actual training installation and hardware.
`resume` without `--execute-training` stops at preparation and reports a pending
prerequisite. `dry-run` prints the command and records no training completion.
QLoRA-to-BF16 merging remains refused by the existing representation gate.

## Producers, boundaries and recovery

The persisted stages are admission, settlement, experience import, replay,
curriculum, preparation, training, merge, candidate registration, experiment plan,
crossed execution, strict decision and activation. Admission calls the authenticated
`GitHubSource.collect` / `admit` route for a missing contribution and verifies an
already admitted exact commitment on restart. The configured credential type is the
existing user/PAT read-only adapter; installation-token compatibility is not claimed.
The cycle pins the full round identity and explicit PR set.

Round opening, baseline execution and `validator.judge` remain their existing
producer surfaces. A cycle waiting for grading reports the named round as pending;
it neither freezes an open contribution window implicitly nor invents a verdict.
Once graded, it calls `SettlementStore.activate/settle_round`, imports the committed
settlement with `ReplayStore.import_round`, and invokes all learning/release
producers itself. External outbox delivery is separate from committed experience.

Per-cycle workspaces and explicit experience IDs keep later imports from expanding
an earlier corpus or curriculum. `curriculum.json` retains the original request IDs
and settled input IDs for the next actual task producer. Mutable `Workspace` stage
files are checked projections, not cycle authority. A missing or changed completed
artifact stops recovery instead of silently rebuilding released evidence.
New contributions can use a new reviewed policy and replay root via the optional
cycle `replay` field. Its complete configuration, role and issuer are frozen in the
new request. Prior cycles continue resolving their original roots; versioning never
rewrites an earlier policy or approval. Preserve historical family/exposure truth
when reviewing each version, as required by the release catalog checks.

Each job is committed before its producer starts and keeps one stable ID. Concurrent
resumes serialize through a process lock. Completed producers are revalidated and
reused. Deterministic producers can reconcile an interrupted publication from their
original authority or frozen inputs. Training runs in `admin.cycle_jobs`; the child
owns durable completion independently of the requesting CLI. A lost CLI response
can be resumed without another training launch. A running supervisor is pending;
a killed/failed supervisor without completion is uncertain and cannot be relaunched
under that job ID. An adapter directory alone is never proof of training. Keep the
job log under `cycle-jobs/` and start a separately reviewed run after resolving the
failure. A partially executed crossed plan keeps its spent-family reservation and
cannot be rerun to cherry-pick results. A completed evaluation whose caller died
is reconciled from the original experiment authority.

## Activation and rollback

```sh
spark-hermes cycle rollback --root private/releases --id sha256:EARLIER_APPROVAL \
  --operation-id incident-123-rollback --expected-generation 2 \
  --expected-candidate sha256:CURRENT_CANDIDATE
```

Use a unique durable operation ID per rollback intent. Repeating that operation,
even after subsequent activations, never overwrites a newer pair. Reusing the ID
with different arguments fails. Rollback requires an earlier activated accepted
pair in the same namespace, revalidates its retained evidence, appends history and
allocates a new epoch; it does not assert a newly measured gain. A refused cycle
retains the incumbent and its failed decision. Stale generation/pair checks prevent
concurrent candidates and ABA baseline reuse from replacing a newer incumbent.

## Controller APIs for next-round integration

`CycleController(root).configure/start/status/resume/dry_run/rollback` are the exact
CLI implementations. `resume(..., through=STAGE)` permits handoffs to existing
operator surfaces without skipping prerequisites. The API `hook(name)` injects
failures at `STAGE:before_producer`, `STAGE:after_producer`, `STAGE:after_commit`,
training submission/launch and activation pointer/commit boundaries.

`active_pair()` returns one snapshot with the exact candidate, approval, model ID,
agent ID, merged model files, agent bytes binding/record, upstream ancestry,
representation, runtime, generation and epoch ID. `epoch_binding(pair)` selects the
identity fields for next-round transport. For subsequent generations, cycle start
requires the round's original challenge epoch to contain this exact `incumbent`
binding. Next-round integration must carry the retained expectation through actual
baseline/task execution, challenge, admission, runner and scoring, and compare the
active epoch before credit. Merely annotating old episodes does not establish that
the activated pair executed. The normal competition runner's derived-model/agent
route and the two-cycle task-generation command are owned by final integration;
this controller API supplies the authority they must consume without rewriting
global model pins, prompts, evaluation oracles or retained evidence.

Fixture tests may use `training.execution=fixture` and replace `evaluation.serving`
with `evaluation.fixture`, a path to a `spark-cycle-serving-fixture-v1` record with
the exact controller `origin` and `cells.Q00/Q10/Q01/Q11` scripted task/attempt maps.
The controller creates representation-bound serving fixtures and still executes
the actual crossed producer, tools, verifiers, sink and decision. Optional injected
GitHub transport is accepted by the Python API only in fixture mode.

Commands exit 0 for success, 2 for invalid/missing/stale evidence, 3 for a valid
refused cycle, and 4 for a pending external prerequisite. Status never treats a
pending, failed, prepared or fixture job as real completed training.

Software upgrades require the explicit [supported runtime transition](runtime-transition.md).
Ordinary cycle continuation still refuses a stale evaluator; a verified imported pair
retains historical parent authority and receives a new target epoch before fresh cycles.
