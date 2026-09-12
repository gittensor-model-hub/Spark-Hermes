# Exact candidate, four-cell evaluation and release

These are CPU-testable production interfaces. Scripted CPU completions, tokenizer/checkpoint
bytes and latency in a fixture namespace are **software fixtures**, never measured learning,
production promotion, payout or hardware attestation. Real candidate training and inference
are separate operator prerequisites. Existing model-only `hermes.promotion.decide/load_run`
comparisons remain available; their serialized reports explicitly grant no activation authority.

## Producer and consumer APIs

- `admin.candidates.CandidateStore(root, mode=None, namespace=None).register(workspace=Workspace(...),
  merged_record=Path(...), agent=Path(...), workload=Path(...), parent=Path(...))` returns a durable
  `candidate` authority record with `id` and `payload`. `resolve(id)` revalidates original artifacts.
- `admin.release.ReleaseAuthority(root, mode=None, namespace=None).configure(candidates=Path(...),
  incumbent=id, data_policy=Path(...), policy=...)` binds the configured candidate issuer, original
  family catalog and complete quality policy. Configuration is immutable. Production is the default
  on creation; existing roots retain their original mode/namespace/issuer.
- `authority.freeze(old=id, new=id, schedule=[...], budget={...}, sampling={...}, serving={...})`
  commits a `crossed-plan`. It reserves all confirmation families before any execution. The exact
  original plan can be retrieved idempotently; a changed plan cannot reuse those families.
- `admin.evaluation.execute_crossed(authority, plan_id, allow_unsandboxed=False)` executes all four
  cells through `ServedModelPolicy`, `run_episode`, `LocalToolExecutor`, real public/withheld
  verification and `JsonlEpisodeSink`. Outputs are isolated by plan/cell/attempt. It returns a
  committed `crossed-evaluation` record and a complete matrix artifact. No arbitrary score-JSON
  import grants producer authority. `authority.evaluation(id)` rechecks the original logs and report.
- `hermes.cotraining.crossed_report(matrix)` is the public pure statistical API. It grants no
  authority. Strict producers and consumers additionally require each original attempt's execution
  eligibility. That result is consumed by `authority.decide(evaluation_id)`, which issues a
  durable `release-decision`. `authority.resolve_decision(id)` revalidates that exact accepted pair.
- `authority.activate(approval_id)` updates the fixture or production incumbent and history in one
  SQLite transaction after revalidation and comparison with the evaluation baseline and its history generation.
  Returning to the same baseline after rollback does not make an old measurement fresh again. Repeating the
  same activation is idempotent. `activate(approval_id, rollback=True)` selects an already activated,
  accepted pair and records rollback intent; it does not claim a fresh gain. `status()` returns
  `candidate`, `approval`, `generation`, `origin`, and ordered history for the cycle controller.
- Existing `ParentAuthority(root).approved_parent(...)` consumes these decisions for subsequent
  SFT. Strict decisions revalidate the complete original release lineage at parent lookup. The
  existing `fixture_approve` helper remains only an initial CPU parent-handoff mechanism and cannot
  authorize the strict activation API. Fixture evidence cannot be upgraded by flags or copied JSON.

All stores are trusted operator producers, like `RoundStore`/`AuthorityStore`: their databases,
configuration and original artifacts must be private and protected from contributor/tool writes.
They protect ingress across configured issuers, not arbitrary code execution inside the trusted
operator process. Execution must use a disposable isolated host/container without access to
operator authority stores, credentials or sealed task files. The existing local executor requires
explicit `--allow-unsandboxed` acknowledgement for a disposable execution environment; it is not
an OS sandbox. The separate cycle task owns next-epoch activation of other subsystem pointers.

## Operator commands

Every command is also available as `python -m admin.<module>` using the same implementation.

```sh
spark-hermes candidates register --root var/candidates \
  --workspace var/admin/cycle-1 --merged-record var/admin/cycle-1/models/sft/merged.json \
  --agent private/agent.json --workload private/confirmation.json --parent private/actual-parent
spark-hermes release policy > private/quality-policy.json
spark-hermes release init --root var/releases --config private/release-config.json
spark-hermes cotraining freeze --root var/releases --config private/experiment-config.json
spark-hermes cotraining run --root var/releases --id sha256:PLAN_ID --allow-unsandboxed
spark-hermes cotraining show --root var/releases --id sha256:EVALUATION_ID
spark-hermes release decide --root var/releases --id sha256:EVALUATION_ID
spark-hermes release activate --root var/releases --id sha256:APPROVAL_ID
spark-hermes release status --root var/releases
spark-hermes release rollback --root var/releases --id sha256:PREVIOUS_APPROVAL_ID
```

Commands return 0 on success, 2 for missing/invalid/stale evidence or operational failure, and
`release decide` returns 3 for a valid evaluation refused by the numerical gates. Refusal leaves
incumbent state unchanged. Failed or interrupted experiments retain their logs and spent-family
reservation; they cannot be rerun to cherry-pick noisy outcomes. Use unused confirmation families.

`release-config.json` contains `candidates` (authority root), `incumbent` (initial exact candidate ID),
`data_policy` (original reviewed catalog path) and `policy` (complete output of `release policy`).
The initial incumbent is a configured starting pair, not a newly measured release approval.

When later reviewed experience adds tasks, the experiment configuration may include an optional
`data_policy` path to a new immutable confirmation catalog. It must preserve aliases, memberships,
partitions and exposure from the configured and previously used catalogs as well as the training
sources. The plan retains the original file hashes and refuses changed/deleted inputs. This does
not change numerical thresholds or release previously spent families. The controller exposes the
same input as `confirmation_policy` in a cycle specification.

Agent artifacts use `spark-agent-v1` with exactly `schema`, `system`, `dialect`, `tool_schemas`
(named actual schemas), and `native_tool_messages` (boolean). The producer executes the captured
system prompt and tools directly. Evaluator/tool/protocol code and the installed Python environment
are independently content-addressed by `hermes.harness.crossed_runtime_identity`; a dirty Git tree
or an installed wheel is supported without fabricated Git metadata. Changing code/environment
invalidates ordinary continuation. Use the explicit [supported runtime transition](runtime-transition.md)
to retain a checked historical baseline/parent, then collect fresh target-runtime evaluation.
Retaining the original installation and full original authority/artifact history is required;
import alone grants no new measured promotion.

Candidates require original committed replay/operator corpus, prepared SFT recipe and parent
provenance, actual merged SFT checkpoint shards/config/tokenizer, and the exact prepared chat
template. Production initial Hub parents must resolve to the actual pinned local Hub snapshot used by
preparation; the cache lookup sets `local_files_only=True` and never downloads. Config files need
not retain Transformers-internal repository/revision fields. Fixture parents instead carry an
explicit labelled repository/revision binding. Subsequent SFT must use the approved previous exact pair.
This strict parent-release path supports merged SFT representations; a QLoRA adapter or a merged
representation different from what was evaluated is refused. Do not relabel adapter evidence as
merged-model evidence. Retain the original prepared/data/parent paths for transitive revalidation.

Supported checkpoint layouts contain either `model.safetensors`, `pytorch_model.bin`, or one
`model.safetensors.index.json` / `pytorch_model.bin.index.json` with all its referenced files.
Index references must be safe basenames of the matching weight format; every reference is hashed,
including nonstandard names. Numbered shards must form one complete series. Unindexed shards,
omitted weight files, multiple indexes and mixed single/indexed representations are refused before
merge success is published. Derived checkpoints forbid symlinks; pinned local Hub-cache blob
symlinks remain supported at the initial-parent boundary.

Strict evaluation preserves every original attempt, verified success and measured resource cost.
Each exported row separately records execution eligibility and explicit refusal reasons. Setup
failures, truncation, stalled or harness-final trajectories, invalid structure, missing or
contradictory verifier/usage evidence, and incomplete integrity checks prevent release in any cell.
The report retains all task/family means and lists eligible attempts and complete families separately;
an unexecuted family cannot supply confirmation support. Ordinary verifier failures remain measured
failures. A complete recovered malformed turn retains its success and costs under crossed-v1;
this does not introduce a new protocol-error threshold.

Fixture and authenticated serving paths check typed usage for every completion. Frozen `max_tokens`
limits that response's **output tokens**, not prompt tokens or the episode's cumulative usage.
Per-response usage and provider completion status are retained in original episode evidence and
checked again at release resolution. A `length`, missing, unknown or otherwise incomplete finish
reason refuses strict release even if partial content parses as a final answer and the verifier passes.
Every response must report `stop` or `tool_calls`, and the last response must report `stop`;
completed tool-call continuations remain eligible. Exact-limit output alone is not truncation.
`completion-usage.jsonl` and `completion-status.jsonl` preserve spent responses when a run aborts
before its next episode can be written. Failed/partial runs keep existing episode logs and confirmation
reservations and cannot be retried on those families. Older logs without required originals need fresh evidence; exported
statistics cannot fill the gap.

Workloads use `spark-crossed-workload-v1` with `tasks: [{task: FULL_TASK_RECORD, repository: ...}]`.
The task record includes actual private `hidden_verify`. The confirmation catalog follows
`docs/learning-boundary.md`; exact task version is `content_digest(dataclasses.asdict(Task))`.
It must preserve known family aliases, memberships and exposure from the training catalogs,
including prior non-release exposure. Every chosen family must be sealed and unused. Predeclare
future confirmation families in the immutable catalog; subsequent candidates may bind a fresh
workload, and the incumbent is evaluated on that new workload without rewriting its old identity.
Catalog truth and unknown semantic family relationships remain reviewed operator responsibilities.

`experiment-config.json` contains `old`, `new`, `schedule` (distinct `{attempt_id, seed}` rows,
reused identically per task/cell), `budget` (`max_steps`, `max_tokens` per completion,
`tool_timeout_s`), `sampling` (`temperature`, `top_p`) and `serving` keyed by both exact model IDs.
Task-local timeout/reasoning/verification budgets are part of the frozen workload. All failures
and expensive attempts remain in both quality and resource denominators. Quality is a mean within
each task, then a mean within each family, then a mean across families. The joint interval resamples
paired families 2,000 times with seed 20260912, using linearly interpolated 2.5/97.5 percentiles.
At least 10 attempts/task and 6 independent families, positive resource baselines, lower bound
strictly greater than predeclared min_gain, family regression tolerance and mean resource ratios
are checked. Interaction is descriptive. Missing/unknown policy fields and policy hash changes fail.

## Trusted serving identity protocol

Each production serving configuration has exactly `url` (HTTPS base), `api_key_env`, `alias`,
`deployment_id`, `engine`, `precision`, `device`, and `environment`. Both model deployments must
have identical engine/precision/device/environment. Secrets are read from the configured environment
variable and never recorded in artifacts. Redirects are refused. TLS authenticates the configured
server; hardware attestation is not claimed.

The configured service implements `POST <url>/identity` with `{nonce: ...}` and returns:

```json
{
  "schema": "spark-serving-identity-v1",
  "model_id": "sha256:CONTENT_DIGEST_OF_REPRESENTATION_AND_FILES",
  "representation": "merged-sft",
  "deployment_id": "immutable-deployment-id",
  "engine": "configured-engine",
  "precision": "bf16",
  "device": "configured-device",
  "environment": "configured-serving-environment-identity",
  "mode": "production",
  "namespace": "default",
  "nonce": "exact-request-nonce"
}
```

The service must derive the representation identity from loaded files, never an alias supplied by
the request. The digest is `content_digest({representation: "merged-sft", files: checkpoint_files})`.
Each OpenAI-shaped `POST <url>/chat/completions` includes fixed sampling, max_tokens, attempt seed,
and `spark_identity: {nonce, deployment_id}`. Its response must include `spark_identity` with the
same complete observed identity and fresh nonce, measured prompt/completion token usage and exactly
one choice with an explicit `finish_reason`. Missing completion status cannot authorize a release.
Missing, stale or differing identity blocks the run. Ordinary OpenAI alias-only endpoints do not
satisfy this protocol. A trusted service-side identity endpoint is a real deployment prerequisite.

Fixture roots alone may configure `{fixture: {path, sha256}}` instead. The artifact uses
`spark-serving-fixture-v1`, exact release origin/model_id and
`agents[agent_digest][task_id][attempt_id]` entries with `responses`, `prompt_tokens`,
`completion_tokens`, and `latency`. Each v1 `responses` entry is a complete scripted response,
recorded with `stop` status by default. An optional `finish_reasons` list of equal length explicitly
models provider truncation or tool-call continuation; its actual values are retained and assessed
by the same eligibility rules. The default applies only to this fixture protocol, never to a missing
production finish reason. These are deterministic scripted completion/resource inputs;
tools, verifiers, trajectory persistence, statistics and authority transitions still run normally.
