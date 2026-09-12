# Competition admission and evidence

The upload API stores prose and returns an envelope receipt. Uploading does not
admit a strategy to a round. The validator admits the public commitment with:

```sh
.venv/bin/python -m validator.pr_admission \
  --repository OWNER/REPOSITORY --pr 123 --round ROUND_ID \
  --head FULL_HEAD_SHA --credential-env GH_TOKEN \
  --store var/rounds --intake-root var/submissions --receipts datasets/receipts.jsonl
```

Configure the named credential in the process environment. This command only runs
authenticated `gh api --method GET` requests: credential identity, PR metadata,
file metadata, base/head strategy registry content, and the base commit's
`datasets/rounds/ROUND_ID.json` assignment. It rechecks the PR after collection.
The PR must be open, ready, and append exactly one commitment; no PR code executes.
The author, commitment miner, uploaded receipt, assigned task, and round must agree.
`validator.judge accept --miner ...` now refuses because a miner name cannot
authenticate a commitment.

`RoundStore` and `Intake` default to production. Their `.identity` files bind mode,
namespace, and issuer. Create an isolated CPU fixture with `mode="fixture"` and
the same explicit namespace on both stores. Reopening preserves the identity;
requesting a different mode or namespace fails. Copies of fixture round snapshots
or receipts are rejected by production stores. Fixture scorecards retain their
origin and do not authorize model release, rewards, or payout.

The private round snapshot contains `admissions[author]` together with the standing
submission. An admission records repository/PR/author/head/base, registry delta and
content hashes, submission ID, bundle digest, receipt origin, round origin, task,
epoch, `round_identity`, and `admission_id`. `round_identity(window)` hashes the
complete private challenge (including baseline), round ID, and origin.
`admission_for(window, miner)` validates these bindings. Duplicate delivery returns
the identical admission; a different commitment by the same miner is refused.
Admission and submission are published in one atomic snapshot under a round lock.

The synchronous execution adapter signature remains:

```python
run(miner_id: str, verified_bundle_path: Path, workspace: Path) -> Path
```

The returned path is the complete JSONL episode log. The judge checks the exact
admitted receipt, miner, origin, files, and digest before execution and rechecks
the bundle afterward. `runner_for` passes this exact directory to `--miner-dir`.
It never chooses an upload by miner or recency. Generated task roots travel through
`epoch["task_root"]` to `--task-root`.

The judge carries the admitted epoch, round, task, origin and bundle commitment in
a scoped call context. `validator.judge.execution_context(miner_id, bundle_path)`
returns a separate copy for a compatible adapter; outside judging it returns `None`.
The scope is restored on success or failure and does not leak into later calls.
Wrappers can keep the three positional arguments when delegating to `runner_for`.
Adapters handing work to another thread must explicitly forward this context:

```python
context = execution_context(miner_id, bundle_path)
forwarded = runner_for(**runner_options, evaluation_context=context)
log = pool.submit(forwarded, miner_id, bundle_path, workspace).result()
```

`runner_for` establishes an independent immutable expectation in the destination
thread, binding the resolved full context and miner directory through the final
runner call. Its generated JSON file is transport: the runner strictly decodes
one complete byte snapshot, rejects missing/replaced/ambiguous or conflicting
transport, and compares every field to the preserved expectation. Validation and
episode stamping use that same checked expectation; prompt composition uses the
same captured map whose digest was checked. Later file edits cannot replace either
snapshot. The post-run receipt and episode checks still apply.

Context authority is an interface within the trusted validator process, not
protection from arbitrary code running in that process. No automatic propagation
across processes is provided. A custom process adapter needs a separately trusted
IPC/expectation handoff and the same final-consumer checks; passing only a writable
context file does not preserve admitted authority.

`runner_for` inherits that admission context even when constructed without an
evaluation context, and rejects conflicting configured identities. Outside a judge
call, an explicit evaluation context must already contain the expected bundle
digest; this is operator input, not proof of admission. The adapter never derives
expected authority from mutable files. A direct
call omitting the entire context remains exploratory and produces no competition
identity. Recording callbacks retain the exact receipt path and receive a checked
handoff. Fixture replay retains its fixture-store restriction and all post-run
receipt and episode checks.

The epoch declares `model_revision`, `harness_digest`, a nonempty `epoch_id`, at
least ten unique string `attempt_ids`, and the complete `score_policy` returned by
`validator.score.policy_record()`. Task pins carry the public verifier script or
its `verify_digest`, and a withheld commitment or explicit
`private_check_required=False`. A missing private result is never a private-free
declaration. The challenge CLI accepts `--epoch-id` and `--attempts` when opening
packets from stamped baseline logs.

The runner's standalone `--evaluation-context` reads an operator-supplied JSON object containing
`epoch`, `origin`, and, for candidates, `round_id` and `bundle_sha256`. Before
inference it checks the installed model pin, observed harness, and bundle bytes.
`run_suite(..., evaluation_context=...)` records attempt IDs before scheduling,
so completion order cannot change attribution. The sink preserves integrity and
identity evidence; `episode_metrics_of` normalizes it without dropping those facts.

Scoring requires explicit boolean public/private results, success, protocol,
integrity, setup and truncation flags; positive integer measured tokens;
nonnegative finite resources; and exact task/model/harness/epoch/attempt coverage.
Original baseline evidence is retained in `Attempt.evidence` and checked against
the snapshot's measurements and the same trust domain. No eligible scorecard or
verdict is persisted for invalid evidence. Valid evidence still uses
`hermes.acceptance.decide`: all attempts correct, minimum ten attempts, unchanged
verification, and a 20% median-token reduction at the seeded bootstrap lower bound.
Scorecards include the full policy, its hash, admission ID, bundle, epoch and origin.

Receipt writes use process locks, fsync, and atomic replacement after complete
bundle publication. An identical retry retains the original timestamp and status.
A crash may leave an unreceipted complete bundle; a retry verifies and publishes
it. Corrupt/truncated receipt stores fail closed and are never silently rewritten.

Bundle identity preserves exact UTF-8 text and filenames. LF, CRLF, CR, mixed line
endings, a missing final newline, a BOM, and composed/decomposed Unicode remain
distinct. The digest is SHA256 of the path-to-text map serialized as sorted compact
JSON with `ensure_ascii=True`, encoded as UTF-8. Receipt byte counts measure file
content bytes. Verification captures actual files with strict UTF-8 decoding and
checks paths, symlinks and the miner contract; the sibling `.bundle.json` file is
never a substitute for those bytes. Invalid UTF-8 files refuse execution, and
unencodable upload text or filenames refuse before publication. Archives and
binary files are unsupported.

The runner checks its bundle digest and composes its prompt from the same captured
surface. Prompt framing still strips surrounding whitespace and adds strategy
headings; that rendering does not change artifact identity. The miner search's
unchanged `full` candidate preserves the original bytes. Existing intact receipts
remain valid without rewriting hashes or timestamps, including CRLF/CR receipts
previously rejected by text-mode verification. A mutated bundle needs its exact
original files restored; do not normalize files or reseal its receipt to recover it.

Changed captured bytes refuse before completion construction, even if the stored
files are subsequently restored. If disk changes after an intact admitted snapshot
was captured, execution may safely finish using that original snapshot; the
post-run receipt check still refuses credit for the changed stored artifact.

Legacy receipts without origin and baselines without original identity/integrity
evidence need reviewed recovery or re-upload/re-baselining. The software does not
invent missing historical evidence.

Continue with [durable settlement and delivery](competition-settlement.md) for the
complete admission → freeze → judge → crown → outbox commands, trusted workflow
configuration, fixture CLI proof and the `spark-settlement-v1` consumer schema.

### Complete execution evidence

Competition baselines, round reconstruction, judging, scoring and settlement require complete
JSONL evidence. `hermesbench.sink.read_episodes(path)` reads one byte snapshot and refuses
unfinished final records, malformed or non-object records, recursive duplicate JSON keys
(including identical duplicates), and nonfinite JSON numbers. An unfinished log can still be
inspected with the explicitly observational `read_episode_prefix(path)`; that prefix does
not certify a completed run. `validator.score.metrics_from_bytes(raw, source=...)` lets
settlement score the exact snapshot it hashes, without reopening it during decoding.

Flat metrics and nested runner/sink records remain supported. Baselines retain deep copies
of the original parsed envelopes in `Attempt.evidence`, including failed-but-valid attempts
and their full costs. Verifier-stamp filtering uses the same normalized view as scoring.
An omitted detailed integrity report is not reconstructed as a claim that every detector
ran: summary-only records remain summary-only. Supplied signals and unassessed details must
agree with the producer's integrity semantics; ordinary warnings also prevent clean credit.

`EpisodeResult` verification results, trajectory success and documented trajectory metadata
constrain the corresponding normalized execution assertions. `JsonlEpisodeSink` now retains
an available verification report alongside its metrics and integrity report; its optional
full trajectory still requires `keep_trajectories=True`. Tool transcripts, verifier stdout
and arbitrary metadata payloads are not interpreted as execution authority. Complete round
snapshots, packets, active scorecards, root identities and receipt records also reject
ambiguous JSON before their existing identity and authorization checks.

CPU fixture evidence remains confined to its fixture trust domain. These checks provide
neither proof of model improvement nor authorization for production promotion or payment.

### PR metadata bytes

Authentication and Git blob hashes establish source bytes; parsing must also preserve
all assertions. `hermes.evidence_json.evidence_value` rejects duplicate keys at every
object depth (including equal values and escaped-equivalent names), invalid constants,
and nonfinite numbers, even when a later duplicate would overwrite the value.
`evidence_object` requires an object; `evidence_records` validates the complete JSONL
snapshot. Literal strings containing `NaN` or JSON-looking text remain ordinary strings.

`GitHubSource.get` uses the general decoder so legitimate diff, label and review arrays
remain supported. Endpoint shape checks still apply. Strategy admission decodes every
base/head registry record and the authenticated base-round object before submission or
save. Historical identity-only commitments remain readable and continue to exclude a
second commitment by the same miner in the same round; malformed historical identities
or ambiguous records require correction at the trusted source. Existing byte-exact
append-only checks, actual fork head repository lookup, blob SHA verification and
idempotent admission remain in force. Requested-ref membership continues to rely on
GitHub's content API; this is not independent Git tree authentication.

The dataset registry gate also validates complete base/head JSONL, the rollout gate
validates registry/round metadata, and training/rollout attestation ingress requires
unambiguous objects before existing proof checks. Dataset and rollout histories must
preserve prior bytes exactly. Training proof manifests, mix manifests, evaluation files,
rollout export proofs and signed-token verification retain their separate existing
validation rules; these ingress checks do not certify every downstream parser or change
proof acceptance policy. Training ingress failures return issues with no computed tier,
which the training gate includes in its rejection.

`GitHubActions` inherits strict response decoding while retaining paginated label/review
reconciliation and identity-bound cleanup of merged historical winners. Malformed
provider-response tests establish robustness, not contributor control of GitHub's
serializer. Credentials must still support `GET /user`; installation/App-token support
is not inferred. These checks do not protect against arbitrary code execution inside
the trusted validator process.

### Typed assignment authority and recovery

Round JSON announcements and persisted assignments use explicit `schema_version: 1`,
`round_id`, `state`, `commitment`, `seed`, `task_ids`, `miner_ids`, and `replicas`.
Version and replicas must be exact integers (booleans, floats and numeric strings
are refused). IDs and seed are nonempty strings; seed retains its existing minimum
32-character requirement and need not be hexadecimal. Task/miner pools are nonempty
JSON lists of unique nonempty strings. Internal producers also accept tuples and
freeze validated lists to tuples. Missing known fields, nulls, mappings in place of
lists, and unsupported states/versions are errors, even if coercion would recover a
matching commitment. Unrevealed commitment announcements cannot authorize admission.
The announcement reveal command checks the same schema before publishing the seed.

Admission compares the authenticated base announcement to the stored assignment.
Its `assignment_identity` hashes canonical round ID, seed commitment, sorted task
and miner pools, and replicas. `round_identity` also includes that assignment hash,
so existing judge, score, crown and settlement checks bind the complete assignment.
Independent ownership checks cannot substitute for source/store agreement. Reordering
pools is harmless; changing seed, pool membership or replicas invalidates admitted
authority. Lifecycle state is checked separately: admission needs open assignments;
closing a previously admitted assignment preserves grading and settlement eligibility.
`RoundStore.load` validates stored assignment data even when an assignment argument
is supplied; a supplied object cannot override malformed or different persisted data.
GitHub PR-number responses must be exact positive integers on both source reads and
on active/historical settlement action reads. Valid list pages and history stay supported.

These are deliberate fail-closed compatibility changes. Old admissions without
`assignment_identity` and old active settlement scopes lacking the extended
`round_identity` do not confer current authority. Reading a legacy unscoped window
for inspection still works; retrying admission never adds a missing binding or
reseals a historical record. Preserve the original state and any committed outbox
before recovery. A trusted operator must recover the original authenticated base
announcement, exact PR head/registry delta, receipt bytes, store origin and evaluation
epoch, and verify their agreement. Restore an intact current-format backup where
available. Otherwise create a fresh round through the normal trusted announcement,
receipt and admission process and regenerate evaluation/settlement evidence in its
original trust domain. Do not edit admission hashes or import old winners as fresh
credits. Previously committed outcomes/actions require operator reconciliation before
any replacement round is activated, so recovery cannot duplicate historical effects.
When originals are unavailable, leave the old round non-authoritative. Fixture state
and recovery never authorize production promotion or payment.
