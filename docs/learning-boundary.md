# Settled experience, replay and approved SFT parents

The learning ingress uses committed `SettlementStore.record(round_id)` records and original
executed trajectories. A crowned PR, exported settlement JSON, mutation description, teacher
answer or `SETTLED` snapshot does not authorize a corpus. Correct but nonwinning or efficiency-refused
executions remain usable; all-fail tasks produce zero SFT rows and zero preference pairs.

All CPU examples below require an explicitly created **fixture** namespace. Their scripted token
counts, tokenizer and checkpoint bytes are software fixtures, not measured learning, production
release approval, or payment authority. Production is the default when a state root is created.

## Operator commands

```sh
# Configure immutable source issuers, rights/family policy and mixture caps.
.venv/bin/python -m admin.replay init --root var/replay/v1 --config replay-config.json
.venv/bin/python -m admin.replay import --root var/replay/v1 --source competition --round round-1
.venv/bin/python -m admin.replay freeze --root var/replay/v1 --workspace var/admin/cycle-1
.venv/bin/python -m admin.cli prepare --root var/admin/cycle-1 --profile rtx5090-poc --offline

# Retain the first round, then add the next round before making another frozen mixture.
.venv/bin/python -m admin.replay import --root var/replay/v1 --source competition --round round-2
.venv/bin/python -m admin.replay freeze --root var/replay/v1 --workspace var/admin/cycle-2
.venv/bin/python -m admin.cli prepare --root var/admin/cycle-2 --profile rtx5090-poc \
  --release-root var/releases --parent-approval sha256:DECISION_ID --offline
.venv/bin/python -m admin.cli train --root var/admin/cycle-2 --dry-run
.venv/bin/python -m admin.curriculum --replay-root var/replay/v1 \
  --config curriculum-config.json --out var/next-requests.json
```

The installed `spark-hermes replay ...`, `spark-hermes curriculum ...` and `spark-hermes parents ...`
commands dispatch to the same module CLIs. `validator.aggregate --replay-root ROOT --source NAME
--round ID --out WORKSPACE` imports and freezes through these same producers; `--out` is a pipeline
workspace whose files live under `corpus/`. Historical raw-log collection no longer grants authority.

For CPU handoff fixtures, pass `--mode fixture --namespace NAME` explicitly when creating replay
and workspace roots. `prepare --fixture-tokenizer` is available only in an immutable fixture workspace.
`admin.parents fixture-approve` also creates/opens only a fixture release root. An existing production
root cannot be relabelled fixture, and fixture evidence cannot be copied into a production namespace.

## Replay configuration and policy

`replay-config.json` contains:

```json
{
  "sources": [{
    "name": "competition",
    "rounds": "/private/rounds",
    "settlement": "/private/settlement",
    "intake": "/private/bundles",
    "receipts": "/private/receipts.jsonl"
  }],
  "policy": "/private/data-policy-v1.json",
  "mixture": {
    "version": "spark-mixture-v1",
    "max_pairs_per_task": 8,
    "max_sft_per_family": 32,
    "max_sft_per_miner": 32,
    "max_pairs_per_family": 32,
    "max_pairs_per_miner": 32
  }
}
```

Each source role retains its own exact issuer. Equal mode/namespace is necessary but does not make
an intake issuer interchangeable with a round issuer or a settlement issuer. Configuration freezes
all three issuers and the original policy file hash in the replay authority database.

The trusted operator supplies `spark-data-policy-v1` with a nonempty `version`, an `origin` carrying
mode/namespace, `family_aliases`, `memberships` and `rights`. Contributors cannot supply policy through
task text. Each canonical family must map to itself; aliases may map transitively to a canonical
family, including forks in another repository. Missing aliases, cycles, duplicate membership keys,
unknown exposure states and incomplete rights fail closed.

```json
{
  "schema": "spark-data-policy-v1",
  "version": "licensed-pool-v1",
  "origin": {"mode": "production", "namespace": "default"},
  "family_aliases": {"counter": "counter", "fork-counter": "counter"},
  "memberships": [{
    "task_id": "counter-task",
    "repository": "owner/repo",
    "version": "sha256:TASK_VERSION",
    "family_id": "counter",
    "partition": "private-competition",
    "exposure": ["selection"]
  }],
  "rights": [{
    "subject": "task:sha256:TASK_VERSION",
    "license": "operator-reviewed-license-reference",
    "attribution": "task owner and source reference",
    "training": true,
    "derivatives": true
  }, {
    "subject": "sha256:ADMISSION_ID",
    "license": "operator-reviewed-contribution-grant",
    "attribution": "contributor and exact contribution reference",
    "training": true,
    "derivatives": true
  }]
}
```

Competition task version is `admin.artifacts.content_digest(window.challenge.task_pins)`; full epoch,
model revision, harness/verifier identity, task ID, admission, receipt, bundle digest, settlement hash,
log byte digest, row digest and attempt ID are separately preserved in each row's `provenance`.
Both task rights (`task:` plus version) and contribution rights (the admission ID) are required.
The operator is responsible for the truth of licensed grants and family mappings; hashing does not
establish legal ownership or discover unknown semantic relatives.

Partitions are `public-development`, `private-competition`, and `sealed-release`. Exposure is an
explicit list drawn from `public`, `disclosed`, `selection`, `training`; empty means unexposed.
Public development requires `public`; settled competition ingress requires `selection`.
A sealed membership anywhere in a canonical family excludes every alias/version from training.
Release admission requires a sealed member and no exposure on any known related member/version.
This conservative family rule prevents renamed/reworded IDs and declared cross-repository aliases
from laundering selection evidence into fresh confirmation.

Policy/configuration is immutable within a replay version. To add rights/members or change exposure,
create a reviewed new policy and replay root, then re-import retained historical settlements under
that policy. Do not edit a frozen policy in place: every preparation rechecks it and refuses changes.
The new reviewed catalog must preserve prior family relationships/exposure; removing historical
knowledge is not a supported recovery procedure.

Mixtures deduplicate exact rendered content, merging contributor/attempt lineage instead of losing
attribution. Deterministic content-ID order applies absolute row caps independently to SFT and pairs.
A deduplicated row charges each represented family/miner once. Pair caps apply within exact
task/epoch/round/bundle groups; candidates with different executed strategies are not compared as
though they shared a prompt. Caps are absolute counts, not fractional promises on a small corpus.

## Operator episode admission

`admin.pipeline.build_corpus(workspace, data_policy=PATH)` uses the same `admit_episode` checks as
competition ingress: original-envelope integrity/protocol flags, explicit private-check results,
positive token measurements, executed metadata, typed complete trajectory steps, paired tool calls,
recorded tools, final response and measurement/trajectory agreement. It also binds the generated
task prompt and public verifier. Operator tasks require withheld checks; explicitly private-free
competition tasks are accepted without fabricating hidden successes.

Operator membership versions use `content_digest(dataclasses.asdict(loaded_task))`, repository from
the reviewed exact membership, and contribution ID `sha256:` plus the original rollout log byte hash.
Generated task files, generate/rollout manifests, policy and log are fingerprinted in the corpus
authority. Rollout manifests now record workspace origin and model. Legacy records missing these
facts require fresh verified provenance; there is no automatic approval migration.

## Frozen artifacts and consumer APIs

`admin.replay.ReplayStore` exposes `configure(config)`, `import_round(source_name, round_id)`,
`experiences(identifiers=None)`, `freeze(workspace)` and `configuration()`. Frozen `corpus/stage.json`
has schema `spark-replay-mixture-v1`, exact inputs, configuration, selected counts, dedup/cap counts,
accepted task IDs, original origin, file SHA-256 hashes, and `{root, identity, id}` authority binding.
The files are `corpus/sft.jsonl` and `corpus/preference.jsonl`. `verify_corpus(workspace, manifest)`
looks up the committed mixture and revalidates original settlement sources, policies and bytes.
It also handles `spark-operator-corpus-v1` through its separate operator-corpus authority.

`admin.artifacts.AuthorityStore(root, role=...)` persists immutable, canonical, content-addressed
records in SQLite. Its `put(kind, payload)` is a **trusted producer interface**, not an untrusted
JSON ingress. A role/issuer binding in the database is checked against `state_identity` on opening.
Exported JSON is never imported as authority. These stores assume the same private operator process
and filesystem trust boundary as RoundStore; they do not protect against arbitrary local code or
database writers. Do not expose `put` to PR workflows, remote contributors or user-authored records.

`admin.data_policy.DataPolicy.membership(task_id=..., repository=..., version=..., purpose="release")`
is the release consumer's family/exposure guard. The release task must use its configured reviewed
policy before collecting/accepting fresh four-cell confirmation. Existing `admin.evaluation` is a
selection measurement surface; this assignment does not grant it production release authority.

## Parent approval interface for release/controller consumers

`admin.parents.ParentAuthority(root)` opens the **release** authority role. Its
`approved_parent(decision_id, identity=workspace.identity, profile=..., repository=..., revision=...)`
returns `(exact_merged_path, binding)` only after durable lookup and transitive revalidation.
`admin.training.prepare_training(..., parent_approval=ID, release_root=ROOT)` uses that path for SFT;
first-cycle SFT retains the pinned Hub base, and existing DPO retains its merged SFT reference.
The actual tokenizer loads locally from the approved parent. All shards, indexes, config, tokenizer,
recipe, profile, base revision, source corpus and evaluation hashes are checked again at launch.
Preparation itself has a durable `prepared-training` record; deleting parent fields from JSON cannot
bypass the launch checks.

The strict `admin.release.ReleaseAuthority` gate writes kind `release-decision` into its configured release
AuthorityStore after evaluating the frozen four-cell evidence. Accepted parent payload:

- `schema: spark-release-decision-v1`, `origin: release_store.identity`, `result: accepted`, `fixture_only: false`.
- `candidate`: exact result of `admin.parents.merged_identity(merged_json_path, identity=release_store.identity)`.
  Real `admin.training.merge` now records `origin`, `stage`, recipe path/hash and corpus authority in `merged.json`.
- `agent`: exact `sha256:` bundle digest; `policy`: complete release policy object; `policy_hash: content_digest(policy)`.
- `evaluation`: `{id, path, sha256, origin, cells: [Q00,Q10,Q01,Q11]}` referring to the checked four-cell artifact.
- `corpus`: `{workspace: original_training_workspace, authority: its_corpus_manifest.authority}`.

This record binds the gate's accepted decision, not an unchecked caller approval flag. There is no
production approval CLI/import. The installed `cotraining`, `release` and `cycle` commands invoke
the actual producers and strict gate; CPU `fixture-approve` does not implement or bypass that gate.
See [crossed release](crossed-release.md) and [cycles](cycles.md) for the complete current records.

## Curriculum

`admin.curriculum.build_curriculum(replay, config=..., diagnostics=optional_path)` reads committed
settled experience. Configuration is `{version: "spark-curriculum-v1", count: 6, min_families: 3,
max_per_family: 2}` (all numeric fields must be positive integers). It ranks genuine family failure
rates, visits families for breadth before additional requests, and refuses unattainable breadth/caps.
The result carries input/config/policy identities, category counts and deterministic request IDs.
Categories distinguish task correctness, withheld generalization and observed tool recovery failures.
All-fail families request new executed, verified experience; no answer or automatic positive is emitted.

Strict competition scoring cannot commit infrastructure-failed/truncated attempts as eligible
settlement entries. Optional diagnostics are therefore a separate, explicitly nonlearning input:
`{origin: replay.identity, logs: [{path, sha256, repository, versions: {task_id: version}, private_required: bool}]}`.
Only infrastructure, truncation or invalid-evidence categories can come from these logs; valid learning
outcomes there refuse and must enter via settlement. They never affect weakness denominators or
create requests by themselves. The public `classify` and `select_requests` helpers expose the same
classification and deterministic selection logic to the task-generation consumer.

## Verification and review observations

`tests/learning_support.py` executes local tools and verifiers before judging/settling, with labelled
scripted policy and controlled authenticated GitHub metadata. `tests/test_learning_boundary.py`
exercises two rounds through real module/installed commands, corrupt/metrics-only sources, rights,
family aliases, exposure, all-fail curriculum, immutable namespaces and approved-parent refusal paths.
`admin.cli selfcheck` retains its genuine CPU operator corpus -> SFT -> DPO handoff.

Relevant integration-review observations: distinct admission/round/settlement/release issuers are
now bound by role; legacy-readable RoundStore state is explicitly refused for learning authority;
committed settlement remains usable after a GRADED-to-SETTLED projection crash. The authenticated
metadata trust boundary remains the private operator process. GitHub credential compatibility,
historical winner cleanup and workflow bootstrap remain the validated competition subsystem's
responsibility and were not changed by this learning assignment.
