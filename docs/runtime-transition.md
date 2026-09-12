# Supported runtime transition

Protocol and writer inventory for explicit operator cutover.

The supported cutover is `spark-runtime-transition-v1`, between retained installed
runtimes that implement this protocol and the current candidate/preparation/release
artifact schemas. An older executable without the protocol is available for
historical inspection only. Missing originals cannot be reconstructed by exporting
or resealing JSON. The operator must retain the original installation, authority
databases and artifacts at their original paths.

The original route is CandidateStore.resolve -> training_recipe ->
ParentAuthority.approved_parent -> ReleaseAuthority.resolve_decision -> original
plan/candidates/evaluation -> complete original episodes and corpus. This recurses
through actual preparation approvals; the Hub pin is ancestry, never a substitute
for an approved derived checkpoint. Runtime identity covers owned admin, eval,
hermes, hermesbench, miner, proof, teacher and validator modules, packaged recipe/harness assets,
interpreter and installed dependency identities. Older runtime identity semantics
are retained only for read-only historical inspection.

Campaign writers are release configuration, approved-parent preparation, confirmation freeze, experiment start
and completion, strict decisions, activation/rollback, candidate registration into
the configured store, workspace enrollment and controller configuration/start/resume. All participate in
the source campaign lock. Runtime protocol records share that lock. Training and
merge subprocesses are supervised inside controller resume; prepared workspaces retain their campaign ownership. The original installed runtime
refuses registering an old workspace in a new authority, reusing a retired strict
parent for new preparation, or configuring another root over retired candidates.
Standalone train/merge also checks workspace retirement before starting. Preparatory
artifacts cannot advance release authority. Competition resolves the exact active
pair and refuses a retired source before execution. Historical status, records and
approval verification remain readable after cutover.

The finite publication states are verified proposal, committed source successor,
and published target baseline. Verification alone leaves the source writable. A
proposal binds the retained runtime record, full original authority snapshot,
source issuer/incumbent/epoch, every reservation and its original plan/catalogs,
and an empty target issuer/candidate store/runtime. Commit serializes source and
target writes, rechecks both runtime installations and all originals, and compares
the full source snapshot. Source advancement, new reservations and target
substitution refuse. The source successor is committed with SQLite FULL durability
before any target authority is published. There is no cancel or second successor
after that point. Repeating the same commit recovers target publication atomically;
another proposal cannot claim the retired source. A crash may leave the source
retired with target publication pending; resume the exact original proposal.
Each successful freeze also commits an issuer-bound reservation record in the
reservation transaction. Removing an entire interrupted reservation group therefore
refuses verification; an uncommitted plan left by a refused freeze is not a reservation.

Target publication copies no historical approval into a new strict decision.
It issues separately typed baseline and parent bridge records that retain original
candidate/approval IDs and exact model, agent, data, recipe and parent lineage.
The target carries all spent confirmation families and all catalog aliases,
memberships and exposure, including failed/interrupted experiments. Every bridge
consumer rechecks the retained original closure and committed cutover. Ordinary
stale-runtime resolution remains an error. Fresh target candidates and four-cell
executions are required for a measured strict promotion. Fixture import, training
and release confer no production promotion, learning or payout claim.

Historical eligibility has two independent checks: the retained installed runtime
verifies original issuance, and current parsers/consumers validate original bytes,
checkpoint completeness, execution completion and numerical eligibility. Neither
an old acceptance boolean nor a freshly exported report can replace either check.

## Operator commands

Use the old installation while its campaign still runs, and retain that installation
without modifying any package, dependency, recipe or original artifact:

```sh
/retained/source/bin/spark-hermes runtime capture --root /campaign/releases
/retained/source/bin/spark-hermes runtime inspect --root /campaign/releases --id RETENTION_ID
```

Install the target separately, using copied files (for example `uv pip install
--link-mode copy`) so editing a disposable installation cannot modify the retained
one through hardlinks. Run the target installation outside either source checkout:

```sh
/installed/target/bin/spark-hermes runtime propose --root /successor/releases \
  --source /campaign/releases --retention RETENTION_ID --namespace SAME_NAMESPACE
/installed/target/bin/spark-hermes runtime commit --root /successor/releases --id PROPOSAL_ID
/installed/target/bin/spark-hermes runtime status --root /campaign/releases
/installed/target/bin/spark-hermes cycle active --root /successor/releases
/installed/target/bin/spark-hermes release status --root /successor/releases
```

For CPU fixtures explicitly add `--mode fixture` when creating the target; production
is the default. Mode/namespace must match, the target must be empty, and the source
must have an activated strict release, an exclusively owned candidate store and no
outstanding supervised train/merge job. Ordinary campaigns may share candidate stores;
that configuration does not support this cutover. Source and target must run on the
same operator host and local filesystem with functioning POSIX locks and SQLite FULL
synchronization. Campaign writes serialize on `/tmp/spark-hermes-campaign-writes.lock`;
this intentionally favors correctness over simultaneous campaign throughput.
Lock reentrancy belongs to the current process and thread; a forked caller must
acquire its own lock. Submitted or running supervised jobs refuse cutover. Reconcile
their original completion before taking a fresh source snapshot; never relaunch an
uncertain job to make transition verification pass.

Keep both roots. If source state changes before commit, create a fresh proposal.
After source commit there is exactly one successor: resume the same target/proposal
until publication finishes. SQLite DELETE journals are required for the atomic target
release/candidate publication; a different journal mode refuses with a recovery reason.
Do not delete the source, bypass a retired writer or substitute another target.

Configure the target controller's replay source using ordinary `cycle init` (see
[cycles.md](cycles.md)), then `cycle start` and `cycle resume`. The baseline's approval
is a `runtime-parent` authority, accepted by `prepare --parent-approval` with the target
`--release-root`. It cannot be used with `release activate` or rollback. The prepared
recipe keeps the exact historical merged checkpoint as `base_model`, its original
provenance and strict approval ancestry. New contribution rounds must bind the target
pair/epoch even at imported generation 0. Fresh confirmation catalogs must preserve
all historical aliases, memberships and exposures, and reserve genuinely unused families.
A new accepted strict release advances the target to generation 1; rejected candidates
leave that incumbent and history unchanged.

`runtime eligibility --root ORIGINAL_RELEASE_ROOT` is a read-only current eligibility
check for retained original artifacts, including older unsupported runtimes. It issues
no bridge, never claims fresh execution, and cannot repair missing original completion
status or incomplete checkpoint layouts. Such sources may remain inspectable but are
ineligible for continued approved-parent authority. Normal `candidates show`, parent
lookup and cycle execution never acquire a historical runtime exception from a flag.
`runtime inspect` and `runtime status` also open existing authority databases read-only,
including unsupported originals. They do not initialize or upgrade an old schema.

Supported strict generations belong to the same campaign, or to earlier campaigns
through already verified runtime bridges. Unbridged cross-authority strict ancestry
refuses: its independently writable reservation history cannot be silently imported.
The only non-strict ancestry exception is the exact configured initial CPU fixture
pair, issued by the existing fixture bootstrap producer. It grants no production
authority. Preparation ownership is part of the checked source snapshot; missing
ownership refuses instead of allowing a new root to reset reservations.
