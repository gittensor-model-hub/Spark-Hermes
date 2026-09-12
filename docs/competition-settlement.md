# Strategy evaluation and durable settlement

These commands use trusted installed code, authenticated PR metadata and the exact
admitted bundle. Start with [competition ingress](competition-ingress.md) to create
and publish a stamped challenge/round and accept private uploads. A registry merge
is not an admission, model approval or payment. The validator retains the private
round snapshot, original baseline attempts and episode logs.

Provision a private persistent **local filesystem** (SQLite locking/fsync must
work; do not use a network filesystem), for example `/srv/spark-hermes`. Back up
the entire state root, including `.identity` files and the settlement database,
using a quiescent snapshot or SQLite's backup API. Do not copy a running database
file alone. Give only the trusted validator account write access. Round,
scorecard, log and outbox inputs are validator-owned state, never PR uploads.
Use the same durable root on every restart and a single delivery owner across
hosts. The CLI has no implicit temporary settlement database. `crown select` and
`crown actions` now require `--store` and `--settlement-root`. Legacy
`datasets/crowns.json` files lack the required identity/provenance and are not
automatically imported; historical incumbent recovery requires reviewed evidence.

The examples use the external interpreter provisioned below with current trusted
checkout code. Configure
`GH_READ_TOKEN` as a bot credential that supports authenticated `GET /user` and
repository/PR/content reads; the metadata adapter currently requires that identity
endpoint, so do not assume an Actions installation token supports it. Missing
credentials and malformed or changing metadata refuse admission. The action bot
also needs PR review/close and issue-label access. Credentials are supplied only
to their corresponding process. Never run a PR-head checkout or provide repository
credentials to evaluation code.

```sh
export SPARK_STATE_ROOT=/srv/spark-hermes
# Existing round/intake stores must share their immutable mode and namespace.
"$SPARK_PYTHON" -m validator.settlement activate \
  --store "$SPARK_STATE_ROOT/rounds" --round ROUND_ID \
  --settlement-root "$SPARK_STATE_ROOT/settlement" --repository OWNER/REPOSITORY

"$SPARK_PYTHON" -m validator.pr_admission \
  --repository OWNER/REPOSITORY --pr 123 --head FULL_HEAD_SHA --round ROUND_ID \
  --credential-env GH_READ_TOKEN --store "$SPARK_STATE_ROOT/rounds" \
  --intake-root "$SPARK_STATE_ROOT/submissions" --receipts "$SPARK_STATE_ROOT/receipts.jsonl"

# After the deadline; early freezes require a recorded --reason.
"$SPARK_PYTHON" -m validator.judge freeze \
  --store "$SPARK_STATE_ROOT/rounds" --round ROUND_ID

# Requires a separately provisioned pinned model service and the runner sandbox.
# This is an operator production command, not part of the CPU proof below.
"$SPARK_PYTHON" -m validator.judge judge \
  --store "$SPARK_STATE_ROOT/rounds" --round ROUND_ID \
  --intake-root "$SPARK_STATE_ROOT/submissions" --receipts "$SPARK_STATE_ROOT/receipts.jsonl" \
  --workspace "$SPARK_STATE_ROOT/judge" --scorecards "$SPARK_STATE_ROOT/scorecards" \
  --base-url http://127.0.0.1:8000/v1 --model PINNED_MODEL --api-key-env NONE --no-settle

"$SPARK_PYTHON" -m validator.crown select \
  --store "$SPARK_STATE_ROOT/rounds" --settlement-root "$SPARK_STATE_ROOT/settlement" \
  --scorecards "$SPARK_STATE_ROOT/scorecards" --episodes "$SPARK_STATE_ROOT/judge" \
  --out "$SPARK_STATE_ROOT/latest-settlement.json"
"$SPARK_PYTHON" -m validator.crown actions \
  --store "$SPARK_STATE_ROOT/rounds" --settlement-root "$SPARK_STATE_ROOT/settlement"
"$SPARK_PYTHON" -m validator.settlement deliver \
  --settlement-root "$SPARK_STATE_ROOT/settlement"
```

The last command is a local dry run: it neither contacts GitHub nor acknowledges
pending actions. Actual GitHub delivery is a separate, explicit operator command:

```sh
"$SPARK_PYTHON" -m validator.settlement deliver \
  --settlement-root "$SPARK_STATE_ROOT/settlement" \
  --credential-env GH_ACTION_TOKEN --allow-external
```

No command emits SN74/chain rewards. Fixture roots refuse live delivery even with
`--allow-external`. Controlled transport receipts are labelled `controlled-github`;
a dry-run receipt is intent only. Neither is proof of live GitHub delivery or payment.

`strategy.yml` exposes explicit trusted admission and evaluation dispatches.
`crown.yml` settles only the persisted active round; scheduled runs never deliver.
Delivery requires a manual `deliver=true` dispatch. Both workflows check out the
repository default branch, disable persisted checkout credentials and use a
`self-hosted, linux, spark-validator` runner with a configured `SPARK_STATE_ROOT`.
Provision `SPARK_PYTHON`, `gh`, sandbox, state volume and model service before
use. `SPARK_PYTHON` is an absolute path to a dedicated Python 3.12+ virtual
environment **outside checkout and /tmp**, for example
`/srv/spark-runtime/bin/python`. Checkout uses its default clean behavior; no
runtime or durable state may depend on ignored checkout files surviving it.
Each job runs `scripts/competition-environment.sh` after checkout, without tokens,
checking the interpreter, current dependency-lock stamp, actual entry-point imports
and resolved state/runtime paths. A missing or outdated runtime stops before work.

Provision from a reviewed default-branch checkout as the trusted runner owner,
with no repository credentials in the environment (requires preinstalled `uv`
and Python 3.12; package access or a populated offline uv cache):

```sh
set -euo pipefail
export SPARK_PYTHON=/srv/spark-runtime/bin/python
export SPARK_STATE_ROOT=/srv/spark-hermes
# Create private directories owned by the validator account first.
umask 077
mkdir -p "$SPARK_STATE_ROOT"
UV_PROJECT_ENVIRONMENT=/srv/spark-runtime uv sync --frozen --no-dev \
  --extra validator --no-install-project --python python3.12 --no-python-downloads
sha256sum uv.lock | cut -d ' ' -f 1 > /srv/spark-runtime/.spark-uv-lock.sha256
bash scripts/competition-environment.sh
"$SPARK_PYTHON" -m validator.settlement --help
"$SPARK_PYTHON" -m validator.judge --help
```

Add `--offline` to `uv sync` when using a prepopulated package cache. Repeat this
provisioning whenever the trusted lock changes; never use a PR-head lock or
package source. The lock stamp records the successful setup, not a substitute for
installing dependencies. Set `SPARK_PYTHON` and `SPARK_STATE_ROOT` variables in
all three Actions environments. Dependencies live in the external environment;
product modules load from each fresh trusted checkout. Do not install an editable
project referring to an old checkout. Protect the runtime from untrusted jobs.
This dedicated runner must never accept PR-head jobs or retain repository
credentials in its host environment. Configure `competition-validator`,
`competition-evaluation` and `competition-delivery` environments, the shared state
variable, model/endpoint variables and separate read/action bot credentials.
Only the metadata and delivery steps receive those credentials. Evaluation
receives no repository token. The workflows do not install PR dependencies.

Acceptance and storage behavior:

- Flat and nested runner/sink episode records use the same evidence checks.
  Duplicate assertions across `metrics`, `evidence` and integrity reports must
  agree in type and value; outer truncation and nested failures remain visible.
  Contradictory baseline logs raise `ChallengeError` before a challenge is built.
  `Attempt.evidence` retains the original envelope, and scoring/settlement validate
  it again. A historical baseline already stripped of failure evidence needs
  trusted original logs or a fresh baseline; this repair cannot recover lost facts.
- `strategy-score-v1` carries every threshold, bootstrap setting and verification
  rule with a canonical hash. Score and crown call the same acceptance function:
  all declared attempts correct, at least ten, and a 20% reduction at the seeded
  95% bootstrap lower bound. A positive 5% saving is refused.
- Active scope binds repository, source issuer, round identity, task and complete
  epoch. Historical files cannot participate. Missing/corrupt active scorecards
  stop settlement; valid refused candidates cannot win. Each card is also
  recomputed from its original episode log before commit.
- Rank accepted entries by descending reduction lower bound, then ascending median
  tool calls, admission receipt time, miner ID, PR, round ID and task ID. These
  final identity fields make a fully equal tie independent of input ordering.
  No winner retains the incumbent. Two barren rounds retain the existing task
  rotation rule; `--challenges` supplies available published tasks. Rotation does
  not create or activate a new round. Activate the next trusted round explicitly.
- One SQLite transaction writes the immutable round result, standing and ordered
  outbox. `round_id` and action keys are unique. Retries return the committed result;
  they do not rerank evidence or recalculate actions from the new standing.
- The round snapshot's SETTLED/salt-release state is a recoverable projection after
  the database commit. Repeating `crown select` repairs a crash between them.
- Delivery serializes through a process lock in global action order. It rechecks
  repository, PR author and evaluated head. Historical crown-label removal accepts
  a merged, closed PR with those same identities and reconciles an already absent
  label. Active label/review/close actions still refuse merged or changed heads.
  Delivery reconciles labels/closed state and
  looks for bot-authored reviews with the exact commit, body and idempotency marker.
  Acknowledged actions are skipped. A timeout or failed acknowledgment remains
  pending; retry reads the remote state before writing. A failed earlier action
  stops later actions. `settlement status` shows attempts, pending errors and receipts.
  Head changes need operator resolution; they do not silently redirect actions.
- This is reconciliation of at-least-once delivery. GitHub has no atomic
  exactly-once review API; after an uncertain response, eventual-consistency delay
  can still permit a duplicate. Stable markers let operators/consumers reconcile
  such duplicates. Do not independently run a second delivery system against the
  same outbox or interpret a retry as a second winner/credit.

# Settled artifact interface for corpus and cycle consumers

`validator.crown select --out PATH` or
`validator.settlement export --settlement-root ROOT --round ROUND_ID --out PATH`
produces `schema="spark-settlement-v1"`. The database is authoritative; the export
is an immutable copy. An export contains:

| Field | Meaning |
| --- | --- |
| `settlement_id`, `scope.round_id`, `scope.round_identity` | Stable settlement and original private challenge identities |
| `origin`, `producer`, `mode`, `namespace` | Exact round issuer and settlement issuer; immutable fixture/production domain |
| `scope.epoch`, `scope.task_id`, `scope.repository` | Complete evaluated epoch, task and authenticated repository |
| `policy`, `policy_hash` | Complete versioned scoring policy and canonical SHA256 |
| `outcome` | This round's winner, accepted ordering and refusal reasons |
| `standing` | Carried incumbent and barren-round counter, distinct from this round's winner |
| `entries[]` | Every evaluated admission and full scorecard, its canonical hash, original episode path and byte SHA256 |
| `actions[]` | Persisted ordered intent, stable key, PR author/head, scope and origin |
| `authorizes_model_promotion`, `authorizes_payment` | Always false; release and reward authorization are separate consumers |

Consumers must verify origin/issuer and namespace against their configured trusted
state, validate hashes and read only the matching original episode bytes. Do not
consume just `standing.winner` as a new win in a barren round, infer live delivery
from `actions`, or upgrade fixture records by renaming files. Delivery receipts
live separately in `settlement status`/`crown actions` because acknowledgment is
mutable while the settlement artifact is immutable. Existing corpus collection
reads the SETTLED round snapshot and `judge/ROUND_ID/MINER.jsonl`; consumers needing
crown provenance should join on the exported round/admission/scorecard identities.

CPU proof without models, network or repository mutations:

```sh
.venv/bin/python tests/settlement_cli_round.py --root /tmp/spark-settlement-fixture
.venv/bin/python -m pytest tests/test_settlement.py tests/test_pr_admission.py -q
```

The script creates a new explicit fixture namespace, admits through controlled
responses to the real metadata adapter, freezes and judges labelled input logs
with the actual CLI, persists settlement and prints pending dry-run intent.
`judge --fixture-episodes` is refused by production stores. The tests kill real
processes before/after persistence, delivery and acknowledgment, restart the real
SQLite store, and exercise the real action adapter with controlled responses.
These fixtures demonstrate CPU software behavior, not measured model learning.
