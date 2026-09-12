# Train Spark Hermes

The operator CLI connects generated tasks, executed rollouts, a verified corpus, bf16 LoRA
training, merging, and held-out evaluation. Installed `spark-hermes` commands also work outside
the checkout; use absolute state/configuration paths there. The first
operator SFT pass starts directly from the pinned Qwen base; it does not require the optional
reasoning and mixed-data adapters in the historical A/B/C curriculum.

Corpus admission now requires explicit reviewed rights and family/exposure metadata. See
[settled experience and repeated SFT](learning-boundary.md) for replay commands, policy schemas,
immutable fixture/production namespaces and approved cycle-two parent preparation. Legacy
corpora without source authority must be rebuilt from verified original evidence.

The confirmed release policy keeps the contribution code, permitted surfaces, recipes and
protocols public and reusable, while derived adapters/checkpoints, licensed data and private
evaluation assets remain operator-controlled. Keep upstream licenses/NOTICE and modification
attribution, and review data and contribution grants separately. An accepted release decision
authorizes an exact same-namespace pair; it does not publish weights or transfer upstream rights.
See [contribution policy](../CONTRIBUTING.md) and [crossed release](crossed-release.md).

## Complete the software workflow without a GPU

The software audit, project map, and remaining runtime inputs are in
[project status](project-status.md). Install and validate the standalone project on a CPU host:

```bash
scripts/install.sh
source .venv/bin/activate
spark-hermes doctor --software-only
spark-hermes selfcheck
spark-hermes cycle demo --root /tmp/spark-cycle-demo --mode fixture
scripts/check.sh
```

`selfcheck` is offline: it executes deterministic CPU tool fixtures, verifies correct and
incorrect outcomes, builds a temporary corpus, prepares SFT and DPO, and validates stale-input
rejection. It downloads no weights and runs no optimizer. Its tokenizer counts and checkpoint
bytes are explicitly test fixtures, removed when the check exits. CI also runs it against the
built wheel from outside the checkout.

The [two-cycle demonstration](cycles.md) uses a fresh explicit fixture root and the actual
controller, admission, judging, settlement, replay, preparation and release producers. External
GitHub, trainer and model responses are labelled CPU fixtures. Only a fixture incumbent can
activate, the second rejected candidate preserves it, and persisted state resumes after restart.

`doctor --software-only` exits according to CPU dependencies/assets, while listing all real
prerequisites. Full `doctor` exits 1 when real prerequisites remain unverified; it does not
contact serving/GitHub/chain services or probe a GPU. `status` displays stage progress plus the
same prerequisite categories. Neither a stage record nor copied fixture weights clear real
readiness. The demonstration summary names its actual cycle workspaces; inspect one with:

```bash
spark-hermes status --root /absolute/path/from/demo/summary --profile rtx5090-poc
spark-hermes doctor --root /absolute/path/from/demo/summary --profile rtx5090-poc --software-only
```

The JSON distinguishes software, verified corpus authority/rights, original private-check
commitments, prepared recipes, merged artifact hashes versus actual training evidence, trusted
serving identity, hardware, optional attestation, and external SN74 onboarding/eligibility.
Offline observations cannot certify optimizer execution or live deployment even when local
artifact hashes match. Fixture private-check results do not replace sealed production checks.

The older corpus under `var/datasets/rollout-2026-08-11/` contains 125 SFT rows and 36 preference
pairs, all from held-out evaluation tasks. It cannot be used for training while those tasks
remain the evaluation suite. Generate separate training tasks before a real run.

## First real training target after CPU validation: 4B

Use the pinned `Qwen/Qwen3.5-4B` for the 32 GB RTX 5090 proof of concept, then the
27B bf16 profile on the PRO 6000 96 GB. The 4B revision is
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; its own upstream chat template is committed
separately. Both profiles use bf16 LoRA, so the proof of concept exercises the same
training method. The POC uses rank 16, microbatch 1, gradient checkpointing, SDPA, and
2048-token sequences. Peak memory still needs a real training measurement.

Create the private training salt using the setup instructions below before generating tasks.
On the RTX 5090 host, install with `scripts/install_train.sh`, activate `.venv`, and serve
this exact 4B pin as `qwen3.5-4b`. Use a fresh root to keep the experiment separate:

```bash
python -m admin.cli doctor --profile rtx5090-poc --root var/admin/poc-4b
python -m admin.cli generate --profile rtx5090-poc --root var/admin/poc-4b \
  --count 16 --concurrency 1 --salt-file var/private/train-salt --allow-unsandboxed
python -m admin.cli rollout --profile rtx5090-poc --root var/admin/poc-4b \
  --repeats 4 --concurrency 1 --salt-file var/private/train-salt --allow-unsandboxed
python -m admin.cli corpus --root var/admin/poc-4b --data-policy var/private/poc-data-policy.json
python -m admin.cli prepare --profile rtx5090-poc --root var/admin/poc-4b --max-steps 10
python -m admin.cli train --root var/admin/poc-4b --dry-run
python -m admin.cli train --root var/admin/poc-4b
python -m admin.cli merge --root var/admin/poc-4b
```

Stop the inference server before training. `--max-steps 10` makes this a pipeline smoke test,
not evidence of improved model quality; the step cap is recorded in the preparation and training
manifests. For a full POC run, use a fresh root and omit the cap. The corpus must contain complete
trajectories within 2048 tokens; do not truncate long episodes to force them into this profile.
If memory permits a longer sequence, pass an explicit `--sequence-len` during preparation.
Before `corpus`, create the reviewed `poc-data-policy.json` using the
[rights/family schema](learning-boundary.md#replay-configuration-and-policy); declare the actual source,
training permission, attribution and exposure for every task and contribution.

The default `--profile bf16` remains the pinned 27B production configuration. Do not compare
4B and 27B scores as though they measured a training improvement of the same model.

## PRO 6000 96 GB setup and baseline

Use the project's 96 GB GPU host for the bf16 27B recipe. Actual peak memory and a working
serving engine for this pin still require measurement; see [serving bring-up](serving-qwen3.8.md).
Do not leave inference and training competing for the same GPU's memory.

```bash
scripts/install_train.sh
source .venv/bin/activate
```

Keep the salt outside the model's task workspaces. Generate it once and reuse it for this task
batch; changing it breaks the withheld-check commitments. For example:

```bash
mkdir -p var/private
python -c 'import pathlib,secrets; p=pathlib.Path("var/private/train-salt"); f=p.open("x"); f.write(secrets.token_hex(32)); f.close(); p.chmod(0o600)'
```

A teacher or the pinned model must already be served through an OpenAI-compatible endpoint.
For a remote endpoint, set its credential through the environment variable named by
`--api-key-env`. Never put the credential in a command-line argument.

Before training, evaluate the served base under a separate run root. Evaluation requires the
private checks matching the committed evaluation suite, and their original salt. The newly
created training salt does **not** open those evaluation commitments.

Write the actual serving configuration to `var/private/serving.json`. For example, after
confirming the server uses these settings (replace the engine version and device as needed):

```json
{
  "precision": "bf16",
  "device": "RTX PRO 6000 Blackwell 96 GB",
  "engine": "sglang <installed-version>",
  "temperature": 0.6,
  "top_p": 0.95,
  "max_model_len": 8192,
  "confidential_computing": false
}
```

Sampling values are passed to every completion request. The other fields are operator
declarations; the client cannot prove the remote server's hardware or precision.
Manifested runs require a clean committed checkout and its `uv.lock`; the runner refuses an
unpinnable checkout before any model request. Keep runtime files under ignored `var/` paths.

```bash
export SPARKDISTILL_WITHHELD_ROOT=/private/evaluation-checks
python -m admin.cli evaluate --root var/admin/base-eval \
  --salt-file /private/evaluation-salt --model qwen3.8-27b \
  --serving-config var/private/serving.json \
  --base-url http://127.0.0.1:8001/v1 --allow-unsandboxed
```

`--allow-unsandboxed` asserts the process is already on a disposable execution host; the harness
executes model-authored commands. The operator CLI does not enable this flag implicitly.

## Collect verified training data

```bash
python -m admin.cli generate --root var/admin/run-1 --count 160 \
  --salt-file var/private/train-salt --allow-unsandboxed
python -m admin.cli rollout --root var/admin/run-1 --repeats 8 \
  --concurrency 4 --salt-file var/private/train-salt --allow-unsandboxed
python -m admin.cli corpus --root var/admin/run-1 --data-policy var/private/data-policy.json
```

Generation writes accepted task files under `tasks/generated/` and private checks under
`tasks/withheld/`. Rollout validates the checks' commitments before making model requests and
uses the pinned `qwen35` dialect. Task IDs and prompt overlap with the held-out suite are checked.
Corpus building refuses unknown tasks, public-only results, simulated trajectories, wrong
dialects, and unmeasured token usage. Integrity-disqualified successes cannot become SFT rows.
Supply the separately reviewed rights/family/exposure policy shown in
[learning ingress](learning-boundary.md). Public dataset access is not proof of training rights.

SFT chooses the cheapest complete verified attempt per task. Preference pairs compare outcomes
within a task; schema definitions accompany the messages. All-failure tasks supply no positive
example. Gather additional successful executions before training if the corpus is empty.

Generation can resume its accepted tasks. Existing rollout and evaluation logs are never
appended to by a new run; preserve a partial run and use a new root. Rebuilding an upstream stage
invalidates downstream completion manifests, while preserving artifact files.

## Prepare, train, merge

```bash
python -m admin.cli prepare --root var/admin/run-1 --sequence-len 8192
python -m admin.cli train --root var/admin/run-1 --dry-run
python -m admin.cli train --root var/admin/run-1
python -m admin.cli merge --root var/admin/run-1
```

Preparation loads the pinned tokenizer, uses the committed Qwen template, measures every
complete rendered example, and refuses examples above the sequence length. It does not silently
truncate or drop them. Raise `--sequence-len` only after measuring GPU memory, or collect shorter
complete trajectories. Hashes bind the source corpus, prepared data, and recipe; modifying them
requires preparation again. Preparation and dry runs never mark training complete.
Use `prepare --offline` when the exact pinned tokenizer is already cached. Prepared-cache
keys include the recipe settings, so changing context length or templates cannot reuse stale
preprocessing. DPO also verifies the merged reference's profile, recipe, tokenizer, and shards.

The generated configuration lives at `models/sft/train.yaml`, adapters at `models/sft/adapter/`,
and the merged checkpoint at `models/sft/adapter/merged/`, all relative to the run root. There is
no random validation split of preference pairs from the same task; evaluation uses separate tasks.

The competition launcher `scripts/train.sh` retains its canonical-dataset restrictions. The
operator CLI prepares its own verified corpus explicitly, so competition policy is not weakened.

Axolotl's documented template selectors include `tokenizer_default` and `jinja`; a bare
`chat_template.jinja` filename is not a selector. The checked-in curriculum uses the former and
the operator recipe embeds the committed template with the latter.
[Axolotl conversation configuration](https://docs.axolotl.ai/docs/dataset-formats/conversation.html).

Optional preference training, after SFT has been merged:

```bash
python -m admin.cli prepare --root var/admin/run-1 --training-stage dpo --sequence-len 8192
python -m admin.cli train --root var/admin/run-1 --training-stage dpo
python -m admin.cli merge --root var/admin/run-1 --training-stage dpo
```

DPO starts a fresh adapter over the merged SFT policy. Preparation renders the shared initial
prompt and both complete trajectories to strings, preserving reasoning, calls, and results.
This is **trajectory-level DPO**: tool-result tokens also contribute to its loss. It is not
assistant-only action preference optimization. Use SFT alone if that objective is unsuitable.
The resulting fields use Axolotl's documented custom DPO format.
[Axolotl DPO formats](https://docs.axolotl.ai/docs/rlhf.html#user_defined.default).

## Evaluate the result

Serve the merged checkpoint after stopping the trainer. Evaluate it under a new root with the
same private evaluation checks, salt, serving precision, settings, and hardware as the baseline.
The evaluation command uses the 19 held-out tasks with ten repeats.

```bash
python -m admin.cli evaluate --root var/admin/candidate-eval \
  --salt-file /private/evaluation-salt --model spark-hermes-candidate \
  --serving-config var/private/serving.json \
  --base-url http://127.0.0.1:8001/v1 --allow-unsandboxed
```

Use `evaluate --print-only` to inspect the command without executing it. A successful training
exit and adapter validation establish that a checkpoint was produced, not that it improved.
The existing `hermes.promotion` gate compares evaluation records with matching serving and
grader metadata. GGUF export is a subsequent distribution step and requires its own evaluation.
This legacy two-run comparison is exploratory evidence. Production cycle activation requires
the exact candidate/agent identities, trusted serving binding and four-cell paired-family
experiment through `spark-hermes candidates`, `cotraining`, `release` and `cycle`; follow
[crossed release](crossed-release.md) and [the cycle runbook](cycles.md). A serving declaration
or model alias alone does not supply that authority.

Evaluation writes `reports/run.json` with the serving declaration and hashed references to
`manifest.json` and `episodes.jsonl`. It refuses incomplete or disagreeing results. Compare:

```bash
python -m hermes.promotion \
  --incumbent var/admin/base-eval/reports/run.json \
  --candidate var/admin/candidate-eval/reports/run.json --json
```

## Validation still required on the GPU host

Run tokenizer preparation against the pinned Hub artifact, Axolotl preprocessing and a short
training smoke test, then measure peak memory for the chosen context length. Validate the merge
by serving it and running the held-out suite. CPU integration tests exercise data and command
handoffs but do not establish CUDA, optimizer, model-class, or serving compatibility.

Live SN74 participation is also external: repository registration and manual approval, the
read-only Gittensor App, miner eligibility and eligible merged PRs must satisfy the current
subnet policy. Local scores, labels, outbox receipts and release decisions do not guarantee
emissions. See [registration](https://docs.gittensor.io/register-repository.html) and
[OSS contribution scoring](https://docs.gittensor.io/oss-contributions.html).
