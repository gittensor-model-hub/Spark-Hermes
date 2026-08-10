# Changelog

All notable changes to SparkDistill are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow semver.

## [Unreleased]

### Changed
- **Teachers: add Kimi K3 (Moonshot) via OpenRouter** (`teacher/providers.py`): third frontier teacher,
  pinned to the OpenRouter slug `moonshotai/kimi-k3` with `kimi-k3` as the logical id. The two are kept
  distinct on purpose — `OpenAICompatibleTeacher` now separates `request_model` (what the gateway is
  called with) from `model` (what the trajectory records and SparkProof verifies). Conflating them does
  not fail at call time; it fails at verification, after the run. Provider key is the vendor
  (`moonshot`), matching `anthropic`/`claude-fable-5`. Pinned to OpenRouter only — the same model
  through another gateway is not the same evidence, and OpenRouter is the one route with a
  generation-id ledger re-check. Pairs with
  [SparkProof #59](https://github.com/gittensor-model-hub/SparkProof/pull/59).
- **Teachers: add Qwen 3.8 Max 2.4T, retire GPT 5.6 Sol from generation** (`teacher/providers.py`):
  new `qwen` provider pinned to the yunwu slug `qwen3.8-max`, alongside Claude Fable 5 and GPT 5.6.
  yunwu speaks the OpenAI chat-completions protocol, so `OpenAICompatibleTeacher` was generalized to
  take a `base_url` and an instance-level provider `name` rather than duplicating the request body —
  the provider recorded on a trajectory is what SparkProof verifies, so a Qwen row labelled `openai`
  would (correctly) fail.
  `gpt-5.6-sol` is removed from the **generation** allowlist and raises a self-explaining error, but
  is deliberately left in SparkProof's `ALLOWED_MODELS`: 23 merged registry entries were proved with
  it and `mix_registry --all` re-reads every one when rebuilding the canonical dataset, so removing
  it would retroactively invalidate the data behind the current frontier. **Retiring generation and
  revoking verification are different actions.**
  Pairs with [SparkProof #58](https://github.com/gittensor-model-hub/SparkProof/pull/58), which adds
  the `qwen` provider and enforces the slug pin in both directions (a qwen bundle cannot claim
  `claude-fable-5`, and an anthropic bundle cannot claim `qwen3.8-max`).
  **Not yet usable end to end:** `dataset_registry.yml` pins SparkProof at `3104e28` (v0.3.0), which
  has no `qwen` provider — the pin must be bumped to a release containing #58 before a Qwen dataset
  can pass the gate. The `qwen3.8-max` slug must also be confirmed against yunwu's `/v1/models`;
  drift fails at verification, not at generation.

### Added
- **DPO training track — gate orchestration + miner prep** (`eval/training_track_gate.py`,
  `eval/prepare_mining_dpo.py`): completes the DPO submission path. The training gate is now
  track-aware end to end — it detects a `rl: dpo` recipe (`pr_training_track`), requires the PR
  body cite the canonical **preference** URL + `pref_sha256` (`validate_pr_body_canonical_pin`
  gains a `track`), honors a **preference** merge-base grace window
  (`_canonical_pref_sha256s_for_pr_window`), and threads `acceptable_pref_shas` through the
  proof-bundle checks into `verify_submission` (which selects the preference pin + the
  `<arch>::dpo` frontier). SFT PRs are byte-identical. `eval.prepare_mining_dpo`
  (`scripts/prepare_mining_dpo.sh`) downloads the canonical preference dataset and writes the
  local chosen/rejected jsonl, pinned to `pref_sha256`. A DPO bundle still **fails closed** until
  the canonical preference pin exists. (Activation, deliberately deferred: publish the canonical
  preference dataset + write the `pref_manifest` pin, and ship the `dpo.yaml` example recipe once
  the training-submission gate distinguishes a new recipe file from a proof-bundle submission.)
- **Canonical reasoning trace — what a student learns instead of teacher prose** (`hermes/state.py`):
  three frontier teachers produce three incompatible reasoning *styles*, and training a student on the
  concatenation teaches verbosity, conflicting strategies and inconsistent tool habits — the styles
  fight each other and none of them is the transferable part. What transfers is the shape of the work:
  `goal → known → unknown → hypothesis → action → expected → observed → decision`.
  A `thinking` step may now carry a `ReasoningState` instead of free text; where it does,
  `hermes.format` trains the labelled form and **drops the prose entirely** — keeping both would train
  the student to emit the structure *and* the accent. `goal` and `action` are required, because a
  state with no next action is a sentiment rather than a step. `expected_signal` paired with
  `observed_signal` is the load-bearing pair: it records a prediction made *before* the evidence
  arrives, which is how a student learns its expectations are testable rather than authoritative.
  `--structured-only` drops partially-normalized rows, which is what matters for a multi-teacher mix:
  a row with one un-normalized step still trains raw prose there and reintroduces the style mixing.
  Compression is **measured and reported, never asserted** (`measure_compression`) — the ratio a real
  50k-token trace achieves depends on that trace, and claiming an undemonstrated number is exactly
  what this pipeline exists to avoid. Fully backwards compatible: trajectories without state are
  unchanged.
- **HermesBench v1 — long-horizon tasks and goal drift** (`hermesbench/`): short episodes cannot
  exhibit the failures that matter at length, so `v1` tasks declare **checkpoints** — sub-objectives
  checkable on their own and sampled repeatedly while the episode runs. An objective observed passing
  and later failing is goal drift / context corruption made concrete, and a single end-of-episode
  `verify` cannot see it: it scores "finished objective A, then destroyed it" identically to "never
  did A". `objective_completion` and `objective_regression_rate` are reported alongside their
  denominator, and short tasks carry `-1.0` rather than `0.0` so a suite mixing horizons never
  averages a 5-step task in as a failed long-horizon one. The measurement deliberately
  **under**-reports (an objective broken and repaired between two samples is invisible) — a false
  accusation of drift would be worse than a missed one.
  **Verification is no longer charged to the action budget**: `max_verification_steps` is separate
  from `max_steps` and tasks name their read-only tools in `verification_tools`. One shared budget
  made the harness contradict itself — `self_check_rate` rewards re-running the tests while the
  budget charged for it and could cut an episode off mid-verification. Tasks declaring no
  `verification_tools` keep the previous single-budget behaviour, so every `v0` task is unchanged.
  Two bugs found writing the first v1 task: `grep -r` matched compiled bytecode in `__pycache__`
  (penalising the agent for an artifact it never created), and under `set -e` bash **never exits on a
  command negated with `!`** — so the task's "nothing may import the shim" check was a silent no-op.
- **SparkRouter Phase A — static registry, eligibility and capability statistics**
  (`hermes/router/spec.py`, `manifest.py`, `capability.py`, `plan.py`): the schemas and deterministic
  rules the rest of the routing plan builds on. **Do not route to a model; route to a verified worker
  configuration** — `AgentModule` pairs a model with its Hermes profile, tools, environment, verifier
  and budget, because a bare model id cannot express what makes a route *valid* and those constraints
  would otherwise be discovered mid-episode as failures. Registry lookups go through aliases
  (`expert.cuda.optimization`), so a model upgrade is a registry edit; repointing an alias raises
  rather than silently moving production traffic.
  **Hard filters run before scoring**, as a separate stage: a CUDA expert on a CPU box is not a worse
  route, it is not a route. Blocks on missing tool, missing/mismatched GPU architecture, context
  limit, modality, action mismatch (the best kernel *writer* is not automatically the best
  *reviewer*), and training rights — which **fail closed**, since unknown rights are not approved
  rights. `NoEligibleModule` carries every exclusion and its reason code.
  **Capability estimates are Beta-posterior means, not raw rates.** 2/2 must not outrank 800/1000;
  ranking on raw success routes everything to whichever model has the least evidence. A never-measured
  pairing scores the prior rather than 0.0 (never having been tried is not evidence of failure), and
  model versions keep separate histories so a regression in a new build cannot hide behind the old
  build's record. Routing mode follows the evidence: `single` on a clear winner, `dual` on a close
  margin / thin coverage / **no deterministic verifier**, `all_eligible` when a bucket has no coverage
  at all — the only way it acquires any.
  `HarnessPin` records every digest that can change what the agent observes (tool schema, system
  prompt, container, lockfile), because a release number is not a pin; `conformance_verified` defaults
  to False so trajectories from an unconfirmed harness cannot silently become training data.
  **Not yet wired to generation or serving**, and Kimi K3 is in neither teacher allowlist — adding it
  needs the same two-repo change as Qwen plus a rights determination.
- **Hermes router — Phase 4** (`hermes/router/`): dispatches a task to the specialist best suited to
  it, or to the generalist when it is not sure. Ships the specialist taxonomy, a deterministic
  `KeywordRouter` baseline, a `ModelRouter` that wraps any `complete(prompt) -> str` callable (so the
  3B–7B classifier drops in without touching the logic), and a routing eval harness.
  **Abstention is the design.** The two routing errors do not cost the same: a firmware task sent to
  the generalist loses some quality, while one sent to `cyber` puts a specialist confidently to work
  outside its training. So the router falls back rather than guesses; `RoutingDecision.abstained`
  distinguishes "I don't know" from a deliberate `general`; cross-domain tasks go to the generalist by
  rule; and `evaluate()` reports `misroute_rate` apart from `abstention_rate`, since trading the
  expensive error for the cheap one is an improvement even when accuracy does not move. Model-router
  guards: a hallucinated specialist, a low self-reported confidence, and a dead endpoint all degrade
  to the generalist instead of dispatching into the void or crashing the run.
  **The shipped `routing_v0` suite is a regression harness, not validation** — its 31 tasks and the
  keyword lists were authored together, so the baseline scores 100% on it by construction, and a
  learned router has no headroom to prove itself. Validating one needs a held-out `routing_v1` written
  by someone who has not seen `domains.py`. `compare()` reports a candidate that merely matches the
  free baseline as not justifying its inference cost. No model is served.
  **Two-tier cascade** (`CascadeRouter`): deterministic rules settle the easy majority for free and a
  small LLM ("tiny router") is consulted only when they genuinely cannot tell — on `routing_v0` the
  free tier handles 74.2% and the model sees 25.8%. Routing everything through an LLM pays full price
  for the easy cases; routing nothing through one sends every ambiguous task to the generalist. Only
  *uncertainty* escalates: `no_evidence`/`too_close` mean the rules cannot tell and a model might,
  while `cross_domain` means the task provably spans three specialists so the generalist already is
  the answer and a second opinion is a wasted call. That distinction rides on a structured
  `reason_code`, not on matching a human-readable message, so rewording cannot silently change what
  gets billed. Every decision records the tier that made it, and `evaluate()` reports
  `escalation_rate` so the saving is measurable. A cascade with no tiny router deployed degrades to
  the free tier rather than erroring, and a dead or reckless tiny router shows up in the visible
  metrics instead of being absorbed.
- **TritonBench stays** (documented in `README.md` and `docs/roadmap-hermes.md`): it is not legacy.
  `runs/frontiers.json` scores both architectures on triton composites, so every `eval:*` label is a
  delta against them; `eval.attested_samples.verify_tritonbench_report` depends on it for anti-forgery;
  and Phase 3 inherits it as Spark-Hermes-CUDA's domain eval. Its scope narrows, its existence does not.
- **Hermes-native agent pipeline — Phase 0 scaffolding** (`hermes/`, `hermesbench/`,
  `hermes/recipes/spark-hermes-27b-alpha/`, `docs/roadmap-hermes.md`): reframes SparkDistill around building
  **verified workers** rather than chat models. `hermes/trajectory.py` defines the agent
  execution-trajectory schema and enforces the invariant that makes it trainable — **every tool call
  has an observed result**, and an episode with no tool use at all is rejected as a chat record.
  `hermes/generate.py` produces synthetic trajectories from the existing pinned teachers, stamping
  every row `metadata.executed = false` because a teacher's imagined tool output is not evidence;
  `hermes/format.py` renders trajectories to Axolotl `chat_template` messages with `<think>` blocks
  and tool-call/tool-result roles. **HermesBench-v0** (`hermesbench/`) scores agents on work rather
  than knowledge: success is the exit status of each task's own `verify` command run against the
  workspace the agent actually modified, never the model's closing summary. The runner executes every
  tool call for real and drops policy-authored `tool_result` steps, so a policy can never be believed
  about an observation it invented, and `task.tools` is enforced at execution so a read-only task
  cannot be talked into handing out a shell. Three tasks ship; a test asserts each one fails on an
  untouched workspace (the pass-on-correct-solution direction was checked by hand, not in CI). Three
  QLoRA stage configs (reasoning → Hermes behavior → tool reliability) target Qwen3.6-27B.
  **Nothing here is a mining track**: no `hermes/`/`hermesbench/` artifact is scored or rewarded, the
  live SN74 dataset and training gates are untouched, and no Phase 0 model has been trained — the
  base model is a placeholder pending availability. See `docs/roadmap-hermes.md` for phases 0–5.
- **Per-track frontier buckets (DPO gets its own baseline)** (`eval/frontiers.py`): the frontier
  is now keyed by `(architecture, track)`. SFT keeps the bare architecture key — `runs/frontiers.json`
  is byte-identical and existing behavior is unchanged — while a DPO run uses an independent
  `<arch>::dpo` bucket. Because `verify_submission` already labels `eval:BASELINE` when the frontier
  is unset, this makes the **first verified DPO run its own phase baseline** (pays 2×, seeds the DPO
  frontier) and every later DPO run **tier (XS–XL) over the DPO frontier** — exactly how SFT was
  bootstrapped, with no cross-contamination between tracks. Frontier load/apply are track-aware
  (`load_frontier_scores(..., track=)`, driven by the bundle's `train_objective`); `runs/frontiers.json`
  now preserves track buckets on read/write. (Inert until the DPO track is live — needs the canonical
  preference pin + gate orchestration.)

## [0.2.0] — 2026-07-26

Verification-hardening and capability release. The training/dataset gates now corroborate the
training GPU against **signed NRAS claims** (not the miner-editable sidecar), bind claim/TDX to
the signed `eat_nonce` and quote REPORTDATA, fail closed on malformed proof-bundle scores and
attested-sample forgery, and keep the accepted-registry snapshot + `runs/frontiers.json` state
consistent with the mix. New capability: a **DPO training track** (preference tuning verified
exactly like SFT), a **CI pipeline** (ruff / pyright / pytest + SHA-pinned Actions + Dependabot),
**CoT recovery** for encrypted/empty teacher reasoning, and a GPU-wheel **vLLM serve stack** for
fast eval. Pairs with — and the dataset gate is now pinned to —
[SparkProof v0.3.0](https://github.com/gittensor-model-hub/SparkProof/releases/tag/v0.3.0)
(DPO correctness pairs + attested `preferences_sha256`).

Package version **0.2.0** (was `0.1.3`).

### Added
- **DPO training-track foundation** (`eval/canonical_dataset.py`, `eval/verify.py`):
  miners can submit an Axolotl `rl: dpo` recipe that trains on a canonical **preference
  dataset** (chosen/rejected pairs) — verified and scored exactly like SFT, since scoring is a
  held-out benchmark delta independent of training method. The single "this is SFT" chokepoint,
  `assert_recipe_uses_canonical_dataset`, is now track-aware: an SFT recipe (no `rl` key) keeps
  the exact prior behavior and must use the SFT mix, while a `rl: dpo` recipe must use the
  canonical preference path (`CANONICAL_PREFERENCE_DATASET_PATH`). The two tracks' accepted
  datasets/pins are disjoint (`canonical_pref_sha256` beside `canonical_sft_sha256`), so a DPO
  submission can never satisfy the SFT pin or vice-versa — SFT verification strength is unchanged.
  Preference pairs come from SparkProof's GPU-validated verified-vs-failed kernels (SparkProof
  `--pair-type correctness`), so the signal is attestable, not human-labeled. The reward-gate
  check `verify.check_canonical_dataset_claim` is likewise track-aware: a bundle that declares
  `train_objective: "dpo"` is verified against the canonical **preference** pin
  (`pref_manifest.pref_sha256` / `canonical_pref_hf_url`) and **fails closed** if no preference
  pin is configured, while SFT bundles keep their exact behavior (including bootstrap
  fail-open-when-unpinned). (Follow-up: publish the canonical preference dataset + `pref_manifest`
  pin, `prepare_mining_dpo`, the gate orchestration — DPO-aware PR-body citation + grace window in
  `training_track_gate` — and the `dpo.yaml` example recipe (deferred so it does not trip the
  training-submission gate), to complete end-to-end DPO submission.)
- **vLLM serve stack installs GPU wheels again**: `scripts/install_serve.sh` now pulls the
  official `vllm==0.25.0+cu129` wheel plus matching PyTorch CUDA builds instead of the
  generic PyPI package (which could install CPU-only torch and spend minutes JIT-compiling
  at engine start). `eval/serve_stack.py` centralizes wheel selection for Hopper H100/H200
  and Blackwell CC nodes; `eval.triton_bench` auto-uses the serve venv, caps `--max-model-len`
  for eval, and disables Blackwell CUDA graphs that regressed short decode runs on vLLM 0.25.
- **CoT recovery for encrypted/empty teacher reasoning** (`teacher/cot_recovery.py`): GPT 5.6
  over chat-completions often returns no usable reasoning (absent, empty, or an encrypted
  `reasoning_details` dump), producing bare-answer SFT rows with no `<think>` block.
  `teacher.generate` now normalizes captured reasoning (dropping encrypted JSON) and, when a
  non-Fable trajectory has no usable trace, asks Claude Fable 5 to explain how to reach the
  answer and attaches that plaintext rationale as the `<think>` trace, tagged
  `metadata.cot_recovery`. On by default; `--no-recover-cot` disables. Best-effort — a flaky
  recovery call never discards an already-generated trajectory.
- **Continuous integration** (`.github/workflows/ci.yml`): a least-privilege `pull_request`
  job runs `ruff check`, `ruff format --check`, `pyright`, and `pytest` on every change (no
  secrets exposed to fork PRs; SparkProof-dependent tests skip via `tests/conftest.py`). All
  GitHub Actions across every workflow are pinned to commit SHAs, and `.github/dependabot.yml`
  keeps the pins and Python (uv) deps current. `tritonbench/` (vendored) is excluded from ruff.

### Changed
- **Dataset gate pins SparkProof to the v0.3.0 release**: the `Dataset registry gate` workflow
  now checks out SparkProof at the immutable `v0.3.0` commit (`3104e28`) instead of the moving
  `main`, so every dataset submission is verified against exactly the released verifier — the one
  carrying the DPO correctness pairs + attested `preferences_sha256` this repo's DPO track
  consumes, plus the v0.3.0 attestation hardening. Reproducible and auditable; bump deliberately
  when the gate should adopt a newer SparkProof release.

### Fixed
- **`serve_stack._gpu_architecture` `UnboundLocalError` on the auto-detect path**: when
  `SPARKDISTILL_GPU_ARCHITECTURE` was unset and `nvidia-smi` succeeded, `normalize_gpu_architecture`
  was referenced but only imported inside the override branch (and `subprocess` only inside the
  `try`), so the function raised instead of returning the detected architecture. Both imports are
  now module-level. Surfaced by the new pyright gate; `frontiers`/`prepare_mining_sft` got small
  type-safety fixes so the gate passes clean.
- **Malformed proof-bundle scores fail closed instead of crashing verify**:
  `assert_fraction_scores` (the ingestion guard for the miner-controlled
  `eval_scores.json["scores"]`) ran `float(value)` directly, so a `null`, list, or
  object score value raised `TypeError` and a non-numeric string raised a bare
  `float()` `ValueError` — both escaped the function's fail-closed contract and crashed
  `eval.verify` / the CI training-track gate (which does not wrap `verify_submission`)
  instead of rejecting the bundle. A JSON `true`/`false` also slipped through as
  `1.0`/`0.0`, and NaN/Infinity (which `json.loads` accepts) could sneak past the range
  check. Every score value is now validated to be a finite, non-boolean real number,
  and a non-object `scores` payload is rejected up front.
- **Registry mix export uses SparkProof publish path**: `eval.mix_registry` now delegates
  to SparkProof's `trajectory_to_messages_record` (same as HF publish) instead of
  `teacher.format`. Empty or failed-validation trajectories are skipped (not coerced
  into empty assistant turns), multi-turn episodes are preferred, and repair rows use
  `prompt_meta.prompt` instead of the validator wrapper. Closes the approach in #213.
- **Accepted-registry snapshot applies the same exportability filter as the mix**:
  follow-up to the SparkProof publish-path export above — `mix_registry_datasets` now
  skips unexportable (empty/failed) rows, but `export_registry_snapshot.collect_accepted_trajectories`
  still accepted every non-duplicate row, so `accepted_registry_snapshot.jsonl` /
  `accepted_task_ids.json` over-reported rows the canonical mix rejects and their dedup
  state diverged (an unexportable row wrongly blocked a later good duplicate). The
  snapshot builder now runs the same exporter and drops the rows the mix drops, so miners'
  pre-generation dedup matches what actually enters the mix.
- **Claim binding uses signed NRAS ``eat_nonce``, not editable JSON** : `check_claim_binding`
  / `check_attestation_integrity` require `eat_nonce` from JWKS-verified platform or
  per-device JWTs (`REMOTE_GPU_CLAIMS`) to equal `claim_sha256(bundle)`. Miner-editable
  `attestation["claims"]` can no longer rebind a stolen valid NRAS token to another
  bundle. Aligns with SparkProof `verify_nras_token(..., expected_nonce=)`.
- **TDX binding reads REPORTDATA from ``quote_b64``, not JSON**: `check_tdx_binding` and
  dataset-track TDX checks extract the 64-byte REPORTDATA from the quote at the TDX v4
  offset and compare to `tdx_report_data(claim_sha256|nonce)`. Forged `tdx.report_data`
  JSON can no longer rebind a genuine quote; JSON/quote mismatches are rejected.
- **Attested GSM8K regression sample can no longer forge a score by duplicating
  problem_ids**: `verify_regression_sample` (validator side) recomputed
  `correct / len(problems)` while iterating over the miner-controlled `responses`
  list without checking coverage, so a bundle whose `responses` duplicated one
  correct answer (or cherry-picked easy problems) recomputed to an arbitrary
  `exact_match` — e.g. 50 copies of one correct answer verified as `1.0`, slipping
  a regressed gsm8k past the attested no-GPU floor. `verify_regression_sample` now
  requires the responses to answer each frozen problem exactly once (the same
  coverage invariant `build_regression_sample` already enforced miner-side).
- **Training-track merge now seeds / raises ``runs/frontiers.json``**: the ledger
  workflow only wrote `ledger.jsonl` + `result.json`, so Hopper `eval:BASELINE` (#120)
  never filled its architecture bucket. `record_merged_ledger_entry` now calls
  `apply_verified_report_to_frontiers` (and backfills the Hopper frontier from
  `2026-07-15-magicrails-hopper-v2`).

## [0.1.3] — 2026-07-21

Training-track CI fail-closes forged attestation JSON; dataset/registry gates no longer
crash on bad prior rows or invalid JSON; sha-pinned exports stay LF+UTF-8 across platforms;
teacher generation survives a single flaky API call. Pairs with
[SparkProof v0.1.3](https://github.com/gittensor-model-hub/SparkProof/releases/tag/v0.1.3)
(Sol→Fable CoT recovery for encrypted GPT Sol reasoning).

Package version **0.1.3** (was still `0.1.1` in `pyproject.toml` after the v0.1.2 tag).

### Fixed
- **Training CI fail-closes GPU + TDX attestation crypto** ([#194](https://github.com/gittensor-model-hub/SparkDistill/pull/194)):
  forged `{"passed": true}` attestation JSON no longer earns eval tiers. `eval.verify` and
  the training-track gate require NRAS JWKS signature + `claim_sha256` nonce binding; when
  a TDX quote is present (or attested eval samples require it), both REPORTDATA binding
  and Intel DCAP/PCS verification must pass. Proof-only bundles whose attested samples
  cover every claimed benchmark still verify entirely on CPU.
- **Unattested training PRs cannot earn eval tiers**: training-track eval labeling requires
  a committed `runs/<run-id>/attestation.json` that passes integrity checks.
- **Repair-tier mix dedupe fallback** (SparkProof [#29](https://github.com/gittensor-model-hub/SparkProof/pull/29)):
  `_PromptDedupeRegistry` fingerprints `metadata.prompt_meta.prompt` before top-level
  `prompt`, matching SparkProof `NoveltyRegistry` for repair-heavy bundles.
- **Dataset registry gate tolerates malformed prior rows when indexing duplicates**
  ([#195](https://github.com/gittensor-model-hub/SparkDistill/pull/195), follow-up to
  [#173](https://github.com/gittensor-model-hub/SparkDistill/pull/173) /
  [#189](https://github.com/gittensor-model-hub/SparkDistill/pull/189)): building `seen_hf` /
  `seen_sha` from existing registry lines no longer calls `hf_repo_from_url` unconditionally.
- **Sha-pinned jsonl exports write LF + UTF-8 on every platform** ([#195](https://github.com/gittensor-model-hub/SparkDistill/pull/195)):
  mining SFT export, registry mix, accepted-registry snapshot, and teacher trajectory writers
  open text files with `newline="\n"` so Windows `\r\n` cannot desync byte hashes.
- **Teacher generation skips a single flaky call instead of aborting the batch**
  ([#195](https://github.com/gittensor-model-hub/SparkDistill/pull/195)): per-prompt API
  failures are logged and skipped; all-fail still raises.
- **Dataset registry gate rejects malformed lines / invalid JSON cleanly**
  ([#173](https://github.com/gittensor-model-hub/SparkDistill/pull/173),
  [#189](https://github.com/gittensor-model-hub/SparkDistill/pull/189)): returns
  `dataset:REJECT` instead of crashing CI with a traceback.
- **GPU corroboration matches hwmodel claims only**
  ([#149](https://github.com/gittensor-model-hub/SparkDistill/pull/149)): only per-device
  `hwmodel` values corroborate `train_gpu`; non-empty claims without `hwmodel` fail closed.

## [0.1.2] — 2026-07-15

Hopper joins Blackwell on both mining tracks, per-architecture frontiers replace the
shared Triton scoreboard, and dataset verification closes the last userland trust gaps
(NRAS JWKS + Intel TDX). Architecture-scoped exact dedupe and a refreshed canonical mining
mix (174→178 rows) land in the same release window.

### Added
- **Canonical mining dataset for every training PR** ([#89], [#97]): `datasets/canonical.json`
  pins `gittensor-model-hub/sparkproof-mining`; recipes must cite the canonical path,
  HF URL, and `sft_sha256`. Training gate rejects local generators, private blends, and
  registry edits; published HF proof bundles are mandatory ([#97]).
- **Training-track canonical pin grace window** ([#121], fixes [#118]): when dataset
  merges advance the pin mid-train (~60 min cycles), the gate accepts any `sft_sha256`
  from the PR merge-base through `main` HEAD. Cite the pin you trained on in the PR body.
- **Hopper H100/H200 dataset generation** ([#104], SparkProof [#20]): registry entries
  carry required `gpu_architecture` (`blackwell` / `hopper`), cross-checked against the
  re-verified bundle. SparkProof stamps architecture-specific prompts and validation.
- **Per-architecture TritonBench frontiers** ([#104], [#109]): `runs/frontiers.json`
  holds separate Blackwell and Hopper buckets; `eval.verify` tiers each bundle against
  its own architecture's frontier instead of comparing hardware-sensitive speed numbers
  across GPUs. `eval.triton_bench` records the architecture a run executed on.
- **Attested no-GPU validator path** ([#101], [#104]): miners export
  `attested_eval_samples.json` on a GPU CC + Intel TDX guest once; validators
  re-check claimed scores from bundled artifacts on CPU alone — no checkpoint
  reproduction, no harness re-run. GSM8K uses a frozen 50-problem set with CPU
  re-grading.
- **Intel TDX for dataset-track bundles** ([#122], SparkProof [#22]): production
  dataset verification requires `gpu_attestation.tdx` with `report_data` bound to the
  dataset nonce. Legacy bundles without a `tdx` key are grandfathered; `"tdx": null`
  rejects on new bundles.
- **NRAS JWKS verification in the dataset gate** ([#53]): `sparkproof-verify --online`
  is now always passed — hand-crafted attestations fail NVIDIA signature verification.
- **Fair dataset reward labels** ([#116]): labels come from canonical-mix
  `rows_selected` after cross-registry dedupe, not raw bundle `verified_rows`.
- **Accepted-registry snapshot for miner-side novelty** ([#119]): CI publishes
  `accepted_registry_snapshot.jsonl` + `accepted_task_ids.json` on the canonical mining
  HF repo so miners can run SparkProof `--registry-snapshot` before spending GPU time.
- **Training-track ledger automation** ([#127]): `training_track_ledger.yml` appends
  `runs/ledger.jsonl` and `runs/<run-id>/result.json` on every merged training PR;
  backfilled [#120].
- **Training-track attested verification in CI** ([#126]): gate runs `eval.verify`'s
  CPU-only checks (claim binding, JWKS, TDX DCAP, attested samples) against the PR's
  committed `attestation.json` and per-arch frontier.
- **6-hour cron safety net** for canonical pin refresh when registry merges skip the
  in-job commit path.
- **Registry merges:** nghetienhiep sparkproof-triton-xl-001 ([#96], 161 rows),
  magicrails-xs-v1 ([#105], 25 rows), sparkproof-triton-xl-002 ([#112], 159 rows).
- **First Hopper training BASELINE** ([#120]): magicrails-hopper-v2 on H200 —
  `eval:BASELINE`, first verified run in the Hopper frontier bucket.

### Changed
- **SN74 eval tier multipliers (2× dataset at same letter)** ([#125]): training-track
  `eval:XL/L/M/S/XS` pay **2×** `dataset:xl/l/m/s/xs` at the same tier;
  `eval:BASELINE` = **2.0**. Documented in `.gittensor/weights.json` and miner guide;
  live config via [gittensor #1635](https://github.com/entrius/gittensor/pull/1635).
- **Mining mix dedupe defaults to `exact`** ([#98]): only identical prompts drop at mix
  time; quality still enforced by SparkProof pre-merge. `dedupe_mode` recorded in
  `mix_manifest.json` ([#98] policy docs). Exact dedupe is **architecture-scoped**
  ([#133], SparkProof [#26]): same prompt on Blackwell vs Hopper is a fresh row.
- **Miner docs: registry snapshot workflow** ([#132]): `sparkproof-publish-dataset
  --mining-repo` and `accepted_registry_snapshot.jsonl` pins in `docs/miner-guide.md`
  and `datasets/README.md`.
- **Canonical pin grows 94 → 133 → 174 → 178 rows** as registry merges land ([#99],
  [#115], [#107], [#134]); `prepare_mining_sft` export format now matches registry
  publish ([#100]).
- **Training GPU claims** accept H100/H200 and B200/B300 in proof bundles ([#98]).
- **Blackwell training defaults to SDPA** with hardened Qwen3.5 train prep ([#84]).
- **Auto-close `dataset:none` PRs** ([#91]); PR template checkboxes accept bold or plain
  ([#82]).
- **GitHub Pages** updated for Hopper support and per-architecture frontiers ([#113]).

### Fixed
- **Stale `datasets/canonical.json` after registry auto-merges** ([#107]): missing git
  identity in `dataset_registry.yml` caused silent pin drift; sibling
  `update_canonical_pin.yml` never fired on token merges.
- **`prepare_mining_sft` hash mismatch** ([#100]): export now uses the same compact JSON
  format as registry publish.
- **`eval.score` NameError** (`TIER_BENCHMARK`) and missing `import os` in
  `registry_gate` pin path — surfaced while wiring Hopper ([#104]).
- **Default missing `gpu_architecture` to Blackwell** for legacy bundles ([#106]).
- **H200 attestation corroboration** ([#126]): genuine H200 nodes report `hwmodel=GH100`
  (same die as H100); old check wrongly required `gh200` tokens.
- **Invalid YAML in `update_canonical_pin.yml`** blocked the workflow entirely ([#98]).
- **Architecture-scoped dataset dedupe** ([#133], SparkProof [#26]): exact dedupe keys
  prompt matches by `gpu_architecture` in mining mix, registry snapshot, and SparkProof
  novelty gate.
- **Canonical mining pin refresh** ([#134]): republished `gittensor-model-hub/sparkproof-mining`
  with arch-aware dedupe — **174 → 178 rows** (+4 cross-arch prompts wrongly dropped).

## [0.1.1] — 2026-07-12

Hardens the proof of training into dual-vendor authenticated, measured-VM
attestation — every gap closeable with today's infrastructure is closed.

### Added
- **Intel TDX measured-VM proof** ([#45]): `eval.attestation` captures a TDX quote
  via configfs-tsm with the bundle's `claim_sha256` in its 64-byte REPORTDATA;
  MRTD (guest-image measurement) recorded. `eval.verify` reports the binding as
  `tdx_bound`. Non-TDX hosts record `"tdx": null`; once-per-boot provisioning
  documented in the miner guide.
- **Intel DCAP verification of TDX quotes** ([#48]): `verify_tdx_quote` (via
  `dcap-qvl`) validates the quote's ECDSA signature, PCK certificate chain to
  Intel's root CA, QE identity, and platform TCB status against live Intel PCS
  collateral — reported as `tdx_signature` (`"UpToDate"` + no advisories is the
  clean pass). Verified live on a real 5,247-byte quote from the Targon TDX guest.
- **NVIDIA JWKS verification of GPU tokens** ([#49]): `verify_gpu_token`
  validates every NRAS-signed JWT in the EAT (platform + per-device, ES384,
  issuer and expiry enforced) against NVIDIA's published JWKS — reported as
  `gpu_signature`. The SDK-local HS256 overall JWT is intentionally not counted
  as evidence. With this, nothing in an attestation is taken on the miner's word:
  both hardware roots of trust (NVIDIA and Intel) are authenticated end-to-end.

### Changed
- Baseline run `2026-07-11-qwen3.5-4b-mining-001` re-attested with GPU + TDX
  claim binding; ledger record carries the strongest available proof ([#46]).
- Website describes the double (GPU + measured-VM) claim binding ([#47]).

### Fixed
- Honest claims were rejected on cross-server generation drift ([#51]): the
  triton composite over the 3-problem quick set moved 2.1pp between vLLM
  server instances, past the 2pp tolerance — found by the full live function
  test. Benchmarks gain a per-benchmark `claim_tolerance_pct` (triton: 5pp
  while the problem set is tiny) and the eval server pins determinism
  (`--seed 0 --no-enable-prefix-caching`).

## [0.1.0] — 2026-07-11

First complete, working release of the SparkDistill miner economy: train a
Triton-specialized student on verified data, prove the run cryptographically,
and verify it from public artifacts alone.

### Added
- **Project foundation**: teacher-trajectory generation (Anthropic/OpenAI),
  `<think>`-format SFT data preparation, Axolotl recipes for Qwen3.5-4B,
  benchmark harness, proof-of-training bundle packaging + Hugging Face
  publishing, immutable run ledger.
- **Dataset track** (`dataset:xs`–`xl`) ([#1], [#4]–[#30]): SparkProof bundles
  verified end-to-end by CI — release gate, GPU CC attestation, sha256 pinning,
  novelty checks — with auto-merged registry PRs (`datasets/registry.jsonl`)
  and automatic aggregation into the canonical mining dataset
  (`gittensor-model-hub/sparkproof-mining`).
- **TritonBench domain benchmark** ([#4], [#32]): vendored Triton 3.7.1 /
  Blackwell harness (thunlp/tritonbench @ 603e28a) — generated kernels are
  compiled and executed on the GPU. Registered as the `triton` improvement
  signal in the eval basket; the general basket (GSM8K, BFCL, HumanEval,
  IFEval, MMLU-Pro, AIME, GPQA-Diamond) acts as the regression guard.
- **Reproducible Blackwell training** ([#33]–[#37]): recipe preparation with
  absolute paths, small-dataset guards (`sample_packing` auto-disable),
  SM-specific FlashAttention 2/3 selection with SDPA fallback, HF→local
  mining-dataset export, pinned training installer.
- **Weights-free, claim-bound proof bundles** ([#41]): bundles carry the claim
  (eval scores, training claims, per-file checkpoint sha256 manifest) — never
  the weights (~12KB instead of ~8.8GB). `proof.bundle` prints a `claim_sha256`
  that `eval.attestation --nonce` binds into the NRAS-signed GPU attestation;
  `eval.verify` recomputes and checks it (`claim_bound`), compares reproduced
  checkpoints (`checkpoint_hash_match`), and accepts locally reproduced
  checkpoints via `--checkpoint`.
- **Deterministic serving for comparable claims** ([#41]): pinned vLLM stack
  (`scripts/install_serve.sh`) and greedy decoding (`temperature: 0.0`) in the
  TritonBench eval configs.
- **BASELINE path + canonical frontier** ([#42]): `eval.verify` labels the
  first verified run on a student/phase `eval:BASELINE`; `runs/frontier.json`
  is the tracked score-to-beat, read by default.
- **First verified baseline on the ledger** ([#40]): Qwen3.5-4B LoRA on the
  canonical mining dataset, trained in ~97s on a Targon RTX PRO 6000 Blackwell
  CC node, attested (nonce-bound), published weights-free, and verified —
  `triton 0.4278` (syntax 100%, exec 0%) / `gsm8k 0.6`.
- **Project website** ([#3], later redesigns): GitHub Pages from `main:/docs`.

### Fixed
- TritonBench evaluation correctness ([#32]): stale-report pickup (mtime-based
  selection), broken `--serve` mode (`--served-model-name`), full runs using the
  quick config, full-vs-quick claim comparison (`triton_quick`), verification
  endpoint hijack via stale `SPARKDISTILL_STUDENT_ENDPOINT`; in the vendored
  harness, correctness scoring no longer gives full credit to kernels that
  merely run (an executed `torch.allclose`/`assert_close` reference check is
  required) and problem `required_patterns` are enforced.
- Attestation claims decode ([#38]): per-GPU submodule tokens (carrying
  `hwmodel`) are decoded so GPU corroboration works — found live when a genuine
  RTX PRO 6000 attestation was rejected.
- lm-eval 0.4.x results parsing ([#39]): date-suffixed output paths and
  filter-suffixed metric keys (`exact_match,strict-match`) — found live when a
  finished gsm8k run crashed on read.
- Training-track claim enforcement ([#2]): 5-hour wall-clock budget, RTX PRO
  6000 requirement, attestation/GPU corroboration.

[#1]: https://github.com/gittensor-model-hub/SparkDistill/pull/1
[#2]: https://github.com/gittensor-model-hub/SparkDistill/pull/2
[#3]: https://github.com/gittensor-model-hub/SparkDistill/pull/3
[#4]: https://github.com/gittensor-model-hub/SparkDistill/pull/4
[#30]: https://github.com/gittensor-model-hub/SparkDistill/pull/30
[#32]: https://github.com/gittensor-model-hub/SparkDistill/pull/32
[#33]: https://github.com/gittensor-model-hub/SparkDistill/pull/33
[#37]: https://github.com/gittensor-model-hub/SparkDistill/pull/37
[#38]: https://github.com/gittensor-model-hub/SparkDistill/pull/38
[#39]: https://github.com/gittensor-model-hub/SparkDistill/pull/39
[#40]: https://github.com/gittensor-model-hub/SparkDistill/pull/40
[#41]: https://github.com/gittensor-model-hub/SparkDistill/pull/41
[#42]: https://github.com/gittensor-model-hub/SparkDistill/pull/42
[#45]: https://github.com/gittensor-model-hub/SparkDistill/pull/45
[#46]: https://github.com/gittensor-model-hub/SparkDistill/pull/46
[#47]: https://github.com/gittensor-model-hub/SparkDistill/pull/47
[#48]: https://github.com/gittensor-model-hub/SparkDistill/pull/48
[#49]: https://github.com/gittensor-model-hub/SparkDistill/pull/49

[#51]: https://github.com/gittensor-model-hub/SparkDistill/pull/51
[#53]: https://github.com/gittensor-model-hub/SparkDistill/pull/53
[#84]: https://github.com/gittensor-model-hub/SparkDistill/pull/84
[#89]: https://github.com/gittensor-model-hub/SparkDistill/pull/89
[#91]: https://github.com/gittensor-model-hub/SparkDistill/pull/91
[#96]: https://github.com/gittensor-model-hub/SparkDistill/pull/96
[#97]: https://github.com/gittensor-model-hub/SparkDistill/pull/97
[#98]: https://github.com/gittensor-model-hub/SparkDistill/pull/98
[#99]: https://github.com/gittensor-model-hub/SparkDistill/pull/99
[#100]: https://github.com/gittensor-model-hub/SparkDistill/pull/100
[#101]: https://github.com/gittensor-model-hub/SparkDistill/pull/101
[#104]: https://github.com/gittensor-model-hub/SparkDistill/pull/104
[#105]: https://github.com/gittensor-model-hub/SparkDistill/pull/105
[#106]: https://github.com/gittensor-model-hub/SparkDistill/pull/106
[#107]: https://github.com/gittensor-model-hub/SparkDistill/pull/107
[#109]: https://github.com/gittensor-model-hub/SparkDistill/pull/109
[#112]: https://github.com/gittensor-model-hub/SparkDistill/pull/112
[#113]: https://github.com/gittensor-model-hub/SparkDistill/pull/113
[#115]: https://github.com/gittensor-model-hub/SparkDistill/pull/115
[#116]: https://github.com/gittensor-model-hub/SparkDistill/pull/116
[#118]: https://github.com/gittensor-model-hub/SparkDistill/issues/118
[#119]: https://github.com/gittensor-model-hub/SparkDistill/pull/119
[#120]: https://github.com/gittensor-model-hub/SparkDistill/pull/120
[#121]: https://github.com/gittensor-model-hub/SparkDistill/pull/121
[#122]: https://github.com/gittensor-model-hub/SparkDistill/pull/122
[#125]: https://github.com/gittensor-model-hub/SparkDistill/pull/125
[#126]: https://github.com/gittensor-model-hub/SparkDistill/pull/126
[#127]: https://github.com/gittensor-model-hub/SparkDistill/pull/127
[#132]: https://github.com/gittensor-model-hub/SparkDistill/pull/132
[#133]: https://github.com/gittensor-model-hub/SparkDistill/pull/133
[#134]: https://github.com/gittensor-model-hub/SparkDistill/pull/134

[Unreleased]: https://github.com/gittensor-model-hub/SparkDistill/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/gittensor-model-hub/SparkDistill/releases/tag/v0.1.3
[0.1.2]: https://github.com/gittensor-model-hub/SparkDistill/releases/tag/v0.1.2
[0.1.1]: https://github.com/gittensor-model-hub/SparkDistill/releases/tag/v0.1.1
[0.1.0]: https://github.com/gittensor-model-hub/SparkDistill/releases/tag/v0.1.0
