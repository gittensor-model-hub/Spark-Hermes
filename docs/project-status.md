# CPU software delivery and real-world prerequisites

Updated 2026-09-12 for the confirmed public contribution/private derived-model policy.
The connected software path includes trusted PR admission, exact uploaded-bundle execution,
strict scoring, durable settlement/outbox, rights-aware replay, approved-parent preparation,
four-cell agent/model evaluation and resumable cycle activation/rollback. The installed
[two-cycle CPU demonstration](cycles.md) exercises these producers with labelled fixtures at
external boundaries. This is software evidence, not a trained model or measured learning gain.

No real optimizer execution, GPU memory measurement, trusted live serving, attested deployment,
SN74 registration or payout is certified by this result. Historical investigation reports such
as [the original pipeline review](pipeline-research-review.md) describe earlier code states;
current conclusions require current code and recorded checks.
The default `var/admin/run-1` has no completed training stages; temporary test fixtures do
not populate it or stand in for a trained checkpoint.

Public code, permitted contribution surfaces, recipes and protocol metadata remain reusable.
Derived checkpoints/adapters, licensed corpus, private checks and deployment credentials are
private by default. Preserve Qwen Apache-2.0 license/NOTICE and modification attribution,
Hermes MIT notices and independent dataset/contribution rights. [Contribution policy](../CONTRIBUTING.md)
also separates local quality intent from externally approved SN74 merged-PR eligibility and payout.

## Project map

| Location | Responsibility |
|---|---|
| `admin/cli.py` | Installed `spark-hermes` operator command |
| `admin/readiness.py` | Offline software, corpus/private-check, artifact and external prerequisite observations |
| `admin/cycles.py`, `admin/cycle_jobs.py` | Durable producer jobs, resume, atomic pair/epoch activation and rollback |
| `admin/cycle_demo.py`, `admin/cycle_fixtures.py` | Installed connected CPU demonstration and explicitly labelled external fixtures |
| `admin/competition_pair.py`, `admin/task_feedback.py` | Exact activated-pair competition execution and committed feedback-to-task synthesis |
| `admin/replay.py`, `admin/curriculum.py`, `admin/data_policy.py` | Settled experience, reviewed rights, replay caps and measured next-task requests |
| `admin/candidates.py`, `admin/cotraining.py`, `admin/release.py`, `admin/serving_identity.py` | Exact artifact identity, crossed evaluation, release authority and trusted serving protocol |
| `admin/pipeline.py`, `admin/split.py` | Stage records, task execution, corpus validation, held-out protection |
| `admin/training.py`, `admin/artifacts.py` | SFT/DPO preparation, input hashes, train/merge commands, reference provenance |
| `admin/evaluation.py`, `hermes/promotion.py` | Held-out evaluation export and improvement gate |
| `admin/selfcheck.py` | Offline CPU integration smoke check |
| `hermes/taskgen/` | Training-task generation and executable verification |
| `hermesbench/` | Tool execution, episode logs, committed evaluation tasks |
| `hermes/recipes/`, `hermes/templates/` | First 4B POC and final 27B settings and pinned chat templates |
| `validator/`, `miner/`, `proof/` | Competition service, submissions, verification interfaces |
| `eval/`, `teacher/` | Dataset preparation, evaluation support, teacher interfaces |
| `tests/`, `scripts/check.sh`, `.github/workflows/ci.yml` | CPU regression, formatting, typing, wheel smoke checks |

Each operator run lives under an ignored directory such as `var/admin/run-1/`:

```text
tasks/generated/          accepted training tasks
tasks/withheld/           private training checks
rollouts/episodes.jsonl   executed conversations and outcomes
corpus/                  verified SFT and preference JSONL
models/sft/              prepared data, recipe, adapter, merged checkpoint
models/dpo/              optional preference pass and merged checkpoint
reports/                 evaluation episodes, manifest, promotion run record
```

Stage records distinguish preparation, training, and merging. Rebuilding an upstream stage
invalidates downstream completion records while preserving files. Existing trained outputs
and episode logs require a fresh run root, preventing accidental reuse or overwrite.

## Gaps fixed

| Gap found | Resolution |
|---|---|
| CPU installation depended on a sibling SparkProof checkout | Standalone frozen dependency installation; optional integration remains explicit |
| Operator and miner modules were missing from distribution | Packaged modules, runtime templates/recipes/tasks, and installed CLI |
| Container used a nonexistent base digest and installed before copying sources | Verified official base pin, dependency-only layer first, source install, restricted build context |
| No offline integration path | CPU self-check executes tools, public/private verifiers, corpus, SFT, DPO, and stale-input rejection |
| Generated tasks were not wired consistently to rollout | Accepted task IDs, generated root, pinned dialect, private-check commitments checked before execution |
| Evaluation examples or simulated results could reach training | Held-out ID/prompt protection, execution evidence, boolean checks, measured tokens, integrity filtering |
| Preference data lost tool context or used truncated/unpriced attempts | Complete rendered trajectories with schemas and both sides filtered |
| Recipe selectors and raw preference format were incompatible | Supported template selection and rendered custom DPO fields |
| Preparation could reuse stale input/cache/reference artifacts | Data/config hashes, configuration-specific caches, merged checkpoint profile/shard/tokenizer validation |
| Train dry run or preparation could appear complete | Separate records and dry-run refusal to write training completion |
| Evaluation did not hand evidence to promotion | Serving configuration, exact attempt counts, matching manifest/log results, hashed export, manifest loading |
| Invalid checkout discovered after paid requests | Harness pin preflight before execution |
| Flags could be silently ignored on the wrong command | Command-specific flag validation |

## Software validation

Current 2026-09-12 integrated delivery: `scripts/check.sh` exited 0 with **3,837 passed,
4 existing skips and 15 warnings** in 45m03s. Ruff lint/format passed; Pyright reported
zero errors and four existing package-export warnings. The pytest warnings are one
Starlette deprecation and fourteen existing multiprocessing-fork warnings. No new behavior
was skipped. The installed wheel passed help, empty/populated doctor/status, offline
selfcheck and the connected two-cycle command outside the checkout. Its 163 packaged
Python files match the checkout and installed copies. The first strict fixture release
activated generation 1; the second numerical refusal preserved it. Full logs and original
artifact paths are recorded in `.local/integrated-delivery-evidence/REPORT.md`.

Historical 2026-09-11 check: **2,657 tests passed, 4 skipped**. These counts describe the earlier
operator snapshot, not the integrated delivery's current full check. Those four skips select between Hub-base
and derived-base recipe checks; the corresponding recipes are covered by the other check.
Ruff lint and formatting passed. Pyright reported zero errors and four existing package-export
warnings. Pytest reported one FastAPI/Starlette deprecation warning. The wheel self-check
passed outside the checkout. The Docker build, offline container self-check, and listing of
the complete 19-task suite passed. Current delivery verification belongs in the worker's recorded
command logs/report; historical passing counts are not a substitute for rerunning the checks.

Run `scripts/install.sh`, then `scripts/check.sh`. The checks require no GPU, model weights,
SSH credentials, or provider key. They include the full CPU test suite, Ruff, Pyright, and
the offline self-check. Some optional SparkProof integration tests skip without that external
package. Package verification additionally installs the wheel and runs the self-check from
outside the repository. CI builds the CPU container and runs its self-check without network
access as its unprivileged user. See the final task report for observed check results.

To repeat the container checks locally:

```bash
docker build -t spark-hermes:cpu-check .
docker run --rm --network none --entrypoint /opt/spark/.venv/bin/spark-hermes \
  spark-hermes:cpu-check selfcheck
docker run --rm --network none spark-hermes:cpu-check --suite v0,v1 --workspace-root /tmp/tasks --list
```

This image contains code and dependencies, not Git history, model weights, or private checks.
Strict crossed and active-pair competition runs fingerprint installed code bytes and environment,
so the wheel supports their manifests without fabricated Git metadata. The legacy exploratory
manifest route still requires its clean Git pin. Keep runtime code read-only and use a separate
writable location for episodes and task workspaces; changing code invalidates retained evidence.

## Inputs needed only when starting real training

These are runtime inputs, not missing project structure:

- A separate verified training corpus. The archived 125 SFT rows and 36 pairs use held-out
  tasks and are deliberately blocked from training.
- Access to the exact model/tokenizer revisions and, for execution, a compatible training
  stack and serving endpoint. Hardware memory and optimizer/serving compatibility still need
  measurement when training is authorized.
- The original private evaluation checks and salt. A new salt cannot open existing
  commitments; the software refuses missing checks instead of reporting public-only success.
- Immutable installed runtime bytes/environment and verified serving identity for strict
  baseline/candidate comparison; a clean Git pin for the legacy exploratory manifest route.
- An approved guest measurement and attestation hardware for the optional confidential
  competition deployment. CPU checks do not certify attestation or model quality.

The 4B proof of concept is the first real training target, using `rtx5090-poc` and pinned
`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` on an RTX 5090 32 GB.
The final `bf16` profile retains
`Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` on PRO 6000 96 GB.
Neither GPU is required to finish the software.
Follow [the training runbook](train-spark-hermes.md) when supplying real runtime inputs.

## Interpreting doctor and status

`spark-hermes doctor --software-only --root WORKSPACE --profile rtx5090-poc` reports CPU
software readiness and still shows production blockers. Full `doctor` exits 1 while external
prerequisites remain unverified. `status` shows stage manifests and the same categories; a
manifest is progress, not proof of real training. These commands inspect local inputs and
never contact a model provider, GitHub, chain service or GPU.

| Category | What the local report can establish | Still required for the real product |
|---|---|---|
| Software | CPU dependencies, packaged assets and exact profile pins | Full check and installed-wheel demonstration |
| Corpus | Committed source/rights/family policy and byte integrity | Actual licensed training data and truthful grants |
| Private checks | Configured bodies/salt match packaged task commitments | Fresh sealed evaluation families and protected access |
| Prepared training | Committed recipe, corpus, parent and input hashes | Real tokenizer and training compatibility |
| Real training | Merged artifact identity and file hashes | Trusted evidence of actual optimizer execution and measured gains |
| Serving | Required authenticated identity protocol | Exact live model/deployment binding on every completion |
| Hardware/attestation | Required profile and optional confidential-proof policy | Measured GPU compatibility; authentic run-bound evidence and approved guest measurement |
| SN74 | Explicit external prerequisites | Repository approval, miner and merged-PR eligibility, subnet-controlled payout |

Fixture corpus/recipe/model validity is reported as an artifact observation. Fixture namespaces
cannot turn those observations into real training, serving, production release or payout readiness.
