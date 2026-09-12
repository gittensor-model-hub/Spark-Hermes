# Spark Hermes: investigation and proposed mission scope

Reviewed 2026-09-12. Status: scope confirmed by the user; engineering implementation in progress.
This is a proposal for the requested system, not a claim that continuous training,
live SN74 rewards, or model quality improvement already works.

## Intended product

Spark Hermes is a Qwen-derived model and Hermes agent stack that improve together
through externally contributed, independently evaluated GitHub pull requests.
Start with the existing pinned `Qwen/Qwen3.5-4B` profile; retain
`Qwen/Qwen3.8-27B` as the final target. Both exact model identifiers exist in the
official Qwen organization. Their architecture is in the `qwen3_5` family; the
27B checkpoint must not be silently replaced with another model or assumed to be
a conventional text-only Qwen3 checkpoint.

The current work is CPU-verifiable engineering, under the user's earlier request
to complete the project without any GPU. Real training and quality measurements
remain required for the final product. Zenith coding-agent investigation and
execution are distinct from training or serving the candidate Spark Hermes model.
No GPU, SSH job, PR publication/merge, repository registration, or on-chain action
has been performed for this mission.

## Proposed public and proprietary boundary

Propose a public contribution repository containing the agent framework, allowed
optimization surfaces, training/evaluation tooling, protocol, and reproducibility
metadata. Keep private training data, unreleased adapters/checkpoints, deployment
secrets, and product assets under explicitly recorded access and licensing rules.
Expose sufficient reference behavior and evaluator evidence for miners to test
useful changes and contest results. Public code remains reusable by competitors.

The Qwen checkpoints are Apache-2.0; Hermes Agent is MIT. These licenses support
commercial derivatives subject to their conditions, not exclusive ownership of
upstream Qwen or Hermes. Preserve license/NOTICE and modification attribution,
and track dataset and contribution rights separately. This proposal changes the
repository's present blanket description of Spark Hermes as an open-weight model;
the user confirmed that release-policy difference before implementation.

## SN74 integration boundary

Gittensor's OSS competition rewards eligible merged PRs to registered repositories.
A public maintained repository, its read-only GitHub App, a registration submission,
and manual acceptance are required. A local `.gittensor/weights.json` does not
enroll Spark Hermes or set its live emissions. The actual subnet registry and
validator policy govern eligibility and payouts.

SparkInfer is a useful example of pinned baselines, correctness-gated evaluation,
trusted labels, and frontier history. Its local weights file is documentary.
Its speedup label implementation requires greater than 2% frontier improvement
and anchors higher tiers against a separate reference. Those speed thresholds and
label economics are not automatically suitable for agent quality or co-training.
Spark Hermes needs its own measured and versioned acceptance policy, plus external
agreement about any trusted quality labels used for SN74 rewards.

## Confirmed engineering gaps

| Area | Current gap | Required observable behavior |
| --- | --- | --- |
| PR admission | `eval/community_pr_policy.py` recognizes training/dataset submissions but auto-closes external strategy commitments. `eval.strategy_track.gate` has no production caller. | Valid strategy, data, and training submissions enter the intended trusted evaluation path; unrelated scorer changes cannot change the submitting miner's current score. |
| Exact candidate execution | `validator/judge.py` verifies one bundle digest, while its runner can select another upload by the same miner. | The committed receipt and exact bytes determine what executes, including multiple-upload and tampering cases. |
| Evidence and scoring | `validator/score.py` can count absent private results, malformed execution, integrity-disqualified rows, and negative token counts. | Required execution evidence, positive measured costs, correct task/epoch, complete attempts, and applicable verifiers are mandatory for credit. |
| Settlement | `validator/crown.py` can crown refused or stale candidates and loses label actions when recalculating after state persistence. Its workflow uses ephemeral state. | One versioned acceptance decision drives scoped settlement and a durable action outbox. Crash/retry cannot lose an action or duplicate credit. |
| Identity and intake | Upload identity is caller-supplied and receipt storage is an unlocked read-modify-write sequence. | Trusted PR/miner/receipt association and concurrent intake preserve attribution and all accepted receipts. |
| Corpus | Competition aggregation and the stricter operator corpus path have different admission rules; history/replay and family-level isolation are missing. | Admitted examples retain execution, contribution, rights, and artifact provenance; replay is deduplicated/capped and sealed task families cannot enter training. |
| Repeated learning | Operator stages end at evaluation. New SFT runs start from the original Hub base; promotion and evolution helpers are not connected to a cycle controller. | A durable controller resumes rounds, prepares the next approved parent, evaluates candidates, retains the incumbent after failure, and activates successful releases exactly once. |
| Release identity | Evaluation records a model alias without binding the exact checkpoint and agent artifact; missing manifests are merely notes in the statistical promotion helper. | Automatic activation rejects missing/stale provenance and binds exact model, agent, data, recipe, environment, workload, and evaluator identities. |
| Co-training measurement | Existing model-only comparison requires a fixed harness; no separate agent/model experimental factors exist. | An explicit four-cell experiment distinguishes agent gain, model gain, and their interaction while holding the evaluator fixed. |
| Curriculum | Task generation does not consume measured learning gaps; evolution reuses its selection holdout and writes mutation text rather than trainable executed experience. | Measured failures guide new families without exposing sealed release tests; accepted changes generate fresh verified trajectories, not synthetic claims of execution. |

These are independently reviewed source findings, supported by targeted CPU
reproductions. Existing targeted competition tests passed 94/94 and training,
promotion, evolution, and task-generation tests passed 114/114. The passing tests
do not cover the gaps above. The software doctor reports ready for CPU operation;
the default operator has no completed training artifacts.

## Agent × model experiment

Treat an agent bundle and model checkpoint as separate experimental factors:

| | Incumbent model M0 | Candidate model M1 |
| --- | --- | --- |
| Incumbent agent A0 | Reference Q00 | Model effect Q01 |
| Candidate agent A1 | Agent effect Q10 | Joint result Q11 |

For a predeclared higher-is-better metric Q, report agent gain `Q10 - Q00`, model
gain `Q01 - Q00`, and interaction `Q11 - Q10 - Q01 + Q00`. Use the same fixed
workload/evaluator, measurement budget, and compatible sampling settings. Carry
per-family correctness and cost/latency regressions alongside the aggregate.
Missing cells, noisy comparisons, or changed evaluator versions do not establish
a joint gain. These are experimental measurements, not a complete miner payout
formula and not causal attribution to every PR in a many-contributor batch.

The differentiation to test is whether independently useful agent changes produce
verified experience that improves the next model, and whether those gains transfer
to untouched task families. Compare agent-only, model-only, joint, and replay
ablations when actual inference/training resources are available. Novelty and
improvement must be demonstrated rather than asserted from this architecture.

## Continuous operation and proof boundary

The proposed durable cycle is:

```text
pin incumbent + evaluator + epoch
→ admit attributed PR/receipt
→ execute exact candidate on competition variants
→ evaluate and settle under one policy
→ admit licensed verified experience + bounded replay
→ prepare/train candidate from declared parent
→ evaluate old/new agent × old/new model
→ strict release decision
→ activate next epoch, or retain incumbent and record failure
→ select the next workload from measured gaps
```

CPU validation must exercise two cycles with explicitly labelled model/training
fixtures, real storage and CLI/API transitions, restart/replay, concurrent intake,
failed promotion, and exactly-once activation. Fixtures prove software behavior;
they cannot establish real learning or reproduce confidential hardware guarantees.
Use public development, private competition, and sealed release partitions with
family/version and exposure records. Retire disclosed variants and never relabel a
reused selection set as untouched release evaluation.

Production prerequisites remain visible: repository admission and reward-policy
agreement with SN74; contribution/data rights; verified corpus and fresh private
checks; exact candidate serving provenance; compatible training/serving stack;
and authorized compute for measured 4B, then 27B, learning cycles. Optional hardware
attestation requires actual hardware and approved measurements. None is satisfied
by CPU self-checks or a local scoring file.

## Sources

- [Gittensor repository registration](https://docs.gittensor.io/register-repository.html)
- [Gittensor OSS contribution scoring](https://docs.gittensor.io/oss-contributions.html)
- [Upstream repository reward registry](https://raw.githubusercontent.com/entrius/gittensor/main/gittensor/validator/weights/master_repositories.json)
- [SparkInfer scoring intent](https://raw.githubusercontent.com/gittensor-ai-lab/sparkinfer/main/.gittensor/weights.json)
- [SparkInfer label implementation](https://raw.githubusercontent.com/gittensor-ai-lab/sparkinfer/main/bench/scripts/label.py)
- [SparkInfer evaluation trust boundaries](https://raw.githubusercontent.com/gittensor-ai-lab/sparkinfer/main/EVAL-TRUST.md)
- [Official Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Official Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Qwen license](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE)
- [Hermes Agent license](https://raw.githubusercontent.com/NousResearch/hermes-agent/main/LICENSE)

Upstream main-branch references describe sources observed on the review date;
they do not prove the revision running on every live subnet validator.
